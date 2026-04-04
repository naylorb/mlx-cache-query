"""End-to-end integration tests for the full mcq pipeline.

RED-GREEN TDD: These tests exercise the INTEGRATION between modules,
not individual units. They test the full CLI pipeline without needing mlx.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from mcq.cli import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_paths(tmp_path):
    db = tmp_path / "registry.db"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(exist_ok=True)
    return (
        patch("mcq.cli.REGISTRY_DB", db),
        patch("mcq.cli.ARTIFACTS_DIR", artifacts),
    )


def _create_obsidian_vault(tmp_path) -> Path:
    """Create a realistic Obsidian vault for testing."""
    vault = tmp_path / "vault"
    vault.mkdir()

    # .obsidian config (should be skipped)
    obs = vault / ".obsidian"
    obs.mkdir()
    (obs / "app.json").write_text('{"theme": "dark"}')

    # Notes with frontmatter and wikilinks
    (vault / "auth.md").write_text(
        "---\ntitle: Authentication\ntags: [security, auth]\n---\n"
        "# Authentication\n\nJWT-based auth middleware.\nSee [[Database]] for user storage.\n"
    )
    (vault / "database.md").write_text(
        "---\ntitle: Database\ntags: [backend]\n---\n"
        "# Database Layer\n\nPostgreSQL with pgbouncer connection pooling.\n"
    )
    (vault / "routes.md").write_text(
        "# API Routes\n\nGET /api/users — list users\nPOST /api/login — [[Authentication]]\n"
    )
    (vault / "empty.md").write_text("")
    (vault / "notes.txt").write_text("Some plain text notes.\n")

    return vault


def _invoke(runner, tmp_path, args):
    p1, p2 = _patch_paths(tmp_path)
    with p1, p2:
        return runner.invoke(main, args, catch_exceptions=False)


def _invoke_pipeline(runner, tmp_path, vault_path):
    """Run the standard ingest → compile pipeline."""
    p1, p2 = _patch_paths(tmp_path)
    with p1, p2:
        runner.invoke(main, ["ingest", str(vault_path), "-n", "test", "-f", "obsidian"])
        return runner


def _extract_json(output: str):
    """Extract JSON from mixed CLI output."""
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("{") or line.startswith("["):
            return json.loads(line)
    raise ValueError(f"No JSON found in: {output}")


# ---------------------------------------------------------------------------
# 1. Obsidian ingestion via CLI
# ---------------------------------------------------------------------------


def test_cli_ingest_obsidian_format(tmp_path):
    """RED: mcq ingest --format obsidian should skip .obsidian/ and parse frontmatter."""
    vault = _create_obsidian_vault(tmp_path)
    runner = CliRunner()
    result = _invoke(runner, tmp_path, ["--json", "ingest", str(vault), "-n", "test", "-f", "obsidian"])
    assert result.exit_code == 0
    data = _extract_json(result.output)
    assert data["name"] == "test"
    # Should have 4 files (auth.md, database.md, routes.md, empty.md, notes.txt)
    # but NOT .obsidian/app.json
    assert data["chunks"] == 5  # 4 .md + 1 .txt


# ---------------------------------------------------------------------------
# 2. Compile via CLI
# ---------------------------------------------------------------------------


def test_cli_compile(tmp_path):
    """RED: mcq compile should generate an index and return JSON."""
    vault = _create_obsidian_vault(tmp_path)
    runner = CliRunner()
    p1, p2 = _patch_paths(tmp_path)
    with p1, p2:
        runner.invoke(main, ["ingest", str(vault), "-n", "test", "-f", "obsidian"])
        result = runner.invoke(main, ["--json", "compile", "test"])
    assert result.exit_code == 0
    data = _extract_json(result.output)
    assert data["has_index"] is True
    assert data["chunks"] > 0


# ---------------------------------------------------------------------------
# 3. Lint via CLI
# ---------------------------------------------------------------------------


def test_cli_lint_finds_issues(tmp_path):
    """RED: mcq lint should detect empty files and broken wikilinks."""
    vault = _create_obsidian_vault(tmp_path)
    runner = CliRunner()
    p1, p2 = _patch_paths(tmp_path)
    with p1, p2:
        runner.invoke(main, ["ingest", str(vault), "-n", "test", "-f", "obsidian"])
        result = runner.invoke(main, ["--json", "lint", "test"])
    assert result.exit_code == 0
    issues = _extract_json(result.output)
    assert isinstance(issues, list)
    # Should find: empty.md is empty
    assert any(i["severity"] == "error" and "empty" in i["file"] for i in issues)


def test_cli_lint_unknown_corpus(tmp_path):
    """RED: mcq lint on unknown corpus should exit 1."""
    runner = CliRunner()
    result = _invoke(runner, tmp_path, ["lint", "nonexistent"])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# 4. Stats via CLI
# ---------------------------------------------------------------------------


def test_cli_stats(tmp_path):
    """RED: mcq stats should return file counts and word counts."""
    vault = _create_obsidian_vault(tmp_path)
    runner = CliRunner()
    p1, p2 = _patch_paths(tmp_path)
    with p1, p2:
        runner.invoke(main, ["ingest", str(vault), "-n", "test"])
        result = runner.invoke(main, ["--json", "stats", "test"])
    assert result.exit_code == 0
    data = _extract_json(result.output)
    assert data["total_files"] > 0
    assert data["total_words"] > 0
    assert ".md" in data["extensions"]


# ---------------------------------------------------------------------------
# 5. Export via CLI
# ---------------------------------------------------------------------------


def test_cli_export_markdown(tmp_path):
    """RED: mcq export --format markdown should produce concatenated markdown."""
    vault = _create_obsidian_vault(tmp_path)
    runner = CliRunner()
    p1, p2 = _patch_paths(tmp_path)
    with p1, p2:
        runner.invoke(main, ["ingest", str(vault), "-n", "test"])
        result = runner.invoke(main, ["export", "test", "-f", "markdown"])
    assert result.exit_code == 0
    assert "# test" in result.output
    assert "auth.md" in result.output


def test_cli_export_json(tmp_path):
    """RED: mcq export --format json should produce valid JSON."""
    vault = _create_obsidian_vault(tmp_path)
    runner = CliRunner()
    p1, p2 = _patch_paths(tmp_path)
    with p1, p2:
        runner.invoke(main, ["ingest", str(vault), "-n", "test"])
        result = runner.invoke(main, ["export", "test", "-f", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["name"] == "test"
    assert len(data["chunks"]) > 0


def test_cli_export_filelist(tmp_path):
    """RED: mcq export --format filelist should list files one per line."""
    vault = _create_obsidian_vault(tmp_path)
    runner = CliRunner()
    p1, p2 = _patch_paths(tmp_path)
    with p1, p2:
        runner.invoke(main, ["ingest", str(vault), "-n", "test"])
        result = runner.invoke(main, ["export", "test", "-f", "filelist"])
    assert result.exit_code == 0
    lines = result.output.strip().split("\n")
    assert len(lines) >= 4  # at least our 4 .md + 1 .txt files
    assert any("auth.md" in l for l in lines)


def test_cli_export_context(tmp_path):
    """RED: mcq export --format context should produce LLM-paste format."""
    vault = _create_obsidian_vault(tmp_path)
    runner = CliRunner()
    p1, p2 = _patch_paths(tmp_path)
    with p1, p2:
        runner.invoke(main, ["ingest", str(vault), "-n", "test"])
        result = runner.invoke(main, ["export", "test", "-f", "context"])
    assert result.exit_code == 0
    assert "<documents>" in result.output
    assert "</documents>" in result.output
    assert "<query>" not in result.output  # context, not prompt


# ---------------------------------------------------------------------------
# 6. Find via CLI
# ---------------------------------------------------------------------------


def test_cli_find_json(tmp_path):
    """RED: mcq find should return ranked results as JSON."""
    vault = _create_obsidian_vault(tmp_path)
    runner = CliRunner()
    p1, p2 = _patch_paths(tmp_path)
    with p1, p2:
        runner.invoke(main, ["ingest", str(vault), "-n", "test"])
        result = runner.invoke(main, ["--json", "find", "test", "authentication JWT"])
    assert result.exit_code == 0
    data = _extract_json(result.output)
    assert isinstance(data, list)
    assert len(data) > 0
    assert data[0]["source_path"] == "auth.md"  # most relevant


def test_cli_find_content_mode(tmp_path):
    """RED: mcq find --content should output full file content."""
    vault = _create_obsidian_vault(tmp_path)
    runner = CliRunner()
    p1, p2 = _patch_paths(tmp_path)
    with p1, p2:
        runner.invoke(main, ["ingest", str(vault), "-n", "test"])
        result = runner.invoke(main, ["find", "test", "authentication", "--content"])
    assert result.exit_code == 0
    assert "[== auth.md ==]" in result.output
    assert "JWT" in result.output


# ---------------------------------------------------------------------------
# 7. Sync via CLI
# ---------------------------------------------------------------------------


def test_cli_sync_init(tmp_path):
    """RED: mcq sync init should initialize git repo."""
    vault = _create_obsidian_vault(tmp_path)
    runner = CliRunner()
    p1, p2 = _patch_paths(tmp_path)
    with p1, p2:
        runner.invoke(main, ["ingest", str(vault), "-n", "test"])
        result = runner.invoke(main, ["--json", "sync", "test", "init"])
    assert result.exit_code == 0
    data = _extract_json(result.output)
    assert data["action"] == "initialized"


def test_cli_sync_status(tmp_path):
    """RED: mcq sync status should show repo state."""
    vault = _create_obsidian_vault(tmp_path)
    runner = CliRunner()
    p1, p2 = _patch_paths(tmp_path)
    with p1, p2:
        runner.invoke(main, ["ingest", str(vault), "-n", "test"])
        runner.invoke(main, ["sync", "test", "init"])
        result = runner.invoke(main, ["--json", "sync", "test", "status"])
    assert result.exit_code == 0
    data = _extract_json(result.output)
    assert data["is_repo"] is True


# ---------------------------------------------------------------------------
# 8. PrefixCompiler v2 — wiki-aware ordering
# ---------------------------------------------------------------------------


def test_prefix_compiler_index_first():
    """RED: PrefixCompiler should put _index.md first in the prompt."""
    from mcq.core.types import Corpus, CorpusChunk
    from mcq.prefix.compiler import PrefixCompiler

    corpus = Corpus(name="test", chunks=[
        CorpusChunk("z_last.md", "# Last file\n", (0, 13)),
        CorpusChunk("a_first.md", "# First file\n", (0, 14)),
        CorpusChunk("_index.md", "# Index\n\nMaster index.\n", (0, 23)),
    ])
    text = PrefixCompiler.build_prompt_text(corpus)
    idx_pos = text.index("_index.md")
    a_pos = text.index("a_first.md")
    z_pos = text.index("z_last.md")
    assert idx_pos < a_pos < z_pos, "_index.md should appear before all other files"


# ---------------------------------------------------------------------------
# 9. Full pipeline: ingest → compile → lint → find → export
# ---------------------------------------------------------------------------


def test_full_pipeline_without_mlx(tmp_path):
    """RED: Full pipeline should work end-to-end without mlx."""
    vault = _create_obsidian_vault(tmp_path)
    runner = CliRunner()
    p1, p2 = _patch_paths(tmp_path)

    with p1, p2:
        # Step 1: Ingest
        r = runner.invoke(main, ["--json", "ingest", str(vault), "-n", "demo", "-f", "obsidian"])
        assert r.exit_code == 0
        ingest_data = _extract_json(r.output)
        assert ingest_data["chunks"] > 0

        # Step 2: Compile
        r = runner.invoke(main, ["--json", "compile", "demo"])
        assert r.exit_code == 0
        compile_data = _extract_json(r.output)
        assert compile_data["has_index"] is True

        # Step 3: Lint
        r = runner.invoke(main, ["--json", "lint", "demo"])
        assert r.exit_code == 0
        lint_data = _extract_json(r.output)
        assert isinstance(lint_data, list)

        # Step 4: Find
        r = runner.invoke(main, ["--json", "find", "demo", "authentication"])
        assert r.exit_code == 0
        find_data = _extract_json(r.output)
        assert len(find_data) > 0

        # Step 5: Export
        r = runner.invoke(main, ["export", "demo", "-f", "json"])
        assert r.exit_code == 0
        export_data = json.loads(r.output)
        assert export_data["name"] == "demo"

        # Step 6: Stats
        r = runner.invoke(main, ["--json", "stats", "demo"])
        assert r.exit_code == 0
        stats_data = _extract_json(r.output)
        assert stats_data["total_files"] > 0

        # Step 7: List should show the corpus
        r = runner.invoke(main, ["--json", "list"])
        assert r.exit_code == 0
        list_data = _extract_json(r.output)
        assert any(c["corpus"] == "demo" for c in list_data)


# ---------------------------------------------------------------------------
# 10. Error handling
# ---------------------------------------------------------------------------


def test_compile_unknown_corpus(tmp_path):
    runner = CliRunner()
    result = _invoke(runner, tmp_path, ["compile", "nonexistent"])
    assert result.exit_code == 1


def test_stats_unknown_corpus(tmp_path):
    runner = CliRunner()
    result = _invoke(runner, tmp_path, ["stats", "nonexistent"])
    assert result.exit_code == 1


def test_export_unknown_corpus(tmp_path):
    runner = CliRunner()
    result = _invoke(runner, tmp_path, ["export", "nonexistent", "-f", "markdown"])
    assert result.exit_code == 1


def test_sync_unknown_corpus(tmp_path):
    runner = CliRunner()
    result = _invoke(runner, tmp_path, ["sync", "nonexistent", "status"])
    assert result.exit_code == 1


def test_find_unknown_corpus(tmp_path):
    runner = CliRunner()
    result = _invoke(runner, tmp_path, ["--json", "find", "nonexistent", "query"])
    assert result.exit_code == 1
