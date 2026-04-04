"""Tests for the mcq CLI.

Exercises the CLI surface without requiring mlx or Apple Silicon.
Registry and ingestor are exercised for real; model loading is patched.
"""
from __future__ import annotations

import json
from unittest.mock import patch

from click.testing import CliRunner

from mcq.cli import main


def _patch_paths(tmp_path):
    db = tmp_path / "registry.db"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(exist_ok=True)
    return (
        patch("mcq.cli.REGISTRY_DB", db),
        patch("mcq.cli.ARTIFACTS_DIR", artifacts),
    )


def _prepare_source(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "hello.txt").write_text("hello world\n")
    return src


def _invoke(runner, tmp_path, args):
    p1, p2 = _patch_paths(tmp_path)
    with p1, p2:
        return runner.invoke(main, args)


def _ingest_corpus(runner, tmp_path, src, name="test"):
    """Helper: ingest a corpus via the internal registry (bypassing build)."""
    from mcq.cache.registry import CacheRegistry
    from mcq.ingest.ingestor import CorpusIngestor

    db = tmp_path / "registry.db"
    corpus = CorpusIngestor.ingest(src, name=name)
    reg = CacheRegistry(db)
    reg.register_corpus(
        name=name,
        source_path=str(src),
        content_hash=corpus.content_hash,
        chunk_count=len(corpus.chunks),
    )


# ── help ─────────────────────────────────────────────────────────────


def test_help_shows_all_commands():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    for cmd in ("build", "query", "list", "info", "delete", "find", "gc", "verify"):
        assert cmd in result.output


# ── list ─────────────────────────────────────────────────────────────


def test_list_empty(tmp_path):
    runner = CliRunner()
    result = _invoke(runner, tmp_path, ["list"])
    assert result.exit_code == 0
    assert "No corpora" in result.output


def test_list_json_empty(tmp_path):
    runner = CliRunner()
    result = _invoke(runner, tmp_path, ["--json", "list"])
    assert result.exit_code == 0
    data = _extract_json_array(result.output)
    assert data == []


def test_list_after_ingest(tmp_path):
    src = _prepare_source(tmp_path)
    runner = CliRunner()
    p1, p2 = _patch_paths(tmp_path)
    with p1, p2:
        _ingest_corpus(runner, tmp_path, src, name="mylib")
        result = runner.invoke(main, ["list"])
    assert result.exit_code == 0
    assert "mylib" in result.output


def test_list_json_after_ingest(tmp_path):
    src = _prepare_source(tmp_path)
    runner = CliRunner()
    p1, p2 = _patch_paths(tmp_path)
    with p1, p2:
        _ingest_corpus(runner, tmp_path, src, name="mylib")
        result = runner.invoke(main, ["--json", "list"])
    assert result.exit_code == 0
    data = _extract_json_array(result.output)
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["corpus"] == "mylib"
    assert data[0]["status"] == "ingested"


# ── info ─────────────────────────────────────────────────────────────


def test_info_unknown_exits_nonzero(tmp_path):
    runner = CliRunner()
    result = _invoke(runner, tmp_path, ["info", "nonexistent"])
    assert result.exit_code != 0


# ── delete ───────────────────────────────────────────────────────────


def test_delete_unknown_exits_nonzero(tmp_path):
    runner = CliRunner()
    result = _invoke(runner, tmp_path, ["delete", "--force", "nonexistent"])
    assert result.exit_code != 0


# ── query ────────────────────────────────────────────────────────────


def test_query_unknown_exits_nonzero(tmp_path):
    runner = CliRunner()
    result = _invoke(runner, tmp_path, ["query", "nonexistent", "hello?"])
    assert result.exit_code != 0


# ── find ─────────────────────────────────────────────────────────────


def test_find_against_ingested_corpus(tmp_path):
    src = _prepare_source(tmp_path)
    runner = CliRunner()
    p1, p2 = _patch_paths(tmp_path)
    with p1, p2:
        _ingest_corpus(runner, tmp_path, src, name="proj")
        result = runner.invoke(main, ["--json", "find", "proj", "hello"])
    assert result.exit_code == 0
    data = _extract_json_array(result.output)
    assert len(data) > 0
    assert data[0]["source_path"] == "hello.txt"


def test_find_unknown_corpus(tmp_path):
    runner = CliRunner()
    result = _invoke(runner, tmp_path, ["find", "nonexistent", "test"])
    assert result.exit_code != 0


# ── gc ───────────────────────────────────────────────────────────────


def test_gc_empty(tmp_path):
    runner = CliRunner()
    result = _invoke(runner, tmp_path, ["gc"])
    assert result.exit_code == 0
    assert "Nothing" in result.output


def test_gc_json_empty(tmp_path):
    runner = CliRunner()
    result = _invoke(runner, tmp_path, ["--json", "gc"])
    assert result.exit_code == 0
    data = _extract_json_object(result.output)
    assert data["removed"] == []
    assert data["bytes_freed"] == 0


# ── verify ───────────────────────────────────────────────────────────


def test_verify_unknown_exits_nonzero(tmp_path):
    runner = CliRunner()
    result = _invoke(runner, tmp_path, ["verify", "nonexistent"])
    assert result.exit_code != 0


# ── corpus name validation ────────────────────────────────────────────


def test_build_rejects_invalid_corpus_name(tmp_path):
    src = _prepare_source(tmp_path)
    runner = CliRunner()
    for bad_name in ["my corpus", "../etc", "", "a/b"]:
        result = _invoke(runner, tmp_path, ["build", str(src), "-n", bad_name])
        assert result.exit_code != 0, f"Expected rejection for name '{bad_name}'"


def test_build_accepts_valid_corpus_names(tmp_path):
    """Valid names should pass validation (build will fail later at model load, that's ok)."""
    src = _prepare_source(tmp_path)
    runner = CliRunner()
    for good_name in ["my-project", "docs_v2", "project.2024"]:
        result = _invoke(runner, tmp_path, ["build", str(src), "-n", good_name])
        # Will fail at model loading (exit 3) — but NOT at name validation (exit 1)
        assert result.exit_code != 1 or "Invalid name" not in result.output


# ── stats ────────────────────────────────────────────────────────────


def test_stats_against_ingested_corpus(tmp_path):
    src = _prepare_source(tmp_path)
    runner = CliRunner()
    p1, p2 = _patch_paths(tmp_path)
    with p1, p2:
        _ingest_corpus(runner, tmp_path, src, name="proj")
        result = runner.invoke(main, ["--json", "stats", "proj"])
    assert result.exit_code == 0
    data = _extract_json_object(result.output)
    assert data["total_files"] == 1
    assert data["total_words"] > 0


# ── helpers ──────────────────────────────────────────────────────────


def _extract_json_object(text: str) -> dict:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise ValueError(f"No JSON object found in output:\n{text}")


def _extract_json_array(text: str) -> list:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("["):
            return json.loads(line)
    raise ValueError(f"No JSON array found in output:\n{text}")
