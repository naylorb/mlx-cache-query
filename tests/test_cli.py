"""Tests for the mcq CLI using Click's CliRunner.

These tests exercise the CLI surface (help, ingest, list, info, delete, query)
without requiring mlx or Apple Silicon -- only the registry and ingestor are
exercised for real; everything else is patched.
"""
from __future__ import annotations

import json
from unittest.mock import patch

from click.testing import CliRunner

from mcq.cli import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_paths(tmp_path):
    """Return context managers that redirect REGISTRY_DB and ARTIFACTS_DIR."""
    db = tmp_path / "registry.db"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(exist_ok=True)
    return (
        patch("mcq.cli.REGISTRY_DB", db),
        patch("mcq.cli.ARTIFACTS_DIR", artifacts),
    )


def _prepare_source(tmp_path):
    """Create a tiny source tree with one .txt file."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "hello.txt").write_text("hello world\n")


def _invoke(runner, tmp_path, args):
    """Invoke CLI with patched paths."""
    p1, p2 = _patch_paths(tmp_path)
    with p1, p2:
        return runner.invoke(main, args)


# ---------------------------------------------------------------------------
# 1. Help
# ---------------------------------------------------------------------------


def test_help_shows_all_commands():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    for cmd in ("ingest", "build", "query", "list", "info", "delete"):
        assert cmd in result.output


# ---------------------------------------------------------------------------
# 2. ingest (human mode)
# ---------------------------------------------------------------------------


def test_ingest_creates_corpus(tmp_path):
    _prepare_source(tmp_path)
    runner = CliRunner()
    result = _invoke(runner, tmp_path, ["ingest", str(tmp_path / "src"), "-n", "test"])
    assert result.exit_code == 0
    # Status output goes to stderr; CliRunner mixes it into result.output
    assert "test" in result.output
    assert "Ingested" in result.output or "files" in result.output


# ---------------------------------------------------------------------------
# 3. ingest --json
# ---------------------------------------------------------------------------


def test_ingest_json_output(tmp_path):
    _prepare_source(tmp_path)
    runner = CliRunner()
    result = _invoke(
        runner, tmp_path, ["--json", "ingest", str(tmp_path / "src"), "-n", "test"]
    )
    assert result.exit_code == 0
    # JSON line is in the output -- find it among possible status lines
    data = _extract_json_object(result.output)
    assert data["name"] == "test"
    assert data["chunks"] == 1
    assert "hash" in data
    assert "source" in data
    assert "elapsed_s" in data
    assert isinstance(data["elapsed_s"], (int, float))


# ---------------------------------------------------------------------------
# 4. list (empty)
# ---------------------------------------------------------------------------


def test_list_empty(tmp_path):
    runner = CliRunner()
    result = _invoke(runner, tmp_path, ["list"])
    assert result.exit_code == 0
    assert "No corpora" in result.output


# ---------------------------------------------------------------------------
# 5. list after ingest
# ---------------------------------------------------------------------------


def test_list_after_ingest(tmp_path):
    _prepare_source(tmp_path)
    runner = CliRunner()
    p1, p2 = _patch_paths(tmp_path)
    with p1, p2:
        runner.invoke(main, ["ingest", str(tmp_path / "src"), "-n", "mylib"])
        result = runner.invoke(main, ["list"])
    assert result.exit_code == 0
    assert "mylib" in result.output
    assert "ingested" in result.output.lower()


# ---------------------------------------------------------------------------
# 6. list --json after ingest
# ---------------------------------------------------------------------------


def test_list_json_after_ingest(tmp_path):
    _prepare_source(tmp_path)
    runner = CliRunner()
    p1, p2 = _patch_paths(tmp_path)
    with p1, p2:
        runner.invoke(main, ["ingest", str(tmp_path / "src"), "-n", "mylib"])
        result = runner.invoke(main, ["--json", "list"])
    assert result.exit_code == 0
    data = _extract_json_array(result.output)
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["corpus"] == "mylib"
    assert data[0]["status"] == "ingested"
    assert data[0]["chunks"] == 1


# ---------------------------------------------------------------------------
# 7. info unknown -> exit 1
# ---------------------------------------------------------------------------


def test_info_unknown_exits_1(tmp_path):
    runner = CliRunner()
    result = _invoke(runner, tmp_path, ["info", "nonexistent"])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# 8. delete unknown -> exit 1
# ---------------------------------------------------------------------------


def test_delete_unknown_exits_1(tmp_path):
    runner = CliRunner()
    result = _invoke(runner, tmp_path, ["delete", "nonexistent"])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# 9. query unknown -> exit 1
# ---------------------------------------------------------------------------


def test_query_unknown_exits_1(tmp_path):
    runner = CliRunner()
    result = _invoke(runner, tmp_path, ["query", "nonexistent", "hello?"])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# 10. list --json when empty returns []
# ---------------------------------------------------------------------------


def test_list_json_empty(tmp_path):
    runner = CliRunner()
    result = _invoke(runner, tmp_path, ["--json", "list"])
    assert result.exit_code == 0
    data = _extract_json_array(result.output)
    assert data == []


# ---------------------------------------------------------------------------
# 11. ingest with --quiet suppresses status
# ---------------------------------------------------------------------------


def test_ingest_quiet_suppresses_status(tmp_path):
    _prepare_source(tmp_path)
    runner = CliRunner()
    result = _invoke(
        runner, tmp_path, ["--quiet", "ingest", str(tmp_path / "src"), "-n", "test"]
    )
    assert result.exit_code == 0
    # With --quiet, the human-facing status lines should be suppressed.
    # stdout should be empty (no --json flag), and status was silenced.
    # Note: Rich Console quiet mode suppresses print calls.
    assert "Ingested" not in result.output


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------


def _extract_json_object(text: str) -> dict:
    """Find and parse the first JSON object ({...}) in mixed output."""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise ValueError(f"No JSON object found in output:\n{text}")


def _extract_json_array(text: str) -> list:
    """Find and parse the first JSON array ([...]) in mixed output."""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("["):
            return json.loads(line)
    raise ValueError(f"No JSON array found in output:\n{text}")
