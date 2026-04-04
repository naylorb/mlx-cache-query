import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_corpus_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with sample text files."""
    (tmp_path / "readme.md").write_text("# Sample Project\n\nThis is a test document.\n")
    (tmp_path / "main.py").write_text("def hello():\n    return 'world'\n")
    (tmp_path / "notes.txt").write_text("Some notes about the project.\n")
    return tmp_path


@pytest.fixture
def tmp_store_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for cache artifact storage."""
    store = tmp_path / "store"
    store.mkdir()
    return store
