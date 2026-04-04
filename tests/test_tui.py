"""Tests for the TUI finder module — tests the app can be instantiated."""
import pytest

try:
    from textual.app import App
    HAS_TEXTUAL = True
except ImportError:
    HAS_TEXTUAL = False

from mcq.core.types import Corpus, CorpusChunk


pytestmark = pytest.mark.skipif(not HAS_TEXTUAL, reason="textual not installed")


def test_finder_app_creates():
    """Verify FinderApp can be instantiated."""
    from mcq.tui.finder import FinderApp
    corpus = Corpus(
        name="test",
        chunks=[
            CorpusChunk("readme.md", "# Hello\nWorld.\n", (0, 16)),
            CorpusChunk("main.py", "print('hi')\n", (0, 12)),
        ],
    )
    app = FinderApp(corpus, "test")
    assert app.corpus_name == "test"
    assert len(app.corpus.chunks) == 2


def test_run_finder_import():
    """Verify run_finder_tui is importable."""
    from mcq.tui.finder import run_finder_tui
    assert callable(run_finder_tui)
