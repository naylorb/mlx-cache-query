"""Tests for the mcq GUI app."""
import pytest

try:
    from textual.app import App
    HAS_TEXTUAL = True
except ImportError:
    HAS_TEXTUAL = False

pytestmark = pytest.mark.skipif(not HAS_TEXTUAL, reason="textual not installed")


def test_app_creates():
    from mcq.tui.app import McqApp
    app = McqApp()
    assert app.TITLE == "mcq"


def test_run_app_import():
    from mcq.tui.app import run_app
    assert callable(run_app)
