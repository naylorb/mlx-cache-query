from mcq.core.types import Corpus, CorpusChunk
from mcq.export.exporter import CorpusExporter
import json


def _make_corpus():
    return Corpus(
        name="test",
        chunks=[
            CorpusChunk("readme.md", "# Hello\n\nWorld.\n", (0, 17)),
            CorpusChunk("main.py", "print('hi')\n", (0, 12)),
        ],
    )


def test_export_markdown():
    corpus = _make_corpus()
    md = CorpusExporter.to_markdown(corpus)
    assert "# test" in md
    assert "## main.py" in md
    assert "## readme.md" in md
    assert "print('hi')" in md


def test_export_json():
    corpus = _make_corpus()
    j = CorpusExporter.to_json(corpus)
    data = json.loads(j)
    assert data["name"] == "test"
    assert len(data["chunks"]) == 2
    assert data["chunks"][0]["source_path"] in ("main.py", "readme.md")


def test_export_context():
    corpus = _make_corpus()
    ctx = CorpusExporter.to_context(corpus)
    assert "<documents>" in ctx
    assert "</documents>" in ctx
    assert "[== main.py ==]" in ctx
    assert "<query>" not in ctx  # context format, not prompt


def test_export_filelist():
    corpus = _make_corpus()
    fl = CorpusExporter.to_filelist(corpus)
    lines = fl.strip().split("\n")
    assert len(lines) == 2
    assert "main.py" in lines
    assert "readme.md" in lines
