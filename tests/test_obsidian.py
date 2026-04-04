from pathlib import Path
from mcq.ingest.ingestor import CorpusIngestor


def test_obsidian_skips_dotobsidian(tmp_path):
    obs = tmp_path / ".obsidian"
    obs.mkdir()
    (obs / "config.json").write_text('{"key": "val"}')
    (tmp_path / "note.md").write_text("# Hello\n")
    corpus = CorpusIngestor.ingest(tmp_path, name="test", format="obsidian")
    paths = [c.source_path for c in corpus.chunks]
    assert "note.md" in paths
    assert not any(".obsidian" in p for p in paths)


def test_obsidian_parses_frontmatter(tmp_path):
    (tmp_path / "note.md").write_text("---\ntitle: My Note\ntags: [ai, ml]\n---\n# Content\nSome text.\n")
    corpus = CorpusIngestor.ingest(tmp_path, name="test", format="obsidian")
    chunk = corpus.chunks[0]
    assert chunk.metadata is not None
    assert "frontmatter" in chunk.metadata
    assert chunk.metadata["frontmatter"]["title"] == "My Note"
    assert "ai" in chunk.metadata.get("tags", [])
    assert "ml" in chunk.metadata.get("tags", [])


def test_obsidian_extracts_wikilinks(tmp_path):
    (tmp_path / "note.md").write_text("See [[Other Note]] and [[Topic|alias]].\n")
    corpus = CorpusIngestor.ingest(tmp_path, name="test", format="obsidian")
    chunk = corpus.chunks[0]
    assert chunk.metadata is not None
    assert "Other Note" in chunk.metadata.get("wikilinks", [])
    assert "Topic" in chunk.metadata.get("wikilinks", [])


def test_obsidian_extracts_inline_tags(tmp_path):
    (tmp_path / "note.md").write_text("# Note\nThis is about #machine-learning and #ai.\n")
    corpus = CorpusIngestor.ingest(tmp_path, name="test", format="obsidian")
    chunk = corpus.chunks[0]
    assert chunk.metadata is not None
    assert "machine-learning" in chunk.metadata.get("tags", [])
    assert "ai" in chunk.metadata.get("tags", [])


def test_obsidian_frontmatter_excluded_from_content(tmp_path):
    (tmp_path / "note.md").write_text("---\ntitle: Test\n---\n# Content\nBody text.\n")
    corpus = CorpusIngestor.ingest(tmp_path, name="test", format="obsidian")
    chunk = corpus.chunks[0]
    assert "---" not in chunk.content or chunk.content.index("---") > 5
    assert "Body text." in chunk.content


def test_auto_format_still_works(tmp_path):
    (tmp_path / "test.py").write_text("x = 1\n")
    corpus = CorpusIngestor.ingest(tmp_path, name="test")
    assert len(corpus.chunks) == 1


def test_qmd_extension_supported(tmp_path):
    (tmp_path / "paper.qmd").write_text("---\ntitle: Paper\n---\n# Introduction\nHello.\n")
    corpus = CorpusIngestor.ingest(tmp_path, name="test")
    assert len(corpus.chunks) == 1
