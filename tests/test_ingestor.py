from pathlib import Path

from mcq.core.types import Corpus
from mcq.ingest.ingestor import CorpusIngestor


def test_ingest_single_file(tmp_path: Path):
    f = tmp_path / "hello.txt"
    f.write_text("Hello, world!\n")
    corpus = CorpusIngestor.ingest(f, name="test")
    assert isinstance(corpus, Corpus)
    assert corpus.name == "test"
    assert len(corpus.chunks) == 1
    assert corpus.chunks[0].source_path == "hello.txt"
    assert corpus.chunks[0].content == "Hello, world!\n"


def test_ingest_directory(tmp_corpus_dir: Path):
    corpus = CorpusIngestor.ingest(tmp_corpus_dir, name="project")
    assert len(corpus.chunks) == 3
    paths = [c.source_path for c in corpus.chunks]
    assert paths == sorted(paths)  # deterministic order


def test_ingest_filters_unsupported_extensions(tmp_path: Path):
    (tmp_path / "good.py").write_text("x = 1\n")
    (tmp_path / "bad.bin").write_bytes(b"\x00\x01\x02")
    corpus = CorpusIngestor.ingest(tmp_path, name="filtered")
    assert len(corpus.chunks) == 1
    assert corpus.chunks[0].source_path == "good.py"


def test_ingest_normalizes_line_endings(tmp_path: Path):
    f = tmp_path / "crlf.txt"
    f.write_bytes(b"line one\r\nline two\r\n")
    corpus = CorpusIngestor.ingest(f, name="crlf")
    assert "\r" not in corpus.chunks[0].content
    assert corpus.chunks[0].content == "line one\nline two\n"


def test_ingest_strips_trailing_whitespace(tmp_path: Path):
    f = tmp_path / "trailing.txt"
    f.write_text("hello   \nworld  \n")
    corpus = CorpusIngestor.ingest(f, name="trailing")
    assert corpus.chunks[0].content == "hello\nworld\n"


def test_ingest_preserves_indentation(tmp_path: Path):
    f = tmp_path / "indented.py"
    f.write_text("def foo():\n    return 42\n")
    corpus = CorpusIngestor.ingest(f, name="indent")
    assert corpus.chunks[0].content == "def foo():\n    return 42\n"


def test_ingest_deterministic_hash(tmp_corpus_dir: Path):
    a = CorpusIngestor.ingest(tmp_corpus_dir, name="a")
    b = CorpusIngestor.ingest(tmp_corpus_dir, name="b")
    assert a.content_hash == b.content_hash


def test_ingest_respects_gitignore(tmp_path: Path):
    (tmp_path / ".gitignore").write_text("*.log\nbuild/\n")
    (tmp_path / "main.py").write_text("x = 1\n")
    (tmp_path / "debug.log").write_text("log stuff\n")
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "output.py").write_text("y = 2\n")
    corpus = CorpusIngestor.ingest(tmp_path, name="gitignore")
    paths = [c.source_path for c in corpus.chunks]
    assert "main.py" in paths
    assert "debug.log" not in paths
    assert "build/output.py" not in paths


def test_ingest_empty_directory(tmp_path: Path):
    corpus = CorpusIngestor.ingest(tmp_path, name="empty")
    assert len(corpus.chunks) == 0
