from mcq.compile.compiler import WikiCompiler
from mcq.core.types import Corpus, CorpusChunk


def _make_corpus():
    return Corpus(
        name="test-wiki",
        chunks=[
            CorpusChunk("readme.md", "# My Project\n\nA cool project.\n", (0, 30)),
            CorpusChunk("docs/guide.md", "# User Guide\n\nHow to use it.\n", (0, 29)),
            CorpusChunk("src/main.py", "def main():\n    pass\n", (0, 20)),
        ],
    )


def test_compile_adds_index():
    corpus = _make_corpus()
    compiled = WikiCompiler.compile(corpus)
    paths = [c.source_path for c in compiled.chunks]
    assert paths[0] == "_index.md"
    assert len(compiled.chunks) == len(corpus.chunks) + 1


def test_index_contains_file_listing():
    corpus = _make_corpus()
    compiled = WikiCompiler.compile(corpus)
    index = compiled.chunks[0]
    assert "readme.md" in index.content
    assert "docs/guide.md" in index.content
    assert "src/main.py" in index.content


def test_index_contains_corpus_name():
    corpus = _make_corpus()
    compiled = WikiCompiler.compile(corpus)
    index = compiled.chunks[0]
    assert "test-wiki" in index.content


def test_index_extracts_headings():
    corpus = _make_corpus()
    compiled = WikiCompiler.compile(corpus)
    index = compiled.chunks[0]
    assert "My Project" in index.content
    assert "User Guide" in index.content


def test_compile_preserves_original_chunks():
    corpus = _make_corpus()
    compiled = WikiCompiler.compile(corpus)
    original_paths = {c.source_path for c in corpus.chunks}
    compiled_paths = {c.source_path for c in compiled.chunks}
    assert original_paths.issubset(compiled_paths)


def test_compile_replaces_existing_index():
    corpus = Corpus(
        name="test",
        chunks=[
            CorpusChunk("_index.md", "old index", (0, 9)),
            CorpusChunk("a.txt", "content", (0, 7)),
        ],
    )
    compiled = WikiCompiler.compile(corpus)
    indexes = [c for c in compiled.chunks if c.source_path == "_index.md"]
    assert len(indexes) == 1
    assert "old index" not in indexes[0].content


def test_extract_tags_from_metadata():
    corpus = Corpus(
        name="test",
        chunks=[
            CorpusChunk("note.md", "# Note\nAbout #ai stuff.\n", (0, 24),
                        metadata={"tags": ["ml", "ai"]}),
        ],
    )
    compiled = WikiCompiler.compile(corpus)
    index = compiled.chunks[0]
    assert "Topics" in index.content
    assert "ai" in index.content
