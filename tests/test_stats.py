from mcq.core.stats import CorpusStats
from mcq.core.types import Corpus, CorpusChunk


def test_corpus_stats():
    corpus = Corpus(
        name="test",
        chunks=[
            CorpusChunk("readme.md", "# Hello\n\nWorld.\n", (0, 17)),
            CorpusChunk("main.py", "def foo():\n    return 42\n", (0, 25)),
            CorpusChunk("lib.py", "x = 1\n", (0, 6)),
        ],
    )
    stats = CorpusStats.from_corpus(corpus)
    assert stats.name == "test"
    assert stats.total_files == 3
    assert stats.total_lines == 6
    assert stats.total_words > 0
    assert ".py" in stats.extensions
    assert ".md" in stats.extensions
    assert stats.extensions[".py"] == 2
    assert stats.extensions[".md"] == 1
    assert stats.largest_file == "main.py"


def test_empty_corpus_stats():
    corpus = Corpus(name="empty", chunks=[])
    stats = CorpusStats.from_corpus(corpus)
    assert stats.total_files == 0
    assert stats.total_bytes == 0
