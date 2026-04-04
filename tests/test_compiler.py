from unittest.mock import MagicMock

from mcq.core.types import Corpus, CorpusChunk, TokenizedPrefix
from mcq.prefix.compiler import PrefixCompiler


def _make_corpus() -> Corpus:
    return Corpus(
        name="test",
        chunks=[
            CorpusChunk("readme.md", "# Hello\n\nWorld.\n", (0, 17)),
            CorpusChunk("main.py", "print('hi')\n", (0, 12)),
        ],
    )


def _make_mock_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.encode.side_effect = lambda text, **_: list(range(len(text) // 2))
    tok.apply_chat_template = None  # force raw template path
    return tok


def test_build_prompt_text():
    corpus = _make_corpus()
    text = PrefixCompiler.build_prompt_text(corpus)
    assert "[== readme.md ==]" in text
    assert "[== main.py ==]" in text
    assert "# Hello" in text
    assert "print('hi')" in text
    assert text.index("main.py") < text.index("readme.md")  # sorted lexicographically


def test_build_prompt_text_sorted():
    corpus = Corpus(
        name="test",
        chunks=[
            CorpusChunk("z.txt", "last", (0, 4)),
            CorpusChunk("a.txt", "first", (0, 5)),
        ],
    )
    text = PrefixCompiler.build_prompt_text(corpus)
    assert text.index("a.txt") < text.index("z.txt")


def test_compile_returns_tokenized_prefix():
    corpus = _make_corpus()
    tok = _make_mock_tokenizer()
    prefix = PrefixCompiler.compile(
        corpus=corpus,
        tokenizer=tok,
        model_id="test-model",
        model_revision="rev123",
    )
    assert isinstance(prefix, TokenizedPrefix)
    assert prefix.model_id == "test-model"
    assert prefix.model_revision == "rev123"
    assert prefix.token_count > 0
    assert prefix.corpus_hash == corpus.content_hash


def test_compile_raises_on_context_overflow():
    corpus = _make_corpus()
    tok = _make_mock_tokenizer()
    try:
        PrefixCompiler.compile(
            corpus=corpus,
            tokenizer=tok,
            model_id="test-model",
            model_revision="rev123",
            max_context=10,
            query_budget=5,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "exceeds" in str(e).lower()
