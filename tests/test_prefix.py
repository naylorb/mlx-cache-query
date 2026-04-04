"""Tests for the prefix compiler and prompt templates."""
from mcq.core.types import Corpus, CorpusChunk
from mcq.prefix.compiler import PrefixCompiler, TEMPLATES


def _make_corpus():
    return Corpus(
        name="test",
        chunks=[
            CorpusChunk("readme.md", "# Hello\n\nWorld.\n", (0, 17)),
            CorpusChunk("main.py", "def foo():\n    return 42\n", (0, 25)),
        ],
    )


def test_build_prompt_text_contains_documents():
    corpus = _make_corpus()
    text = PrefixCompiler.build_prompt_text(corpus)
    assert "<documents>" in text
    assert "</documents>" in text
    assert "<query>" in text
    assert "[== readme.md ==]" in text
    assert "[== main.py ==]" in text
    assert "# Hello" in text
    assert "def foo():" in text


def test_build_prompt_text_sorted_order():
    corpus = _make_corpus()
    text = PrefixCompiler.build_prompt_text(corpus)
    main_pos = text.index("[== main.py ==]")
    readme_pos = text.index("[== readme.md ==]")
    assert main_pos < readme_pos  # alphabetical


def test_templates_exist():
    assert "general" in TEMPLATES
    assert "code" in TEMPLATES
    assert "raw" in TEMPLATES


def test_code_template_includes_preamble():
    corpus = _make_corpus()
    text = PrefixCompiler.build_prompt_text(corpus, template="code")
    assert "code expert" in text.lower()


def test_raw_template_minimal():
    corpus = _make_corpus()
    text = PrefixCompiler.build_prompt_text(corpus, template="raw")
    assert text.startswith("<documents>")


def test_general_template_default():
    corpus = _make_corpus()
    text = PrefixCompiler.build_prompt_text(corpus, template="general")
    assert "answering questions" in text.lower()
