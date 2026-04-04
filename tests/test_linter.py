from mcq.compile.linter import WikiLinter, LintIssue
from mcq.core.types import Corpus, CorpusChunk


def test_lint_empty_file():
    corpus = Corpus(name="test", chunks=[
        CorpusChunk("empty.md", "", (0, 0)),
        CorpusChunk("good.md", "# Hello\n\nContent here.\n", (0, 24)),
    ])
    issues = WikiLinter.lint(corpus)
    errors = [i for i in issues if i.severity == "error"]
    assert any("empty" in i.message.lower() for i in errors)


def test_lint_broken_wikilinks():
    corpus = Corpus(name="test", chunks=[
        CorpusChunk("note.md", "See [[NonExistent]] for details.\n", (0, 33)),
        CorpusChunk("other.md", "# Other\nContent.\n", (0, 18)),
    ])
    issues = WikiLinter.lint(corpus)
    warnings = [i for i in issues if i.severity == "warning"]
    assert any("NonExistent" in i.message for i in warnings)


def test_lint_valid_wikilinks():
    corpus = Corpus(name="test", chunks=[
        CorpusChunk("note.md", "See [[other]] for details.\n", (0, 27)),
        CorpusChunk("other.md", "# Other\nContent.\n", (0, 18)),
    ])
    issues = WikiLinter.lint(corpus)
    wikilink_issues = [i for i in issues if "wikilink" in i.message.lower()]
    assert len(wikilink_issues) == 0


def test_lint_large_file():
    big_content = "word " * 5000  # ~25000 chars
    corpus = Corpus(name="test", chunks=[
        CorpusChunk("big.md", f"# Big\n{big_content}", (0, len(big_content))),
    ])
    issues = WikiLinter.lint(corpus)
    info_issues = [i for i in issues if i.severity == "info"]
    assert any("large" in i.message.lower() for i in info_issues)


def test_lint_no_headings():
    corpus = Corpus(name="test", chunks=[
        CorpusChunk("flat.md", "Just some text without any structure.\n", (0, 38)),
    ])
    issues = WikiLinter.lint(corpus)
    assert any("heading" in i.message.lower() for i in issues)


def test_lint_clean_corpus():
    corpus = Corpus(name="test", chunks=[
        CorpusChunk("readme.md", "# Readme\n\nThis is a good document with content.\n", (0, 48)),
        CorpusChunk("guide.md", "# Guide\n\nAnother well-structured document.\n", (0, 43)),
    ])
    issues = WikiLinter.lint(corpus)
    errors = [i for i in issues if i.severity == "error"]
    assert len(errors) == 0


def test_lint_duplicate_content():
    content = "# Topic\n\n" + " ".join(f"word{i} is interesting" for i in range(50))
    corpus = Corpus(name="test", chunks=[
        CorpusChunk("file1.md", content, (0, len(content))),
        CorpusChunk("file2.md", content, (0, len(content))),
    ])
    issues = WikiLinter.lint(corpus)
    assert any("similar" in i.message.lower() for i in issues)
