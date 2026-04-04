from mcq.core.types import Corpus, CorpusChunk
from mcq.search.finder import TextFinder, SearchResult


def _make_corpus():
    return Corpus(
        name="test",
        chunks=[
            CorpusChunk("auth.py", "def authenticate(user, password):\n    '''Authenticate a user.'''\n    return check_credentials(user, password)\n", (0, 100)),
            CorpusChunk("readme.md", "# Project\n\nThis is a web application for managing tasks.\n", (0, 60)),
            CorpusChunk("db.py", "def connect_database():\n    '''Connect to the database (PostgreSQL).'''\n    return psycopg2.connect(DSN)\n", (0, 90)),
            CorpusChunk("routes.py", "def login_route(request):\n    user = authenticate(request.user, request.password)\n    return redirect('/dashboard')\n", (0, 120)),
        ],
    )


def test_search_returns_ranked_results():
    corpus = _make_corpus()
    results = TextFinder.search(corpus, "authenticate user")
    assert len(results) > 0
    assert isinstance(results[0], SearchResult)
    # auth.py and routes.py should rank highest (both mention authenticate)
    top_paths = [r.chunk.source_path for r in results[:2]]
    assert "auth.py" in top_paths


def test_search_returns_snippets():
    corpus = _make_corpus()
    results = TextFinder.search(corpus, "database")
    assert len(results) > 0
    assert len(results[0].snippet) > 0


def test_search_respects_top_k():
    corpus = _make_corpus()
    results = TextFinder.search(corpus, "the", top_k=2)
    assert len(results) <= 2


def test_search_empty_query():
    corpus = _make_corpus()
    results = TextFinder.search(corpus, "")
    assert results == []


def test_search_no_matches():
    corpus = _make_corpus()
    results = TextFinder.search(corpus, "xyznonexistentterm")
    assert results == []


def test_search_scores_decrease():
    corpus = _make_corpus()
    results = TextFinder.search(corpus, "authenticate user password")
    if len(results) > 1:
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score
