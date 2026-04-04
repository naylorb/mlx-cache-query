from pathlib import Path
from mcq.core.learnings import LearningStore


def test_record_and_retrieve_query(tmp_path):
    db = tmp_path / "learn.db"
    store = LearningStore(db)
    store.record_query("brain", "what is auth?", source_path="auth.py", relevance_score=0.9)
    stats = store.get_stats("brain")
    assert stats["total_learnings"] == 1


def test_relevant_sources(tmp_path):
    db = tmp_path / "learn.db"
    store = LearningStore(db)
    store.record_query("brain", "authentication login", source_path="auth.py", relevance_score=0.9)
    store.record_query("brain", "database connection", source_path="db.py", relevance_score=0.8)
    store.record_query("brain", "auth middleware", source_path="middleware.py", relevance_score=0.7)

    # Query about "authentication" should surface auth.py and middleware.py
    sources = store.get_relevant_sources("brain", ["authentication", "user"])
    assert "auth.py" in sources


def test_corrections(tmp_path):
    db = tmp_path / "learn.db"
    store = LearningStore(db)
    store.record_correction("brain", "What is X?", "Actually X means Y")
    corrections = store.get_corrections("brain")
    assert len(corrections) == 1
    assert corrections[0].correction == "Actually X means Y"


def test_stats(tmp_path):
    db = tmp_path / "learn.db"
    store = LearningStore(db)
    store.record_query("brain", "q1", relevance_score=0.5)
    store.record_query("brain", "q2", relevance_score=0.6)
    store.record_correction("brain", "q1", "correction")
    stats = store.get_stats("brain")
    assert stats["total_learnings"] == 2
    assert stats["corrections"] == 1
    assert stats["unique_queries"] == 2


def test_clear(tmp_path):
    db = tmp_path / "learn.db"
    store = LearningStore(db)
    store.record_query("brain", "q1")
    store.record_query("other", "q2")
    store.clear("brain")
    assert store.get_stats("brain")["total_learnings"] == 0
    assert store.get_stats("other")["total_learnings"] == 1


def test_empty_relevant_sources(tmp_path):
    db = tmp_path / "learn.db"
    store = LearningStore(db)
    sources = store.get_relevant_sources("brain", ["anything"])
    assert sources == []
