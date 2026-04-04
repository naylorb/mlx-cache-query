"""Learning system — mcq gets better over time.

Tracks query patterns, source relevance, and user corrections.
Stored in SQLite alongside the registry. Each learning is a
signal that biases future search and query routing.

Inspired by Pal (agno-agi/pal) — every interaction improves the next.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS learnings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    corpus_name TEXT NOT NULL,
    query TEXT NOT NULL,
    source_path TEXT,
    relevance_score REAL DEFAULT 0.0,
    user_feedback TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    corpus_name TEXT NOT NULL,
    original_query TEXT NOT NULL,
    correction TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_learnings_corpus ON learnings(corpus_name);
CREATE INDEX IF NOT EXISTS idx_learnings_query ON learnings(query);
"""


@dataclass
class Learning:
    corpus_name: str
    query: str
    source_path: str | None
    relevance_score: float
    user_feedback: str | None
    created_at: str


@dataclass
class Correction:
    corpus_name: str
    original_query: str
    correction: str
    created_at: str


class LearningStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def record_query(
        self,
        corpus_name: str,
        query: str,
        source_path: str | None = None,
        relevance_score: float = 0.0,
        user_feedback: str | None = None,
    ) -> None:
        """Record a query and its result for future learning."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO learnings (corpus_name, query, source_path, relevance_score, user_feedback, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (corpus_name, query, source_path, relevance_score, user_feedback, now),
        )
        self._conn.commit()

    def record_correction(
        self,
        corpus_name: str,
        original_query: str,
        correction: str,
    ) -> None:
        """Record a user correction — these always override learned patterns."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO corrections (corpus_name, original_query, correction, created_at) "
            "VALUES (?, ?, ?, ?)",
            (corpus_name, original_query, correction, now),
        )
        self._conn.commit()

    def get_relevant_sources(
        self, corpus_name: str, query_terms: list[str], top_k: int = 5
    ) -> list[str]:
        """Get source paths that were relevant to similar queries in the past.

        Uses simple term overlap to find related past queries, then returns
        the sources that scored highest for those queries.
        """
        # Get all learnings for this corpus
        rows = self._conn.execute(
            "SELECT query, source_path, relevance_score FROM learnings "
            "WHERE corpus_name = ? AND source_path IS NOT NULL "
            "ORDER BY relevance_score DESC",
            (corpus_name,),
        ).fetchall()

        if not rows:
            return []

        # Score each past learning by term overlap with current query
        source_scores: dict[str, float] = {}
        query_set = set(t.lower() for t in query_terms)

        for past_query, source_path, score in rows:
            past_terms = set(past_query.lower().split())
            overlap = len(query_set & past_terms)
            if overlap > 0:
                boost = overlap * score
                source_scores[source_path] = source_scores.get(source_path, 0) + boost

        sorted_sources = sorted(source_scores.items(), key=lambda x: -x[1])
        return [s for s, _ in sorted_sources[:top_k]]

    def get_corrections(self, corpus_name: str) -> list[Correction]:
        """Get all user corrections for a corpus."""
        rows = self._conn.execute(
            "SELECT corpus_name, original_query, correction, created_at "
            "FROM corrections WHERE corpus_name = ? ORDER BY created_at DESC",
            (corpus_name,),
        ).fetchall()
        return [Correction(*r) for r in rows]

    def get_stats(self, corpus_name: str | None = None) -> dict:
        """Get learning statistics."""
        if corpus_name:
            total = self._conn.execute(
                "SELECT COUNT(*) FROM learnings WHERE corpus_name = ?",
                (corpus_name,),
            ).fetchone()[0]
            corrections = self._conn.execute(
                "SELECT COUNT(*) FROM corrections WHERE corpus_name = ?",
                (corpus_name,),
            ).fetchone()[0]
            unique_queries = self._conn.execute(
                "SELECT COUNT(DISTINCT query) FROM learnings WHERE corpus_name = ?",
                (corpus_name,),
            ).fetchone()[0]
        else:
            total = self._conn.execute("SELECT COUNT(*) FROM learnings").fetchone()[0]
            corrections = self._conn.execute("SELECT COUNT(*) FROM corrections").fetchone()[0]
            unique_queries = self._conn.execute("SELECT COUNT(DISTINCT query) FROM learnings").fetchone()[0]

        return {
            "total_learnings": total,
            "corrections": corrections,
            "unique_queries": unique_queries,
        }

    def clear(self, corpus_name: str | None = None) -> None:
        """Clear learnings, optionally for a specific corpus only."""
        if corpus_name:
            self._conn.execute("DELETE FROM learnings WHERE corpus_name = ?", (corpus_name,))
            self._conn.execute("DELETE FROM corrections WHERE corpus_name = ?", (corpus_name,))
        else:
            self._conn.execute("DELETE FROM learnings")
            self._conn.execute("DELETE FROM corrections")
        self._conn.commit()
