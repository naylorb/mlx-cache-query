"""SQLite registry for corpora and cache artifacts.

This is the metadata index — the "git index" of mcq.
Every build and ingest registers here. Queries look up cache
artifacts by corpus name.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from mcq.core.types import ArtifactRef

_SCHEMA = """
CREATE TABLE IF NOT EXISTS corpora (
    name TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    registered_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_hash TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    model_revision TEXT NOT NULL,
    corpus_hash TEXT NOT NULL,
    corpus_name TEXT NOT NULL,
    prefix_token_count INTEGER NOT NULL,
    prompt_template_version TEXT NOT NULL,
    normalization_version TEXT NOT NULL,
    build_timestamp TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    mlx_lm_version TEXT NOT NULL
);
"""

_ARTIFACT_COLS = (
    "artifact_hash", "model_id", "model_revision", "corpus_hash",
    "corpus_name", "prefix_token_count", "prompt_template_version",
    "normalization_version", "build_timestamp", "file_size_bytes",
    "file_path", "mlx_lm_version",
)


class CacheRegistry:
    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ── Corpora ──────────────────────────────────────────────────────

    def register_corpus(
        self, name: str, source_path: str, content_hash: str, chunk_count: int
    ) -> None:
        from datetime import datetime, timezone
        self._conn.execute(
            "INSERT OR REPLACE INTO corpora "
            "(name, source_path, content_hash, chunk_count, registered_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, source_path, content_hash, chunk_count,
             datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()

    def get_corpus(self, name: str) -> dict | None:
        cols = ("name", "source_path", "content_hash", "chunk_count", "registered_at")
        row = self._conn.execute(
            f"SELECT {', '.join(cols)} FROM corpora WHERE name = ?", (name,)
        ).fetchone()
        return dict(zip(cols, row)) if row else None

    def list_corpora(self) -> list[dict]:
        cols = ("name", "source_path", "content_hash", "chunk_count", "registered_at")
        rows = self._conn.execute(
            f"SELECT {', '.join(cols)} FROM corpora ORDER BY registered_at DESC"
        ).fetchall()
        return [dict(zip(cols, r)) for r in rows]

    def delete_corpus(self, name: str) -> None:
        self._conn.execute("DELETE FROM corpora WHERE name = ?", (name,))
        self._conn.commit()

    # ── Artifacts ────────────────────────────────────────────────────

    def register(self, ref: ArtifactRef) -> None:
        cols = ", ".join(_ARTIFACT_COLS)
        placeholders = ", ".join("?" for _ in _ARTIFACT_COLS)
        values = tuple(getattr(ref, c) for c in _ARTIFACT_COLS)
        self._conn.execute(
            f"INSERT OR REPLACE INTO artifacts ({cols}) VALUES ({placeholders})",
            values,
        )
        self._conn.commit()

    def get_by_corpus_name(
        self, corpus_name: str, model_id: str | None = None
    ) -> list[ArtifactRef]:
        if model_id:
            rows = self._conn.execute(
                f"SELECT {', '.join(_ARTIFACT_COLS)} FROM artifacts "
                "WHERE corpus_name = ? AND model_id = ? ORDER BY build_timestamp DESC",
                (corpus_name, model_id),
            ).fetchall()
        else:
            rows = self._conn.execute(
                f"SELECT {', '.join(_ARTIFACT_COLS)} FROM artifacts "
                "WHERE corpus_name = ? ORDER BY build_timestamp DESC",
                (corpus_name,),
            ).fetchall()
        return [ArtifactRef(**dict(zip(_ARTIFACT_COLS, r))) for r in rows]

    def list_all(self) -> list[ArtifactRef]:
        rows = self._conn.execute(
            f"SELECT {', '.join(_ARTIFACT_COLS)} FROM artifacts "
            "ORDER BY build_timestamp DESC"
        ).fetchall()
        return [ArtifactRef(**dict(zip(_ARTIFACT_COLS, r))) for r in rows]

    def delete(self, artifact_hash: str) -> None:
        self._conn.execute(
            "DELETE FROM artifacts WHERE artifact_hash = ?", (artifact_hash,)
        )
        self._conn.commit()

    def get_orphaned_hashes(self) -> list[str]:
        """Find artifact hashes whose corpus no longer exists."""
        rows = self._conn.execute(
            "SELECT a.artifact_hash FROM artifacts a "
            "LEFT JOIN corpora c ON a.corpus_name = c.name "
            "WHERE c.name IS NULL"
        ).fetchall()
        return [r[0] for r in rows]
