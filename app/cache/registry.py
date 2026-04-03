from __future__ import annotations

import sqlite3
from pathlib import Path

from app.core.types import ArtifactRef

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

_COLUMNS = (
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
        # DEVIATION: use executescript() instead of execute() because _SCHEMA
        # contains multiple statements and execute() only runs the first one.
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def register_corpus(
        self, name: str, source_path: str, content_hash: str, chunk_count: int
    ) -> None:
        from datetime import datetime, timezone
        self._conn.execute(
            "INSERT OR REPLACE INTO corpora (name, source_path, content_hash, chunk_count, registered_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, source_path, content_hash, chunk_count, datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()

    def get_corpus(self, name: str) -> dict | None:
        row = self._conn.execute(
            "SELECT name, source_path, content_hash, chunk_count, registered_at FROM corpora WHERE name = ?",
            (name,),
        ).fetchone()
        if not row:
            return None
        return dict(zip(("name", "source_path", "content_hash", "chunk_count", "registered_at"), row))

    def list_corpora(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT name, source_path, content_hash, chunk_count, registered_at FROM corpora ORDER BY registered_at DESC"
        ).fetchall()
        return [dict(zip(("name", "source_path", "content_hash", "chunk_count", "registered_at"), r)) for r in rows]

    def delete_corpus(self, name: str) -> None:
        self._conn.execute("DELETE FROM corpora WHERE name = ?", (name,))
        self._conn.commit()

    def register(self, ref: ArtifactRef) -> None:
        cols = ", ".join(_COLUMNS)
        placeholders = ", ".join("?" for _ in _COLUMNS)
        values = tuple(getattr(ref, c) for c in _COLUMNS)
        self._conn.execute(
            f"INSERT OR REPLACE INTO artifacts ({cols}) VALUES ({placeholders})",
            values,
        )
        self._conn.commit()

    def get_by_artifact_hash(self, artifact_hash: str) -> ArtifactRef | None:
        row = self._conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM artifacts WHERE artifact_hash = ?",
            (artifact_hash,),
        ).fetchone()
        return self._row_to_ref(row) if row else None

    def get_by_corpus_name(
        self, corpus_name: str, model_id: str | None = None
    ) -> list[ArtifactRef]:
        if model_id:
            rows = self._conn.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM artifacts WHERE corpus_name = ? AND model_id = ?",
                (corpus_name, model_id),
            ).fetchall()
        else:
            rows = self._conn.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM artifacts WHERE corpus_name = ?",
                (corpus_name,),
            ).fetchall()
        return [self._row_to_ref(r) for r in rows]

    def list_all(self) -> list[ArtifactRef]:
        rows = self._conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM artifacts ORDER BY build_timestamp DESC"
        ).fetchall()
        return [self._row_to_ref(r) for r in rows]

    def delete(self, artifact_hash: str) -> None:
        self._conn.execute(
            "DELETE FROM artifacts WHERE artifact_hash = ?", (artifact_hash,)
        )
        self._conn.commit()

    @staticmethod
    def _row_to_ref(row: tuple) -> ArtifactRef:
        return ArtifactRef(**dict(zip(_COLUMNS, row)))
