"""Core data types for mcq.

Every type here is a plain dataclass — no framework dependencies, no magic.
These are the atoms that every module trades in.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CorpusChunk:
    """A single ingested file (or file segment)."""
    source_path: str
    content: str
    byte_range: tuple[int, int]
    metadata: dict | None = field(default=None, hash=False, compare=False)


@dataclass
class Corpus:
    """A named collection of ingested chunks."""
    name: str
    chunks: list[CorpusChunk]

    @property
    def content_hash(self) -> str:
        """Deterministic SHA-256 over sorted (path, content) pairs."""
        hasher = hashlib.sha256()
        for path, content in sorted(
            (c.source_path, c.content) for c in self.chunks
        ):
            hasher.update(path.encode())
            hasher.update(b"\x00")
            hasher.update(content.encode())
            hasher.update(b"\x00")
        return hasher.hexdigest()


@dataclass
class TokenizedPrefix:
    """The tokenized form of a corpus, ready to become a KV cache."""
    model_id: str
    model_revision: str
    tokens: list[int]
    corpus_hash: str

    @property
    def token_count(self) -> int:
        return len(self.tokens)

    @property
    def prefix_hash(self) -> str:
        """Content-addressed hash: same model + same tokens = same hash."""
        hasher = hashlib.sha256()
        hasher.update(self.model_revision.encode())
        hasher.update(b"\x00")
        for t in self.tokens:
            hasher.update(t.to_bytes(4, "little"))
        return hasher.hexdigest()


@dataclass(frozen=True)
class ArtifactRef:
    """A reference to a persisted KV cache artifact on disk.

    This is the "git object" of mcq — content-addressed, immutable,
    carrying all the metadata needed to verify and reload the cache.
    """
    artifact_hash: str
    model_id: str
    model_revision: str
    corpus_hash: str
    corpus_name: str
    prefix_token_count: int
    prompt_template_version: str
    normalization_version: str
    build_timestamp: str
    file_size_bytes: int
    file_path: str
    mlx_lm_version: str
