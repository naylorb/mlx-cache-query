from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CorpusChunk:
    source_path: str
    content: str
    byte_range: tuple[int, int]


@dataclass
class Corpus:
    name: str
    chunks: list[CorpusChunk]

    @property
    def content_hash(self) -> str:
        hasher = hashlib.sha256()
        for path, content in sorted(
            (c.source_path, c.content) for c in self.chunks
        ):
            hasher.update(path.encode("utf-8"))
            hasher.update(b"\x00")
            hasher.update(content.encode("utf-8"))
            hasher.update(b"\x00")
        return hasher.hexdigest()


@dataclass
class TokenizedPrefix:
    model_id: str
    model_revision: str
    tokens: list[int]
    corpus_hash: str

    @property
    def token_count(self) -> int:
        return len(self.tokens)

    @property
    def prefix_hash(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(self.model_revision.encode("utf-8"))
        hasher.update(b"\x00")
        for t in self.tokens:
            hasher.update(t.to_bytes(4, "little"))
        return hasher.hexdigest()


@dataclass(frozen=True)
class ArtifactRef:
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
