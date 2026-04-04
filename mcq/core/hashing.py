from __future__ import annotations

import hashlib


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def artifact_hash_from_prefix(model_revision: str, prefix_tokens: list[int]) -> str:
    hasher = hashlib.sha256()
    hasher.update(model_revision.encode("utf-8"))
    hasher.update(b"\x00")
    for t in prefix_tokens:
        hasher.update(t.to_bytes(4, "little"))
    return hasher.hexdigest()
