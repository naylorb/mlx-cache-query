"""Content-addressed hashing utilities."""
from __future__ import annotations

import hashlib


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_content_hash(path_str: str, content: str) -> str:
    """Hash a single file's contribution to a corpus."""
    hasher = hashlib.sha256()
    hasher.update(path_str.encode())
    hasher.update(b"\x00")
    hasher.update(content.encode())
    return hasher.hexdigest()
