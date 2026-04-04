"""Garbage collection for orphaned cache artifacts.

Like `git gc` — finds artifacts on disk that no longer have a
registry entry, or whose corpus has been deleted, and removes them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mcq.cache.registry import CacheRegistry
from mcq.cache.store import CacheStore


@dataclass
class GcResult:
    removed_files: list[str]
    bytes_freed: int


def collect_garbage(store: CacheStore, registry: CacheRegistry) -> GcResult:
    """Remove orphaned artifacts and return what was cleaned up."""
    # Get all known artifact hashes from registry
    all_refs = registry.list_all()
    known_hashes = {r.artifact_hash for r in all_refs}

    # Find files on disk that aren't in the registry
    removed: list[str] = []
    freed = 0

    for path in store.list_artifact_files():
        file_hash = path.stem  # filename without .safetensors
        if file_hash not in known_hashes:
            size = path.stat().st_size
            path.unlink()
            removed.append(str(path))
            freed += size

    # Also clean up registry entries whose corpus no longer exists
    orphaned = registry.get_orphaned_hashes()
    for artifact_hash in orphaned:
        artifact_path = store.artifact_path(artifact_hash)
        if artifact_path.exists():
            freed += artifact_path.stat().st_size
            artifact_path.unlink()
            removed.append(str(artifact_path))
        registry.delete(artifact_hash)

    return GcResult(removed_files=removed, bytes_freed=freed)
