"""Tests for garbage collection."""
from pathlib import Path

from mcq.cache.gc import collect_garbage, GcResult
from mcq.cache.registry import CacheRegistry
from mcq.cache.store import CacheStore
from mcq.core.types import ArtifactRef


def _make_ref(artifact_hash: str, corpus_name: str = "test", file_path: str = "") -> ArtifactRef:
    return ArtifactRef(
        artifact_hash=artifact_hash,
        model_id="test-model",
        model_revision="rev",
        corpus_hash="c" * 64,
        corpus_name=corpus_name,
        prefix_token_count=100,
        prompt_template_version="v1",
        normalization_version="v1",
        build_timestamp="2026-04-03T12:00:00Z",
        file_size_bytes=1024,
        file_path=file_path,
        mlx_lm_version="0.22.0",
    )


def test_gc_removes_orphan_files(tmp_path: Path):
    """Files on disk with no registry entry should be removed."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    db = tmp_path / "registry.db"

    # Create an orphan file (not in registry)
    orphan = artifacts / "deadbeef.safetensors"
    orphan.write_bytes(b"x" * 512)

    store = CacheStore(artifacts)
    registry = CacheRegistry(db)

    result = collect_garbage(store, registry)
    assert len(result.removed_files) == 1
    assert result.bytes_freed == 512
    assert not orphan.exists()


def test_gc_keeps_registered_artifacts(tmp_path: Path):
    """Files that have a registry entry should NOT be removed."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    db = tmp_path / "registry.db"

    hash_val = "a" * 64
    registered_file = artifacts / f"{hash_val}.safetensors"
    registered_file.write_bytes(b"x" * 256)

    store = CacheStore(artifacts)
    registry = CacheRegistry(db)
    # Must also register the corpus so the artifact isn't considered orphaned
    registry.register_corpus(name="test", source_path="/tmp/src",
                             content_hash="c" * 64, chunk_count=1)
    ref = _make_ref(artifact_hash=hash_val, file_path=str(registered_file))
    registry.register(ref)

    result = collect_garbage(store, registry)
    assert len(result.removed_files) == 0
    assert result.bytes_freed == 0
    assert registered_file.exists()


def test_gc_empty_store(tmp_path: Path):
    """GC on empty store should do nothing."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    store = CacheStore(artifacts)
    registry = CacheRegistry(tmp_path / "registry.db")

    result = collect_garbage(store, registry)
    assert result.removed_files == []
    assert result.bytes_freed == 0
