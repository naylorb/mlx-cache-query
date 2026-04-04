from pathlib import Path
from unittest.mock import patch, MagicMock

from mcq.cache.store import CacheStore
from mcq.cache.registry import CacheRegistry
from mcq.core.types import ArtifactRef, TokenizedPrefix


def _make_prefix() -> TokenizedPrefix:
    return TokenizedPrefix(
        model_id="test-model",
        model_revision="rev123",
        tokens=[1, 2, 3, 4, 5],
        corpus_hash="c" * 64,
    )


def test_artifact_path(tmp_store_dir: Path):
    store = CacheStore(artifacts_dir=tmp_store_dir)
    prefix = _make_prefix()
    path = store.artifact_path(prefix.prefix_hash)
    assert path.parent == tmp_store_dir
    assert path.name == f"{prefix.prefix_hash}.safetensors"


def test_save_and_load_roundtrip(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    db = tmp_path / "registry.db"
    store = CacheStore(artifacts_dir=artifacts)
    registry = CacheRegistry(db)
    prefix = _make_prefix()

    fake_cache = [MagicMock()]
    metadata = {"model": "test-model"}

    def fake_save(path, cache, metadata=None):
        """Mock save that creates a real file so stat() works."""
        Path(path).write_bytes(b"fake-cache-data")

    with patch("mcq.cache.store.save_prompt_cache", side_effect=fake_save) as mock_save, \
         patch("mcq.cache.store.load_prompt_cache") as mock_load:
        mock_load.return_value = (fake_cache, metadata)

        ref = store.save(
            cache=fake_cache,
            prefix=prefix,
            corpus_name="test-corpus",
            mlx_lm_version="0.22.0",
            registry=registry,
        )

        assert isinstance(ref, ArtifactRef)
        assert ref.artifact_hash == prefix.prefix_hash
        assert ref.corpus_name == "test-corpus"
        assert ref.file_size_bytes > 0
        mock_save.assert_called_once()

        loaded_cache, loaded_meta = store.load(ref)
        mock_load.assert_called_once()
        assert loaded_cache == fake_cache


def test_delete_removes_file_and_registry(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    db = tmp_path / "registry.db"
    store = CacheStore(artifacts_dir=artifacts)
    registry = CacheRegistry(db)

    # Create a fake artifact file
    fake_hash = "f" * 64
    fake_file = artifacts / f"{fake_hash}.safetensors"
    fake_file.write_text("fake")

    ref = ArtifactRef(
        artifact_hash=fake_hash,
        model_id="m",
        model_revision="r",
        corpus_hash="c" * 64,
        corpus_name="test",
        prefix_token_count=5,
        prompt_template_version="v1",
        normalization_version="v1",
        build_timestamp="2026-04-03T12:00:00Z",
        file_size_bytes=4,
        file_path=str(fake_file),
        mlx_lm_version="0.22.0",
    )
    registry.register(ref)

    store.delete(ref, registry)
    assert not fake_file.exists()
    assert registry.get_by_corpus_name("test") == []
