from pathlib import Path

from mcq.cache.registry import CacheRegistry
from mcq.core.types import ArtifactRef


def _make_ref(**overrides) -> ArtifactRef:
    defaults = dict(
        artifact_hash="a" * 64,
        model_id="test-model",
        model_revision="rev123",
        corpus_hash="b" * 64,
        corpus_name="my-corpus",
        prefix_token_count=1000,
        prompt_template_version="v1",
        normalization_version="v1",
        build_timestamp="2026-04-03T12:00:00Z",
        file_size_bytes=1024,
        file_path="/tmp/test.safetensors",
        mlx_lm_version="0.22.0",
    )
    defaults.update(overrides)
    return ArtifactRef(**defaults)


def test_register_and_get(tmp_path: Path):
    db = tmp_path / "registry.db"
    reg = CacheRegistry(db)
    ref = _make_ref()
    reg.register(ref)
    got = reg.get_by_artifact_hash(ref.artifact_hash)
    assert got is not None
    assert got.artifact_hash == ref.artifact_hash
    assert got.model_id == ref.model_id
    assert got.corpus_name == ref.corpus_name


def test_get_missing_returns_none(tmp_path: Path):
    reg = CacheRegistry(tmp_path / "registry.db")
    assert reg.get_by_artifact_hash("nonexistent") is None


def test_get_by_corpus_name(tmp_path: Path):
    reg = CacheRegistry(tmp_path / "registry.db")
    reg.register(_make_ref(corpus_name="docs"))
    results = reg.get_by_corpus_name("docs")
    assert len(results) == 1
    assert results[0].corpus_name == "docs"


def test_get_by_corpus_name_with_model(tmp_path: Path):
    reg = CacheRegistry(tmp_path / "registry.db")
    reg.register(_make_ref(artifact_hash="a" * 64, model_id="model-a"))
    reg.register(_make_ref(artifact_hash="b" * 64, model_id="model-b"))
    results = reg.get_by_corpus_name("my-corpus", model_id="model-a")
    assert len(results) == 1
    assert results[0].model_id == "model-a"


def test_list_all(tmp_path: Path):
    reg = CacheRegistry(tmp_path / "registry.db")
    reg.register(_make_ref(artifact_hash="a" * 64, corpus_name="one"))
    reg.register(_make_ref(artifact_hash="b" * 64, corpus_name="two"))
    all_refs = reg.list_all()
    assert len(all_refs) == 2


def test_delete(tmp_path: Path):
    reg = CacheRegistry(tmp_path / "registry.db")
    ref = _make_ref()
    reg.register(ref)
    reg.delete(ref.artifact_hash)
    assert reg.get_by_artifact_hash(ref.artifact_hash) is None


def test_register_duplicate_replaces(tmp_path: Path):
    reg = CacheRegistry(tmp_path / "registry.db")
    ref1 = _make_ref(file_size_bytes=100)
    ref2 = _make_ref(file_size_bytes=200)
    reg.register(ref1)
    reg.register(ref2)
    got = reg.get_by_artifact_hash(ref1.artifact_hash)
    assert got.file_size_bytes == 200
