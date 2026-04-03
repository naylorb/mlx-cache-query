"""
Integration smoke test — requires Apple Silicon, mlx, and network for first model download.
Run with: pytest tests/test_integration_smoke.py -v -m slow
"""
import os
from pathlib import Path

import pytest

# Skip entirely if not on Apple Silicon or no mlx
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not HAS_MLX, reason="mlx not available"),
]


def test_full_pipeline_smoke(tmp_path: Path):
    """End-to-end: ingest -> compile -> build -> save -> load -> query."""
    from mlx_lm import load
    import mlx_lm
    from huggingface_hub import model_info

    from app.cache.builder import CacheBuilder
    from app.cache.registry import CacheRegistry
    from app.cache.store import CacheStore
    from app.ingest.ingestor import CorpusIngestor
    from app.inference.engine import QueryEngine
    from app.prefix.compiler import PrefixCompiler

    # Create test corpus
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "test.md").write_text(
        "# MLX Cache Query\n\n"
        "This tool builds KV caches from documents.\n"
        "It runs on Apple Silicon using MLX.\n"
    )

    # Use smallest feasible model for smoke test
    model_id = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

    # Ingest
    corpus = CorpusIngestor.ingest(corpus_dir, name="smoke-test")
    assert len(corpus.chunks) == 1

    # Load model
    mlx_model, tokenizer = load(model_id)
    try:
        info = model_info(model_id)
        revision = info.sha
    except Exception:
        revision = "unknown"

    # Compile
    prefix = PrefixCompiler.compile(
        corpus=corpus,
        tokenizer=tokenizer,
        model_id=model_id,
        model_revision=revision,
    )
    assert prefix.token_count > 0
    assert prefix.token_count < 1000  # small corpus

    # Build cache
    cache = CacheBuilder.build(prefix, mlx_model, tokenizer)
    assert len(cache) > 0  # should have per-layer cache objects

    # Save
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    db_path = tmp_path / "registry.db"
    store = CacheStore(artifacts_dir)
    registry = CacheRegistry(db_path)

    ref = store.save(
        cache=cache,
        prefix=prefix,
        corpus_name="smoke-test",
        mlx_lm_version=mlx_lm.__version__,
        registry=registry,
    )
    assert Path(ref.file_path).exists()
    assert ref.file_size_bytes > 0

    # Load
    loaded_cache, metadata = store.load(ref)
    assert len(loaded_cache) == len(cache)

    # Query
    result = QueryEngine.query(
        model=mlx_model,
        tokenizer=tokenizer,
        prompt_cache=loaded_cache,
        question="What does this tool do?",
        max_tokens=50,
    )
    assert len(result.text) > 0
    assert result.total_tokens > 0

    # Registry lookup
    found = registry.get_by_corpus_name("smoke-test", model_id=model_id)
    assert len(found) == 1
    assert found[0].artifact_hash == ref.artifact_hash
