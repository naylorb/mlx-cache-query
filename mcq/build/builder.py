"""Build pipeline — the heart of mcq.

Orchestrates: ingest → tokenize prefix → prefill KV cache → save artifact.
This is the equivalent of `git add && git commit` — it takes source
documents and produces a content-addressed cache artifact.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from mcq.cache.registry import CacheRegistry
from mcq.cache.store import CacheStore
from mcq.core.types import ArtifactRef, TokenizedPrefix


@dataclass
class BuildResult:
    ref: ArtifactRef
    ingest_time: float
    tokenize_time: float
    prefill_time: float
    save_time: float

    @property
    def total_time(self) -> float:
        return self.ingest_time + self.tokenize_time + self.prefill_time + self.save_time


@dataclass
class BuildProgress:
    """Callback data for build stages."""
    stage: str
    detail: str


def build_cache(
    source: Path,
    name: str,
    model_id: str,
    registry: CacheRegistry,
    store: CacheStore,
    on_progress: callable | None = None,
) -> BuildResult:
    """Full build pipeline: source → artifact.

    Args:
        source: Path to file or directory
        name: Corpus name
        model_id: HuggingFace model ID
        registry: Registry to record the artifact
        store: Store to save the artifact
        on_progress: Optional callback for stage updates
    """
    def _emit(stage: str, detail: str) -> None:
        if on_progress:
            on_progress(BuildProgress(stage=stage, detail=detail))

    # ── Stage 1: Ingest ──────────────────────────────────────────
    _emit("ingest", "Reading source files...")
    from mcq.ingest.ingestor import CorpusIngestor

    t0 = time.perf_counter()
    corpus = CorpusIngestor.ingest(source, name=name)
    t_ingest = time.perf_counter() - t0

    _emit("ingest", f"{len(corpus.chunks)} files in {t_ingest:.2f}s")

    # Register/update corpus
    registry.register_corpus(
        name=name,
        source_path=str(source),
        content_hash=corpus.content_hash,
        chunk_count=len(corpus.chunks),
    )

    # ── Stage 2: Tokenize ────────────────────────────────────────
    _emit("tokenize", f"Loading model {model_id}...")
    from huggingface_hub import snapshot_download
    from mlx_lm import load
    import mlx_lm

    snapshot_download(model_id)
    revision = "local"
    try:
        from huggingface_hub import model_info
        info = model_info(model_id)
        revision = info.sha
    except Exception:
        pass

    mlx_model, tokenizer = load(model_id)

    from mcq.prefix.compiler import PrefixCompiler
    from mcq.core.constants import DEFAULT_QUERY_BUDGET

    t0 = time.perf_counter()
    max_context = getattr(mlx_model, "max_position_embeddings", None) or getattr(
        getattr(mlx_model, "config", None), "max_position_embeddings", 32768
    )
    prefix = PrefixCompiler.compile(
        corpus=corpus,
        tokenizer=tokenizer,
        model_id=model_id,
        model_revision=revision,
        max_context=max_context,
        query_budget=DEFAULT_QUERY_BUDGET,
    )
    t_tokenize = time.perf_counter() - t0

    _emit("tokenize", f"{prefix.token_count} tokens in {t_tokenize:.2f}s")

    # ── Stage 3: Prefill KV cache ────────────────────────────────
    _emit("prefill", "Building KV cache...")
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.generate import generate_step

    t0 = time.perf_counter()
    prompt_cache = make_prompt_cache(mlx_model)
    prompt_array = mx.array(prefix.tokens)

    for _ in generate_step(
        prompt=prompt_array,
        model=mlx_model,
        max_tokens=0,
        prompt_cache=prompt_cache,
    ):
        pass

    t_prefill = time.perf_counter() - t0
    _emit("prefill", f"Cache built in {t_prefill:.2f}s")

    # ── Stage 4: Save artifact ───────────────────────────────────
    _emit("save", "Saving artifact...")

    t0 = time.perf_counter()
    ref = store.save(
        cache=prompt_cache,
        prefix=prefix,
        corpus_name=name,
        mlx_lm_version=mlx_lm.__version__,
        registry=registry,
    )
    t_save = time.perf_counter() - t0

    _emit("save", f"Saved {ref.file_size_bytes / 1024 / 1024:.1f} MB in {t_save:.2f}s")

    return BuildResult(
        ref=ref,
        ingest_time=t_ingest,
        tokenize_time=t_tokenize,
        prefill_time=t_prefill,
        save_time=t_save,
    )
