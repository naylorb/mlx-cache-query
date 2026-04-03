from __future__ import annotations

import sys
import time
from pathlib import Path

import click

from app.core.constants import (
    APP_DIR,
    ARTIFACTS_DIR,
    DEFAULT_MODEL,
    DEFAULT_QUERY_BUDGET,
    REGISTRY_DB,
)


def _ensure_dirs() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


@click.group()
def main() -> None:
    """mcq — local MLX prompt-cache query tool."""
    _ensure_dirs()


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--name", "-n", required=True, help="Corpus name for reference")
def ingest(path: str, name: str) -> None:
    """Ingest a file or directory into a corpus."""
    from app.cache.registry import CacheRegistry
    from app.ingest.ingestor import CorpusIngestor

    source = Path(path).resolve()
    t0 = time.perf_counter()
    corpus = CorpusIngestor.ingest(source, name=name)
    elapsed = time.perf_counter() - t0

    # Persist corpus registration
    registry = CacheRegistry(REGISTRY_DB)
    registry.register_corpus(
        name=name,
        source_path=str(source),
        content_hash=corpus.content_hash,
        chunk_count=len(corpus.chunks),
    )

    click.echo(f"Corpus '{name}': {len(corpus.chunks)} files, hash={corpus.content_hash[:16]}...")
    click.echo(f"Ingested in {elapsed:.2f}s")
    click.echo(f"Content hash: {corpus.content_hash}")
    click.echo(f"Source: {source}")


@main.command()
@click.argument("name")
@click.option("--model", "-m", default=DEFAULT_MODEL, help="Model ID", show_default=True)
def build(name: str, model: str) -> None:
    """Build a KV cache artifact from a registered corpus."""
    from huggingface_hub import snapshot_download
    from mlx_lm import load
    import mlx_lm

    from app.cache.builder import CacheBuilder
    from app.cache.registry import CacheRegistry
    from app.cache.store import CacheStore
    from app.ingest.ingestor import CorpusIngestor
    from app.prefix.compiler import PrefixCompiler

    # Look up registered corpus
    registry = CacheRegistry(REGISTRY_DB)
    corpus_info = registry.get_corpus(name)
    if not corpus_info:
        click.echo(f"Corpus '{name}' not found. Run 'mcq ingest' first.")
        sys.exit(1)

    corpus_path = corpus_info["source_path"]

    # Re-ingest from source to get current content
    click.echo(f"Ingesting '{corpus_path}'...")
    t0 = time.perf_counter()
    corpus = CorpusIngestor.ingest(Path(corpus_path), name=name)
    t_ingest = time.perf_counter() - t0
    click.echo(f"  {len(corpus.chunks)} files in {t_ingest:.2f}s")

    # Check for staleness
    if corpus.content_hash != corpus_info["content_hash"]:
        click.echo(f"  Warning: corpus content has changed since registration. Updating.")
        registry.register_corpus(name, corpus_path, corpus.content_hash, len(corpus.chunks))

    # Load model
    click.echo(f"Loading model '{model}'...")
    t0 = time.perf_counter()
    model_path = snapshot_download(model)
    revision = "local"
    try:
        # Try to get the revision from the snapshot info
        from huggingface_hub import model_info
        info = model_info(model)
        revision = info.sha
    except Exception:
        pass

    mlx_model, tokenizer = load(model)
    t_load = time.perf_counter() - t0
    click.echo(f"  Model loaded in {t_load:.2f}s")

    # Compile prefix
    click.echo("Compiling prefix...")
    t0 = time.perf_counter()
    # Get model context window from config if available
    max_context = getattr(mlx_model, "max_position_embeddings", None) or getattr(
        getattr(mlx_model, "config", None), "max_position_embeddings", 32768
    )
    prefix = PrefixCompiler.compile(
        corpus=corpus,
        tokenizer=tokenizer,
        model_id=model,
        model_revision=revision,
        max_context=max_context,
    )
    t_compile = time.perf_counter() - t0
    click.echo(f"  {prefix.token_count} tokens in {t_compile:.2f}s (limit: {max_context - DEFAULT_QUERY_BUDGET})")

    # Build cache
    click.echo("Building KV cache (this may take a while)...")
    t0 = time.perf_counter()
    cache = CacheBuilder.build(prefix, mlx_model, tokenizer)
    t_build = time.perf_counter() - t0
    click.echo(f"  Cache built in {t_build:.2f}s")

    # Save
    click.echo("Saving artifact...")
    t0 = time.perf_counter()
    store = CacheStore(ARTIFACTS_DIR)
    registry = CacheRegistry(REGISTRY_DB)
    ref = store.save(
        cache=cache,
        prefix=prefix,
        corpus_name=name,
        mlx_lm_version=mlx_lm.__version__,
        registry=registry,
    )
    t_save = time.perf_counter() - t0
    click.echo(f"  Saved in {t_save:.2f}s ({ref.file_size_bytes / 1024 / 1024:.1f} MB)")

    click.echo(f"\nArtifact: {ref.artifact_hash[:16]}...")
    click.echo(f"  Path: {ref.file_path}")
    click.echo(f"  Tokens: {ref.prefix_token_count}")
    click.echo(f"  Total build time: {t_ingest + t_compile + t_build + t_save:.2f}s (excl. model load)")


@main.command()
@click.argument("name")
@click.option("--model", "-m", default=DEFAULT_MODEL, help="Model ID", show_default=True)
@click.option("--max-tokens", default=512, help="Max tokens to generate", show_default=True)
def query(name: str, model: str, max_tokens: int) -> None:
    """Interactive query against a cached corpus."""
    from mlx_lm import load

    from app.cache.registry import CacheRegistry
    from app.cache.store import CacheStore
    from app.inference.engine import QueryEngine

    registry = CacheRegistry(REGISTRY_DB)
    store = CacheStore(ARTIFACTS_DIR)

    refs = registry.get_by_corpus_name(name, model_id=model)
    if not refs:
        click.echo(f"No cache found for corpus '{name}' with model '{model}'")
        click.echo("Run 'mcq build' first.")
        sys.exit(1)

    ref = refs[0]  # most recent
    click.echo(f"Loading model '{model}'...")
    mlx_model, tokenizer = load(model)

    click.echo(f"Loading cache ({ref.file_size_bytes / 1024 / 1024:.1f} MB)...")
    t0 = time.perf_counter()
    prompt_cache, _ = store.load(ref)
    t_load = time.perf_counter() - t0
    click.echo(f"  Cache loaded in {t_load:.3f}s")
    click.echo(f"  Prefix: {ref.prefix_token_count} tokens")
    click.echo()

    click.echo("Ready. Type your question (Ctrl+C to exit):")
    while True:
        try:
            question = click.prompt("Q", prompt_suffix="> ")
            if not question.strip():
                continue

            click.echo("A> ", nl=False)
            for chunk in QueryEngine.stream_query(
                model=mlx_model,
                tokenizer=tokenizer,
                prompt_cache=prompt_cache,
                question=question,
                max_tokens=max_tokens,
            ):
                click.echo(chunk, nl=False)
            click.echo()
            click.echo()
        except (KeyboardInterrupt, EOFError):
            click.echo("\nBye.")
            break


@main.command("list")
def list_cmd() -> None:
    """List all registered corpora and cached artifacts."""
    from app.cache.registry import CacheRegistry

    registry = CacheRegistry(REGISTRY_DB)
    refs = registry.list_all()

    if not refs:
        click.echo("No cached artifacts found.")
        return

    for ref in refs:
        size_mb = ref.file_size_bytes / 1024 / 1024
        click.echo(
            f"  {ref.corpus_name:20s}  model={ref.model_id:40s}  "
            f"tokens={ref.prefix_token_count:6d}  size={size_mb:6.1f}MB  "
            f"hash={ref.artifact_hash[:12]}..."
        )


@main.command()
@click.argument("name")
def info(name: str) -> None:
    """Show details for a corpus and its cache artifacts."""
    from app.cache.registry import CacheRegistry

    registry = CacheRegistry(REGISTRY_DB)
    refs = registry.get_by_corpus_name(name)

    if not refs:
        click.echo(f"No artifacts found for corpus '{name}'.")
        return

    for ref in refs:
        click.echo(f"Corpus: {ref.corpus_name}")
        click.echo(f"  Artifact hash:    {ref.artifact_hash}")
        click.echo(f"  Model:            {ref.model_id}")
        click.echo(f"  Model revision:   {ref.model_revision}")
        click.echo(f"  Corpus hash:      {ref.corpus_hash}")
        click.echo(f"  Prefix tokens:    {ref.prefix_token_count}")
        click.echo(f"  Template version: {ref.prompt_template_version}")
        click.echo(f"  File size:        {ref.file_size_bytes / 1024 / 1024:.1f} MB")
        click.echo(f"  File path:        {ref.file_path}")
        click.echo(f"  Built:            {ref.build_timestamp}")
        click.echo(f"  mlx-lm version:   {ref.mlx_lm_version}")
        click.echo()


@main.command()
@click.argument("name")
@click.option("--cache-only", is_flag=True, help="Delete only cache artifacts, keep corpus registration")
def delete(name: str, cache_only: bool) -> None:
    """Delete a corpus and its cache artifacts."""
    from app.cache.registry import CacheRegistry
    from app.cache.store import CacheStore

    registry = CacheRegistry(REGISTRY_DB)
    store = CacheStore(ARTIFACTS_DIR)
    refs = registry.get_by_corpus_name(name)

    if not refs:
        click.echo(f"No artifacts found for corpus '{name}'.")
        return

    for ref in refs:
        store.delete(ref, registry)
        click.echo(f"Deleted: {ref.artifact_hash[:16]}... ({ref.model_id})")

    click.echo(f"Removed {len(refs)} artifact(s).")
