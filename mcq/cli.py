from __future__ import annotations

import sys
import time
from pathlib import Path

import click

from mcq.console import emit_json, is_interactive, is_piped, output, status
from mcq.core.constants import (
    APP_DIR,
    ARTIFACTS_DIR,
    DEFAULT_MODEL,
    DEFAULT_QUERY_BUDGET,
    REGISTRY_DB,
)


def _ensure_dirs() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Group
# ---------------------------------------------------------------------------


@click.group()
@click.option("--json", "use_json", is_flag=True, envvar="MCQ_JSON", help="Output JSON to stdout")
@click.option("--quiet", "-q", is_flag=True, envvar="MCQ_QUIET", help="Suppress status output")
@click.pass_context
def main(ctx: click.Context, use_json: bool, quiet: bool) -> None:
    """mcq \u2014 local MLX prompt-cache query tool."""
    ctx.ensure_object(dict)
    ctx.obj["json"] = use_json
    ctx.obj["quiet"] = quiet
    if quiet:
        status.quiet = True
    _ensure_dirs()


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--name", "-n", required=True, help="Corpus name for reference")
@click.option("--format", "-f", "fmt", default="auto", type=click.Choice(["auto", "obsidian"]), help="Ingestion format")
@click.pass_context
def ingest(ctx: click.Context, path: str, name: str, fmt: str) -> None:
    """Ingest a file or directory into a corpus."""
    from mcq.cache.registry import CacheRegistry
    from mcq.ingest.ingestor import CorpusIngestor

    use_json = ctx.obj["json"]

    source = Path(path).resolve()
    t0 = time.perf_counter()
    corpus = CorpusIngestor.ingest(source, name=name, format=fmt)
    elapsed = time.perf_counter() - t0

    registry = CacheRegistry(REGISTRY_DB)
    registry.register_corpus(
        name=name,
        source_path=str(source),
        content_hash=corpus.content_hash,
        chunk_count=len(corpus.chunks),
    )

    if use_json:
        emit_json({
            "name": name,
            "hash": corpus.content_hash,
            "chunks": len(corpus.chunks),
            "source": str(source),
            "elapsed_s": round(elapsed, 3),
        })
    else:
        status.print(f"Corpus [bold]'{name}'[/bold]: {len(corpus.chunks)} files, hash={corpus.content_hash[:16]}...")
        status.print(f"Ingested in {elapsed:.2f}s")
        status.print(f"Content hash: {corpus.content_hash}")
        status.print(f"Source: {source}")


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


@main.command()
@click.argument("name")
@click.option("--model", "-m", default=DEFAULT_MODEL, envvar="MCQ_MODEL", help="Model ID", show_default=True)
@click.pass_context
def build(ctx: click.Context, name: str, model: str) -> None:
    """Build a KV cache artifact from a registered corpus."""
    from huggingface_hub import snapshot_download
    from mlx_lm import load
    import mlx_lm

    from mcq.cache.builder import CacheBuilder
    from mcq.cache.registry import CacheRegistry
    from mcq.cache.store import CacheStore
    from mcq.ingest.ingestor import CorpusIngestor
    from mcq.prefix.compiler import PrefixCompiler

    use_json = ctx.obj["json"]

    # Look up registered corpus
    registry = CacheRegistry(REGISTRY_DB)
    corpus_info = registry.get_corpus(name)
    if not corpus_info:
        status.print(f"[red]Error:[/red] Corpus '{name}' not found. Run 'mcq ingest' first.")
        raise SystemExit(1)

    corpus_path = corpus_info["source_path"]

    # Re-ingest
    with status.status("[bold]Ingesting corpus...") as spinner:
        t0 = time.perf_counter()
        corpus = CorpusIngestor.ingest(Path(corpus_path), name=name)
        t_ingest = time.perf_counter() - t0
        spinner.update(f"[bold]Ingested {len(corpus.chunks)} files in {t_ingest:.2f}s")

    status.print(f"  {len(corpus.chunks)} files in {t_ingest:.2f}s")

    # Check for staleness
    if corpus.content_hash != corpus_info["content_hash"]:
        status.print("[yellow]  Warning: corpus content has changed since registration. Updating.[/yellow]")
        registry.register_corpus(name, corpus_path, corpus.content_hash, len(corpus.chunks))

    # Load model
    with status.status(f"[bold]Loading model '{model}'...") as spinner:
        t0 = time.perf_counter()
        model_path = snapshot_download(model)
        revision = "local"
        try:
            from huggingface_hub import model_info
            info = model_info(model)
            revision = info.sha
        except Exception:
            pass

        mlx_model, tokenizer = load(model)
        t_load = time.perf_counter() - t0
        spinner.update(f"[bold]Model loaded in {t_load:.2f}s")

    status.print(f"  Model loaded in {t_load:.2f}s")

    # Compile prefix
    with status.status("[bold]Compiling prefix...") as spinner:
        t0 = time.perf_counter()
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

    status.print(f"  {prefix.token_count} tokens in {t_compile:.2f}s (limit: {max_context - DEFAULT_QUERY_BUDGET})")

    # Build cache
    with status.status("[bold]Building KV cache (this may take a while)...") as spinner:
        t0 = time.perf_counter()
        cache = CacheBuilder.build(prefix, mlx_model, tokenizer)
        t_build = time.perf_counter() - t0

    status.print(f"  Cache built in {t_build:.2f}s")

    # Save
    with status.status("[bold]Saving artifact...") as spinner:
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

    status.print(f"  Saved in {t_save:.2f}s ({ref.file_size_bytes / 1024 / 1024:.1f} MB)")

    if use_json:
        emit_json({
            "artifact_hash": ref.artifact_hash,
            "corpus_name": name,
            "model_id": ref.model_id,
            "prefix_token_count": ref.prefix_token_count,
            "file_size_bytes": ref.file_size_bytes,
            "file_path": ref.file_path,
            "build_time_s": round(t_ingest + t_compile + t_build + t_save, 2),
        })
    else:
        status.print(f"\nArtifact: {ref.artifact_hash[:16]}...")
        status.print(f"  Path: {ref.file_path}")
        status.print(f"  Tokens: {ref.prefix_token_count}")
        status.print(f"  Total build time: {t_ingest + t_compile + t_build + t_save:.2f}s (excl. model load)")


# ---------------------------------------------------------------------------
# query helpers
# ---------------------------------------------------------------------------


def _run_single_query(
    mlx_model,
    tokenizer,
    prompt_cache: list,
    question: str,
    max_tokens: int,
    use_json: bool,
) -> None:
    """Run a single query. Streams raw text to stdout, or emits JSON."""
    import copy
    import time as _time

    from mcq.inference.engine import QueryEngine

    cache = copy.deepcopy(prompt_cache)
    query_text = QueryEngine.format_query_prompt(question)

    from mlx_lm import stream_generate

    t0 = _time.perf_counter()
    ttft: float | None = None
    chunks: list[str] = []
    token_count = 0

    for response in stream_generate(
        mlx_model,
        tokenizer,
        prompt=query_text,
        max_tokens=max_tokens,
        prompt_cache=cache,
    ):
        if ttft is None:
            ttft = (_time.perf_counter() - t0) * 1000
        chunks.append(response.text)
        token_count += 1
        if not use_json:
            sys.stdout.write(response.text)
            sys.stdout.flush()

    total_s = _time.perf_counter() - t0
    decode_tokens = max(token_count - 1, 1)
    decode_time = total_s - (ttft / 1000 if ttft else 0)
    tps = decode_tokens / max(decode_time, 0.001)

    if use_json:
        emit_json({
            "text": "".join(chunks),
            "ttft_ms": round(ttft or 0.0, 1),
            "tokens_per_sec": round(tps, 1),
            "tokens": token_count,
        })
    else:
        sys.stdout.write("\n")
        sys.stdout.flush()
        status.print(f"[dim]({token_count} tokens, {tps:.1f} tok/s, TTFT {ttft or 0:.0f}ms)[/dim]")


def _run_interactive(
    mlx_model,
    tokenizer,
    prompt_cache: list,
    max_tokens: int,
) -> None:
    """Interactive REPL. Prompt on stderr, answers to stdout."""
    from mcq.inference.engine import QueryEngine

    status.print("[bold]Ready.[/bold] Type your question (Ctrl+C to exit):")
    while True:
        try:
            # Print prompt to stderr so it doesn't mix with piped output
            status.print()
            question = click.prompt("Q", prompt_suffix="> ", err=True)
            if not question.strip():
                continue

            status.print("A> ", end="")
            for chunk in QueryEngine.stream_query(
                model=mlx_model,
                tokenizer=tokenizer,
                prompt_cache=prompt_cache,
                question=question,
                max_tokens=max_tokens,
            ):
                sys.stdout.write(chunk)
                sys.stdout.flush()
            sys.stdout.write("\n")
            sys.stdout.flush()
        except (KeyboardInterrupt, EOFError):
            status.print("\nBye.")
            break


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


@main.command()
@click.argument("name")
@click.argument("question", required=False, default=None)
@click.option("--model", "-m", default=DEFAULT_MODEL, envvar="MCQ_MODEL", help="Model ID", show_default=True)
@click.option("--max-tokens", default=512, envvar="MCQ_MAX_TOKENS", help="Max tokens to generate", show_default=True)
@click.pass_context
def query(ctx: click.Context, name: str, question: str | None, model: str, max_tokens: int) -> None:
    """Query against a cached corpus.

    Supports three modes:

    \b
    1. One-shot:    mcq query myproject "What does auth do?"
    2. Piped:       echo "question" | mcq query myproject
    3. Interactive:  mcq query myproject
    """
    from mlx_lm import load

    from mcq.cache.registry import CacheRegistry
    from mcq.cache.store import CacheStore

    use_json = ctx.obj["json"]

    registry = CacheRegistry(REGISTRY_DB)
    store = CacheStore(ARTIFACTS_DIR)

    refs = registry.get_by_corpus_name(name, model_id=model)
    if not refs:
        status.print(f"[red]Error:[/red] No cache found for corpus '{name}' with model '{model}'")
        status.print("Run 'mcq build' first.")
        raise SystemExit(1)

    ref = refs[0]  # most recent

    with status.status(f"[bold]Loading model '{model}'..."):
        mlx_model, tokenizer = load(model)

    with status.status(f"[bold]Loading cache ({ref.file_size_bytes / 1024 / 1024:.1f} MB)..."):
        t0 = time.perf_counter()
        prompt_cache, _ = store.load(ref)
        t_load = time.perf_counter() - t0

    status.print(f"Cache loaded in {t_load:.3f}s ({ref.prefix_token_count} prefix tokens)")

    if question:
        # One-shot mode
        _run_single_query(mlx_model, tokenizer, prompt_cache, question, max_tokens, use_json)
    elif not is_interactive():
        # Pipe mode -- read from stdin
        question = sys.stdin.read().strip()
        if not question:
            status.print("[red]Error:[/red] No question provided on stdin.")
            raise SystemExit(1)
        _run_single_query(mlx_model, tokenizer, prompt_cache, question, max_tokens, use_json)
    else:
        # Interactive REPL
        _run_interactive(mlx_model, tokenizer, prompt_cache, max_tokens)


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@main.command("list")
@click.pass_context
def list_cmd(ctx: click.Context) -> None:
    """List all registered corpora and cached artifacts."""
    from mcq.cache.registry import CacheRegistry

    use_json = ctx.obj["json"]
    registry = CacheRegistry(REGISTRY_DB)

    corpora = registry.list_corpora()
    refs = registry.list_all()

    # Build a set of corpus names that have cache artifacts
    cached_names = {r.corpus_name for r in refs}

    if use_json:
        rows = []
        # Include ingested-only corpora
        for c in corpora:
            name = c["name"]
            if name not in cached_names:
                rows.append({
                    "corpus": name,
                    "status": "ingested",
                    "source": c["source_path"],
                    "chunks": c["chunk_count"],
                })
        # Include cached artifacts
        for ref in refs:
            rows.append({
                "corpus": ref.corpus_name,
                "status": "cached",
                "model": ref.model_id,
                "tokens": ref.prefix_token_count,
                "size_bytes": ref.file_size_bytes,
                "artifact_hash": ref.artifact_hash,
                "built": ref.build_timestamp,
            })
        emit_json(rows)
        return

    if not corpora and not refs:
        status.print("No corpora or cached artifacts found.")
        return

    from rich.table import Table

    table = Table(title="Corpora & Artifacts")
    table.add_column("Corpus", style="bold")
    table.add_column("Status")
    table.add_column("Model")
    table.add_column("Tokens", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Built")

    # Show ingested-only corpora first
    for c in corpora:
        name = c["name"]
        if name not in cached_names:
            table.add_row(
                name,
                "[yellow]ingested[/yellow]",
                "-",
                "-",
                "-",
                "-",
            )

    # Show cached artifacts
    for ref in refs:
        size_mb = ref.file_size_bytes / 1024 / 1024
        table.add_row(
            ref.corpus_name,
            "[green]cached[/green]",
            ref.model_id,
            str(ref.prefix_token_count),
            f"{size_mb:.1f} MB",
            ref.build_timestamp,
        )

    output.print(table)


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


@main.command()
@click.argument("name")
@click.pass_context
def info(ctx: click.Context, name: str) -> None:
    """Show details for a corpus and its cache artifacts."""
    from mcq.cache.registry import CacheRegistry

    use_json = ctx.obj["json"]
    registry = CacheRegistry(REGISTRY_DB)
    refs = registry.get_by_corpus_name(name)

    if not refs:
        status.print(f"[red]Error:[/red] No artifacts found for corpus '{name}'.")
        raise SystemExit(1)

    if use_json:
        emit_json([
            {
                "artifact_hash": r.artifact_hash,
                "model_id": r.model_id,
                "model_revision": r.model_revision,
                "corpus_hash": r.corpus_hash,
                "corpus_name": r.corpus_name,
                "prefix_token_count": r.prefix_token_count,
                "prompt_template_version": r.prompt_template_version,
                "normalization_version": r.normalization_version,
                "build_timestamp": r.build_timestamp,
                "file_size_bytes": r.file_size_bytes,
                "file_path": r.file_path,
                "mlx_lm_version": r.mlx_lm_version,
            }
            for r in refs
        ])
        return

    from rich.panel import Panel
    from rich.text import Text

    for ref in refs:
        size_mb = ref.file_size_bytes / 1024 / 1024
        body = Text()
        body.append("Artifact hash:    ", style="bold")
        body.append(f"{ref.artifact_hash}\n")
        body.append("Model:            ", style="bold")
        body.append(f"{ref.model_id}\n")
        body.append("Model revision:   ", style="bold")
        body.append(f"{ref.model_revision}\n")
        body.append("Corpus hash:      ", style="bold")
        body.append(f"{ref.corpus_hash}\n")
        body.append("Prefix tokens:    ", style="bold")
        body.append(f"{ref.prefix_token_count}\n")
        body.append("Template version: ", style="bold")
        body.append(f"{ref.prompt_template_version}\n")
        body.append("File size:        ", style="bold")
        body.append(f"{size_mb:.1f} MB\n")
        body.append("File path:        ", style="bold")
        body.append(f"{ref.file_path}\n")
        body.append("Built:            ", style="bold")
        body.append(f"{ref.build_timestamp}\n")
        body.append("mlx-lm version:   ", style="bold")
        body.append(ref.mlx_lm_version)

        panel = Panel(body, title=f"[bold]{ref.corpus_name}[/bold]", border_style="blue")
        output.print(panel)


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


@main.command()
@click.argument("name")
@click.option("--cache-only", is_flag=True, help="Delete only cache artifacts, keep corpus registration")
@click.pass_context
def delete(ctx: click.Context, name: str, cache_only: bool) -> None:
    """Delete a corpus and its cache artifacts."""
    from mcq.cache.registry import CacheRegistry
    from mcq.cache.store import CacheStore

    use_json = ctx.obj["json"]
    registry = CacheRegistry(REGISTRY_DB)
    store = CacheStore(ARTIFACTS_DIR)
    refs = registry.get_by_corpus_name(name)

    if not refs:
        status.print(f"[red]Error:[/red] No artifacts found for corpus '{name}'.")
        raise SystemExit(1)

    deleted = []
    for ref in refs:
        store.delete(ref, registry)
        deleted.append(ref.artifact_hash)
        status.print(f"Deleted: {ref.artifact_hash[:16]}... ({ref.model_id})")

    if use_json:
        emit_json({"corpus": name, "deleted": deleted, "count": len(deleted)})
    else:
        status.print(f"Removed {len(refs)} artifact(s).")


# ---------------------------------------------------------------------------
# compile
# ---------------------------------------------------------------------------


@main.command()
@click.argument("name")
@click.pass_context
def compile(ctx: click.Context, name: str) -> None:
    """Compile a corpus into a wiki-structured corpus with an index.

    Generates an _index.md with file summaries and topic extraction.
    Re-ingests the corpus from its registered source path.
    """
    from mcq.cache.registry import CacheRegistry
    from mcq.compile.compiler import WikiCompiler
    from mcq.ingest.ingestor import CorpusIngestor

    use_json = ctx.obj["json"]
    registry = CacheRegistry(REGISTRY_DB)
    corpus_info = registry.get_corpus(name)

    if not corpus_info:
        status.print(f"[red]Error:[/red] Corpus '{name}' not found. Run 'mcq ingest' first.")
        raise SystemExit(1)

    source_path = corpus_info["source_path"]

    with status.status("[bold]Ingesting corpus..."):
        t0 = time.perf_counter()
        corpus = CorpusIngestor.ingest(Path(source_path), name=name)
        t_ingest = time.perf_counter() - t0

    status.print(f"  Ingested {len(corpus.chunks)} files in {t_ingest:.2f}s")

    with status.status("[bold]Compiling wiki index..."):
        t0 = time.perf_counter()
        compiled = WikiCompiler.compile(corpus)
        t_compile = time.perf_counter() - t0

    status.print(f"  Compiled wiki with {len(compiled.chunks)} chunks in {t_compile:.2f}s")

    # Update corpus registration with new hash
    registry.register_corpus(
        name=name,
        source_path=source_path,
        content_hash=compiled.content_hash,
        chunk_count=len(compiled.chunks),
    )

    if use_json:
        emit_json({
            "name": name,
            "chunks": len(compiled.chunks),
            "hash": compiled.content_hash,
            "has_index": True,
            "elapsed_s": round(t_ingest + t_compile, 3),
        })
    else:
        status.print(f"\n[green]✓[/green] Wiki compiled for '{name}' ({len(compiled.chunks)} chunks)")


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--name", "-n", required=True, help="Corpus name")
@click.option("--model", "-m", default=DEFAULT_MODEL, envvar="MCQ_MODEL", help="Model ID", show_default=True)
@click.option("--format", "-f", "fmt", default="auto", type=click.Choice(["auto", "obsidian"]), help="Ingestion format")
@click.option("--no-compile", is_flag=True, help="Skip wiki compilation step")
@click.pass_context
def setup(ctx: click.Context, path: str, name: str, model: str, fmt: str, no_compile: bool) -> None:
    """One-command pipeline: ingest → compile → build.

    \b
    Example: mcq setup ~/research/papers -n papers
    """
    # Invoke ingest
    ctx.invoke(ingest, path=path, name=name, fmt=fmt)

    # Invoke compile (unless --no-compile)
    if not no_compile:
        ctx.invoke(compile, name=name)

    # Invoke build
    ctx.invoke(build, name=name, model=model)


# ---------------------------------------------------------------------------
# find
# ---------------------------------------------------------------------------


@main.command()
@click.argument("name")
@click.argument("query", required=False, default=None)
@click.option("--top", "-k", "top_k", default=10, help="Number of results", show_default=True)
@click.option("--content", is_flag=True, help="Output full content of matching chunks (for piping)")
@click.pass_context
def find(ctx: click.Context, name: str, query: str | None, top_k: int, content: bool) -> None:
    """Search a corpus for relevant content.

    \b
    Examples:
      mcq find myproject "authentication"
      mcq find brain "threat modeling" --top 5
      mcq find brain "auth" --content | fabric -p summarize
    """
    from mcq.cache.registry import CacheRegistry
    from mcq.ingest.ingestor import CorpusIngestor
    from mcq.search.finder import TextFinder

    use_json = ctx.obj["json"]
    registry = CacheRegistry(REGISTRY_DB)
    corpus_info = registry.get_corpus(name)

    if not corpus_info:
        status.print(f"[red]Error:[/red] Corpus '{name}' not found.")
        raise SystemExit(1)

    # Read question from stdin if not provided
    if not query:
        if not is_interactive():
            query = sys.stdin.read().strip()
        if not query:
            status.print("[red]Error:[/red] No search query provided.")
            raise SystemExit(1)

    with status.status("[bold]Searching..."):
        corpus = CorpusIngestor.ingest(Path(corpus_info["source_path"]), name=name)
        results = TextFinder.search(corpus, query, top_k=top_k)

    if use_json:
        emit_json([
            {
                "source_path": r.chunk.source_path,
                "score": round(r.score, 3),
                "snippet": r.snippet,
            }
            for r in results
        ])
    elif content:
        # Pipe-friendly: output full content of matching chunks
        for r in results:
            sys.stdout.write(f"[== {r.chunk.source_path} ==]\n")
            sys.stdout.write(r.chunk.content)
            sys.stdout.write("\n")
        sys.stdout.flush()
    else:
        if not results:
            status.print("No results found.")
            return

        from rich.table import Table
        table = Table(title=f"Search: '{query}'")
        table.add_column("Score", justify="right", style="bold")
        table.add_column("File")
        table.add_column("Snippet", max_width=80)

        for r in results:
            table.add_row(
                f"{r.score:.2f}",
                r.chunk.source_path,
                r.snippet.replace("\n", " ")[:120],
            )
        output.print(table)


# ---------------------------------------------------------------------------
# watch
# ---------------------------------------------------------------------------


@main.command()
@click.argument("name")
@click.pass_context
def watch(ctx: click.Context, name: str) -> None:
    """Watch a corpus source for changes and prompt to rebuild.

    \b
    Example: mcq watch myproject
    """
    from mcq.cache.registry import CacheRegistry
    from mcq.ingest.ingestor import CorpusIngestor
    from mcq.watch.watcher import FileWatcher

    registry = CacheRegistry(REGISTRY_DB)
    corpus_info = registry.get_corpus(name)

    if not corpus_info:
        status.print(f"[red]Error:[/red] Corpus '{name}' not found.")
        raise SystemExit(1)

    source_path = corpus_info["source_path"]
    status.print(f"Watching [bold]{source_path}[/bold] for changes...")
    status.print("Press Ctrl+C to stop.\n")

    def on_change(changed_path: str):
        status.print(f"[yellow]Changed:[/yellow] {changed_path}")
        # Re-hash to check if corpus actually changed
        corpus = CorpusIngestor.ingest(Path(source_path), name=name)
        if corpus.content_hash != corpus_info["content_hash"]:
            status.print(f"[bold]Corpus content changed.[/bold] Run 'mcq build {name}' to rebuild cache.")
            corpus_info["content_hash"] = corpus.content_hash
        else:
            status.print("[dim]Content hash unchanged — no rebuild needed.[/dim]")

    watcher = FileWatcher(Path(source_path), on_change=on_change)
    try:
        watcher.run_forever()
    except KeyboardInterrupt:
        status.print("\nStopped watching.")


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@main.command()
@click.option("--port", "-p", default=8420, envvar="MCQ_PORT", help="Port to listen on", show_default=True)
@click.option("--host", default="127.0.0.1", envvar="MCQ_HOST", help="Host to bind to", show_default=True)
@click.pass_context
def serve(ctx: click.Context, port: int, host: str) -> None:
    """Start the mcq API server.

    \b
    Endpoints:
      GET /health          — health check
      GET /corpora         — list corpora
      GET /artifacts       — list cache artifacts
      GET /info/{name}     — corpus details
      GET /find/{name}?q=  — text search
    """
    try:
        import uvicorn
    except ImportError:
        status.print("[red]Error:[/red] Install API dependencies: pip install 'mlx-cache-query[api]'")
        raise SystemExit(1)

    from mcq.api.server import create_app

    status.print(f"Starting mcq API server on [bold]http://{host}:{port}[/bold]")
    status.print("Press Ctrl+C to stop.\n")
    uvicorn.run(create_app(), host=host, port=port, log_level="info")
