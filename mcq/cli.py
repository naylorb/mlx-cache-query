"""mcq CLI — the user-facing surface.

Commands:
    build   Build a KV cache from documents
    query   Query against a cached corpus
    list    Show all caches
    info    Show details for a corpus
    find    BM25 text search (no model needed)
    delete  Remove a corpus and its caches
    gc      Garbage collect orphaned artifacts

Design:
    - stderr: chrome (spinners, progress, human messages)
    - stdout: data (answers, JSON, pipeable output)
    - --json: structured output everywhere
    - stdin: pipe questions in
    - exit codes: 0=ok, 1=query failed, 2=cache not found, 3=build failed
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import click

from mcq.console import emit_json, emit_text, is_interactive, is_piped, status
from mcq.core.constants import (
    APP_DIR,
    ARTIFACTS_DIR,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    EXIT_BUILD_FAILED,
    EXIT_CACHE_NOT_FOUND,
    EXIT_OK,
    REGISTRY_DB,
)


def _ensure_dirs() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Group ────────────────────────────────────────────────────────────


@click.group()
@click.version_option(package_name="mlx-cache-query")
@click.option("--json", "use_json", is_flag=True, envvar="MCQ_JSON",
              help="JSON output to stdout")
@click.option("--quiet", "-q", is_flag=True, envvar="MCQ_QUIET",
              help="Suppress status output on stderr")
@click.pass_context
def main(ctx: click.Context, use_json: bool, quiet: bool) -> None:
    """mcq — persistent KV caches for local LLMs.

    Build once, query instantly, pipe everywhere.
    """
    ctx.ensure_object(dict)
    ctx.obj["json"] = use_json
    ctx.obj["quiet"] = quiet
    if quiet:
        status.quiet = True
    _ensure_dirs()


# ── build ────────────────────────────────────────────────────────────


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--name", "-n", required=True, help="Corpus name")
@click.option("--model", "-m", default=DEFAULT_MODEL, envvar="MCQ_MODEL",
              help="Model ID", show_default=True)
@click.option("--template", "-t", default="general",
              type=click.Choice(["general", "code", "raw"]),
              help="Prompt template", show_default=True)
@click.pass_context
def build(ctx: click.Context, path: str, name: str, model: str, template: str) -> None:
    """Build a KV cache from a file or directory.

    \b
    Examples:
      mcq build ./docs -n my-project
      mcq build ./src -n codebase --template code
      mcq build paper.pdf -n paper
    """
    from mcq.cache.registry import CacheRegistry
    from mcq.cache.store import CacheStore
    from mcq.ingest.ingestor import CorpusIngestor
    from mcq.prefix.compiler import PrefixCompiler

    use_json = ctx.obj["json"]
    source = Path(path).resolve()
    registry = CacheRegistry(REGISTRY_DB)
    store = CacheStore(ARTIFACTS_DIR)

    # Stage 1: Ingest
    with status.status("[bold]Ingesting..."):
        t0 = time.perf_counter()
        corpus = CorpusIngestor.ingest(source, name=name)
        t_ingest = time.perf_counter() - t0

    status.print(f"  {len(corpus.chunks)} files in {t_ingest:.2f}s")

    registry.register_corpus(
        name=name,
        source_path=str(source),
        content_hash=corpus.content_hash,
        chunk_count=len(corpus.chunks),
    )

    # Stage 2: Load model + tokenize
    with status.status(f"[bold]Loading model {model}..."):
        t0 = time.perf_counter()
        from huggingface_hub import snapshot_download
        from mlx_lm import load
        import mlx_lm

        snapshot_download(model)
        revision = "local"
        try:
            from huggingface_hub import model_info
            info = model_info(model)
            revision = info.sha
        except Exception:
            pass

        mlx_model, tokenizer = load(model)
        t_load = time.perf_counter() - t0

    status.print(f"  Model loaded in {t_load:.2f}s")

    with status.status("[bold]Tokenizing prefix..."):
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
            template=template,
        )
        t_tokenize = time.perf_counter() - t0

    status.print(f"  {prefix.token_count} tokens in {t_tokenize:.2f}s")

    # Stage 3: Prefill KV cache
    with status.status("[bold]Building KV cache..."):
        t0 = time.perf_counter()
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.generate import generate_step

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

    status.print(f"  Cache built in {t_prefill:.2f}s")

    # Stage 4: Save
    with status.status("[bold]Saving artifact..."):
        t0 = time.perf_counter()
        ref = store.save(
            cache=prompt_cache,
            prefix=prefix,
            corpus_name=name,
            mlx_lm_version=mlx_lm.__version__,
            registry=registry,
        )
        t_save = time.perf_counter() - t0

    size_mb = ref.file_size_bytes / 1024 / 1024
    total = t_ingest + t_tokenize + t_prefill + t_save

    if use_json:
        emit_json({
            "artifact_hash": ref.artifact_hash,
            "corpus_name": name,
            "model_id": ref.model_id,
            "prefix_token_count": ref.prefix_token_count,
            "file_size_bytes": ref.file_size_bytes,
            "file_path": ref.file_path,
            "build_time_s": round(total, 2),
        })
    else:
        status.print(f"\n  {ref.artifact_hash[:16]}... ({size_mb:.1f} MB, {total:.1f}s)")


# ── query ────────────────────────────────────────────────────────────


@main.command()
@click.argument("name")
@click.argument("question", required=False, default=None)
@click.option("--model", "-m", default=DEFAULT_MODEL, envvar="MCQ_MODEL",
              help="Model ID", show_default=True)
@click.option("--max-tokens", default=DEFAULT_MAX_TOKENS, envvar="MCQ_MAX_TOKENS",
              help="Max tokens to generate", show_default=True)
@click.option("--no-stream", is_flag=True, help="Wait for full response")
@click.pass_context
def query(
    ctx: click.Context,
    name: str,
    question: str | None,
    model: str,
    max_tokens: int,
    no_stream: bool,
) -> None:
    """Query against a cached corpus.

    \b
    Three modes:
      mcq query myproject "What does auth do?"    # one-shot
      echo "question" | mcq query myproject        # piped
      mcq query myproject                          # interactive REPL
    """
    from mlx_lm import load

    from mcq.cache.registry import CacheRegistry
    from mcq.cache.store import CacheStore

    use_json = ctx.obj["json"]

    registry = CacheRegistry(REGISTRY_DB)
    store = CacheStore(ARTIFACTS_DIR)

    refs = registry.get_by_corpus_name(name, model_id=model)
    if not refs:
        status.print(f"[red]Error:[/red] No cache for '{name}' with model '{model}'")
        status.print("Run: mcq build <path> -n " + name)
        raise SystemExit(EXIT_CACHE_NOT_FOUND)

    ref = refs[0]

    with status.status(f"[bold]Loading model {model}..."):
        mlx_model, tokenizer = load(model)

    with status.status(f"[bold]Loading cache ({ref.file_size_bytes / 1024 / 1024:.1f} MB)..."):
        t0 = time.perf_counter()
        prompt_cache, _ = store.load(ref)
        t_load = time.perf_counter() - t0

    status.print(f"Cache loaded in {t_load:.3f}s ({ref.prefix_token_count} tokens)")

    if question:
        _run_single_query(mlx_model, tokenizer, prompt_cache, question,
                          max_tokens, use_json, no_stream)
    elif not is_interactive():
        question = sys.stdin.read().strip()
        if not question:
            status.print("[red]Error:[/red] No question on stdin.")
            raise SystemExit(EXIT_CACHE_NOT_FOUND)
        _run_single_query(mlx_model, tokenizer, prompt_cache, question,
                          max_tokens, use_json, no_stream)
    else:
        _run_interactive(mlx_model, tokenizer, prompt_cache, max_tokens)


def _run_single_query(
    mlx_model, tokenizer, prompt_cache, question, max_tokens, use_json, no_stream=False,
) -> None:
    """One-shot query. Streams to stdout or emits JSON."""
    import copy

    from mcq.query.engine import QueryEngine

    cache = copy.deepcopy(prompt_cache)
    query_text = QueryEngine.format_query_prompt(question)

    from mlx_lm import stream_generate

    t0 = time.perf_counter()
    ttft: float | None = None
    chunks: list[str] = []
    token_count = 0

    for response in stream_generate(
        mlx_model, tokenizer,
        prompt=query_text,
        max_tokens=max_tokens,
        prompt_cache=cache,
    ):
        if ttft is None:
            ttft = (time.perf_counter() - t0) * 1000
        chunks.append(response.text)
        token_count += 1
        if not use_json and not no_stream:
            emit_text(response.text)

    total_s = time.perf_counter() - t0
    decode_tokens = max(token_count - 1, 1)
    decode_time = total_s - (ttft / 1000 if ttft else 0)
    tps = decode_tokens / max(decode_time, 0.001)

    full_text = "".join(chunks)

    if use_json:
        emit_json({
            "text": full_text,
            "ttft_ms": round(ttft or 0.0, 1),
            "tokens_per_sec": round(tps, 1),
            "tokens": token_count,
        })
    elif no_stream:
        emit_text(full_text, end="\n")
        status.print(f"[dim]({token_count} tokens, {tps:.1f} tok/s, TTFT {ttft or 0:.0f}ms)[/dim]")
    else:
        emit_text("", end="\n")
        status.print(f"[dim]({token_count} tokens, {tps:.1f} tok/s, TTFT {ttft or 0:.0f}ms)[/dim]")


def _run_interactive(mlx_model, tokenizer, prompt_cache, max_tokens) -> None:
    """Interactive REPL. Prompt on stderr, answers on stdout."""
    from mcq.query.engine import QueryEngine

    status.print("[bold]Ready.[/bold] Type your question (Ctrl+C to exit):\n")
    while True:
        try:
            question = click.prompt("Q", prompt_suffix="> ", err=True)
            if not question.strip():
                continue
            for chunk in QueryEngine.stream_query(
                model=mlx_model,
                tokenizer=tokenizer,
                prompt_cache=prompt_cache,
                question=question,
                max_tokens=max_tokens,
            ):
                emit_text(chunk)
            emit_text("", end="\n\n")
        except (KeyboardInterrupt, EOFError):
            status.print("\nBye.")
            break


# ── list ─────────────────────────────────────────────────────────────


@main.command("list")
@click.pass_context
def list_cmd(ctx: click.Context) -> None:
    """List all corpora and cached artifacts."""
    from mcq.cache.registry import CacheRegistry

    use_json = ctx.obj["json"]
    registry = CacheRegistry(REGISTRY_DB)
    corpora = registry.list_corpora()
    refs = registry.list_all()
    cached_names = {r.corpus_name for r in refs}

    if use_json:
        rows = []
        for c in corpora:
            if c["name"] not in cached_names:
                rows.append({
                    "corpus": c["name"], "status": "ingested",
                    "source": c["source_path"], "chunks": c["chunk_count"],
                })
        for ref in refs:
            rows.append({
                "corpus": ref.corpus_name, "status": "cached",
                "model": ref.model_id, "tokens": ref.prefix_token_count,
                "size_bytes": ref.file_size_bytes,
                "artifact_hash": ref.artifact_hash,
                "built": ref.build_timestamp,
            })
        emit_json(rows)
        return

    if not corpora and not refs:
        status.print("No corpora or cached artifacts. Run: mcq build <path> -n <name>")
        return

    from rich.table import Table
    from mcq.console import output

    table = Table(title="mcq caches")
    table.add_column("Corpus", style="bold")
    table.add_column("Status")
    table.add_column("Model")
    table.add_column("Tokens", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Built")

    for c in corpora:
        if c["name"] not in cached_names:
            table.add_row(c["name"], "[yellow]ingested[/yellow]", "-", "-", "-", "-")

    for ref in refs:
        size_mb = ref.file_size_bytes / 1024 / 1024
        model_short = ref.model_id.split("/")[-1] if "/" in ref.model_id else ref.model_id
        table.add_row(
            ref.corpus_name,
            "[green]cached[/green]",
            model_short,
            str(ref.prefix_token_count),
            f"{size_mb:.1f} MB",
            ref.build_timestamp[:10],
        )

    output.print(table)


# ── info ───────────────────────────────────────────────────────────���─


@main.command()
@click.argument("name")
@click.pass_context
def info(ctx: click.Context, name: str) -> None:
    """Show details for a corpus and its cache artifacts."""
    from mcq.cache.registry import CacheRegistry

    use_json = ctx.obj["json"]
    registry = CacheRegistry(REGISTRY_DB)

    corpus_info = registry.get_corpus(name)
    refs = registry.get_by_corpus_name(name)

    if not corpus_info and not refs:
        status.print(f"[red]Error:[/red] '{name}' not found.")
        raise SystemExit(EXIT_CACHE_NOT_FOUND)

    if use_json:
        data = {"corpus": corpus_info, "artifacts": []}
        for r in refs:
            data["artifacts"].append({
                "artifact_hash": r.artifact_hash,
                "model_id": r.model_id,
                "model_revision": r.model_revision,
                "corpus_hash": r.corpus_hash,
                "prefix_token_count": r.prefix_token_count,
                "prompt_template_version": r.prompt_template_version,
                "build_timestamp": r.build_timestamp,
                "file_size_bytes": r.file_size_bytes,
                "file_path": r.file_path,
                "mlx_lm_version": r.mlx_lm_version,
            })
        emit_json(data)
        return

    from rich.panel import Panel
    from rich.text import Text
    from mcq.console import output

    if corpus_info:
        status.print(f"[bold]{name}[/bold]")
        status.print(f"  Source: {corpus_info['source_path']}")
        status.print(f"  Files:  {corpus_info['chunk_count']}")
        status.print(f"  Hash:   {corpus_info['content_hash'][:16]}...")
        status.print()

    for ref in refs:
        size_mb = ref.file_size_bytes / 1024 / 1024
        body = Text()
        body.append("Hash:     ", style="bold")
        body.append(f"{ref.artifact_hash[:16]}...\n")
        body.append("Model:    ", style="bold")
        body.append(f"{ref.model_id}\n")
        body.append("Tokens:   ", style="bold")
        body.append(f"{ref.prefix_token_count}\n")
        body.append("Size:     ", style="bold")
        body.append(f"{size_mb:.1f} MB\n")
        body.append("Built:    ", style="bold")
        body.append(f"{ref.build_timestamp}\n")
        body.append("Template: ", style="bold")
        body.append(f"{ref.prompt_template_version}\n")
        body.append("Path:     ", style="bold")
        body.append(ref.file_path)

        panel = Panel(body, title=f"[bold blue]{ref.corpus_name}[/bold blue]",
                      border_style="blue")
        output.print(panel)


# ── find ─────────────────────────────────────────────────────────────


@main.command()
@click.argument("name")
@click.argument("query_text", required=False, default=None, metavar="QUERY")
@click.option("--top", "-k", "top_k", default=10,
              help="Number of results", show_default=True)
@click.option("--content", is_flag=True,
              help="Output full chunk content (for piping)")
@click.pass_context
def find(ctx: click.Context, name: str, query_text: str | None, top_k: int, content: bool) -> None:
    """Search a corpus for relevant content (no model needed).

    \b
    Examples:
      mcq find myproject "authentication"
      mcq find codebase "error handling" --top 5
      mcq find docs "auth" --content | head -100
    """
    from mcq.cache.registry import CacheRegistry
    from mcq.ingest.ingestor import CorpusIngestor
    from mcq.search.finder import TextFinder

    use_json = ctx.obj["json"]
    registry = CacheRegistry(REGISTRY_DB)
    corpus_info = registry.get_corpus(name)

    if not corpus_info:
        status.print(f"[red]Error:[/red] Corpus '{name}' not found.")
        raise SystemExit(EXIT_CACHE_NOT_FOUND)

    if not query_text:
        if not is_interactive():
            query_text = sys.stdin.read().strip()
        if not query_text:
            status.print("[red]Error:[/red] No search query.")
            raise SystemExit(1)

    with status.status("[bold]Searching..."):
        corpus = CorpusIngestor.ingest(Path(corpus_info["source_path"]), name=name)
        results = TextFinder.search(corpus, query_text, top_k=top_k)

    if use_json:
        emit_json([
            {"source_path": r.chunk.source_path, "score": round(r.score, 3),
             "snippet": r.snippet}
            for r in results
        ])
    elif content:
        for r in results:
            emit_text(f"[== {r.chunk.source_path} ==]\n")
            emit_text(r.chunk.content)
            emit_text("\n")
    else:
        if not results:
            status.print("No results found.")
            return

        from rich.table import Table
        from mcq.console import output

        table = Table(title=f"Search: '{query_text}'")
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


# ── delete ───────────────────────────────────────────────────────────


@main.command()
@click.argument("name")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
@click.pass_context
def delete(ctx: click.Context, name: str, force: bool) -> None:
    """Delete a corpus and its cache artifacts."""
    from mcq.cache.registry import CacheRegistry
    from mcq.cache.store import CacheStore

    use_json = ctx.obj["json"]
    registry = CacheRegistry(REGISTRY_DB)
    store = CacheStore(ARTIFACTS_DIR)
    refs = registry.get_by_corpus_name(name)

    if not refs and not registry.get_corpus(name):
        status.print(f"[red]Error:[/red] '{name}' not found.")
        raise SystemExit(EXIT_CACHE_NOT_FOUND)

    if not force and not use_json:
        n_artifacts = len(refs)
        total_mb = sum(r.file_size_bytes for r in refs) / 1024 / 1024
        if not click.confirm(
            f"Delete '{name}' ({n_artifacts} artifact(s), {total_mb:.1f} MB)?",
            err=True,
        ):
            status.print("Aborted.")
            return

    deleted = []
    for ref in refs:
        store.delete(ref, registry)
        deleted.append(ref.artifact_hash)

    registry.delete_corpus(name)

    if use_json:
        emit_json({"corpus": name, "deleted": deleted, "count": len(deleted)})
    else:
        status.print(f"Deleted '{name}' ({len(refs)} artifact(s)).")


# ── gc ───────────────────────────────────────────────────────────────


@main.command()
@click.pass_context
def gc(ctx: click.Context) -> None:
    """Garbage collect orphaned cache artifacts."""
    from mcq.cache.gc import collect_garbage
    from mcq.cache.registry import CacheRegistry
    from mcq.cache.store import CacheStore

    use_json = ctx.obj["json"]
    registry = CacheRegistry(REGISTRY_DB)
    store = CacheStore(ARTIFACTS_DIR)

    result = collect_garbage(store, registry)

    if use_json:
        emit_json({
            "removed": result.removed_files,
            "bytes_freed": result.bytes_freed,
        })
    else:
        if not result.removed_files:
            status.print("Nothing to collect.")
        else:
            freed_mb = result.bytes_freed / 1024 / 1024
            status.print(f"Removed {len(result.removed_files)} orphan(s), freed {freed_mb:.1f} MB.")


# ── stats ────────────────────────────────────────────────────────────


@main.command()
@click.argument("name")
@click.pass_context
def stats(ctx: click.Context, name: str) -> None:
    """Show corpus statistics — files, words, lines, extensions."""
    from mcq.cache.registry import CacheRegistry
    from mcq.core.stats import CorpusStats
    from mcq.ingest.ingestor import CorpusIngestor

    use_json = ctx.obj["json"]
    registry = CacheRegistry(REGISTRY_DB)
    corpus_info = registry.get_corpus(name)

    if not corpus_info:
        status.print(f"[red]Error:[/red] Corpus '{name}' not found.")
        raise SystemExit(EXIT_CACHE_NOT_FOUND)

    corpus = CorpusIngestor.ingest(Path(corpus_info["source_path"]), name=name)
    s = CorpusStats.from_corpus(corpus)

    if use_json:
        emit_json({
            "name": s.name, "total_files": s.total_files,
            "total_bytes": s.total_bytes, "total_lines": s.total_lines,
            "total_words": s.total_words, "extensions": s.extensions,
            "largest_file": s.largest_file,
            "largest_file_bytes": s.largest_file_bytes,
        })
    else:
        from rich.panel import Panel
        from rich.text import Text
        from mcq.console import output

        body = Text()
        body.append("Files:        ", style="bold")
        body.append(f"{s.total_files}\n")
        body.append("Total bytes:  ", style="bold")
        body.append(f"{s.total_bytes:,}\n")
        body.append("Total lines:  ", style="bold")
        body.append(f"{s.total_lines:,}\n")
        body.append("Total words:  ", style="bold")
        body.append(f"{s.total_words:,}\n")
        body.append("Largest file: ", style="bold")
        body.append(f"{s.largest_file} ({s.largest_file_bytes:,} bytes)\n")
        body.append("\nExtensions:\n", style="bold")
        for ext, count in s.extensions.items():
            body.append(f"  {ext:10s}  {count} files\n")

        panel = Panel(body, title=f"[bold]{s.name}[/bold]", border_style="blue")
        output.print(panel)


# ── verify ───────────────────────────────────────────────────────────


@main.command()
@click.argument("name")
@click.pass_context
def verify(ctx: click.Context, name: str) -> None:
    """Verify cache artifact integrity — check that files exist on disk."""
    from mcq.cache.registry import CacheRegistry

    use_json = ctx.obj["json"]
    registry = CacheRegistry(REGISTRY_DB)
    refs = registry.get_by_corpus_name(name)

    if not refs:
        status.print(f"[red]Error:[/red] '{name}' not found.")
        raise SystemExit(EXIT_CACHE_NOT_FOUND)

    issues = []
    for ref in refs:
        path = Path(ref.file_path)
        if not path.exists():
            issues.append({"artifact_hash": ref.artifact_hash, "issue": "file missing",
                           "path": ref.file_path})
        elif path.stat().st_size != ref.file_size_bytes:
            issues.append({"artifact_hash": ref.artifact_hash, "issue": "size mismatch",
                           "expected": ref.file_size_bytes,
                           "actual": path.stat().st_size})

    if use_json:
        emit_json({"corpus": name, "artifacts": len(refs), "issues": issues})
    else:
        if not issues:
            status.print(f"[green]OK[/green] {name}: {len(refs)} artifact(s) verified.")
        else:
            for i in issues:
                status.print(f"[red]FAIL[/red] {i['artifact_hash'][:16]}... — {i['issue']}")
            raise SystemExit(1)
