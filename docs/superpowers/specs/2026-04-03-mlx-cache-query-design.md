# mlx-cache-query — Design Spec

## 1. Product Intent

A local-first Apple Silicon tool that turns private documents and codebases into reusable MLX prompt-cache artifacts. Users ask questions with low latency without sending data to the cloud.

This is **not a RAG system**. The persisted KV cache artifact is the central product primitive — not an embedding index, not a vector store. The user builds a cache once, reloads it instantly, and queries repeatedly.

## 2. Constraints

- Apple Silicon only (V1)
- Local-first, privacy-first — no cloud dependencies at query time
- Benchmark-driven development
- Deterministic cache identity and invalidation
- Text-first V1; visual PDF/VLM is V2
- Raw cache path before TurboQuant path
- Target hardware: 8GB MacBook (M-series)

## 3. Model Strategy

- Model-parameterized from the start — cache artifacts are inherently model-specific
- Soft default: `mlx-community/Qwen2.5-3B-Instruct-4bit` (~1.7GB weights, comfortable on 8GB)
- User specifies model at build time; CLI/docs suggest the default
- Cache identity includes model_id, so switching models produces a separate artifact
- Fallback option: `Qwen2.5-1.5B-Instruct-4bit` (~0.9GB) for tighter memory budgets

## 4. Corpus Size Targets

- **Validate first:** 8K token prefix (~288MB KV cache, ~5-10 pages of text)
- **Design ceiling:** 32K tokens (~1.1GB KV cache — tight on 8GB, requires measured validation)
- Reserved query budget: 2048 tokens (subtracted from context window)

### Memory math (Qwen2.5-3B-Instruct)
- 36 layers, 2 KV heads, head dim 128, float16
- Per token per layer: `2 (K,V) * 2 (KV heads) * 128 * 2 bytes = 1024 bytes`
- 8K tokens: `1024 * 8192 * 36 = ~288 MiB`
- 32K tokens: `1024 * 32768 * 36 = ~1.1 GiB`
- 8GB budget: ~1.7GB weights + 288MB cache + ~72MB query buffer + OS = ~4GB peak. Feasible.
- 32K on 8GB: ~1.7GB + 1.1GB + 72MB + OS = ~5GB peak. Tight but possible. Must benchmark.

## 5. Data Model

Five primary types flow through the pipeline:

### Corpus
Normalized text content from a source. A list of ordered `CorpusChunk` objects, each with:
- `source_path: str` — relative path from corpus root
- `content: str` — normalized text content
- `byte_range: tuple[int, int]` — original byte range in source file

Content hash: `sha256(sorted((chunk.source_path, chunk.content) for chunk in chunks))`. Includes file paths because they appear in the prompt template — different layouts produce different prompts.

### TokenizedPrefix
Model-specific token sequence compiled from a Corpus:
- `model_id: str`
- `tokens: list[int]` — the full prefix token sequence
- `token_count: int`
- `corpus_hash: str`
- `prefix_hash: str` — `sha256(model_id | tokenizer_hash | prefix_tokens)`

Prompt structure:
```
<system>You are answering questions about the following document(s).</system>
<documents>
[== path/to/file1.md ==]
<content>
[== path/to/file2.py ==]
<content>
</documents>
<query>
```

### KVCache
MLX cache state from running the tokenized prefix through the model in prefill mode. This is a list of per-layer cache objects (e.g., `mlx_lm.models.cache.KVCache`, `RotatingKVCache`) — not raw arrays. Each cache object carries `state` (the K/V tensors) and `meta_state` (offsets, class identity). The expensive artifact — ~10-30s build time for 8K tokens on M-series.

**Important:** `mlx-lm` provides native `make_prompt_cache()`, `save_prompt_cache()`, and `load_prompt_cache()` APIs. Our `CacheBuilder` and `CacheStore` wrap these — we do not implement custom serialization.

### ArtifactRef
Pointer to a saved cache on disk, with full build provenance:
- `artifact_hash: str` — content-addressed filename (`sha256(model_revision | prefix_token_ids)`)
- `model_id: str` — HuggingFace model identifier
- `model_revision: str` — pinned commit hash (not mutable `main`)
- `corpus_hash: str` — hash of `(source_path, content)` pairs
- `corpus_name: str` — user-facing name
- `prefix_token_count: int`
- `prompt_template_version: str` — e.g., "v1"
- `normalization_version: str` — e.g., "v1"
- `build_timestamp: str` (ISO 8601)
- `file_size_bytes: int`
- `file_path: str`
- `mlx_lm_version: str` — for reproducibility

Stored in SQLite registry.

### QuerySession
Ephemeral runtime object: a loaded KVCache + model + tokenizer, ready to accept user questions and stream answers.

## 6. Architecture: Pipeline-of-Functions

Linear pipeline. Each stage is a plain Python class with typed inputs/outputs. CLI and API are thin wrappers that call the same pipeline. No framework, no plugin system, no event bus.

```
ingest(source) -> Corpus
compile(corpus, model_id) -> TokenizedPrefix
build(prefix, model) -> KVCache
save(cache, store_path) -> ArtifactRef
load(artifact_ref) -> KVCache
query(cache, model, question) -> Stream[str]
```

### Why this approach
- Each function independently testable and benchmarkable
- No framework lock-in
- Trivially inspectable — print any intermediate
- Linear pipeline matches the actual data flow
- Deterministic hashing handles invalidation without a DAG runner

## 7. Pipeline Stages

### CorpusIngestor
- Input: source path (file, directory, or glob)
- Reads files, filters by supported types: `.txt`, `.md`, `.py`, `.rs`, `.js`, `.ts`, `.json`, `.toml`, `.yaml`, `.pdf` (text extraction)
- Normalizes line endings to `\n`, strips trailing whitespace per line. No content-altering normalization (preserves indentation for Python, YAML, Makefiles, etc.)
- Sorts chunks deterministically by relative path (lexicographic)
- Computes corpus content hash
- PDF text extraction via `pymupdf` (fitz)
- Codebase ingestion respects `.gitignore`
- Output: `Corpus`

### PrefixCompiler
- Input: `Corpus` + `model_id`
- Loads tokenizer for the model (via `mlx-lm`)
- Wraps corpus in structured prompt template (see Section 5)
- Tokenizes the full prefix
- Validates prefix fits within `model_context_window - query_budget` (default query_budget=2048)
- Output: `TokenizedPrefix`

### CacheBuilder
- Input: `TokenizedPrefix` + loaded model
- Creates cache via `mlx_lm.utils.make_prompt_cache(model)`
- Runs token sequence through model using `mlx_lm.utils.generate_step()` with `max_tokens=0` to populate the cache (prefill only)
- Output: populated cache object list (preserves cache class types and meta_state)

### CacheStore
- Save: uses `mlx_lm.utils.save_prompt_cache(path, cache, metadata=...)` to write cache to `~/.mlx-cache-query/artifacts/<artifact_hash>.safetensors`, registers metadata in SQLite
- Load: uses `mlx_lm.utils.load_prompt_cache(path, return_metadata=True)` to restore full cache objects
- Delete: removes file + SQLite row
- Content-addressed naming: `sha256(model_revision | prefix_token_ids)`

### CacheRegistry
- SQLite wrapper at `~/.mlx-cache-query/registry.db`
- Tracks all ArtifactRef metadata
- Lookups by: corpus hash, model ID, artifact hash, corpus name
- Stale detection: if corpus content changes, hash changes, old artifact is stale
- No auto-deletion of stale artifacts — user decides

### QueryEngine
- Input: loaded cache objects + model + tokenizer + user question string
- Tokenizes the question
- Feeds question tokens through the model with `prompt_cache=` to extend the cached prefix
- The cache object tracks the token boundary internally — `QueryEngine` does not manage offsets manually
- Runs autoregressive decoding via `mlx_lm.utils.generate_step()`, yields tokens as a stream
- Reports: TTFT (time to first token), decode throughput (tokens/sec)
- **Important:** each query mutates the cache (appends KV state for query tokens). For repeated queries against the same prefix, reload the cache from disk each time or deepcopy before querying.

## 8. CLI Interface

Entry point: `mcq` (mlx-cache-query), built with `click`.

```
mcq ingest <path> [--name <corpus_name>]      # register a corpus
mcq build <corpus_name> [--model <model_id>]   # compile prefix + build cache
mcq query <corpus_name> [--model <model_id>]   # interactive query against cached artifact
mcq list                                        # show registered corpora and cached artifacts
mcq info <corpus_name>                          # show corpus details, cache status, timings
mcq delete <corpus_name> [--cache-only]         # remove corpus and/or its cache artifacts
```

Default model: `mlx-community/Qwen2.5-3B-Instruct-4bit`

## 9. Storage Layout

### User-facing paths
```
~/.mlx-cache-query/
├── artifacts/          # cache files (safetensors)
│   ├── <hash1>.safetensors
│   └── <hash2>.safetensors
└── registry.db         # SQLite metadata
```

### Repo structure
```
mlx-cache-query/
├── app/
│   ├── __init__.py
│   ├── cli.py                  # click CLI entry point
│   ├── core/
│   │   ├── types.py            # Corpus, TokenizedPrefix, ArtifactRef, etc.
│   │   └── hashing.py          # deterministic hashing utilities
│   ├── ingest/
│   │   └── ingestor.py         # CorpusIngestor
│   ├── prefix/
│   │   └── compiler.py         # PrefixCompiler
│   ├── cache/
│   │   ├── builder.py          # CacheBuilder
│   │   ├── store.py            # CacheStore (file I/O)
│   │   └── registry.py         # CacheRegistry (SQLite)
│   ├── inference/
│   │   └── engine.py           # QueryEngine
│   ├── eval/
│   │   └── bench.py            # BenchmarkRunner
│   ├── api/
│   │   └── server.py           # FastAPI (after CLI is solid)
│   └── ui/                     # Minimal web UI (after API)
├── tests/
│   ├── test_ingestor.py
│   ├── test_compiler.py
│   ├── test_builder.py
│   ├── test_store.py
│   └── test_engine.py
├── docs/
├── pyproject.toml
└── README.md
```

## 10. Dependencies

### Core (V1 CLI)
- `mlx` — array ops, Metal acceleration
- `mlx-lm` — model loading, tokenizer, generation
- `click` — CLI framework
- `pymupdf` — PDF text extraction
- `safetensors` — cache serialization format

### Deferred (API/UI phase)
- `fastapi`, `uvicorn` — HTTP API
- `pytest` — testing (installed from start, used from start)

## 11. Deterministic Identity & Invalidation

Cache identity is derived from the compiled prefix — the most reliable identity signal:
```
artifact_hash = sha256(model_revision | prefix_token_ids)
```

The prefix token IDs already encode: model identity, tokenizer behavior, corpus content, file paths, prompt template, and normalization. If any of these change, the token sequence changes, and the hash changes.

- `model_revision`: pinned HuggingFace commit hash (not mutable `main` ref)
- `prefix_token_ids`: the exact `int[]` output of `PrefixCompiler`

### Invalidation rules
- **Corpus changes** → different prefix tokens → new artifact_hash → old artifact is stale
- **Model changes** → different model_revision → new artifact_hash → separate artifact
- **Tokenizer update** → different prefix tokens → new artifact_hash → new artifact
- **Template change** → different prefix tokens → new artifact_hash → new artifact
- **No auto-deletion** — stale artifacts remain until user runs `mcq delete`
- **No partial invalidation** — the entire prefix is rebuilt. Prefix is atomic.

### Build provenance
Each artifact stores a build manifest in SQLite:
- `model_id`, `model_revision` (pinned commit hash)
- `corpus_hash` (path+content hash)
- `prefix_token_count`
- `prompt_template_version` (string, e.g., "v1")
- `normalization_version` (string, e.g., "v1")
- `build_timestamp` (ISO 8601)
- `mlx_lm_version` (for reproducibility)

## 12. Benchmark Targets (V1)

Measured on 8GB M-series MacBook with Qwen2.5-3B-Instruct-4bit, 8K token prefix:

| Metric | Target | Notes |
|--------|--------|-------|
| Cache build time | <30s | One-time cost |
| Cache save to disk | <2s | safetensors write |
| Cache load from disk | <1s | safetensors read + mx.array |
| TTFT (first token) | <500ms | From cache-loaded state |
| Decode throughput | >30 tok/s | Autoregressive generation |
| Corpus ingest + hash | <1s | For 8K tokens of source text |

These are targets, not guarantees. The benchmark runner measures and reports actuals.

## 13. V1 Scope Boundary

### In scope
- Text, markdown, Python, and common code file ingestion
- PDF text extraction (not visual/OCR)
- Deterministic prefix compilation and hashing
- Raw KV cache build, save, load
- Interactive CLI query with streaming
- SQLite registry
- Basic timing benchmarks
- Local filesystem artifact store

### Out of scope (V2+)
- TurboQuant compressed caches
- Visual PDF ingestion (VLM)
- FastAPI server
- Web UI
- Vector DB / hybrid RAG
- Multi-model cache sharing
- Cloud sync
- Multi-tenant / auth
- Cross-platform support

## 14. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Wrong cache identity → silent invalid reuse | Incorrect answers | Hash prefix token IDs directly (encodes all inputs). Round-trip test: rebuild cache, compare token IDs |
| Memory estimates wrong for target hardware | OOM or false promises | Benchmark peak unified memory at 8K and 32K on actual 8GB hardware before claiming support |
| mlx-lm cache API changes | Breaks save/load | Pin mlx-lm version, isolate behind CacheBuilder/CacheStore interfaces |
| Unpinned Hub revision → non-determinism | Cache identity breaks | Always resolve and store the model's commit hash, not mutable `main` ref |
| Corpus normalization corrupts content | Wrong answers from mangled code | Normalize only line endings and trailing whitespace. Never alter content structure. |
| Query mutates cache state | Stale cache on second query | Reload from disk or deepcopy for each query session |
| Prefix prompt template affects model quality | Bad answers | Keep template simple, test with real questions, make template configurable |

## 15. Future Extension Points

- **TurboQuant codec:** `CacheCodec` interface between `CacheBuilder` and `CacheStore`. Raw codec passes through, TurboQuant codec compresses/decompresses. Same pipeline, swappable codec.
- **VLM / visual PDF:** Extend `CorpusIngestor` to produce image tokens. `PrefixCompiler` interleaves text and image tokens. Requires VLM-capable model.
- **FastAPI server:** Thin wrapper around the same pipeline functions. SSE streaming for query responses.
- **Web UI:** Minimal interface for corpus management and query. Talks to FastAPI.
