# mcq — Product Vision & Roadmap

## The Core Thesis

**KV caches are for depth. Embeddings are for breadth. mcq combines both.**

Traditional RAG retrieves text chunks and stuffs them into a context window — the model sees fragments. mcq builds a persistent KV cache where the model has *attended to every token* with full self-attention. The difference: the model has actually **read** your documents, not just seen excerpts.

For small corpora (<32K tokens), a pure KV cache is all you need. For large collections (Obsidian vaults, codebases, paper libraries), you need embeddings to **select** what goes into the cache, then the cache for **deep understanding**.

This is not RAG. This is not keyword search. This is **local semantic memory with persistent attention state**.

## Who This Is For

**The ML researcher** (Karpathy archetype): 200 arxiv PDFs, experiment code across 30 repos, lecture notes. Needs to query "what papers discuss KV cache compression for long contexts?" and get an answer that synthesizes across papers — not a list of filenames.

**The security researcher / knowledge worker** (Miessler archetype): 5000-note Obsidian vault, pattern libraries, collected wisdom. Wants to pipe notes through AI like Fabric, but locally, against their own knowledge base. `mcq query brain "what patterns do I have for threat modeling?" | fabric -p summarize`.

**The senior engineer** (Cherny archetype): Large TypeScript/Rust codebase, design docs, ADRs. Needs "how does the auth middleware work?" answered from the actual code, not from an LLM hallucinating about it.

**Common thread**: Power users with large private knowledge bases who refuse to send their data to the cloud. They live in terminals. They pipe things. They want tools that compose.

---

## User Stories

### V2: Knowledge Base Mode

**US-1: Obsidian vault as a queryable brain**
> As a knowledge worker with a 3000-note Obsidian vault, I want to ingest my entire vault and ask questions about my own notes, so I can find connections and synthesize knowledge I've already captured.

Acceptance criteria:
- `mcq ingest ~/obsidian/brain -n brain --format obsidian`
- Parses YAML frontmatter, extracts tags and aliases
- Resolves `[[wikilinks]]` to include linked context
- Skips `.obsidian/` config directory
- Handles embedded images gracefully (skip or describe path)
- Respects folder hierarchy as organizational signal
- Tags from frontmatter and inline `#tag` are queryable metadata

**US-2: Auto-rebuild on file changes**
> As a researcher who updates notes daily, I want mcq to detect when my corpus has changed and offer to rebuild, so my cache is never stale.

Acceptance criteria:
- `mcq watch ~/obsidian/brain -n brain` — watches for changes
- On file change: re-hash corpus, compare to registry
- If stale: auto-rebuild (or prompt in interactive mode)
- Uses filesystem events (fsevents on macOS) not polling
- Background daemon mode: `mcq watch --daemon`

**US-3: QMD/Quarto document support**
> As an academic who writes papers in Quarto (.qmd), I want to ingest my research project including code chunks, prose, and citations.

Acceptance criteria:
- `.qmd` added to supported extensions
- Code chunks (```` ```{python} ... ``` ````) preserved with language annotation
- YAML frontmatter extracted
- Cross-references (`@fig-results`, `@tbl-data`) preserved
- Bibliography entries resolvable if `.bib` file present

**US-4: Multi-corpus management**
> As a user with separate projects, I want to maintain multiple corpora and switch between them easily.

Acceptance criteria:
- `mcq list` shows all corpora with status (ingested/cached/stale)
- `mcq query` tab-completes corpus names
- Corpora are independent — different models, different sizes
- `mcq gc` garbage-collects stale/orphaned artifacts

### V3: Hybrid Search — Embeddings + KV Cache

**US-5: Semantic search across large collections**
> As a researcher with 200 PDFs, I want to semantically search for "attention mechanism efficiency" and find the most relevant papers — not just keyword matches.

Acceptance criteria:
- `mcq index ~/papers -n papers` — computes embeddings for all chunks
- Uses `nomic-embed-text` via Ollama (local, no cloud)
- Stores vectors in local sqlite-vss or numpy arrays
- `mcq find papers "attention mechanism efficiency"` — returns ranked results
- Sub-second search over 10K+ chunks

**US-6: Fuzzy finder TUI**
> As a terminal power user, I want an interactive fuzzy finder (like fzf/telescope) that lets me semantically search, preview content, and select files to query against.

Acceptance criteria:
- `mcq find brain` — opens interactive TUI
- Type to search: results update in real-time
- Preview pane shows matched content with highlighted context
- Tab to select multiple results
- Enter to open in `$EDITOR`
- Ctrl+Q to query: selected files become the query corpus
- `mcq find brain "auth" --top 5` — non-interactive, pipe-friendly
- Ranked by semantic similarity, not just fuzzy string match

**US-7: Dynamic cache from search results**
> As a user with a vault too large for a single KV cache (>32K tokens), I want to search for relevant content and build a focused cache from just those results.

Acceptance criteria:
- `mcq find brain "threat modeling" --top 10 | mcq query --stdin`
- Or: `mcq query brain "threat modeling" --retrieve 10`
- Embeddings retrieve top-k relevant chunks
- Those chunks are compiled into a temporary KV cache
- Query runs against the focused cache
- Cache can optionally be persisted for repeated queries
- The user sees: "Retrieved 10 notes (4,231 tokens) → building cache..."

**US-8: Fabric-style piping with local knowledge**
> As a Fabric user, I want to compose mcq with Fabric patterns — query my local knowledge base and pipe results through AI patterns.

Acceptance criteria:
- `mcq query brain "what do I know about zero trust?" | fabric -p extract_wisdom`
- `mcq find brain "security" --top 5 --content | fabric -p summarize`
- `cat report.md | mcq query codebase "does this match our implementation?"`
- All composable via stdout/stdin
- `--content` flag on `find` outputs actual text, not just filenames

### V4: Visual & Multimodal

**US-9: Screenshot query**
> As a developer debugging a complex error, I want to take a screenshot and ask "what's wrong here?" using a local VLM.

Acceptance criteria:
- `mcq screenshot "what's wrong with this error?"` — captures screen, queries VLM
- Uses `screencapture` on macOS (silent mode)
- Runs through local VLM (e.g., Qwen2-VL-2B-Instruct-4bit)
- Can combine with corpus context: `mcq screenshot --corpus myproject "is this related to our auth bug?"`
- `mcq query myproject --image ~/screenshots/error.png "explain this"`

**US-10: Visual PDF understanding**
> As a researcher reading papers with figures and tables, I want the model to understand the visual layout — not just extracted text.

Acceptance criteria:
- `mcq ingest paper.pdf -n paper --visual` — uses VLM for PDF pages
- Each page rendered as an image → processed by VLM
- Figures, tables, equations understood visually
- Falls back to text extraction when VLM unavailable
- Model: `Qwen2-VL` family on MLX

**US-11: Context-aware screen query**
> Like Gemini on Android — I want to hold a hotkey, see what's on my screen, and ask a question about it with full context from my knowledge base.

Acceptance criteria:
- Global hotkey (via macOS accessibility) triggers capture + query
- Screenshot + question → VLM
- Optional: include corpus context for domain-specific answers
- Menu bar agent: `mcq agent --menu-bar`
- Response shown as notification or overlay

### V5: Ecosystem & Integration

**US-12: Obsidian plugin**
> As an Obsidian user, I want a sidebar where I can query my vault using mcq without leaving Obsidian.

Acceptance criteria:
- Obsidian community plugin
- Sidebar panel with query input
- Streams answers in real-time
- Links to source notes in answers
- Auto-triggers rebuild when vault changes
- Communicates with `mcq serve` (local HTTP API)

**US-13: Raycast / Alfred extension**
> As a macOS power user, I want to query my knowledge base from Raycast/Alfred with a keyboard shortcut.

Acceptance criteria:
- Raycast extension: type query, see streaming answer
- Alfred workflow: keyword trigger, result display
- Uses `mcq query --json` under the hood
- Fast startup (cache pre-loaded by daemon)

**US-14: API server for IDE integration**
> As a developer, I want VS Code / Neovim to query my codebase cache for contextual answers while I code.

Acceptance criteria:
- `mcq serve --port 8420` — local FastAPI server
- SSE streaming for query responses
- `/query`, `/find`, `/list`, `/info` endpoints
- VS Code extension: sidebar panel
- Neovim plugin: `:McqQuery "how does auth work?"`

---

## Architecture Evolution

```
V1 (current):
  Files → Ingest → Compile → Build KV Cache → Query
                                     │
                                     ├→ Save to disk
                                     └→ Load from disk

V2 (knowledge base):
  Files → Ingest (format-aware) → Compile → Build → Save
    ↑                                                  │
    └── Watch (fsevents) ──── Stale? ──── Rebuild ◄────┘

V3 (hybrid search):
  Files → Ingest → Embed (nomic-embed-text)  → Vector Store
                                                    │
  Question → Embed ──── Similarity Search ──────────┘
                              │
                         top-k chunks
                              │
                    Compile → Build temp cache → Query (deep)

V4 (multimodal):
  Screenshot ─→ VLM ─→ Answer
  PDF (visual) ─→ VLM page-by-page ─→ Rich ingest
  Image + Corpus ─→ VLM + KV cache ─→ Context-aware answer

V5 (ecosystem):
  mcq serve (HTTP API)
     ├── Obsidian plugin
     ├── Raycast extension
     ├── VS Code extension
     ├── Neovim plugin
     └── Menu bar agent
```

## Technical Decisions by Phase

### V2: Knowledge Base
| Decision | Choice | Why |
|----------|--------|-----|
| File watching | `watchdog` (Python) | Cross-platform, fsevents on macOS |
| Obsidian parsing | Custom (regex + yaml) | Simple format, no need for a library |
| QMD parsing | `pyyaml` + regex | Quarto frontmatter is YAML, chunks are fenced |
| Daemon mode | `mcq watch --daemon` | Simple background process, PID file |

### V3: Hybrid Search
| Decision | Choice | Why |
|----------|--------|-----|
| Embedding model | `nomic-embed-text` via MLX or Ollama | Best local quality/speed on Apple Silicon |
| Vector storage | `sqlite-vss` or `numpy` + flat index | No external DB, fits local-first philosophy |
| TUI framework | `textual` (by Rich author) | Python, runs in terminal, beautiful |
| Fuzzy matching | Embedding similarity + BM25 hybrid | Semantic + lexical for best recall |
| Chunk strategy | Sliding window with overlap | Preserves context at boundaries |

### V4: Multimodal
| Decision | Choice | Why |
|----------|--------|-----|
| VLM model | `Qwen2-VL-2B-Instruct-4bit` (~1.2GB) | Fits 8GB, good quality |
| Screenshot capture | `screencapture -x` (macOS native) | Silent, fast, no deps |
| PDF rendering | `pymupdf` page-to-image | Already a dependency |
| Image tokenization | VLM native (handles internally) | mlx-vlm provides this |

### V5: Ecosystem
| Decision | Choice | Why |
|----------|--------|-----|
| API server | FastAPI + SSE | Async, streaming, well-known |
| Obsidian plugin | TypeScript, uses HTTP API | Standard Obsidian plugin arch |
| Cache daemon | Long-running `mcq serve` | Keeps model + cache in memory for fast queries |

## The Competitive Landscape

| Tool | What it does | mcq's advantage |
|------|-------------|-----------------|
| Fabric | Cloud LLM + prompt patterns | mcq is local-first, has persistent memory |
| Simon Willison's `llm` | Great CLI, cloud-dependent | mcq runs on-device, zero cloud |
| Cursor / Copilot | IDE-integrated, cloud | mcq is standalone, works anywhere |
| privateGPT / localGPT | RAG with chunks | mcq uses KV cache (full attention, not fragments) |
| Spotlight / Alfred | Keyword file search | mcq is semantic, understands content |
| NotebookLM | Google's document AI | mcq is local, private, extensible |

**mcq's unique position**: Local-first semantic memory with persistent attention state. Not RAG. Not keyword search. The model has **read** your documents.

## Milestones

| Phase | Codename | Target | Key Deliverable |
|-------|----------|--------|-----------------|
| V1 | Foundation | ✅ Done | Core pipeline + CLI |
| V1.5 | Polish | ✅ Done | Charm-quality UX, pipe support |
| V2 | Memory | Next | Obsidian/QMD, watch mode, multi-corpus |
| V3 | Search | After V2 | Local embeddings, fuzzy finder TUI, hybrid query |
| V4 | Vision | After V3 | Screenshots, visual PDF, VLM integration |
| V5 | Ecosystem | After V4 | API server, Obsidian plugin, Raycast, menu bar |

## Design Principles (Permanent)

1. **Local-first, always.** No cloud dependency at query time. Ever.
2. **Stdout is sacred.** Data on stdout, chrome on stderr. Always pipeable.
3. **Compose, don't contain.** Work with Fabric, fzf, jq, pbcopy. Don't reinvent them.
4. **Fast by default.** Cache load <1s. Search <100ms. TTFT <500ms.
5. **Beautiful when interactive, invisible when piped.** Detect TTY, adapt.
6. **Progressive disclosure.** Simple defaults, power flags. `mcq query x "y"` just works.
7. **Deterministic.** Same input → same cache → same hash. Always reproducible.
8. **The cache is the product.** Everything revolves around building, managing, and querying persistent KV state.
