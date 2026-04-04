"""mcq app — Spotlight-style local knowledge base interface.

The main screen is a clean chat + corpus browser.
Press Ctrl+K to summon a floating Spotlight search overlay —
just like macOS Spotlight or Raycast. Escape dismisses it.

Launch with: mcq app
"""
from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, Center, Middle
from textual.screen import ModalScreen
from textual.widgets import (
    Header, Footer, Input, Static, ListView, ListItem,
    Label, RichLog,
)
from textual import on, work

from mcq.core.constants import REGISTRY_DB, ARTIFACTS_DIR
from mcq.core.types import Corpus, CorpusChunk
from mcq.search.finder import TextFinder, SearchResult


# ---------------------------------------------------------------------------
# Spotlight Overlay (Modal)
# ---------------------------------------------------------------------------


class SpotlightResultItem(ListItem):
    """A single result in the Spotlight overlay."""
    def __init__(self, result: SearchResult) -> None:
        super().__init__()
        self.result = result

    def compose(self) -> ComposeResult:
        score = f"  [{self.result.score:.1f}]" if self.result.score > 0 else ""
        snippet = self.result.snippet.replace("\n", " ")[:80]
        yield Label(
            f"[bold]{self.result.chunk.source_path}[/bold]{score}\n"
            f"[dim]{snippet}[/dim]"
        )


class SpotlightScreen(ModalScreen[str | None]):
    """Raycast/Spotlight-style floating search overlay.

    Summoned with Ctrl+K, dismissed with Escape.
    Results update live as you type. Enter selects a result.
    """

    CSS = """
    SpotlightScreen {
        align: center middle;
    }

    #spotlight-container {
        width: 80;
        max-width: 90%;
        height: auto;
        max-height: 70%;
        background: $surface;
        border: heavy $accent;
        padding: 1 2;
    }

    #spotlight-title {
        text-align: center;
        text-style: bold;
        color: $accent;
        padding: 0 0 1 0;
    }

    #spotlight-input {
        margin: 0 0 1 0;
    }

    #spotlight-results {
        height: auto;
        max-height: 20;
        min-height: 3;
    }

    #spotlight-preview {
        height: auto;
        max-height: 12;
        border-top: solid $primary-darken-2;
        padding: 1 1 0 1;
        margin: 1 0 0 0;
        overflow-y: auto;
    }

    #spotlight-hint {
        text-align: center;
        color: $text-muted;
        padding: 1 0 0 0;
    }

    SpotlightResultItem {
        padding: 0 1;
        height: auto;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss_spotlight", "Close", show=False),
    ]

    def __init__(self, corpus: Corpus | None, corpus_name: str) -> None:
        super().__init__()
        self._corpus = corpus
        self._corpus_name = corpus_name

    def compose(self) -> ComposeResult:
        with Vertical(id="spotlight-container"):
            yield Static(f"[bold]mcq[/bold] [dim]·[/dim] {self._corpus_name}", id="spotlight-title")
            yield Input(placeholder="Search...", id="spotlight-input")
            yield ListView(id="spotlight-results")
            yield Static("", id="spotlight-preview")
            yield Static("↑↓ navigate · Enter select · Esc close · Tab to chat", id="spotlight-hint")

    def on_mount(self) -> None:
        self.query_one("#spotlight-input").focus()
        if self._corpus:
            self._show_all()

    def _show_all(self) -> None:
        results = self.query_one("#spotlight-results", ListView)
        results.clear()
        if not self._corpus:
            return
        for chunk in sorted(self._corpus.chunks, key=lambda c: c.source_path)[:15]:
            r = SearchResult(chunk=chunk, score=0.0, snippet=chunk.content[:100])
            results.append(SpotlightResultItem(r))

    @on(Input.Changed, "#spotlight-input")
    def on_search(self, event: Input.Changed) -> None:
        if not self._corpus:
            return
        query = event.value.strip()
        results_list = self.query_one("#spotlight-results", ListView)
        results_list.clear()

        if not query:
            self._show_all()
            self.query_one("#spotlight-preview", Static).update("")
            return

        results = TextFinder.search(self._corpus, query, top_k=10)
        for r in results:
            results_list.append(SpotlightResultItem(r))

    @on(ListView.Highlighted, "#spotlight-results")
    def on_highlight(self, event: ListView.Highlighted) -> None:
        preview = self.query_one("#spotlight-preview", Static)
        if event.item and isinstance(event.item, SpotlightResultItem):
            chunk = event.item.result.chunk
            lines = chunk.content.split("\n")[:15]
            text = "\n".join(lines)
            if len(chunk.content.split("\n")) > 15:
                text += "\n[dim]...[/dim]"
            preview.update(text)
        else:
            preview.update("")

    @on(ListView.Selected, "#spotlight-results")
    def on_select(self, event: ListView.Selected) -> None:
        if event.item and isinstance(event.item, SpotlightResultItem):
            self.dismiss(event.item.result.chunk.source_path)

    @on(Input.Submitted, "#spotlight-input")
    def on_submit(self, event: Input.Submitted) -> None:
        """On Enter in search bar, select first result or pass query to chat."""
        query = event.value.strip()
        if not query:
            return
        # If there are results, select the first one
        results_list = self.query_one("#spotlight-results", ListView)
        if results_list.children:
            first = results_list.children[0]
            if isinstance(first, SpotlightResultItem):
                self.dismiss(f"__query__:{query}")
                return
        self.dismiss(f"__query__:{query}")

    def action_dismiss_spotlight(self) -> None:
        self.dismiss(None)


# ---------------------------------------------------------------------------
# Corpus Sidebar
# ---------------------------------------------------------------------------


class CorpusItem(ListItem):
    def __init__(self, name: str, corpus_status: str, chunks: int) -> None:
        super().__init__()
        self.corpus_name = name
        self.corpus_status = corpus_status
        self.chunk_count = chunks

    def compose(self) -> ComposeResult:
        icon = "●" if self.corpus_status == "cached" else "○"
        color = "green" if self.corpus_status == "cached" else "yellow"
        yield Label(f"[{color}]{icon}[/{color}] [bold]{self.corpus_name}[/bold] [dim]{self.chunk_count}f[/dim]")


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------


class McqApp(App):
    """mcq — local knowledge base interface."""

    CSS = """
    Screen {
        layout: horizontal;
    }

    #sidebar {
        width: 26;
        border-right: solid $primary-darken-2;
        padding: 0;
    }

    #sidebar-title {
        text-align: center;
        padding: 1;
        text-style: bold;
        color: $accent;
        border-bottom: solid $primary-darken-2;
    }

    #corpus-list {
        height: 1fr;
    }

    #corpus-list ListItem {
        padding: 0 1;
    }

    #main-area {
        width: 1fr;
        layout: vertical;
    }

    #chat-log {
        height: 1fr;
        padding: 1 2;
        overflow-y: auto;
    }

    #chat-input-area {
        dock: bottom;
        height: 3;
        padding: 0 1;
    }

    #chat-input {
        width: 100%;
    }

    #hint-bar {
        dock: bottom;
        height: 1;
        padding: 0 2;
        background: $surface;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("ctrl+k", "spotlight", "Spotlight", show=True, priority=True),
        Binding("ctrl+j", "focus_chat", "Chat", show=True),
        Binding("ctrl+q", "quit", "Quit", show=True),
        Binding("ctrl+r", "refresh", "Refresh", show=True),
    ]

    TITLE = "mcq"
    SUB_TITLE = "local knowledge base"

    def __init__(self) -> None:
        super().__init__()
        self._corpora: dict[str, dict] = {}
        self._active_corpus: Corpus | None = None
        self._active_corpus_name: str = "(none)"

    def compose(self) -> ComposeResult:
        yield Header()

        with Horizontal():
            with Vertical(id="sidebar"):
                yield Static("[bold]Corpora[/bold]", id="sidebar-title")
                yield ListView(id="corpus-list")

            with Vertical(id="main-area"):
                yield RichLog(id="chat-log", wrap=True, markup=True)
                with Horizontal(id="chat-input-area"):
                    yield Input(placeholder="Ask a question... (Ctrl+J) · Ctrl+K for Spotlight", id="chat-input")

        yield Static("  Ctrl+K Spotlight  ·  Ctrl+J Chat  ·  Ctrl+R Refresh  ·  Ctrl+Q Quit", id="hint-bar")

    def on_mount(self) -> None:
        self._load_corpora()
        chat = self.query_one("#chat-log", RichLog)
        chat.write("[bold]Welcome to mcq[/bold]\n")
        chat.write("Select a corpus from the sidebar, then:\n")
        chat.write("  [bold]Ctrl+K[/bold]  Spotlight search (Raycast-style overlay)\n")
        chat.write("  [bold]Ctrl+J[/bold]  Focus chat input\n")
        chat.write("  Type a question to chat with your documents\n")

    # --- Data Loading ---

    def _load_corpora(self) -> None:
        try:
            from mcq.cache.registry import CacheRegistry
            registry = CacheRegistry(REGISTRY_DB)
            corpora = registry.list_corpora()
            artifacts = registry.list_all()
            cached_names = {a.corpus_name for a in artifacts}

            corpus_list = self.query_one("#corpus-list", ListView)
            corpus_list.clear()
            self._corpora = {}

            for c in corpora:
                name = c["name"]
                self._corpora[name] = c
                corpus_list.append(CorpusItem(
                    name=name,
                    corpus_status="cached" if name in cached_names else "ingested",
                    chunks=c["chunk_count"],
                ))

            if not corpora:
                chat = self.query_one("#chat-log", RichLog)
                chat.write("\n[yellow]No corpora found. Run:[/yellow]")
                chat.write("  mcq ingest <path> -n <name>")
                chat.write("  mcq build <name>\n")
        except Exception as e:
            chat = self.query_one("#chat-log", RichLog)
            chat.write(f"[red]Error: {e}[/red]")

    def _load_corpus(self, name: str) -> None:
        info = self._corpora.get(name)
        if not info:
            return
        try:
            from mcq.ingest.ingestor import CorpusIngestor
            self._active_corpus = CorpusIngestor.ingest(
                Path(info["source_path"]), name=name
            )
            self._active_corpus_name = name
            chat = self.query_one("#chat-log", RichLog)
            chat.write(f"\n[green]● {name}[/green] loaded ({len(self._active_corpus.chunks)} files)")
            chat.write(f"  Press [bold]Ctrl+K[/bold] to search, or type a question below.\n")
        except Exception as e:
            chat = self.query_one("#chat-log", RichLog)
            chat.write(f"[red]Error loading '{name}': {e}[/red]")

    # --- Event Handlers ---

    @on(ListView.Selected, "#corpus-list")
    def on_corpus_selected(self, event: ListView.Selected) -> None:
        if isinstance(event.item, CorpusItem):
            self._load_corpus(event.item.corpus_name)

    @on(Input.Submitted, "#chat-input")
    def on_chat_submit(self, event: Input.Submitted) -> None:
        question = event.value.strip()
        if not question:
            return
        event.input.value = ""
        self._handle_chat(question)

    def _handle_chat(self, question: str) -> None:
        chat = self.query_one("#chat-log", RichLog)
        chat.write(f"\n[bold green]You:[/bold green] {question}")

        if not self._active_corpus:
            chat.write("[yellow]Select a corpus first from the sidebar.[/yellow]")
            return

        # Show BM25 search results immediately
        results = TextFinder.search(self._active_corpus, question, top_k=3)
        if results:
            chat.write(f"[dim]Found {len(results)} relevant files:[/dim]")
            for r in results:
                snippet = r.snippet.replace("\n", " ")[:100]
                chat.write(f"  [bold]{r.chunk.source_path}[/bold] [dim]({r.score:.1f})[/dim]")
                chat.write(f"  [dim]{snippet}[/dim]")

        # Try LLM query in background
        self._try_llm_query(question)

    @work(thread=True)
    def _try_llm_query(self, question: str) -> None:
        try:
            from mcq.cache.registry import CacheRegistry
            from mcq.cache.store import CacheStore
            from mcq.inference.engine import QueryEngine
            from mlx_lm import load
            from mcq.core.config import McqConfig

            config = McqConfig.load()
            registry = CacheRegistry(REGISTRY_DB)
            refs = registry.get_by_corpus_name(self._active_corpus_name, model_id=config.model)

            if not refs:
                self.call_from_thread(
                    self.query_one("#chat-log", RichLog).write,
                    f"\n[yellow]No cache built. Run: mcq build {self._active_corpus_name}[/yellow]"
                )
                return

            ref = refs[0]
            store = CacheStore(ARTIFACTS_DIR)
            mlx_model, tokenizer = load(config.model)
            prompt_cache, _ = store.load(ref)
            result = QueryEngine.query(mlx_model, tokenizer, prompt_cache, question, max_tokens=config.max_tokens)

            self.call_from_thread(
                self.query_one("#chat-log", RichLog).write,
                f"\n[bold blue]mcq:[/bold blue] {result.text}\n"
                f"[dim]({result.total_tokens} tok · {result.decode_tokens_per_sec:.0f} tok/s · TTFT {result.ttft_ms:.0f}ms)[/dim]"
            )
        except ImportError:
            self.call_from_thread(
                self.query_one("#chat-log", RichLog).write,
                "\n[dim]LLM query requires Apple Silicon. Showing search results only.[/dim]"
            )
        except Exception as e:
            self.call_from_thread(
                self.query_one("#chat-log", RichLog).write,
                f"\n[red]Query error: {e}[/red]"
            )

    # --- Actions ---

    def action_spotlight(self) -> None:
        """Summon the Spotlight search overlay."""
        def handle_spotlight_result(result: str | None) -> None:
            if result is None:
                return
            if result.startswith("__query__:"):
                # User pressed Enter on search — send to chat
                q = result[len("__query__:"):]
                self._handle_chat(q)
            else:
                # User selected a file — show preview in chat
                chat = self.query_one("#chat-log", RichLog)
                chat.write(f"\n[bold]Selected:[/bold] {result}")
                if self._active_corpus:
                    for chunk in self._active_corpus.chunks:
                        if chunk.source_path == result:
                            lines = chunk.content.split("\n")[:30]
                            chat.write("\n".join(lines))
                            if len(chunk.content.split("\n")) > 30:
                                chat.write("[dim]...[/dim]")
                            break

        self.push_screen(
            SpotlightScreen(self._active_corpus, self._active_corpus_name),
            callback=handle_spotlight_result,
        )

    def action_focus_chat(self) -> None:
        self.query_one("#chat-input").focus()

    def action_refresh(self) -> None:
        self._load_corpora()
        chat = self.query_one("#chat-log", RichLog)
        chat.write("[dim]Refreshed corpus list.[/dim]")


def run_app() -> None:
    """Launch the mcq GUI app."""
    app = McqApp()
    app.run()
