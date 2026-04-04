"""Interactive fuzzy finder TUI using Textual."""
from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Static, ListView, ListItem, Label
from textual.containers import Vertical, Horizontal
from textual.reactive import reactive
from textual import on

from mcq.core.types import Corpus
from mcq.search.finder import TextFinder, SearchResult


class PreviewPanel(Static):
    """Shows preview of selected search result."""
    DEFAULT_CSS = """
    PreviewPanel {
        height: 1fr;
        border: solid $accent;
        padding: 1 2;
        overflow-y: auto;
    }
    """


class ResultItem(ListItem):
    """A single search result in the list."""
    def __init__(self, result: SearchResult) -> None:
        super().__init__()
        self.result = result

    def compose(self) -> ComposeResult:
        score = f"{self.result.score:.2f}"
        yield Label(f"[bold]{self.result.chunk.source_path}[/bold]  [dim]({score})[/dim]")


class FinderApp(App):
    """Interactive search TUI for mcq."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #search-input {
        dock: top;
        margin: 1 2;
    }
    #results-list {
        height: 40%;
        border: solid $primary;
        margin: 0 2;
    }
    #preview {
        height: 1fr;
        margin: 0 2 1 2;
        border: solid $accent;
        padding: 1 2;
        overflow-y: auto;
    }
    #status-bar {
        dock: bottom;
        height: 1;
        padding: 0 2;
        background: $surface;
        color: $text-muted;
    }
    ResultItem {
        padding: 0 1;
    }
    ResultItem > Label {
        width: 100%;
    }
    """

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("escape", "quit", "Quit"),
    ]

    def __init__(self, corpus: Corpus, corpus_name: str) -> None:
        super().__init__()
        self.corpus = corpus
        self.corpus_name = corpus_name
        self._results: list[SearchResult] = []
        self.selected_path: str | None = None  # Set when user presses Enter

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Input(placeholder="Type to search...", id="search-input")
        yield ListView(id="results-list")
        yield Static("", id="preview")
        yield Static(
            f"  [bold]{self.corpus_name}[/bold] · {len(self.corpus.chunks)} files · ↑↓ Navigate · Enter: Select · Ctrl+Q: Quit",
            id="status-bar",
        )

    def on_mount(self) -> None:
        self.title = f"mcq find: {self.corpus_name}"
        # Show all files initially
        self._show_all()

    def _show_all(self) -> None:
        """Show all corpus chunks (no search filter)."""
        list_view = self.query_one("#results-list", ListView)
        list_view.clear()
        for chunk in sorted(self.corpus.chunks, key=lambda c: c.source_path):
            result = SearchResult(chunk=chunk, score=0.0, snippet=chunk.content[:200])
            item = ResultItem(result)
            list_view.append(item)

    @on(Input.Changed, "#search-input")
    def on_search_changed(self, event: Input.Changed) -> None:
        query = event.value.strip()
        list_view = self.query_one("#results-list", ListView)
        list_view.clear()

        if not query:
            self._show_all()
            return

        self._results = TextFinder.search(self.corpus, query, top_k=20)
        for result in self._results:
            item = ResultItem(result)
            list_view.append(item)

    @on(ListView.Highlighted, "#results-list")
    def on_result_highlighted(self, event: ListView.Highlighted) -> None:
        preview = self.query_one("#preview", Static)
        if event.item and isinstance(event.item, ResultItem):
            chunk = event.item.result.chunk
            # Show first ~40 lines of content
            lines = chunk.content.split("\n")[:40]
            text = "\n".join(lines)
            if len(chunk.content.split("\n")) > 40:
                text += "\n..."
            preview.update(f"[bold]{chunk.source_path}[/bold]\n\n{text}")
        else:
            preview.update("")

    @on(ListView.Selected, "#results-list")
    def on_result_selected(self, event: ListView.Selected) -> None:
        if event.item and isinstance(event.item, ResultItem):
            self.selected_path = event.item.result.chunk.source_path
            self.exit(self.selected_path)


def run_finder_tui(corpus: Corpus, corpus_name: str) -> str | None:
    """Run the interactive finder TUI. Returns selected file path or None."""
    app = FinderApp(corpus, corpus_name)
    result = app.run()
    return result
