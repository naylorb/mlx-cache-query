"""FileWatcher — detects changes in corpus source directories."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from mcq.core.constants import SUPPORTED_EXTENSIONS, SKIP_DIRS


class _ChangeHandler(FileSystemEventHandler):
    def __init__(self, callback: Callable[[str], None]) -> None:
        self._callback = callback
        self._last_event = 0.0
        self._debounce_s = 1.0  # debounce to avoid rapid-fire rebuilds

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        # Skip non-content files
        if path.suffix not in SUPPORTED_EXTENSIONS:
            return
        # Skip known non-content directories
        try:
            parts = path.parts
            if any(part in SKIP_DIRS for part in parts):
                return
        except Exception:
            pass
        # Debounce
        now = time.monotonic()
        if now - self._last_event < self._debounce_s:
            return
        self._last_event = now
        self._callback(str(path))


class FileWatcher:
    def __init__(self, path: Path, on_change: Callable[[str], None]) -> None:
        self._path = Path(path)
        self._observer = Observer()
        self._handler = _ChangeHandler(on_change)

    def start(self) -> None:
        """Start watching. Non-blocking — runs in a background thread."""
        self._observer.schedule(self._handler, str(self._path), recursive=True)
        self._observer.start()

    def stop(self) -> None:
        """Stop watching."""
        self._observer.stop()
        self._observer.join(timeout=5)

    def run_forever(self) -> None:
        """Block until interrupted."""
        self.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
