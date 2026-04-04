import time
from pathlib import Path
from unittest.mock import MagicMock
from mcq.watch.watcher import FileWatcher


def test_watcher_creation(tmp_path):
    callback = MagicMock()
    watcher = FileWatcher(tmp_path, on_change=callback)
    assert watcher is not None


def test_watcher_detects_file_change(tmp_path):
    changes = []
    def on_change(path):
        changes.append(path)

    (tmp_path / "test.py").write_text("x = 1\n")
    watcher = FileWatcher(tmp_path, on_change=on_change)
    watcher.start()

    time.sleep(0.5)  # let watcher settle
    (tmp_path / "test.py").write_text("x = 2\n")
    time.sleep(2)  # wait for debounce

    watcher.stop()
    assert len(changes) >= 1


def test_watcher_ignores_non_code_files(tmp_path):
    changes = []
    def on_change(path):
        changes.append(path)

    watcher = FileWatcher(tmp_path, on_change=on_change)
    watcher.start()

    time.sleep(0.5)
    (tmp_path / "test.bin").write_bytes(b"\x00\x01")
    time.sleep(2)

    watcher.stop()
    assert len(changes) == 0
