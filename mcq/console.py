"""Two-fd output contract.

stderr → chrome (spinners, progress, human-facing status)
stdout → data (answers, JSON, pipeable output)

When stdout is a pipe: no color, no markup — clean data.
When stderr is not a TTY: status goes quiet.
This is the Charm.sh way.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from typing import Any

from rich.console import Console

# Chrome: always stderr. Spinners, progress, human messages.
status = Console(stderr=True, highlight=False)

# Data: stdout. Respects TTY detection automatically.
output = Console(highlight=False)


def is_interactive() -> bool:
    """True when stdin is a real terminal (not piped)."""
    return sys.stdin.isatty()


def emit_json(data: Any) -> None:
    """Write a single JSON object/array to stdout. No rich, no color."""
    if hasattr(data, "__dataclass_fields__"):
        data = asdict(data)
    sys.stdout.write(json.dumps(data, default=str) + "\n")
    sys.stdout.flush()


def emit_text(text: str, end: str = "") -> None:
    """Write raw text to stdout. For streaming tokens."""
    sys.stdout.write(text + end)
    sys.stdout.flush()
