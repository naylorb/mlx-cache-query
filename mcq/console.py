"""
Charm-style output separation.

status  → stderr (spinners, progress, human-facing chrome)
output  → stdout (data: answers, JSON, pipeable output)

When stdout is a pipe: output has no color, no markup.
When stderr is a pipe: status goes quiet.
Rich auto-detects TTY on each Console independently.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from typing import Any

from rich.console import Console

# Human-facing status: always stderr
status = Console(stderr=True, highlight=False)

# Machine-facing data: stdout, respects pipe detection
output = Console(highlight=False)


def is_interactive() -> bool:
    """True when stdin is a real terminal (not piped)."""
    return sys.stdin.isatty()


def is_piped() -> bool:
    """True when stdout is going to a pipe (not a terminal)."""
    return not sys.stdout.isatty()


def print_json_or_human(
    data: Any,
    *,
    use_json: bool,
    human_fn: callable,
) -> None:
    """Print structured data as JSON (to stdout) or call human_fn for pretty output.

    human_fn receives `data` and should use `status` or `output` to print.
    """
    if use_json:
        d = asdict(data) if hasattr(data, "__dataclass_fields__") else data
        output.print_json(json.dumps(d, default=str))
    else:
        human_fn(data)


def emit_json(data: Any) -> None:
    """Write a single JSON object to stdout."""
    if hasattr(data, "__dataclass_fields__"):
        data = asdict(data)
    sys.stdout.write(json.dumps(data, default=str) + "\n")
    sys.stdout.flush()
