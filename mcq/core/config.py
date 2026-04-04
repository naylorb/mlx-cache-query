"""Configuration: ~/.mcq.toml or MCQ_CONFIG env var.

CLI flags > config file > defaults. Simple and predictable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_CONFIG_PATH = Path(os.environ.get("MCQ_CONFIG", Path.home() / ".mcq.toml"))


@dataclass
class McqConfig:
    model: str = "mlx-community/Qwen2.5-3B-Instruct-4bit"
    max_tokens: int = 512
    query_budget: int = 2048

    @classmethod
    def load(cls) -> McqConfig:
        config = cls()
        if not _CONFIG_PATH.exists():
            return config
        try:
            data = _parse_toml(_CONFIG_PATH.read_text())
            for key, value in data.items():
                if hasattr(config, key):
                    expected = type(getattr(config, key))
                    setattr(config, key, expected(value))
        except Exception:
            pass
        return config


def _parse_toml(text: str) -> dict:
    """Minimal TOML parser for flat key=value files."""
    data = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("["):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        try:
            data[key] = int(val)
        except ValueError:
            data[key] = val
    return data
