"""Configuration file support for mcq.

Loads settings from ~/.mcq.toml (or MCQ_CONFIG env var).
CLI flags override config file values. Config file overrides defaults.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_CONFIG_PATH = Path(os.environ.get("MCQ_CONFIG", Path.home() / ".mcq.toml"))


@dataclass
class McqConfig:
    model: str = "mlx-community/Qwen2.5-3B-Instruct-4bit"
    max_tokens: int = 512
    query_budget: int = 2048
    port: int = 8420
    host: str = "127.0.0.1"
    editor: str = os.environ.get("EDITOR", "vim")

    @classmethod
    def load(cls) -> "McqConfig":
        """Load config from ~/.mcq.toml, falling back to defaults."""
        config = cls()
        if not _CONFIG_PATH.exists():
            return config
        try:
            text = _CONFIG_PATH.read_text()
            data = _parse_toml(text)
            for key, value in data.items():
                if hasattr(config, key):
                    expected_type = type(getattr(config, key))
                    setattr(config, key, expected_type(value))
        except Exception:
            pass  # Bad config? Use defaults.
        return config

    def save(self) -> None:
        """Save current config to ~/.mcq.toml."""
        lines = [
            "# mcq configuration",
            f'model = "{self.model}"',
            f"max_tokens = {self.max_tokens}",
            f"query_budget = {self.query_budget}",
            f"port = {self.port}",
            f'host = "{self.host}"',
        ]
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CONFIG_PATH.write_text("\n".join(lines) + "\n")


def _parse_toml(text: str) -> dict:
    """Minimal TOML parser for flat key=value files. No dependency needed."""
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
        # Try to parse as int
        try:
            data[key] = int(val)
        except ValueError:
            data[key] = val
    return data
