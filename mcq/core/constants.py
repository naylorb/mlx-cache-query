"""Global constants for mcq.

Paths, defaults, and version markers. Nothing here should import
from any other mcq module.
"""
from pathlib import Path

# ── Defaults ─────────────────────────────────────────────────────────
DEFAULT_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
DEFAULT_QUERY_BUDGET = 2048
DEFAULT_MAX_TOKENS = 512

# ── Version markers (bump when format changes) ──────────────────────
PROMPT_TEMPLATE_VERSION = "v1"
NORMALIZATION_VERSION = "v1"

# ── Paths ────────────────────────────────────────────────────────────
APP_DIR = Path.home() / ".mcq"
ARTIFACTS_DIR = APP_DIR / "artifacts"
REGISTRY_DB = APP_DIR / "registry.db"

# ── Ingestion ────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = frozenset({
    ".txt", ".md", ".py", ".rs", ".js", ".ts", ".jsx", ".tsx",
    ".json", ".toml", ".yaml", ".yml", ".cfg", ".ini",
    ".c", ".h", ".cpp", ".hpp", ".java", ".go", ".rb", ".sh",
    ".pdf", ".qmd",
})

SKIP_DIRS = frozenset({
    ".obsidian", ".git", "__pycache__", "node_modules", ".venv", "venv",
})

# ── Exit codes (API contract) ───────────────────────────────────────
EXIT_OK = 0
EXIT_QUERY_FAILED = 1
EXIT_CACHE_NOT_FOUND = 2
EXIT_BUILD_FAILED = 3
