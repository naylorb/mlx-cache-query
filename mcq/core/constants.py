from pathlib import Path

DEFAULT_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
DEFAULT_QUERY_BUDGET = 2048
PROMPT_TEMPLATE_VERSION = "v1"
NORMALIZATION_VERSION = "v1"

APP_DIR = Path.home() / ".mlx-cache-query"
ARTIFACTS_DIR = APP_DIR / "artifacts"
REGISTRY_DB = APP_DIR / "registry.db"

SUPPORTED_EXTENSIONS = frozenset({
    ".txt", ".md", ".py", ".rs", ".js", ".ts", ".jsx", ".tsx",
    ".json", ".toml", ".yaml", ".yml", ".cfg", ".ini",
    ".c", ".h", ".cpp", ".hpp", ".java", ".go", ".rb", ".sh",
    ".pdf",
})
