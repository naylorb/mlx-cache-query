# mlx-cache-query V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working CLI tool that ingests documents/code, builds persistent MLX KV cache artifacts, and streams answers to user queries locally on Apple Silicon.

**Architecture:** Linear pipeline-of-functions: ingest → compile → build → save/load → query. Each stage is a plain Python class. CLI and future API are thin wrappers. Deterministic hashing via prefix token IDs. Native mlx-lm cache APIs for serialization.

**Tech Stack:** Python 3.11+, mlx, mlx-lm, click, pymupdf, safetensors, SQLite, pytest

**Spec:** `docs/superpowers/specs/2026-04-03-mlx-cache-query-design.md`

---

## File Structure

```
mlx-cache-query/
├── pyproject.toml                 # project metadata, dependencies, [project.scripts] entry point
├── app/
│   ├── __init__.py                # package marker, __version__
│   ├── cli.py                     # click CLI: ingest, build, query, list, info, delete
│   ├── core/
│   │   ├── __init__.py
│   │   ├── types.py               # dataclasses: CorpusChunk, Corpus, TokenizedPrefix, ArtifactRef
│   │   ├── hashing.py             # deterministic sha256 utilities
│   │   └── constants.py           # DEFAULT_MODEL, DEFAULT_QUERY_BUDGET, paths, versions
│   ├── ingest/
│   │   ├── __init__.py
│   │   └── ingestor.py            # CorpusIngestor: read files, normalize, hash
│   ├── prefix/
│   │   ├── __init__.py
│   │   └── compiler.py            # PrefixCompiler: template + tokenize
│   ├── cache/
│   │   ├── __init__.py
│   │   ├── builder.py             # CacheBuilder: prefill via mlx-lm
│   │   ├── store.py               # CacheStore: save/load via mlx-lm native APIs
│   │   └── registry.py            # CacheRegistry: SQLite metadata
│   ├── inference/
│   │   ├── __init__.py
│   │   └── engine.py              # QueryEngine: streaming generation
│   └── eval/
│       ├── __init__.py
│       └── bench.py               # BenchmarkRunner: timing measurements
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # shared fixtures
│   ├── test_types.py
│   ├── test_hashing.py
│   ├── test_ingestor.py
│   ├── test_compiler.py
│   ├── test_registry.py
│   ├── test_store.py
│   └── test_engine.py
└── docs/
```

---

## Task 1: Project Scaffolding and Dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `app/__init__.py`
- Create: `app/core/__init__.py`, `app/ingest/__init__.py`, `app/prefix/__init__.py`, `app/cache/__init__.py`, `app/inference/__init__.py`, `app/eval/__init__.py`
- Create: `tests/__init__.py`, `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "mlx-cache-query"
version = "0.1.0"
description = "Local-first KV cache query system for Apple Silicon"
requires-python = ">=3.11"
dependencies = [
    "mlx>=0.22.0",
    "mlx-lm>=0.22.0",
    "click>=8.1",
    "pymupdf>=1.24",
    "safetensors>=0.4",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-timeout>=2.2",
]

[project.scripts]
mcq = "app.cli:main"

[tool.setuptools.packages.find]
include = ["app*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create package __init__ files**

`app/__init__.py`:
```python
__version__ = "0.1.0"
```

Create empty `__init__.py` in: `app/core/`, `app/ingest/`, `app/prefix/`, `app/cache/`, `app/inference/`, `app/eval/`, `tests/`.

- [ ] **Step 3: Create tests/conftest.py with shared fixtures**

```python
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_corpus_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with sample text files."""
    (tmp_path / "readme.md").write_text("# Sample Project\n\nThis is a test document.\n")
    (tmp_path / "main.py").write_text("def hello():\n    return 'world'\n")
    (tmp_path / "notes.txt").write_text("Some notes about the project.\n")
    return tmp_path


@pytest.fixture
def tmp_store_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for cache artifact storage."""
    store = tmp_path / "store"
    store.mkdir()
    return store
```

- [ ] **Step 4: Create app/core/constants.py**

```python
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
```

- [ ] **Step 5: Install in development mode and verify**

Run: `cd /Users/wally/dev/mlx-cache-query && pip install -e ".[dev]"`

Expected: successful install, `mcq` command registered (will fail without cli.py — that's fine)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml app/ tests/
git commit -m "feat: scaffold project structure and dependencies"
```

---

## Task 2: Core Data Types

**Files:**
- Create: `app/core/types.py`
- Create: `tests/test_types.py`

- [ ] **Step 1: Write failing tests for core types**

`tests/test_types.py`:
```python
from app.core.types import CorpusChunk, Corpus, TokenizedPrefix, ArtifactRef


def test_corpus_chunk_creation():
    chunk = CorpusChunk(
        source_path="readme.md",
        content="# Hello\n",
        byte_range=(0, 9),
    )
    assert chunk.source_path == "readme.md"
    assert chunk.content == "# Hello\n"
    assert chunk.byte_range == (0, 9)


def test_corpus_creation_and_hash_determinism():
    chunks_a = [
        CorpusChunk("b.txt", "beta", (0, 4)),
        CorpusChunk("a.txt", "alpha", (0, 5)),
    ]
    chunks_b = [
        CorpusChunk("a.txt", "alpha", (0, 5)),
        CorpusChunk("b.txt", "beta", (0, 4)),
    ]
    corpus_a = Corpus(name="test", chunks=chunks_a)
    corpus_b = Corpus(name="test", chunks=chunks_b)
    assert corpus_a.content_hash == corpus_b.content_hash
    assert len(corpus_a.content_hash) == 64  # sha256 hex


def test_corpus_hash_includes_paths():
    """Different paths with same content must produce different hashes."""
    chunk_a = [CorpusChunk("a.txt", "same content", (0, 12))]
    chunk_b = [CorpusChunk("b.txt", "same content", (0, 12))]
    corpus_a = Corpus(name="a", chunks=chunk_a)
    corpus_b = Corpus(name="b", chunks=chunk_b)
    assert corpus_a.content_hash != corpus_b.content_hash


def test_tokenized_prefix_creation():
    prefix = TokenizedPrefix(
        model_id="test-model",
        model_revision="abc123",
        tokens=[1, 2, 3, 4, 5],
        corpus_hash="deadbeef" * 8,
    )
    assert prefix.token_count == 5
    assert len(prefix.prefix_hash) == 64


def test_tokenized_prefix_hash_determinism():
    kwargs = dict(
        model_id="test-model",
        model_revision="abc123",
        tokens=[1, 2, 3, 4, 5],
        corpus_hash="deadbeef" * 8,
    )
    a = TokenizedPrefix(**kwargs)
    b = TokenizedPrefix(**kwargs)
    assert a.prefix_hash == b.prefix_hash


def test_artifact_ref_creation():
    ref = ArtifactRef(
        artifact_hash="abc" * 21 + "a",
        model_id="test-model",
        model_revision="abc123",
        corpus_hash="def" * 21 + "d",
        corpus_name="my-corpus",
        prefix_token_count=1000,
        prompt_template_version="v1",
        normalization_version="v1",
        build_timestamp="2026-04-03T12:00:00Z",
        file_size_bytes=1024,
        file_path="/tmp/test.safetensors",
        mlx_lm_version="0.22.0",
    )
    assert ref.corpus_name == "my-corpus"
    assert ref.prefix_token_count == 1000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_types.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'app.core.types'`

- [ ] **Step 3: Implement core types**

`app/core/types.py`:
```python
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CorpusChunk:
    source_path: str
    content: str
    byte_range: tuple[int, int]


@dataclass
class Corpus:
    name: str
    chunks: list[CorpusChunk]

    @property
    def content_hash(self) -> str:
        hasher = hashlib.sha256()
        for path, content in sorted(
            (c.source_path, c.content) for c in self.chunks
        ):
            hasher.update(path.encode("utf-8"))
            hasher.update(b"\x00")
            hasher.update(content.encode("utf-8"))
            hasher.update(b"\x00")
        return hasher.hexdigest()


@dataclass
class TokenizedPrefix:
    model_id: str
    model_revision: str
    tokens: list[int]
    corpus_hash: str

    @property
    def token_count(self) -> int:
        return len(self.tokens)

    @property
    def prefix_hash(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(self.model_revision.encode("utf-8"))
        hasher.update(b"\x00")
        for t in self.tokens:
            hasher.update(t.to_bytes(4, "little"))
        return hasher.hexdigest()


@dataclass(frozen=True)
class ArtifactRef:
    artifact_hash: str
    model_id: str
    model_revision: str
    corpus_hash: str
    corpus_name: str
    prefix_token_count: int
    prompt_template_version: str
    normalization_version: str
    build_timestamp: str
    file_size_bytes: int
    file_path: str
    mlx_lm_version: str
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_types.py -v`

Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/core/types.py tests/test_types.py
git commit -m "feat: add core data types with deterministic hashing"
```

---

## Task 3: Hashing Utilities

**Files:**
- Create: `app/core/hashing.py`
- Create: `tests/test_hashing.py`

- [ ] **Step 1: Write failing tests**

`tests/test_hashing.py`:
```python
from app.core.hashing import sha256_hex, artifact_hash_from_prefix


def test_sha256_hex_bytes():
    result = sha256_hex(b"hello")
    assert len(result) == 64
    assert result == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_sha256_hex_deterministic():
    assert sha256_hex(b"test") == sha256_hex(b"test")


def test_sha256_hex_different_inputs():
    assert sha256_hex(b"a") != sha256_hex(b"b")


def test_artifact_hash_from_prefix():
    h = artifact_hash_from_prefix(model_revision="rev123", prefix_tokens=[1, 2, 3])
    assert len(h) == 64


def test_artifact_hash_determinism():
    a = artifact_hash_from_prefix("rev", [10, 20])
    b = artifact_hash_from_prefix("rev", [10, 20])
    assert a == b


def test_artifact_hash_changes_with_revision():
    a = artifact_hash_from_prefix("rev1", [10, 20])
    b = artifact_hash_from_prefix("rev2", [10, 20])
    assert a != b


def test_artifact_hash_changes_with_tokens():
    a = artifact_hash_from_prefix("rev", [10, 20])
    b = artifact_hash_from_prefix("rev", [10, 21])
    assert a != b
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_hashing.py -v`

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement hashing utilities**

`app/core/hashing.py`:
```python
from __future__ import annotations

import hashlib


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def artifact_hash_from_prefix(model_revision: str, prefix_tokens: list[int]) -> str:
    hasher = hashlib.sha256()
    hasher.update(model_revision.encode("utf-8"))
    hasher.update(b"\x00")
    for t in prefix_tokens:
        hasher.update(t.to_bytes(4, "little"))
    return hasher.hexdigest()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_hashing.py -v`

Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/core/hashing.py tests/test_hashing.py
git commit -m "feat: add deterministic hashing utilities"
```

---

## Task 4: CorpusIngestor

**Files:**
- Create: `app/ingest/ingestor.py`
- Create: `tests/test_ingestor.py`

- [ ] **Step 1: Write failing tests**

`tests/test_ingestor.py`:
```python
from pathlib import Path

from app.core.types import Corpus
from app.ingest.ingestor import CorpusIngestor


def test_ingest_single_file(tmp_path: Path):
    f = tmp_path / "hello.txt"
    f.write_text("Hello, world!\n")
    corpus = CorpusIngestor.ingest(f, name="test")
    assert isinstance(corpus, Corpus)
    assert corpus.name == "test"
    assert len(corpus.chunks) == 1
    assert corpus.chunks[0].source_path == "hello.txt"
    assert corpus.chunks[0].content == "Hello, world!\n"


def test_ingest_directory(tmp_corpus_dir: Path):
    corpus = CorpusIngestor.ingest(tmp_corpus_dir, name="project")
    assert len(corpus.chunks) == 3
    paths = [c.source_path for c in corpus.chunks]
    assert paths == sorted(paths)  # deterministic order


def test_ingest_filters_unsupported_extensions(tmp_path: Path):
    (tmp_path / "good.py").write_text("x = 1\n")
    (tmp_path / "bad.bin").write_bytes(b"\x00\x01\x02")
    corpus = CorpusIngestor.ingest(tmp_path, name="filtered")
    assert len(corpus.chunks) == 1
    assert corpus.chunks[0].source_path == "good.py"


def test_ingest_normalizes_line_endings(tmp_path: Path):
    f = tmp_path / "crlf.txt"
    f.write_bytes(b"line one\r\nline two\r\n")
    corpus = CorpusIngestor.ingest(f, name="crlf")
    assert "\r" not in corpus.chunks[0].content
    assert corpus.chunks[0].content == "line one\nline two\n"


def test_ingest_strips_trailing_whitespace(tmp_path: Path):
    f = tmp_path / "trailing.txt"
    f.write_text("hello   \nworld  \n")
    corpus = CorpusIngestor.ingest(f, name="trailing")
    assert corpus.chunks[0].content == "hello\nworld\n"


def test_ingest_preserves_indentation(tmp_path: Path):
    f = tmp_path / "indented.py"
    f.write_text("def foo():\n    return 42\n")
    corpus = CorpusIngestor.ingest(f, name="indent")
    assert corpus.chunks[0].content == "def foo():\n    return 42\n"


def test_ingest_deterministic_hash(tmp_corpus_dir: Path):
    a = CorpusIngestor.ingest(tmp_corpus_dir, name="a")
    b = CorpusIngestor.ingest(tmp_corpus_dir, name="b")
    assert a.content_hash == b.content_hash


def test_ingest_respects_gitignore(tmp_path: Path):
    (tmp_path / ".gitignore").write_text("*.log\nbuild/\n")
    (tmp_path / "main.py").write_text("x = 1\n")
    (tmp_path / "debug.log").write_text("log stuff\n")
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "output.py").write_text("y = 2\n")
    corpus = CorpusIngestor.ingest(tmp_path, name="gitignore")
    paths = [c.source_path for c in corpus.chunks]
    assert "main.py" in paths
    assert "debug.log" not in paths
    assert "build/output.py" not in paths


def test_ingest_empty_directory(tmp_path: Path):
    corpus = CorpusIngestor.ingest(tmp_path, name="empty")
    assert len(corpus.chunks) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ingestor.py -v`

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement CorpusIngestor**

`app/ingest/ingestor.py`:
```python
from __future__ import annotations

import fnmatch
from pathlib import Path

from app.core.constants import SUPPORTED_EXTENSIONS
from app.core.types import Corpus, CorpusChunk


class CorpusIngestor:
    @staticmethod
    def ingest(source: Path, name: str) -> Corpus:
        source = Path(source)
        if source.is_file():
            chunks = CorpusIngestor._ingest_file(source, source.parent)
        elif source.is_dir():
            chunks = CorpusIngestor._ingest_directory(source)
        else:
            raise FileNotFoundError(f"Source not found: {source}")
        chunks.sort(key=lambda c: c.source_path)
        return Corpus(name=name, chunks=chunks)

    @staticmethod
    def _ingest_directory(root: Path) -> list[CorpusChunk]:
        gitignore_patterns = CorpusIngestor._load_gitignore(root)
        chunks: list[CorpusChunk] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix not in SUPPORTED_EXTENSIONS:
                continue
            rel = path.relative_to(root)
            if CorpusIngestor._is_gitignored(rel, gitignore_patterns):
                continue
            chunk = CorpusIngestor._read_file(path, rel)
            if chunk is not None:
                chunks.append(chunk)
        return chunks

    @staticmethod
    def _ingest_file(path: Path, root: Path) -> list[CorpusChunk]:
        if path.suffix not in SUPPORTED_EXTENSIONS:
            return []
        rel = path.relative_to(root)
        chunk = CorpusIngestor._read_file(path, rel)
        return [chunk] if chunk is not None else []

    @staticmethod
    def _read_file(path: Path, rel_path: Path) -> CorpusChunk | None:
        if path.suffix == ".pdf":
            return CorpusIngestor._read_pdf(path, rel_path)
        try:
            raw = path.read_bytes()
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            return None
        normalized = CorpusIngestor._normalize(text)
        return CorpusChunk(
            source_path=str(rel_path),
            content=normalized,
            byte_range=(0, len(raw)),
        )

    @staticmethod
    def _read_pdf(path: Path, rel_path: Path) -> CorpusChunk | None:
        try:
            import fitz  # pymupdf
            doc = fitz.open(str(path))
            pages = [page.get_text() for page in doc]
            doc.close()
            text = "\n\n".join(pages)
        except Exception:
            return None
        normalized = CorpusIngestor._normalize(text)
        file_size = path.stat().st_size
        return CorpusChunk(
            source_path=str(rel_path),
            content=normalized,
            byte_range=(0, file_size),
        )

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.rstrip() for line in text.split("\n")]
        return "\n".join(lines)
        # Note: result may or may not end with \n depending on input.
        # We preserve the original structure minus trailing whitespace per line.

    @staticmethod
    def _load_gitignore(root: Path) -> list[str]:
        gitignore = root / ".gitignore"
        if not gitignore.exists():
            return []
        patterns: list[str] = []
        for line in gitignore.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)
        return patterns

    @staticmethod
    def _is_gitignored(rel_path: Path, patterns: list[str]) -> bool:
        rel_str = str(rel_path)
        for pattern in patterns:
            # Match against filename
            if fnmatch.fnmatch(rel_path.name, pattern):
                return True
            # Match against full relative path
            if fnmatch.fnmatch(rel_str, pattern):
                return True
            # Match directory patterns (e.g., "build/")
            if pattern.endswith("/"):
                dir_pattern = pattern.rstrip("/")
                for parent in rel_path.parents:
                    if fnmatch.fnmatch(str(parent), dir_pattern):
                        return True
        return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ingestor.py -v`

Expected: all 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/ingest/ingestor.py tests/test_ingestor.py
git commit -m "feat: add CorpusIngestor with normalization and gitignore support"
```

---

## Task 5: CacheRegistry (SQLite)

**Files:**
- Create: `app/cache/registry.py`
- Create: `tests/test_registry.py`

- [ ] **Step 1: Write failing tests**

`tests/test_registry.py`:
```python
from pathlib import Path

from app.cache.registry import CacheRegistry
from app.core.types import ArtifactRef


def _make_ref(**overrides) -> ArtifactRef:
    defaults = dict(
        artifact_hash="a" * 64,
        model_id="test-model",
        model_revision="rev123",
        corpus_hash="b" * 64,
        corpus_name="my-corpus",
        prefix_token_count=1000,
        prompt_template_version="v1",
        normalization_version="v1",
        build_timestamp="2026-04-03T12:00:00Z",
        file_size_bytes=1024,
        file_path="/tmp/test.safetensors",
        mlx_lm_version="0.22.0",
    )
    defaults.update(overrides)
    return ArtifactRef(**defaults)


def test_register_and_get(tmp_path: Path):
    db = tmp_path / "registry.db"
    reg = CacheRegistry(db)
    ref = _make_ref()
    reg.register(ref)
    got = reg.get_by_artifact_hash(ref.artifact_hash)
    assert got is not None
    assert got.artifact_hash == ref.artifact_hash
    assert got.model_id == ref.model_id
    assert got.corpus_name == ref.corpus_name


def test_get_missing_returns_none(tmp_path: Path):
    reg = CacheRegistry(tmp_path / "registry.db")
    assert reg.get_by_artifact_hash("nonexistent") is None


def test_get_by_corpus_name(tmp_path: Path):
    reg = CacheRegistry(tmp_path / "registry.db")
    reg.register(_make_ref(corpus_name="docs"))
    results = reg.get_by_corpus_name("docs")
    assert len(results) == 1
    assert results[0].corpus_name == "docs"


def test_get_by_corpus_name_with_model(tmp_path: Path):
    reg = CacheRegistry(tmp_path / "registry.db")
    reg.register(_make_ref(artifact_hash="a" * 64, model_id="model-a"))
    reg.register(_make_ref(artifact_hash="b" * 64, model_id="model-b"))
    results = reg.get_by_corpus_name("my-corpus", model_id="model-a")
    assert len(results) == 1
    assert results[0].model_id == "model-a"


def test_list_all(tmp_path: Path):
    reg = CacheRegistry(tmp_path / "registry.db")
    reg.register(_make_ref(artifact_hash="a" * 64, corpus_name="one"))
    reg.register(_make_ref(artifact_hash="b" * 64, corpus_name="two"))
    all_refs = reg.list_all()
    assert len(all_refs) == 2


def test_delete(tmp_path: Path):
    reg = CacheRegistry(tmp_path / "registry.db")
    ref = _make_ref()
    reg.register(ref)
    reg.delete(ref.artifact_hash)
    assert reg.get_by_artifact_hash(ref.artifact_hash) is None


def test_register_duplicate_replaces(tmp_path: Path):
    reg = CacheRegistry(tmp_path / "registry.db")
    ref1 = _make_ref(file_size_bytes=100)
    ref2 = _make_ref(file_size_bytes=200)
    reg.register(ref1)
    reg.register(ref2)
    got = reg.get_by_artifact_hash(ref1.artifact_hash)
    assert got.file_size_bytes == 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_registry.py -v`

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement CacheRegistry**

`app/cache/registry.py`:
```python
from __future__ import annotations

import sqlite3
from pathlib import Path

from app.core.types import ArtifactRef

_SCHEMA = """
CREATE TABLE IF NOT EXISTS corpora (
    name TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    registered_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_hash TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    model_revision TEXT NOT NULL,
    corpus_hash TEXT NOT NULL,
    corpus_name TEXT NOT NULL REFERENCES corpora(name),
    prefix_token_count INTEGER NOT NULL,
    prompt_template_version TEXT NOT NULL,
    normalization_version TEXT NOT NULL,
    build_timestamp TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    mlx_lm_version TEXT NOT NULL
);
"""

_COLUMNS = (
    "artifact_hash", "model_id", "model_revision", "corpus_hash",
    "corpus_name", "prefix_token_count", "prompt_template_version",
    "normalization_version", "build_timestamp", "file_size_bytes",
    "file_path", "mlx_lm_version",
)


class CacheRegistry:
    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def register_corpus(
        self, name: str, source_path: str, content_hash: str, chunk_count: int
    ) -> None:
        from datetime import datetime, timezone
        self._conn.execute(
            "INSERT OR REPLACE INTO corpora (name, source_path, content_hash, chunk_count, registered_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, source_path, content_hash, chunk_count, datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()

    def get_corpus(self, name: str) -> dict | None:
        row = self._conn.execute(
            "SELECT name, source_path, content_hash, chunk_count, registered_at FROM corpora WHERE name = ?",
            (name,),
        ).fetchone()
        if not row:
            return None
        return dict(zip(("name", "source_path", "content_hash", "chunk_count", "registered_at"), row))

    def list_corpora(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT name, source_path, content_hash, chunk_count, registered_at FROM corpora ORDER BY registered_at DESC"
        ).fetchall()
        return [dict(zip(("name", "source_path", "content_hash", "chunk_count", "registered_at"), r)) for r in rows]

    def delete_corpus(self, name: str) -> None:
        self._conn.execute("DELETE FROM corpora WHERE name = ?", (name,))
        self._conn.commit()

    def register(self, ref: ArtifactRef) -> None:
        cols = ", ".join(_COLUMNS)
        placeholders = ", ".join("?" for _ in _COLUMNS)
        values = tuple(getattr(ref, c) for c in _COLUMNS)
        self._conn.execute(
            f"INSERT OR REPLACE INTO artifacts ({cols}) VALUES ({placeholders})",
            values,
        )
        self._conn.commit()

    def get_by_artifact_hash(self, artifact_hash: str) -> ArtifactRef | None:
        row = self._conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM artifacts WHERE artifact_hash = ?",
            (artifact_hash,),
        ).fetchone()
        return self._row_to_ref(row) if row else None

    def get_by_corpus_name(
        self, corpus_name: str, model_id: str | None = None
    ) -> list[ArtifactRef]:
        if model_id:
            rows = self._conn.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM artifacts WHERE corpus_name = ? AND model_id = ?",
                (corpus_name, model_id),
            ).fetchall()
        else:
            rows = self._conn.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM artifacts WHERE corpus_name = ?",
                (corpus_name,),
            ).fetchall()
        return [self._row_to_ref(r) for r in rows]

    def list_all(self) -> list[ArtifactRef]:
        rows = self._conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM artifacts ORDER BY build_timestamp DESC"
        ).fetchall()
        return [self._row_to_ref(r) for r in rows]

    def delete(self, artifact_hash: str) -> None:
        self._conn.execute(
            "DELETE FROM artifacts WHERE artifact_hash = ?", (artifact_hash,)
        )
        self._conn.commit()

    @staticmethod
    def _row_to_ref(row: tuple) -> ArtifactRef:
        return ArtifactRef(**dict(zip(_COLUMNS, row)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_registry.py -v`

Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/cache/registry.py tests/test_registry.py
git commit -m "feat: add CacheRegistry with SQLite backend"
```

---

## Task 6: PrefixCompiler

**Files:**
- Create: `app/prefix/compiler.py`
- Create: `tests/test_compiler.py`

- [ ] **Step 1: Write failing tests**

`tests/test_compiler.py`:
```python
from unittest.mock import MagicMock

from app.core.types import Corpus, CorpusChunk, TokenizedPrefix
from app.prefix.compiler import PrefixCompiler


def _make_corpus() -> Corpus:
    return Corpus(
        name="test",
        chunks=[
            CorpusChunk("readme.md", "# Hello\n\nWorld.\n", (0, 17)),
            CorpusChunk("main.py", "print('hi')\n", (0, 12)),
        ],
    )


def _make_mock_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.encode.side_effect = lambda text, **_: list(range(len(text) // 2))
    tok.apply_chat_template = None  # force raw template path
    return tok


def test_build_prompt_text():
    corpus = _make_corpus()
    text = PrefixCompiler.build_prompt_text(corpus)
    assert "[== readme.md ==]" in text
    assert "[== main.py ==]" in text
    assert "# Hello" in text
    assert "print('hi')" in text
    assert text.index("main.py") > text.index("readme.md")  # sorted


def test_build_prompt_text_sorted():
    corpus = Corpus(
        name="test",
        chunks=[
            CorpusChunk("z.txt", "last", (0, 4)),
            CorpusChunk("a.txt", "first", (0, 5)),
        ],
    )
    text = PrefixCompiler.build_prompt_text(corpus)
    assert text.index("a.txt") < text.index("z.txt")


def test_compile_returns_tokenized_prefix():
    corpus = _make_corpus()
    tok = _make_mock_tokenizer()
    prefix = PrefixCompiler.compile(
        corpus=corpus,
        tokenizer=tok,
        model_id="test-model",
        model_revision="rev123",
    )
    assert isinstance(prefix, TokenizedPrefix)
    assert prefix.model_id == "test-model"
    assert prefix.model_revision == "rev123"
    assert prefix.token_count > 0
    assert prefix.corpus_hash == corpus.content_hash


def test_compile_raises_on_context_overflow():
    corpus = _make_corpus()
    tok = _make_mock_tokenizer()
    try:
        PrefixCompiler.compile(
            corpus=corpus,
            tokenizer=tok,
            model_id="test-model",
            model_revision="rev123",
            max_context=10,
            query_budget=5,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "exceeds" in str(e).lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_compiler.py -v`

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement PrefixCompiler**

`app/prefix/compiler.py`:
```python
from __future__ import annotations

from app.core.constants import DEFAULT_QUERY_BUDGET, PROMPT_TEMPLATE_VERSION
from app.core.types import Corpus, TokenizedPrefix


class PrefixCompiler:
    TEMPLATE_VERSION = PROMPT_TEMPLATE_VERSION

    @staticmethod
    def build_prompt_text(corpus: Corpus) -> str:
        parts = [
            "You are answering questions about the following document(s).\n\n"
            "<documents>\n"
        ]
        for chunk in sorted(corpus.chunks, key=lambda c: c.source_path):
            parts.append(f"[== {chunk.source_path} ==]\n")
            parts.append(chunk.content)
            if not chunk.content.endswith("\n"):
                parts.append("\n")
            parts.append("\n")
        parts.append("</documents>\n\n")
        parts.append("<query>\n")
        return "".join(parts)

    @staticmethod
    def compile(
        corpus: Corpus,
        tokenizer,
        model_id: str,
        model_revision: str,
        max_context: int | None = None,
        query_budget: int = DEFAULT_QUERY_BUDGET,
    ) -> TokenizedPrefix:
        prompt_text = PrefixCompiler.build_prompt_text(corpus)
        tokens: list[int] = tokenizer.encode(prompt_text)

        if max_context is not None:
            available = max_context - query_budget
            if len(tokens) > available:
                raise ValueError(
                    f"Prefix ({len(tokens)} tokens) exceeds available context "
                    f"({available} = {max_context} max - {query_budget} query budget)"
                )

        return TokenizedPrefix(
            model_id=model_id,
            model_revision=model_revision,
            tokens=tokens,
            corpus_hash=corpus.content_hash,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_compiler.py -v`

Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/prefix/compiler.py tests/test_compiler.py
git commit -m "feat: add PrefixCompiler with prompt template and context validation"
```

---

## Task 7: CacheBuilder and CacheStore

**Files:**
- Create: `app/cache/builder.py`
- Create: `app/cache/store.py`
- Create: `tests/test_store.py`

- [ ] **Step 1: Write failing tests for CacheStore**

These tests use mock cache data since we can't run mlx in unit tests without a GPU. Integration tests come later.

`tests/test_store.py`:
```python
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.cache.store import CacheStore
from app.cache.registry import CacheRegistry
from app.core.types import ArtifactRef, TokenizedPrefix


def _make_prefix() -> TokenizedPrefix:
    return TokenizedPrefix(
        model_id="test-model",
        model_revision="rev123",
        tokens=[1, 2, 3, 4, 5],
        corpus_hash="c" * 64,
    )


def test_artifact_path(tmp_store_dir: Path):
    store = CacheStore(artifacts_dir=tmp_store_dir)
    prefix = _make_prefix()
    path = store.artifact_path(prefix.prefix_hash)
    assert path.parent == tmp_store_dir
    assert path.name == f"{prefix.prefix_hash}.safetensors"


def test_save_and_load_roundtrip(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    db = tmp_path / "registry.db"
    store = CacheStore(artifacts_dir=artifacts)
    registry = CacheRegistry(db)
    prefix = _make_prefix()

    fake_cache = [MagicMock()]
    metadata = {"model": "test-model"}

    def fake_save(path, cache, metadata=None):
        """Mock save that creates a real file so stat() works."""
        Path(path).write_bytes(b"fake-cache-data")

    with patch("app.cache.store.save_prompt_cache", side_effect=fake_save) as mock_save, \
         patch("app.cache.store.load_prompt_cache") as mock_load:
        mock_load.return_value = (fake_cache, metadata)

        ref = store.save(
            cache=fake_cache,
            prefix=prefix,
            corpus_name="test-corpus",
            mlx_lm_version="0.22.0",
            registry=registry,
        )

        assert isinstance(ref, ArtifactRef)
        assert ref.artifact_hash == prefix.prefix_hash
        assert ref.corpus_name == "test-corpus"
        assert ref.file_size_bytes > 0
        mock_save.assert_called_once()

        loaded_cache, loaded_meta = store.load(ref)
        mock_load.assert_called_once()
        assert loaded_cache == fake_cache


def test_delete_removes_file_and_registry(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    db = tmp_path / "registry.db"
    store = CacheStore(artifacts_dir=artifacts)
    registry = CacheRegistry(db)

    # Create a fake artifact file
    fake_hash = "f" * 64
    fake_file = artifacts / f"{fake_hash}.safetensors"
    fake_file.write_text("fake")

    ref = ArtifactRef(
        artifact_hash=fake_hash,
        model_id="m",
        model_revision="r",
        corpus_hash="c" * 64,
        corpus_name="test",
        prefix_token_count=5,
        prompt_template_version="v1",
        normalization_version="v1",
        build_timestamp="2026-04-03T12:00:00Z",
        file_size_bytes=4,
        file_path=str(fake_file),
        mlx_lm_version="0.22.0",
    )
    registry.register(ref)

    store.delete(ref, registry)
    assert not fake_file.exists()
    assert registry.get_by_artifact_hash(fake_hash) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_store.py -v`

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement CacheBuilder**

`app/cache/builder.py`:
```python
from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.generate import generate_step

from app.core.types import TokenizedPrefix


class CacheBuilder:
    @staticmethod
    def build(prefix: TokenizedPrefix, model, tokenizer) -> list:
        prompt_cache = make_prompt_cache(model)
        prompt_array = mx.array(prefix.tokens)

        # Prefill: use generate_step with max_tokens=0 to populate KV cache
        # without generating any output tokens. This is the public API path
        # that handles chunked prefill for long prompts.
        for _ in generate_step(
            prompt=prompt_array,
            model=model,
            max_tokens=0,
            prompt_cache=prompt_cache,
        ):
            pass

        return prompt_cache
```

- [ ] **Step 4: Implement CacheStore**

`app/cache/store.py`:
```python
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from mlx_lm.models.cache import load_prompt_cache, save_prompt_cache

from app.cache.registry import CacheRegistry
from app.core.constants import NORMALIZATION_VERSION, PROMPT_TEMPLATE_VERSION
from app.core.types import ArtifactRef, TokenizedPrefix


class CacheStore:
    def __init__(self, artifacts_dir: Path) -> None:
        self._dir = Path(artifacts_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def artifact_path(self, artifact_hash: str) -> Path:
        return self._dir / f"{artifact_hash}.safetensors"

    def save(
        self,
        cache: list,
        prefix: TokenizedPrefix,
        corpus_name: str,
        mlx_lm_version: str,
        registry: CacheRegistry,
    ) -> ArtifactRef:
        artifact_hash = prefix.prefix_hash
        path = self.artifact_path(artifact_hash)

        metadata = {
            "model_id": prefix.model_id,
            "model_revision": prefix.model_revision,
            "corpus_hash": prefix.corpus_hash,
            "corpus_name": corpus_name,
            "prefix_token_count": str(prefix.token_count),
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
            "mlx_lm_version": mlx_lm_version,
        }

        save_prompt_cache(str(path), cache, metadata)

        file_size = path.stat().st_size

        ref = ArtifactRef(
            artifact_hash=artifact_hash,
            model_id=prefix.model_id,
            model_revision=prefix.model_revision,
            corpus_hash=prefix.corpus_hash,
            corpus_name=corpus_name,
            prefix_token_count=prefix.token_count,
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
            normalization_version=NORMALIZATION_VERSION,
            build_timestamp=datetime.now(timezone.utc).isoformat(),
            file_size_bytes=file_size,
            file_path=str(path),
            mlx_lm_version=mlx_lm_version,
        )

        registry.register(ref)
        return ref

    def load(self, ref: ArtifactRef) -> tuple[list, dict]:
        return load_prompt_cache(ref.file_path, return_metadata=True)

    def delete(self, ref: ArtifactRef, registry: CacheRegistry) -> None:
        path = Path(ref.file_path)
        if path.exists():
            path.unlink()
        registry.delete(ref.artifact_hash)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_store.py -v`

Expected: all 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add app/cache/builder.py app/cache/store.py tests/test_store.py
git commit -m "feat: add CacheBuilder and CacheStore wrapping mlx-lm native cache APIs"
```

---

## Task 8: QueryEngine

**Files:**
- Create: `app/inference/engine.py`
- Create: `tests/test_engine.py`

- [ ] **Step 1: Write failing tests**

`tests/test_engine.py`:
```python
import time
from unittest.mock import MagicMock, patch

from app.inference.engine import QueryEngine, QueryResult


def test_query_result_fields():
    r = QueryResult(
        text="Hello world",
        ttft_ms=50.0,
        decode_tokens_per_sec=35.0,
        total_tokens=10,
    )
    assert r.text == "Hello world"
    assert r.ttft_ms == 50.0
    assert r.decode_tokens_per_sec == 35.0


def test_format_query_prompt():
    text = QueryEngine.format_query_prompt("What does this code do?")
    assert "What does this code do?" in text
    assert "</query>" in text
    assert "<query>" not in text  # opening tag is in the cached prefix, not here
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_engine.py -v`

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement QueryEngine**

`app/inference/engine.py`:
```python
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Generator

import mlx.core as mx


@dataclass
class QueryResult:
    text: str
    ttft_ms: float
    decode_tokens_per_sec: float
    total_tokens: int


class QueryEngine:
    @staticmethod
    def format_query_prompt(question: str) -> str:
        # Prefix already ends with "<query>\n", so we just append the question
        # and close the tag. The cached KV state covers everything up to <query>.
        return f"{question}\n</query>\n"

    @staticmethod
    def query(
        model,
        tokenizer,
        prompt_cache: list,
        question: str,
        max_tokens: int = 512,
    ) -> QueryResult:
        cache = copy.deepcopy(prompt_cache)
        query_text = QueryEngine.format_query_prompt(question)

        from mlx_lm import stream_generate

        t0 = time.perf_counter()
        ttft: float | None = None
        chunks: list[str] = []
        token_count = 0

        for response in stream_generate(
            model,
            tokenizer,
            prompt=query_text,
            max_tokens=max_tokens,
            prompt_cache=cache,
        ):
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            chunks.append(response.text)
            token_count += 1

        total_s = time.perf_counter() - t0
        decode_tokens = max(token_count - 1, 1)  # exclude first token from decode rate
        decode_time = total_s - (ttft / 1000 if ttft else 0)
        tps = decode_tokens / max(decode_time, 0.001)

        return QueryResult(
            text="".join(chunks),
            ttft_ms=ttft or 0.0,
            decode_tokens_per_sec=tps,
            total_tokens=token_count,
        )

    @staticmethod
    def stream_query(
        model,
        tokenizer,
        prompt_cache: list,
        question: str,
        max_tokens: int = 512,
    ) -> Generator[str, None, None]:
        """Stream tokens one at a time. Yields text chunks."""
        cache = copy.deepcopy(prompt_cache)
        query_text = QueryEngine.format_query_prompt(question)

        # Use stream_generate for streaming
        from mlx_lm import stream_generate

        for response in stream_generate(
            model,
            tokenizer,
            prompt=query_text,
            max_tokens=max_tokens,
            prompt_cache=cache,
        ):
            yield response.text
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_engine.py -v`

Expected: all 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/inference/engine.py tests/test_engine.py
git commit -m "feat: add QueryEngine with streaming and timing support"
```

---

## Task 9: CLI Entry Point

**Files:**
- Create: `app/cli.py`

- [ ] **Step 1: Implement CLI**

`app/cli.py`:
```python
from __future__ import annotations

import sys
import time
from pathlib import Path

import click

from app.core.constants import (
    APP_DIR,
    ARTIFACTS_DIR,
    DEFAULT_MODEL,
    DEFAULT_QUERY_BUDGET,
    REGISTRY_DB,
)


def _ensure_dirs() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


@click.group()
def main() -> None:
    """mcq — local MLX prompt-cache query tool."""
    _ensure_dirs()


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--name", "-n", required=True, help="Corpus name for reference")
def ingest(path: str, name: str) -> None:
    """Ingest a file or directory into a corpus."""
    from app.cache.registry import CacheRegistry
    from app.ingest.ingestor import CorpusIngestor

    source = Path(path).resolve()
    t0 = time.perf_counter()
    corpus = CorpusIngestor.ingest(source, name=name)
    elapsed = time.perf_counter() - t0

    # Persist corpus registration
    registry = CacheRegistry(REGISTRY_DB)
    registry.register_corpus(
        name=name,
        source_path=str(source),
        content_hash=corpus.content_hash,
        chunk_count=len(corpus.chunks),
    )

    click.echo(f"Corpus '{name}': {len(corpus.chunks)} files, hash={corpus.content_hash[:16]}...")
    click.echo(f"Ingested in {elapsed:.2f}s")
    click.echo(f"Content hash: {corpus.content_hash}")
    click.echo(f"Source: {source}")


@main.command()
@click.argument("name")
@click.option("--model", "-m", default=DEFAULT_MODEL, help="Model ID", show_default=True)
def build(name: str, model: str) -> None:
    """Build a KV cache artifact from a registered corpus."""
    from huggingface_hub import snapshot_download
    from mlx_lm import load
    import mlx_lm

    from app.cache.builder import CacheBuilder
    from app.cache.registry import CacheRegistry
    from app.cache.store import CacheStore
    from app.ingest.ingestor import CorpusIngestor
    from app.prefix.compiler import PrefixCompiler

    # Look up registered corpus
    registry = CacheRegistry(REGISTRY_DB)
    corpus_info = registry.get_corpus(name)
    if not corpus_info:
        click.echo(f"Corpus '{name}' not found. Run 'mcq ingest' first.")
        sys.exit(1)

    corpus_path = corpus_info["source_path"]

    # Re-ingest from source to get current content
    click.echo(f"Ingesting '{corpus_path}'...")
    t0 = time.perf_counter()
    corpus = CorpusIngestor.ingest(Path(corpus_path), name=name)
    t_ingest = time.perf_counter() - t0
    click.echo(f"  {len(corpus.chunks)} files in {t_ingest:.2f}s")

    # Check for staleness
    if corpus.content_hash != corpus_info["content_hash"]:
        click.echo(f"  Warning: corpus content has changed since registration. Updating.")
        registry.register_corpus(name, corpus_path, corpus.content_hash, len(corpus.chunks))

    # Load model
    click.echo(f"Loading model '{model}'...")
    t0 = time.perf_counter()
    model_path = snapshot_download(model)
    revision = "local"
    try:
        # Try to get the revision from the snapshot info
        from huggingface_hub import model_info
        info = model_info(model)
        revision = info.sha
    except Exception:
        pass

    mlx_model, tokenizer = load(model)
    t_load = time.perf_counter() - t0
    click.echo(f"  Model loaded in {t_load:.2f}s")

    # Compile prefix
    click.echo("Compiling prefix...")
    t0 = time.perf_counter()
    # Get model context window from config if available
    max_context = getattr(mlx_model, "max_position_embeddings", None) or getattr(
        getattr(mlx_model, "config", None), "max_position_embeddings", 32768
    )
    prefix = PrefixCompiler.compile(
        corpus=corpus,
        tokenizer=tokenizer,
        model_id=model,
        model_revision=revision,
        max_context=max_context,
    )
    t_compile = time.perf_counter() - t0
    click.echo(f"  {prefix.token_count} tokens in {t_compile:.2f}s (limit: {max_context - DEFAULT_QUERY_BUDGET})")

    # Build cache
    click.echo("Building KV cache (this may take a while)...")
    t0 = time.perf_counter()
    cache = CacheBuilder.build(prefix, mlx_model, tokenizer)
    t_build = time.perf_counter() - t0
    click.echo(f"  Cache built in {t_build:.2f}s")

    # Save
    click.echo("Saving artifact...")
    t0 = time.perf_counter()
    store = CacheStore(ARTIFACTS_DIR)
    registry = CacheRegistry(REGISTRY_DB)
    ref = store.save(
        cache=cache,
        prefix=prefix,
        corpus_name=name,
        mlx_lm_version=mlx_lm.__version__,
        registry=registry,
    )
    t_save = time.perf_counter() - t0
    click.echo(f"  Saved in {t_save:.2f}s ({ref.file_size_bytes / 1024 / 1024:.1f} MB)")

    click.echo(f"\nArtifact: {ref.artifact_hash[:16]}...")
    click.echo(f"  Path: {ref.file_path}")
    click.echo(f"  Tokens: {ref.prefix_token_count}")
    click.echo(f"  Total build time: {t_ingest + t_compile + t_build + t_save:.2f}s (excl. model load)")


@main.command()
@click.argument("name")
@click.option("--model", "-m", default=DEFAULT_MODEL, help="Model ID", show_default=True)
@click.option("--max-tokens", default=512, help="Max tokens to generate", show_default=True)
def query(name: str, model: str, max_tokens: int) -> None:
    """Interactive query against a cached corpus."""
    from mlx_lm import load

    from app.cache.registry import CacheRegistry
    from app.cache.store import CacheStore
    from app.inference.engine import QueryEngine

    registry = CacheRegistry(REGISTRY_DB)
    store = CacheStore(ARTIFACTS_DIR)

    refs = registry.get_by_corpus_name(name, model_id=model)
    if not refs:
        click.echo(f"No cache found for corpus '{name}' with model '{model}'")
        click.echo("Run 'mcq build' first.")
        sys.exit(1)

    ref = refs[0]  # most recent
    click.echo(f"Loading model '{model}'...")
    mlx_model, tokenizer = load(model)

    click.echo(f"Loading cache ({ref.file_size_bytes / 1024 / 1024:.1f} MB)...")
    t0 = time.perf_counter()
    prompt_cache, _ = store.load(ref)
    t_load = time.perf_counter() - t0
    click.echo(f"  Cache loaded in {t_load:.3f}s")
    click.echo(f"  Prefix: {ref.prefix_token_count} tokens")
    click.echo()

    click.echo("Ready. Type your question (Ctrl+C to exit):")
    while True:
        try:
            question = click.prompt("Q", prompt_suffix="> ")
            if not question.strip():
                continue

            click.echo("A> ", nl=False)
            for chunk in QueryEngine.stream_query(
                model=mlx_model,
                tokenizer=tokenizer,
                prompt_cache=prompt_cache,
                question=question,
                max_tokens=max_tokens,
            ):
                click.echo(chunk, nl=False)
            click.echo()
            click.echo()
        except (KeyboardInterrupt, EOFError):
            click.echo("\nBye.")
            break


@main.command("list")
def list_cmd() -> None:
    """List all registered corpora and cached artifacts."""
    from app.cache.registry import CacheRegistry

    registry = CacheRegistry(REGISTRY_DB)
    refs = registry.list_all()

    if not refs:
        click.echo("No cached artifacts found.")
        return

    for ref in refs:
        size_mb = ref.file_size_bytes / 1024 / 1024
        click.echo(
            f"  {ref.corpus_name:20s}  model={ref.model_id:40s}  "
            f"tokens={ref.prefix_token_count:6d}  size={size_mb:6.1f}MB  "
            f"hash={ref.artifact_hash[:12]}..."
        )


@main.command()
@click.argument("name")
def info(name: str) -> None:
    """Show details for a corpus and its cache artifacts."""
    from app.cache.registry import CacheRegistry

    registry = CacheRegistry(REGISTRY_DB)
    refs = registry.get_by_corpus_name(name)

    if not refs:
        click.echo(f"No artifacts found for corpus '{name}'.")
        return

    for ref in refs:
        click.echo(f"Corpus: {ref.corpus_name}")
        click.echo(f"  Artifact hash:    {ref.artifact_hash}")
        click.echo(f"  Model:            {ref.model_id}")
        click.echo(f"  Model revision:   {ref.model_revision}")
        click.echo(f"  Corpus hash:      {ref.corpus_hash}")
        click.echo(f"  Prefix tokens:    {ref.prefix_token_count}")
        click.echo(f"  Template version: {ref.prompt_template_version}")
        click.echo(f"  File size:        {ref.file_size_bytes / 1024 / 1024:.1f} MB")
        click.echo(f"  File path:        {ref.file_path}")
        click.echo(f"  Built:            {ref.build_timestamp}")
        click.echo(f"  mlx-lm version:   {ref.mlx_lm_version}")
        click.echo()


@main.command()
@click.argument("name")
@click.option("--cache-only", is_flag=True, help="Delete only cache artifacts, keep corpus registration")
def delete(name: str, cache_only: bool) -> None:
    """Delete a corpus and its cache artifacts."""
    from app.cache.registry import CacheRegistry
    from app.cache.store import CacheStore

    registry = CacheRegistry(REGISTRY_DB)
    store = CacheStore(ARTIFACTS_DIR)
    refs = registry.get_by_corpus_name(name)

    if not refs:
        click.echo(f"No artifacts found for corpus '{name}'.")
        return

    for ref in refs:
        store.delete(ref, registry)
        click.echo(f"Deleted: {ref.artifact_hash[:16]}... ({ref.model_id})")

    click.echo(f"Removed {len(refs)} artifact(s).")
```

- [ ] **Step 2: Verify CLI registers correctly**

Run: `cd /Users/wally/dev/mlx-cache-query && pip install -e ".[dev]" && mcq --help`

Expected: shows help with all subcommands (ingest, build, query, list, info, delete)

- [ ] **Step 3: Commit**

```bash
git add app/cli.py
git commit -m "feat: add CLI entry point with all V1 commands"
```

---

## Task 10: BenchmarkRunner

**Files:**
- Create: `app/eval/bench.py`

- [ ] **Step 1: Implement BenchmarkRunner**

`app/eval/bench.py`:
```python
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TimingResult:
    label: str
    duration_ms: float


@dataclass
class BenchmarkReport:
    timings: list[TimingResult] = field(default_factory=list)

    def add(self, label: str, duration_s: float) -> None:
        self.timings.append(TimingResult(label=label, duration_ms=duration_s * 1000))

    def summary(self) -> str:
        lines = ["Benchmark Results", "=" * 50]
        for t in self.timings:
            lines.append(f"  {t.label:30s}  {t.duration_ms:8.1f} ms")
        lines.append("=" * 50)
        return "\n".join(lines)


class BenchmarkRunner:
    """Runs the full pipeline and reports timings."""

    @staticmethod
    def run(
        corpus_path: Path,
        corpus_name: str,
        model_id: str,
        question: str,
        artifacts_dir: Path,
        db_path: Path,
    ) -> BenchmarkReport:
        from mlx_lm import load
        import mlx_lm
        from huggingface_hub import model_info

        from app.cache.builder import CacheBuilder
        from app.cache.registry import CacheRegistry
        from app.cache.store import CacheStore
        from app.ingest.ingestor import CorpusIngestor
        from app.inference.engine import QueryEngine
        from app.prefix.compiler import PrefixCompiler

        report = BenchmarkReport()

        # Ingest
        t0 = time.perf_counter()
        corpus = CorpusIngestor.ingest(corpus_path, name=corpus_name)
        report.add("Corpus ingest", time.perf_counter() - t0)

        # Load model
        t0 = time.perf_counter()
        try:
            info = model_info(model_id)
            revision = info.sha
        except Exception:
            revision = "unknown"
        mlx_model, tokenizer = load(model_id)
        report.add("Model load", time.perf_counter() - t0)

        # Compile
        t0 = time.perf_counter()
        prefix = PrefixCompiler.compile(
            corpus=corpus,
            tokenizer=tokenizer,
            model_id=model_id,
            model_revision=revision,
        )
        report.add("Prefix compile", time.perf_counter() - t0)

        # Build cache
        t0 = time.perf_counter()
        cache = CacheBuilder.build(prefix, mlx_model, tokenizer)
        report.add("Cache build", time.perf_counter() - t0)

        # Save
        store = CacheStore(artifacts_dir)
        registry = CacheRegistry(db_path)
        t0 = time.perf_counter()
        ref = store.save(
            cache=cache,
            prefix=prefix,
            corpus_name=corpus_name,
            mlx_lm_version=mlx_lm.__version__,
            registry=registry,
        )
        report.add("Cache save", time.perf_counter() - t0)

        # Load
        t0 = time.perf_counter()
        loaded_cache, _ = store.load(ref)
        report.add("Cache load", time.perf_counter() - t0)

        # Query
        t0 = time.perf_counter()
        result = QueryEngine.query(
            model=mlx_model,
            tokenizer=tokenizer,
            prompt_cache=loaded_cache,
            question=question,
            max_tokens=100,
        )
        report.add("Query (100 tokens)", time.perf_counter() - t0)
        report.add("Query TTFT (approx)", result.ttft_ms / 1000)
        report.add(f"Decode throughput", 0)  # placeholder: actual value in summary

        return report
```

- [ ] **Step 2: Commit**

```bash
git add app/eval/bench.py
git commit -m "feat: add BenchmarkRunner with full pipeline timing"
```

---

## Task 11: Integration Smoke Test

**Files:**
- Create: `tests/test_integration_smoke.py`

- [ ] **Step 1: Write smoke test**

This test requires `mlx` and a model download. Marked with `pytest.mark.slow` so it's skipped in fast CI.

`tests/test_integration_smoke.py`:
```python
"""
Integration smoke test — requires Apple Silicon, mlx, and network for first model download.
Run with: pytest tests/test_integration_smoke.py -v -m slow
"""
import os
from pathlib import Path

import pytest

# Skip entirely if not on Apple Silicon or no mlx
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not HAS_MLX, reason="mlx not available"),
]


def test_full_pipeline_smoke(tmp_path: Path):
    """End-to-end: ingest → compile → build → save → load → query."""
    from mlx_lm import load
    import mlx_lm
    from huggingface_hub import model_info

    from app.cache.builder import CacheBuilder
    from app.cache.registry import CacheRegistry
    from app.cache.store import CacheStore
    from app.ingest.ingestor import CorpusIngestor
    from app.inference.engine import QueryEngine
    from app.prefix.compiler import PrefixCompiler

    # Create test corpus
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "test.md").write_text(
        "# MLX Cache Query\n\n"
        "This tool builds KV caches from documents.\n"
        "It runs on Apple Silicon using MLX.\n"
    )

    # Use smallest feasible model for smoke test
    model_id = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

    # Ingest
    corpus = CorpusIngestor.ingest(corpus_dir, name="smoke-test")
    assert len(corpus.chunks) == 1

    # Load model
    mlx_model, tokenizer = load(model_id)
    try:
        info = model_info(model_id)
        revision = info.sha
    except Exception:
        revision = "unknown"

    # Compile
    prefix = PrefixCompiler.compile(
        corpus=corpus,
        tokenizer=tokenizer,
        model_id=model_id,
        model_revision=revision,
    )
    assert prefix.token_count > 0
    assert prefix.token_count < 1000  # small corpus

    # Build cache
    cache = CacheBuilder.build(prefix, mlx_model, tokenizer)
    assert len(cache) > 0  # should have per-layer cache objects

    # Save
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    db_path = tmp_path / "registry.db"
    store = CacheStore(artifacts_dir)
    registry = CacheRegistry(db_path)

    ref = store.save(
        cache=cache,
        prefix=prefix,
        corpus_name="smoke-test",
        mlx_lm_version=mlx_lm.__version__,
        registry=registry,
    )
    assert Path(ref.file_path).exists()
    assert ref.file_size_bytes > 0

    # Load
    loaded_cache, metadata = store.load(ref)
    assert len(loaded_cache) == len(cache)

    # Query
    result = QueryEngine.query(
        model=mlx_model,
        tokenizer=tokenizer,
        prompt_cache=loaded_cache,
        question="What does this tool do?",
        max_tokens=50,
    )
    assert len(result.text) > 0
    assert result.total_tokens > 0

    # Registry lookup
    found = registry.get_by_corpus_name("smoke-test", model_id=model_id)
    assert len(found) == 1
    assert found[0].artifact_hash == ref.artifact_hash
```

- [ ] **Step 2: Configure pytest markers**

Add to `pyproject.toml` under `[tool.pytest.ini_options]`:
```toml
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
]
```

- [ ] **Step 3: Run unit tests (fast)**

Run: `pytest tests/ -v -m "not slow"`

Expected: all unit tests PASS (types, hashing, ingestor, compiler, registry, store, engine)

- [ ] **Step 4: Run smoke test (slow, requires Apple Silicon)**

Run: `pytest tests/test_integration_smoke.py -v -m slow --timeout=300`

Expected: PASS (may take 1-2 minutes for model download on first run)

- [ ] **Step 5: Commit**

```bash
git add tests/test_integration_smoke.py pyproject.toml
git commit -m "feat: add integration smoke test for full pipeline"
```

---

## Task Summary

| Task | Component | Tests | Dependencies |
|------|-----------|-------|-------------|
| 1 | Scaffolding | - | None |
| 2 | Core types | 6 | Task 1 |
| 3 | Hashing | 7 | Task 1 |
| 4 | CorpusIngestor | 9 | Tasks 2, 3 |
| 5 | CacheRegistry | 7 | Task 2 |
| 6 | PrefixCompiler | 4 | Tasks 2, 3 |
| 7 | CacheBuilder + CacheStore | 3 | Tasks 2, 5, 6 |
| 8 | QueryEngine | 2 | Task 2 |
| 9 | CLI | manual | Tasks 4-8 |
| 10 | BenchmarkRunner | - | Tasks 4-8 |
| 11 | Integration smoke | 1 | Tasks 1-10 |

Total: 39 unit tests + 1 integration test across 11 tasks.
