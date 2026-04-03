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
