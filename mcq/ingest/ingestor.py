from __future__ import annotations

import fnmatch
import re
from pathlib import Path

from mcq.core.constants import SKIP_DIRS, SUPPORTED_EXTENSIONS
from mcq.core.types import Corpus, CorpusChunk


class CorpusIngestor:
    @staticmethod
    def ingest(source: Path, name: str, format: str = "auto") -> Corpus:
        source = Path(source)
        if source.is_file():
            chunks = CorpusIngestor._ingest_file(source, source.parent, format=format)
        elif source.is_dir():
            chunks = CorpusIngestor._ingest_directory(source, format=format)
        else:
            raise FileNotFoundError(f"Source not found: {source}")
        chunks.sort(key=lambda c: c.source_path)
        return Corpus(name=name, chunks=chunks)

    @staticmethod
    def _ingest_directory(root: Path, format: str = "auto") -> list[CorpusChunk]:
        gitignore_patterns = CorpusIngestor._load_gitignore(root)
        obsidian = format == "obsidian"
        chunks: list[CorpusChunk] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix not in SUPPORTED_EXTENSIONS:
                continue
            rel = path.relative_to(root)
            # Skip known non-content directories
            if any(part in SKIP_DIRS for part in rel.parts):
                continue
            if CorpusIngestor._is_gitignored(rel, gitignore_patterns):
                continue
            chunk = CorpusIngestor._read_file(path, rel, obsidian=obsidian)
            if chunk is not None:
                chunks.append(chunk)
        return chunks

    @staticmethod
    def _ingest_file(path: Path, root: Path, format: str = "auto") -> list[CorpusChunk]:
        if path.suffix not in SUPPORTED_EXTENSIONS:
            return []
        rel = path.relative_to(root)
        obsidian = format == "obsidian"
        chunk = CorpusIngestor._read_file(path, rel, obsidian=obsidian)
        return [chunk] if chunk is not None else []

    @staticmethod
    def _read_file(path: Path, rel_path: Path, obsidian: bool = False) -> CorpusChunk | None:
        if path.suffix == ".pdf":
            return CorpusIngestor._read_pdf(path, rel_path)
        try:
            raw = path.read_bytes()
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            return None
        normalized = CorpusIngestor._normalize(text)

        metadata = None
        if obsidian and path.suffix == ".md":
            normalized, metadata = CorpusIngestor._parse_obsidian_md(normalized)

        return CorpusChunk(
            source_path=str(rel_path),
            content=normalized,
            byte_range=(0, len(raw)),
            metadata=metadata,
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
    def _parse_frontmatter(text: str) -> tuple[dict, str]:
        """Extract YAML frontmatter as raw key-value pairs without PyYAML."""
        if not text.startswith("---\n"):
            return {}, text
        end = text.find("\n---\n", 4)
        if end == -1:
            return {}, text
        fm_text = text[4:end]
        body = text[end + 5:]
        fm: dict = {}
        for line in fm_text.splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                key = key.strip()
                val = val.strip()
                if val.startswith("[") and val.endswith("]"):
                    # Simple list: [tag1, tag2]
                    fm[key] = [v.strip().strip('"').strip("'") for v in val[1:-1].split(",") if v.strip()]
                elif val:
                    fm[key] = val.strip('"').strip("'")
        return fm, body

    @staticmethod
    def _parse_obsidian_md(text: str) -> tuple[str, dict | None]:
        """Parse an Obsidian markdown file. Returns (content, metadata)."""
        metadata: dict = {}
        content = text

        # Parse YAML frontmatter
        fm, body = CorpusIngestor._parse_frontmatter(text)
        if fm:
            metadata["frontmatter"] = fm
            content = body
            # Extract tags from frontmatter
            if "tags" in fm:
                tags = fm["tags"]
                if isinstance(tags, list):
                    metadata.setdefault("tags", []).extend(tags)
                elif isinstance(tags, str):
                    metadata.setdefault("tags", []).append(tags)

        # Extract inline #tags
        inline_tags = re.findall(r'(?<!\w)#([a-zA-Z][\w/-]*)', content)
        if inline_tags:
            metadata.setdefault("tags", []).extend(inline_tags)
            # Deduplicate
            metadata["tags"] = sorted(set(metadata.get("tags", [])))

        # Extract [[wikilinks]]
        wikilinks = re.findall(r'\[\[([^\]|]+?)(?:\|[^\]]+?)?\]\]', content)
        if wikilinks:
            metadata["wikilinks"] = sorted(set(wikilinks))

        return content, metadata if metadata else None

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
