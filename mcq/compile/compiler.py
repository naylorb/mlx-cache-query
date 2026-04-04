"""WikiCompiler — generates structured wiki content from a Corpus.

Produces an enriched Corpus with an _index.md and structured organization.
This is the "offline" compilation step — no LLM needed. Generates:
- _index.md: master index of all files with brief descriptions
- Organized content ordering (index first, then alphabetical)
"""
from __future__ import annotations

import re
from collections import defaultdict
from mcq.core.types import Corpus, CorpusChunk


class WikiCompiler:
    @staticmethod
    def compile(corpus: Corpus) -> Corpus:
        """Compile a corpus into a wiki-structured corpus with an index."""
        # Filter out any existing _index.md before building the new index
        filtered_chunks = [c for c in corpus.chunks if c.source_path != "_index.md"]
        filtered_corpus = Corpus(name=corpus.name, chunks=filtered_chunks)

        index_content = WikiCompiler._build_index(filtered_corpus)
        index_chunk = CorpusChunk(
            source_path="_index.md",
            content=index_content,
            byte_range=(0, len(index_content.encode())),
        )

        chunks = list(filtered_chunks)
        chunks.insert(0, index_chunk)

        return Corpus(name=corpus.name, chunks=chunks)

    @staticmethod
    def _build_index(corpus: Corpus) -> str:
        """Build a master index with file listing and extracted headings."""
        lines = [
            f"# Index: {corpus.name}\n",
            f"Total files: {len(corpus.chunks)}\n",
            f"Content hash: {corpus.content_hash}\n",
            "",
            "## Files\n",
        ]

        # Group by directory
        dirs: dict[str, list[CorpusChunk]] = defaultdict(list)
        for chunk in sorted(corpus.chunks, key=lambda c: c.source_path):
            parts = chunk.source_path.split("/")
            dir_name = "/".join(parts[:-1]) if len(parts) > 1 else "."
            dirs[dir_name].append(chunk)

        for dir_name in sorted(dirs.keys()):
            if dir_name != ".":
                lines.append(f"### {dir_name}/\n")
            for chunk in dirs[dir_name]:
                summary = WikiCompiler._extract_summary(chunk)
                lines.append(f"- **{chunk.source_path}**: {summary}")
            lines.append("")

        # Extract key concepts/topics
        tags = WikiCompiler._extract_all_tags(corpus)
        if tags:
            lines.append("## Topics\n")
            for tag, count in sorted(tags.items(), key=lambda x: -x[1])[:20]:
                lines.append(f"- {tag} ({count} mentions)")
            lines.append("")

        return "\n".join(lines) + "\n"

    @staticmethod
    def _extract_summary(chunk: CorpusChunk) -> str:
        """Extract a brief summary from the first heading or first line."""
        lines = chunk.content.strip().split("\n")
        for line in lines[:10]:
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
            if line.startswith("## "):
                return line[3:].strip()
            if line and not line.startswith("---") and not line.startswith("```"):
                return line[:100] + ("..." if len(line) > 100 else "")
        return "(empty)"

    @staticmethod
    def _extract_all_tags(corpus: Corpus) -> dict[str, int]:
        """Extract tags from all chunks (frontmatter + inline)."""
        tag_counts: dict[str, int] = defaultdict(int)
        for chunk in corpus.chunks:
            # From metadata
            if chunk.metadata and "tags" in chunk.metadata:
                for tag in chunk.metadata["tags"]:
                    tag_counts[tag] += 1
            # From inline #tags in content
            inline = re.findall(r'(?<!\w)#([a-zA-Z][\w/-]*)', chunk.content)
            for tag in inline:
                tag_counts[tag] += 1
        return dict(tag_counts)
