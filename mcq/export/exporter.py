"""Export corpus content in various formats for piping and integration."""
from __future__ import annotations

import json
from mcq.core.types import Corpus


class CorpusExporter:
    @staticmethod
    def to_markdown(corpus: Corpus) -> str:
        """Export corpus as a single concatenated markdown document."""
        parts = [f"# {corpus.name}\n\n"]
        for chunk in sorted(corpus.chunks, key=lambda c: c.source_path):
            parts.append(f"## {chunk.source_path}\n\n")
            parts.append(chunk.content)
            if not chunk.content.endswith("\n"):
                parts.append("\n")
            parts.append("\n")
        return "".join(parts)

    @staticmethod
    def to_json(corpus: Corpus) -> str:
        """Export corpus as JSON."""
        data = {
            "name": corpus.name,
            "content_hash": corpus.content_hash,
            "chunks": [
                {
                    "source_path": c.source_path,
                    "content": c.content,
                    "byte_range": list(c.byte_range),
                    "metadata": c.metadata,
                }
                for c in corpus.chunks
            ],
        }
        return json.dumps(data, indent=2, default=str)

    @staticmethod
    def to_context(corpus: Corpus) -> str:
        """Export corpus in the prompt template format (ready to paste into an LLM).

        This is the same format as PrefixCompiler.build_prompt_text but without
        the <query> tag, making it suitable for pasting into any LLM interface.
        """
        parts = ["<documents>\n"]
        for chunk in sorted(corpus.chunks, key=lambda c: c.source_path):
            parts.append(f"[== {chunk.source_path} ==]\n")
            parts.append(chunk.content)
            if not chunk.content.endswith("\n"):
                parts.append("\n")
            parts.append("\n")
        parts.append("</documents>\n")
        return "".join(parts)

    @staticmethod
    def to_filelist(corpus: Corpus) -> str:
        """Export just the file paths, one per line (for piping to other tools)."""
        return "\n".join(c.source_path for c in sorted(corpus.chunks, key=lambda c: c.source_path)) + "\n"
