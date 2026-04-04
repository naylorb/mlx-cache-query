"""Corpus statistics computation."""
from __future__ import annotations

from dataclasses import dataclass
from mcq.core.types import Corpus


@dataclass
class CorpusStats:
    name: str
    total_files: int
    total_bytes: int
    total_lines: int
    total_words: int
    extensions: dict[str, int]
    largest_file: str
    largest_file_bytes: int

    @classmethod
    def from_corpus(cls, corpus: Corpus) -> "CorpusStats":
        extensions: dict[str, int] = {}
        total_bytes = 0
        total_lines = 0
        total_words = 0
        largest = ("", 0)

        for chunk in corpus.chunks:
            # Extension count
            ext = "." + chunk.source_path.rsplit(".", 1)[-1] if "." in chunk.source_path else "(none)"
            extensions[ext] = extensions.get(ext, 0) + 1

            # Size stats
            chunk_bytes = len(chunk.content.encode("utf-8"))
            total_bytes += chunk_bytes
            total_lines += chunk.content.count("\n")
            total_words += len(chunk.content.split())

            if chunk_bytes > largest[1]:
                largest = (chunk.source_path, chunk_bytes)

        return cls(
            name=corpus.name,
            total_files=len(corpus.chunks),
            total_bytes=total_bytes,
            total_lines=total_lines,
            total_words=total_words,
            extensions=dict(sorted(extensions.items(), key=lambda x: -x[1])),
            largest_file=largest[0],
            largest_file_bytes=largest[1],
        )
