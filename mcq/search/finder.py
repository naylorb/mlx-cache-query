"""TextFinder — BM25-style text search across corpus chunks."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from collections import Counter

from mcq.core.types import Corpus, CorpusChunk


@dataclass
class SearchResult:
    chunk: CorpusChunk
    score: float
    snippet: str  # context around the best match


class TextFinder:
    @staticmethod
    def search(corpus: Corpus, query: str, top_k: int = 10) -> list[SearchResult]:
        """Search corpus chunks for relevance to query. Returns ranked results."""
        query_terms = TextFinder._tokenize(query.lower())
        if not query_terms:
            return []

        # Build document frequencies
        n = len(corpus.chunks)
        df: dict[str, int] = Counter()
        chunk_term_freqs: list[Counter] = []
        chunk_lengths: list[int] = []

        for chunk in corpus.chunks:
            terms = TextFinder._tokenize(chunk.content.lower())
            tf = Counter(terms)
            chunk_term_freqs.append(tf)
            chunk_lengths.append(len(terms))
            for term in set(terms):
                df[term] += 1

        avg_dl = sum(chunk_lengths) / max(n, 1)
        k1, b = 1.5, 0.75  # BM25 parameters

        results = []
        for i, chunk in enumerate(corpus.chunks):
            score = 0.0
            tf = chunk_term_freqs[i]
            dl = chunk_lengths[i]

            for term in query_terms:
                if term not in tf:
                    continue
                term_tf = tf[term]
                term_df = df.get(term, 0)
                idf = math.log((n - term_df + 0.5) / (term_df + 0.5) + 1)
                numerator = term_tf * (k1 + 1)
                denominator = term_tf + k1 * (1 - b + b * dl / max(avg_dl, 1))
                score += idf * numerator / denominator

            if score > 0:
                snippet = TextFinder._extract_snippet(chunk.content, query_terms)
                results.append(SearchResult(chunk=chunk, score=score, snippet=snippet))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple word tokenization."""
        return re.findall(r'\w+', text)

    @staticmethod
    def _extract_snippet(content: str, query_terms: list[str], context_chars: int = 150) -> str:
        """Extract a snippet around the first match of any query term."""
        content_lower = content.lower()
        best_pos = len(content)

        for term in query_terms:
            pos = content_lower.find(term)
            if pos != -1 and pos < best_pos:
                best_pos = pos

        if best_pos == len(content):
            return content[:context_chars * 2].strip() + "..."

        start = max(0, best_pos - context_chars)
        end = min(len(content), best_pos + context_chars)
        snippet = content[start:end].strip()

        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."

        return snippet
