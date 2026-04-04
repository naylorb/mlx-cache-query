"""Query engine — the payoff.

Loads a pre-built KV cache, appends the user's question, and generates
an answer. Supports streaming (for CLI) and batch (for JSON output).

The cached prefix ends with `<query>\\n`. We append:
    {question}
    </query>

Then generate.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Generator


@dataclass
class QueryResult:
    text: str
    ttft_ms: float
    decode_tokens_per_sec: float
    total_tokens: int


class QueryEngine:
    @staticmethod
    def format_query_prompt(question: str) -> str:
        return f"{question}\n</query>\n"

    @staticmethod
    def query(
        model,
        tokenizer,
        prompt_cache: list,
        question: str,
        max_tokens: int = 512,
    ) -> QueryResult:
        """Run a complete query and return the result."""
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
        decode_tokens = max(token_count - 1, 1)
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

        from mlx_lm import stream_generate

        for response in stream_generate(
            model,
            tokenizer,
            prompt=query_text,
            max_tokens=max_tokens,
            prompt_cache=cache,
        ):
            yield response.text
