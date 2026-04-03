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
