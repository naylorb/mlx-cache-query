"""Prefix compiler — corpus text → token sequence.

The prompt template wraps the corpus in a structure the model can
reason over. This is the context engineering surface — the quality
of what goes into the cache determines the quality of answers.

Templates:
  general  — default Q&A over documents
  code     — optimized for code repositories
  raw      — minimal wrapping, maximum context

The template ends with `<query>\\n` — the KV cache covers everything
up to that point. At query time we append the question and `</query>`.
"""
from __future__ import annotations

from mcq.core.constants import DEFAULT_QUERY_BUDGET, PROMPT_TEMPLATE_VERSION
from mcq.core.types import Corpus, TokenizedPrefix

TEMPLATES = {
    "general": (
        "You are answering questions about the following document(s). "
        "Be precise and cite file names when relevant.\n\n"
    ),
    "code": (
        "You are a code expert answering questions about the following codebase. "
        "Reference specific files, functions, and line ranges. "
        "Be precise and technical.\n\n"
    ),
    "raw": "",
}


class PrefixCompiler:
    TEMPLATE_VERSION = PROMPT_TEMPLATE_VERSION

    @staticmethod
    def build_prompt_text(corpus: Corpus, template: str = "general") -> str:
        """Build the full prompt text that will be cached.

        Structure:
            {system preamble}
            <documents>
            [== path/to/file.md ==]
            {content}

            [== path/to/other.py ==]
            {content}
            </documents>

            <query>
        """
        preamble = TEMPLATES.get(template, TEMPLATES["general"])
        parts = [preamble, "<documents>\n"]

        for chunk in sorted(corpus.chunks, key=lambda c: c.source_path):
            parts.append(f"[== {chunk.source_path} ==]\n")
            parts.append(chunk.content)
            if not chunk.content.endswith("\n"):
                parts.append("\n")
            parts.append("\n")

        parts.append("</documents>\n\n<query>\n")
        return "".join(parts)

    @staticmethod
    def compile(
        corpus: Corpus,
        tokenizer,
        model_id: str,
        model_revision: str,
        max_context: int | None = None,
        query_budget: int = DEFAULT_QUERY_BUDGET,
        template: str = "general",
    ) -> TokenizedPrefix:
        prompt_text = PrefixCompiler.build_prompt_text(corpus, template=template)
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
