from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.generate import generate_step

from mcq.core.types import TokenizedPrefix


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
