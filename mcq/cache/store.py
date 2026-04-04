from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

# DEVIATION: Lazy import to allow unit tests to run without mlx-lm installed.
# Tests mock these names via patch("mcq.cache.store.save_prompt_cache", ...).
try:
    from mlx_lm.models.cache import load_prompt_cache, save_prompt_cache
except ImportError:
    load_prompt_cache = None  # type: ignore[assignment]
    save_prompt_cache = None  # type: ignore[assignment]

from mcq.cache.registry import CacheRegistry
from mcq.core.constants import NORMALIZATION_VERSION, PROMPT_TEMPLATE_VERSION
from mcq.core.types import ArtifactRef, TokenizedPrefix


class CacheStore:
    def __init__(self, artifacts_dir: Path) -> None:
        self._dir = Path(artifacts_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def artifact_path(self, artifact_hash: str) -> Path:
        return self._dir / f"{artifact_hash}.safetensors"

    def save(
        self,
        cache: list,
        prefix: TokenizedPrefix,
        corpus_name: str,
        mlx_lm_version: str,
        registry: CacheRegistry,
    ) -> ArtifactRef:
        artifact_hash = prefix.prefix_hash
        path = self.artifact_path(artifact_hash)

        metadata = {
            "model_id": prefix.model_id,
            "model_revision": prefix.model_revision,
            "corpus_hash": prefix.corpus_hash,
            "corpus_name": corpus_name,
            "prefix_token_count": str(prefix.token_count),
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
            "mlx_lm_version": mlx_lm_version,
        }

        save_prompt_cache(str(path), cache, metadata)

        file_size = path.stat().st_size

        ref = ArtifactRef(
            artifact_hash=artifact_hash,
            model_id=prefix.model_id,
            model_revision=prefix.model_revision,
            corpus_hash=prefix.corpus_hash,
            corpus_name=corpus_name,
            prefix_token_count=prefix.token_count,
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
            normalization_version=NORMALIZATION_VERSION,
            build_timestamp=datetime.now(timezone.utc).isoformat(),
            file_size_bytes=file_size,
            file_path=str(path),
            mlx_lm_version=mlx_lm_version,
        )

        registry.register(ref)
        return ref

    def load(self, ref: ArtifactRef) -> tuple[list, dict]:
        return load_prompt_cache(ref.file_path, return_metadata=True)

    def delete(self, ref: ArtifactRef, registry: CacheRegistry) -> None:
        path = Path(ref.file_path)
        if path.exists():
            path.unlink()
        registry.delete(ref.artifact_hash)
