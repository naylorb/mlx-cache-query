"""mcq API server — local HTTP API for IDE and plugin integration.

Run with: mcq serve --port 8420
Or directly: uvicorn mcq.api.server:create_app --factory --port 8420
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse

from mcq.core.constants import ARTIFACTS_DIR, REGISTRY_DB


def create_app() -> FastAPI:
    app = FastAPI(title="mcq", description="Local MLX prompt-cache query API", version="0.1.0")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/corpora")
    async def list_corpora():
        from mcq.cache.registry import CacheRegistry
        registry = CacheRegistry(REGISTRY_DB)
        corpora = registry.list_corpora()
        artifacts = registry.list_all()
        cached_names = {a.corpus_name for a in artifacts}

        result = []
        for c in corpora:
            name = c["name"]
            result.append({
                "name": name,
                "status": "cached" if name in cached_names else "ingested",
                "source_path": c["source_path"],
                "content_hash": c["content_hash"],
                "chunk_count": c["chunk_count"],
            })
        return result

    @app.get("/artifacts")
    async def list_artifacts():
        from mcq.cache.registry import CacheRegistry
        registry = CacheRegistry(REGISTRY_DB)
        refs = registry.list_all()
        return [
            {
                "artifact_hash": r.artifact_hash,
                "corpus_name": r.corpus_name,
                "model_id": r.model_id,
                "prefix_token_count": r.prefix_token_count,
                "file_size_bytes": r.file_size_bytes,
                "build_timestamp": r.build_timestamp,
            }
            for r in refs
        ]

    @app.get("/info/{corpus_name}")
    async def corpus_info(corpus_name: str):
        from mcq.cache.registry import CacheRegistry
        registry = CacheRegistry(REGISTRY_DB)
        refs = registry.get_by_corpus_name(corpus_name)
        if not refs:
            raise HTTPException(status_code=404, detail=f"Corpus '{corpus_name}' not found")
        return [
            {
                "artifact_hash": r.artifact_hash,
                "model_id": r.model_id,
                "model_revision": r.model_revision,
                "corpus_hash": r.corpus_hash,
                "corpus_name": r.corpus_name,
                "prefix_token_count": r.prefix_token_count,
                "prompt_template_version": r.prompt_template_version,
                "normalization_version": r.normalization_version,
                "build_timestamp": r.build_timestamp,
                "file_size_bytes": r.file_size_bytes,
                "file_path": r.file_path,
                "mlx_lm_version": r.mlx_lm_version,
            }
            for r in refs
        ]

    @app.get("/find/{corpus_name}")
    async def find(corpus_name: str, q: str = Query(...), top_k: int = Query(10)):
        """BM25 text search across a corpus."""
        from mcq.cache.registry import CacheRegistry
        from mcq.ingest.ingestor import CorpusIngestor

        registry = CacheRegistry(REGISTRY_DB)
        corpus_info = registry.get_corpus(corpus_name)
        if not corpus_info:
            raise HTTPException(status_code=404, detail=f"Corpus '{corpus_name}' not found")

        corpus = CorpusIngestor.ingest(Path(corpus_info["source_path"]), name=corpus_name)

        from mcq.search.finder import TextFinder
        results = TextFinder.search(corpus, q, top_k=top_k)
        return [
            {
                "source_path": r.chunk.source_path,
                "score": round(r.score, 3),
                "snippet": r.snippet,
            }
            for r in results
        ]

    @app.post("/query/{corpus_name}")
    async def query_corpus(
        corpus_name: str,
        question: str = Query(...),
        model: str = Query(default="mlx-community/Qwen2.5-3B-Instruct-4bit"),
        max_tokens: int = Query(default=512),
    ):
        """Query a cached corpus. Returns the full answer."""
        from mcq.cache.registry import CacheRegistry
        registry = CacheRegistry(REGISTRY_DB)
        refs = registry.get_by_corpus_name(corpus_name, model_id=model)
        if not refs:
            raise HTTPException(status_code=404, detail=f"No cache for corpus '{corpus_name}' with model '{model}'")

        try:
            from mcq.cache.store import CacheStore
            from mcq.inference.engine import QueryEngine
            from mlx_lm import load

            ref = refs[0]
            store = CacheStore(ARTIFACTS_DIR)
            mlx_model, tokenizer = load(model)
            prompt_cache, _ = store.load(ref)
            result = QueryEngine.query(mlx_model, tokenizer, prompt_cache, question, max_tokens=max_tokens)
            return {
                "text": result.text,
                "ttft_ms": round(result.ttft_ms, 1),
                "tokens_per_sec": round(result.decode_tokens_per_sec, 1),
                "total_tokens": result.total_tokens,
            }
        except ImportError:
            raise HTTPException(status_code=503, detail="mlx not available — query requires Apple Silicon")

    @app.get("/export/{corpus_name}")
    async def export_corpus(
        corpus_name: str,
        format: str = Query(default="markdown", enum=["markdown", "json", "context", "filelist"]),
    ):
        """Export corpus content in various formats."""
        from mcq.cache.registry import CacheRegistry
        from mcq.ingest.ingestor import CorpusIngestor
        from mcq.export.exporter import CorpusExporter

        registry = CacheRegistry(REGISTRY_DB)
        corpus_info = registry.get_corpus(corpus_name)
        if not corpus_info:
            raise HTTPException(status_code=404, detail=f"Corpus '{corpus_name}' not found")

        corpus = CorpusIngestor.ingest(Path(corpus_info["source_path"]), name=corpus_name)

        exporters = {
            "markdown": CorpusExporter.to_markdown,
            "json": CorpusExporter.to_json,
            "context": CorpusExporter.to_context,
            "filelist": CorpusExporter.to_filelist,
        }
        content = exporters[format](corpus)

        media_type = "application/json" if format == "json" else "text/plain"
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(content=content, media_type=media_type)

    return app
