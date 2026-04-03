from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TimingResult:
    label: str
    duration_ms: float


@dataclass
class BenchmarkReport:
    timings: list[TimingResult] = field(default_factory=list)

    def add(self, label: str, duration_s: float) -> None:
        self.timings.append(TimingResult(label=label, duration_ms=duration_s * 1000))

    def summary(self) -> str:
        lines = ["Benchmark Results", "=" * 50]
        for t in self.timings:
            lines.append(f"  {t.label:30s}  {t.duration_ms:8.1f} ms")
        lines.append("=" * 50)
        return "\n".join(lines)


class BenchmarkRunner:
    """Runs the full pipeline and reports timings."""

    @staticmethod
    def run(
        corpus_path: Path,
        corpus_name: str,
        model_id: str,
        question: str,
        artifacts_dir: Path,
        db_path: Path,
    ) -> BenchmarkReport:
        from mlx_lm import load
        import mlx_lm
        from huggingface_hub import model_info

        from app.cache.builder import CacheBuilder
        from app.cache.registry import CacheRegistry
        from app.cache.store import CacheStore
        from app.ingest.ingestor import CorpusIngestor
        from app.inference.engine import QueryEngine
        from app.prefix.compiler import PrefixCompiler

        report = BenchmarkReport()

        # Ingest
        t0 = time.perf_counter()
        corpus = CorpusIngestor.ingest(corpus_path, name=corpus_name)
        report.add("Corpus ingest", time.perf_counter() - t0)

        # Load model
        t0 = time.perf_counter()
        try:
            info = model_info(model_id)
            revision = info.sha
        except Exception:
            revision = "unknown"
        mlx_model, tokenizer = load(model_id)
        report.add("Model load", time.perf_counter() - t0)

        # Compile
        t0 = time.perf_counter()
        prefix = PrefixCompiler.compile(
            corpus=corpus,
            tokenizer=tokenizer,
            model_id=model_id,
            model_revision=revision,
        )
        report.add("Prefix compile", time.perf_counter() - t0)

        # Build cache
        t0 = time.perf_counter()
        cache = CacheBuilder.build(prefix, mlx_model, tokenizer)
        report.add("Cache build", time.perf_counter() - t0)

        # Save
        store = CacheStore(artifacts_dir)
        registry = CacheRegistry(db_path)
        t0 = time.perf_counter()
        ref = store.save(
            cache=cache,
            prefix=prefix,
            corpus_name=corpus_name,
            mlx_lm_version=mlx_lm.__version__,
            registry=registry,
        )
        report.add("Cache save", time.perf_counter() - t0)

        # Load
        t0 = time.perf_counter()
        loaded_cache, _ = store.load(ref)
        report.add("Cache load", time.perf_counter() - t0)

        # Query
        t0 = time.perf_counter()
        result = QueryEngine.query(
            model=mlx_model,
            tokenizer=tokenizer,
            prompt_cache=loaded_cache,
            question=question,
            max_tokens=100,
        )
        report.add("Query (100 tokens)", time.perf_counter() - t0)
        report.add("Query TTFT (approx)", result.ttft_ms / 1000)
        report.add(f"Decode throughput", 0)  # placeholder: actual value in summary

        return report
