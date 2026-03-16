from __future__ import annotations

import json
import platform
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from vectormigrate.models import QueryCase, SearchHit
from vectormigrate.routing import RoutingDecision
from vectormigrate.validation import validate_benchmark_iterations


class BenchmarkAdapter(Protocol):
    def fit(
        self,
        source_vectors: NDArray[np.float64],
        target_vectors: NDArray[np.float64],
    ) -> Any: ...

    def transform(self, vectors: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def mean_cosine_similarity(
        self,
        source_vectors: NDArray[np.float64],
        target_vectors: NDArray[np.float64],
    ) -> float: ...


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    iterations: int
    mean_latency_ms: float
    p95_latency_ms: float
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "mean_latency_ms": self.mean_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "metrics": dict(self.metrics),
        }


def export_benchmark_report(
    results: Sequence[BenchmarkResult],
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": [result.to_dict() for result in results],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def build_benchmark_protocol(
    *,
    name: str,
    command: Sequence[str],
    iterations: int,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    validate_benchmark_iterations(iterations)
    return {
        "name": name,
        "command": list(command),
        "iterations": iterations,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "notes": list(notes),
    }


def export_benchmark_bundle(
    results: Sequence[BenchmarkResult],
    output_dir: str | Path,
    *,
    protocol: dict[str, Any],
) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    export_benchmark_report(results, path / "benchmark_report.json")
    protocol_path = path / "benchmark_protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")
    return protocol_path


def benchmark_callable(
    name: str,
    fn: Callable[[], object],
    iterations: int = 50,
) -> BenchmarkResult:
    validate_benchmark_iterations(iterations)
    latencies: list[float] = []
    for _ in range(iterations):
        start = perf_counter()
        fn()
        latencies.append((perf_counter() - start) * 1000)
    return BenchmarkResult(
        name=name,
        iterations=iterations,
        mean_latency_ms=float(np.mean(latencies)),
        p95_latency_ms=float(np.percentile(latencies, 95)),
        metrics={},
    )


def benchmark_adapter_regression(
    name: str,
    adapter: BenchmarkAdapter,
    source_vectors: NDArray[np.float64],
    target_vectors: NDArray[np.float64],
    iterations: int = 10,
) -> BenchmarkResult:
    validate_benchmark_iterations(iterations)
    fit_latencies: list[float] = []
    transform_latencies: list[float] = []
    metric_values: list[float] = []
    for _ in range(iterations):
        start = perf_counter()
        adapter.fit(source_vectors, target_vectors)
        fit_latencies.append((perf_counter() - start) * 1000)
        start = perf_counter()
        adapter.transform(source_vectors)
        transform_latencies.append((perf_counter() - start) * 1000)
        metric_values.append(adapter.mean_cosine_similarity(source_vectors, target_vectors))
    latencies = fit_latencies + transform_latencies
    return BenchmarkResult(
        name=name,
        iterations=iterations,
        mean_latency_ms=float(np.mean(latencies)),
        p95_latency_ms=float(np.percentile(latencies, 95)),
        metrics={
            "mean_fit_latency_ms": float(np.mean(fit_latencies)),
            "mean_transform_latency_ms": float(np.mean(transform_latencies)),
            "mean_cosine_similarity": float(np.mean(metric_values)),
        },
    )


def benchmark_router_modes(
    decisions: Sequence[RoutingDecision],
) -> dict[str, float]:
    if not decisions:
        return {"adapter_share": 0.0, "dual_read_share": 0.0}
    adapter_count = sum(1 for decision in decisions if decision.mode == "adapter")
    total = len(decisions)
    return {
        "adapter_share": adapter_count / total,
        "dual_read_share": 1.0 - (adapter_count / total),
    }


def benchmark_search_results(
    queries: Sequence[QueryCase],
    search_fn: Callable[[str, int], Sequence[SearchHit]],
    top_k: int = 5,
) -> BenchmarkResult:
    latencies: list[float] = []
    for query in queries:
        start = perf_counter()
        search_fn(query.text, top_k)
        latencies.append((perf_counter() - start) * 1000)
    return BenchmarkResult(
        name="search_results",
        iterations=len(queries),
        mean_latency_ms=float(np.mean(latencies) if latencies else 0.0),
        p95_latency_ms=float(np.percentile(latencies, 95) if latencies else 0.0),
        metrics={"query_count": float(len(queries))},
    )
