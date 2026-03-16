from __future__ import annotations

import numpy as np

from vectormigrate.benchmarks import (
    benchmark_adapter_regression,
    benchmark_callable,
    benchmark_router_modes,
    benchmark_search_results,
    build_benchmark_protocol,
)
from vectormigrate.compat import LowRankAffineAdapter
from vectormigrate.models import QueryCase, SearchHit
from vectormigrate.routing import RoutingDecision


def test_benchmark_callable_returns_latency_metrics() -> None:
    result = benchmark_callable("noop", lambda: sum([1, 2, 3]), iterations=3)
    assert result.iterations == 3
    assert result.mean_latency_ms >= 0.0


def test_benchmark_adapter_regression_reports_similarity() -> None:
    rng = np.random.default_rng(3)
    source = rng.normal(size=(20, 4))
    target = source + 0.25
    result = benchmark_adapter_regression(
        "affine",
        LowRankAffineAdapter(rank=4),
        source,
        target,
        iterations=2,
    )
    assert result.metrics["mean_cosine_similarity"] > 0.9


def test_benchmark_router_modes_reports_shares() -> None:
    shares = benchmark_router_modes(
        [
            RoutingDecision("adapter", 0.9, (SearchHit("a", 1.0, "x"),), "ok"),
            RoutingDecision("dual_read", 0.2, (SearchHit("b", 1.0, "x"),), "fallback"),
        ]
    )
    assert shares["adapter_share"] == 0.5


def test_benchmark_search_results_reports_query_count() -> None:
    queries = [QueryCase("q1", "query", {"doc-1": 1.0})]
    result = benchmark_search_results(
        queries,
        lambda _text, _k: [SearchHit("doc-1", 1.0, "ns")],
    )
    assert result.metrics["query_count"] == 1.0


def test_build_benchmark_protocol_reports_runtime_metadata() -> None:
    protocol = build_benchmark_protocol(
        name="demo",
        command=["python3", "-m", "vectormigrate.cli", "benchmark-demo"],
        iterations=2,
        notes=["synthetic"],
    )
    assert protocol["name"] == "demo"
    assert protocol["iterations"] == 2
    assert protocol["command"][0] == "python3"
