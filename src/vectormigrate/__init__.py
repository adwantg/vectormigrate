"""vectormigrate public package exports."""

__version__ = "1.0.1"

from vectormigrate.artifacts import export_run_artifact_bundle, validate_artifact_bundle
from vectormigrate.benchmarks import (
    BenchmarkResult,
    benchmark_adapter_regression,
    benchmark_callable,
    benchmark_router_modes,
    benchmark_search_results,
    build_benchmark_protocol,
    export_benchmark_bundle,
    export_benchmark_report,
)
from vectormigrate.community import write_governance_file
from vectormigrate.compat import (
    LowRankAffineAdapter,
    OrthogonalProcrustesAdapter,
    ResidualMLPAdapter,
)
from vectormigrate.datasets import PairedVectorDataset
from vectormigrate.embedder import DeterministicHashEmbedder
from vectormigrate.evaluation import compare_search_paths
from vectormigrate.io import load_documents, load_query_cases
from vectormigrate.models import (
    Document,
    EmbeddingABI,
    MigrationEvent,
    MigrationPlan,
    MigrationState,
    QueryCase,
    SearchHit,
    ShadowMetrics,
)
from vectormigrate.notebook import execute_notebook_smoke, execute_notebook_subprocess_smoke
from vectormigrate.orchestrator import MigrationOrchestrator
from vectormigrate.plugins import BackendPlugin, PluginRegistry
from vectormigrate.registry import SQLiteRegistry
from vectormigrate.reporting import build_migration_report, export_migration_report
from vectormigrate.routing import ConfidenceGatedSearchRouter, RoutingDecision
from vectormigrate.telemetry import (
    InMemoryTelemetrySink,
    JsonlTelemetrySink,
    OnlineShadowEvaluator,
    OpenTelemetryBridgeSink,
)
from vectormigrate.vector_store import InMemoryVectorBackend

__all__ = [
    "BackendPlugin",
    "BenchmarkResult",
    "ConfidenceGatedSearchRouter",
    "DeterministicHashEmbedder",
    "Document",
    "EmbeddingABI",
    "InMemoryTelemetrySink",
    "InMemoryVectorBackend",
    "JsonlTelemetrySink",
    "LowRankAffineAdapter",
    "MigrationEvent",
    "MigrationOrchestrator",
    "MigrationPlan",
    "MigrationState",
    "OnlineShadowEvaluator",
    "OpenTelemetryBridgeSink",
    "OrthogonalProcrustesAdapter",
    "PairedVectorDataset",
    "PluginRegistry",
    "QueryCase",
    "ResidualMLPAdapter",
    "RoutingDecision",
    "SQLiteRegistry",
    "SearchHit",
    "ShadowMetrics",
    "__version__",
    "benchmark_adapter_regression",
    "benchmark_callable",
    "benchmark_router_modes",
    "benchmark_search_results",
    "build_benchmark_protocol",
    "build_migration_report",
    "compare_search_paths",
    "execute_notebook_smoke",
    "execute_notebook_subprocess_smoke",
    "export_benchmark_bundle",
    "export_benchmark_report",
    "export_migration_report",
    "export_run_artifact_bundle",
    "load_documents",
    "load_query_cases",
    "validate_artifact_bundle",
    "write_governance_file",
]
