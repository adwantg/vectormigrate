from __future__ import annotations

import json
from pathlib import Path

from vectormigrate.artifacts import export_run_artifact_bundle, validate_artifact_bundle
from vectormigrate.benchmarks import (
    BenchmarkResult,
    build_benchmark_protocol,
    export_benchmark_bundle,
    export_benchmark_report,
)
from vectormigrate.community import write_governance_file
from vectormigrate.models import EmbeddingABI, MigrationPlan
from vectormigrate.plugins import BackendPlugin, PluginRegistry
from vectormigrate.registry import SQLiteRegistry
from vectormigrate.telemetry import (
    InMemoryTelemetrySink,
    JsonlTelemetrySink,
    OnlineShadowEvaluator,
    OpenTelemetryBridgeSink,
)


def test_export_run_artifact_bundle(tmp_path: Path) -> None:
    registry = SQLiteRegistry(tmp_path / "state.sqlite")
    source = registry.register_abi(EmbeddingABI("legacy", "vectormigrate", "v1", dimensions=8))
    target = registry.register_abi(EmbeddingABI("upgrade", "vectormigrate", "v2", dimensions=8))
    plan = registry.create_plan(
        MigrationPlan(
            source_abi_id=source.abi_id or "",
            target_abi_id=target.abi_id or "",
            alias_name="active",
        )
    )

    manifest_path = export_run_artifact_bundle(
        registry,
        plan.plan_id,
        tmp_path / "artifacts",
        extra_sections={"benchmarks": {"mean_latency_ms": 1.0}},
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["files"]["report"] == "migration_report.json"
    assert manifest["files"]["benchmarks"] == "benchmarks.json"
    validated = validate_artifact_bundle(manifest_path)
    assert validated["plan_id"] == plan.plan_id
    assert "benchmarks" in validated["resolved_files"]


def test_plugin_registry_registers_and_lists_plugins() -> None:
    registry = PluginRegistry()
    registry.register(BackendPlugin(name="fake", factory=lambda: "ok", description="demo plugin"))

    plugins = registry.list_plugins()

    assert plugins[0].name == "fake"
    assert registry.get("fake").factory() == "ok"


def test_plugin_registry_loads_entry_points(monkeypatch) -> None:
    registry = PluginRegistry()

    def factory() -> str:
        """example plugin"""

        return "loaded"

    class FakeEntryPoint:
        name = "fake-entry"

        @staticmethod
        def load():
            return factory

    class FakeEntryPoints:
        @staticmethod
        def select(*, group: str):
            assert group == "vectormigrate.backends"
            return [FakeEntryPoint()]

    monkeypatch.setattr("vectormigrate.plugins.entry_points", lambda: FakeEntryPoints())

    plugins = registry.load_entry_point_plugins()

    assert plugins[0].name == "fake-entry"
    assert registry.get("fake-entry").factory() == "loaded"


def test_online_shadow_evaluator_records_and_summarizes() -> None:
    sink = InMemoryTelemetrySink()
    evaluator = OnlineShadowEvaluator(sink=sink, top_k=2)

    evaluator.record("q1", ["doc-1"], ["doc-1"], {"doc-1": 3.0})
    summary = evaluator.summary()

    assert summary["query_count"] == 1
    assert len(sink.events) == 2


def test_jsonl_telemetry_sink_writes_records(tmp_path: Path) -> None:
    sink = JsonlTelemetrySink(tmp_path / "telemetry.jsonl")
    sink.emit("shadow_query_recorded", {"query_id": "q1"})

    lines = (tmp_path / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[0])
    assert payload["event_type"] == "shadow_query_recorded"


def test_open_telemetry_bridge_sink_forwards_events() -> None:
    recorded: list[tuple[str, dict[str, object]]] = []

    class FakeCollector:
        def record_event(self, name: str, attributes):
            recorded.append((name, dict(attributes)))

    sink = OpenTelemetryBridgeSink(FakeCollector())
    sink.emit("shadow_summary_computed", {"query_count": 2})

    assert recorded == [("shadow_summary_computed", {"query_count": 2})]


def test_export_benchmark_report(tmp_path: Path) -> None:
    path = export_benchmark_report(
        [BenchmarkResult("demo", 2, 1.0, 1.2, {"query_count": 2.0})],
        tmp_path / "benchmark_report.json",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["results"][0]["name"] == "demo"


def test_export_benchmark_bundle_writes_protocol_and_report(tmp_path: Path) -> None:
    protocol = build_benchmark_protocol(
        name="demo-benchmark",
        command=["python3", "-m", "vectormigrate.cli", "benchmark-demo"],
        iterations=2,
        notes=["synthetic run"],
    )
    protocol_path = export_benchmark_bundle(
        [BenchmarkResult("demo", 2, 1.0, 1.2, {"query_count": 2.0})],
        tmp_path / "benchmark_bundle",
        protocol=protocol,
    )
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    report_payload = json.loads(
        (protocol_path.parent / "benchmark_report.json").read_text(encoding="utf-8")
    )
    assert payload["name"] == "demo-benchmark"
    assert report_payload["results"][0]["name"] == "demo"


def test_write_governance_file(tmp_path: Path) -> None:
    path = write_governance_file(tmp_path / "COMMUNITY.md")
    assert "Community Adapter Governance" in path.read_text(encoding="utf-8")
