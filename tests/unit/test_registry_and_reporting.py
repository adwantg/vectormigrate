from __future__ import annotations

import json
from pathlib import Path

from vectormigrate.models import EmbeddingABI, MigrationPlan, MigrationState
from vectormigrate.registry import SQLiteRegistry
from vectormigrate.reporting import build_migration_report, export_migration_report


def test_registry_records_plan_events_and_transitions(tmp_path: Path) -> None:
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

    registry.transition_plan(
        plan.plan_id,
        MigrationState.PROVISIONED,
        actor="alice",
        reason="initial setup",
    )
    events = registry.list_events(plan.plan_id)

    assert events[0].event_type == "PLAN_CREATED"
    assert events[1].actor == "alice"
    assert events[1].from_state == "DRAFT"
    assert events[1].to_state == "PROVISIONED"


def test_reporting_exports_plan_bundle(tmp_path: Path) -> None:
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

    report = build_migration_report(registry, plan.plan_id)
    output_path = export_migration_report(registry, plan.plan_id, tmp_path / "report.json")

    assert report["plan"]["plan_id"] == plan.plan_id
    exported = json.loads(output_path.read_text(encoding="utf-8"))
    assert exported["source_abi"]["abi_id"] == source.abi_id
