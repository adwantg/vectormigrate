from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vectormigrate.registry import SQLiteRegistry


def build_migration_report(registry: SQLiteRegistry, plan_id: str) -> dict[str, Any]:
    plan = registry.require_plan(plan_id)
    source_abi = registry.require_abi(plan.source_abi_id)
    target_abi = registry.require_abi(plan.target_abi_id)
    events = registry.list_events(plan_id)
    return {
        "plan": plan.to_dict(),
        "source_abi": source_abi.to_dict(),
        "target_abi": target_abi.to_dict(),
        "events": [event.to_dict() for event in events],
    }


def export_migration_report(
    registry: SQLiteRegistry,
    plan_id: str,
    output_path: str | Path,
) -> Path:
    report = build_migration_report(registry, plan_id)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path
