from __future__ import annotations

import json
import tempfile
from pathlib import Path

from vectormigrate import (
    EmbeddingABI,
    InMemoryTelemetrySink,
    MigrationPlan,
    OnlineShadowEvaluator,
    SQLiteRegistry,
    export_run_artifact_bundle,
)


def main() -> None:
    temp_dir = Path(tempfile.mkdtemp(prefix="vectormigrate-artifacts-"))
    registry = SQLiteRegistry(temp_dir / "state.sqlite")
    source = registry.register_abi(EmbeddingABI("legacy", "vectormigrate", "v1", dimensions=8))
    target = registry.register_abi(EmbeddingABI("upgrade", "vectormigrate", "v2", dimensions=8))
    plan = registry.create_plan(
        MigrationPlan(
            source_abi_id=source.abi_id or "",
            target_abi_id=target.abi_id or "",
            alias_name="retrieval_active",
        )
    )

    sink = InMemoryTelemetrySink()
    evaluator = OnlineShadowEvaluator(sink=sink, top_k=2)
    evaluator.record("q1", ["doc-1"], ["doc-1"], {"doc-1": 3.0})
    summary = evaluator.summary()
    manifest_path = export_run_artifact_bundle(registry, plan.plan_id, temp_dir / "artifacts")

    print(
        json.dumps(
            {
                "telemetry_events": len(sink.events),
                "shadow_query_count": summary["query_count"],
                "artifact_manifest": str(manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
