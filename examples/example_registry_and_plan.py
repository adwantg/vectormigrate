from __future__ import annotations

import json
import tempfile
from pathlib import Path

from vectormigrate import (
    EmbeddingABI,
    InMemoryVectorBackend,
    MigrationOrchestrator,
    MigrationPlan,
    SQLiteRegistry,
)


def main() -> None:
    temp_dir = Path(tempfile.mkdtemp(prefix="vectormigrate-example-"))
    registry = SQLiteRegistry(temp_dir / "state.sqlite")
    backend = InMemoryVectorBackend()
    orchestrator = MigrationOrchestrator(registry, backend)

    legacy = orchestrator.register_abi(
        EmbeddingABI("legacy-model", "vectormigrate", "v1", dimensions=16)
    )
    target = orchestrator.register_abi(
        EmbeddingABI("target-model", "vectormigrate", "v2", dimensions=16)
    )
    plan = orchestrator.create_plan(
        MigrationPlan(
            source_abi_id=legacy.abi_id or "",
            target_abi_id=target.abi_id or "",
            alias_name="retrieval_active",
        )
    )
    provisioned = orchestrator.provision_plan(plan.plan_id)
    print(
        json.dumps(
            {
                "legacy_abi_id": legacy.abi_id,
                "target_abi_id": target.abi_id,
                "plan_id": provisioned.plan_id,
                "state": provisioned.state.value,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
