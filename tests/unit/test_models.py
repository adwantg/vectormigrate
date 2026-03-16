from __future__ import annotations

import pytest

from vectormigrate.models import EmbeddingABI, MigrationEvent, MigrationPlan, MigrationState


def test_embedding_abi_generates_stable_id() -> None:
    abi = EmbeddingABI(
        model_id="text-embedding-3-large",
        provider="OpenAI",
        version="2026.03",
        dimensions=3072,
        chunker_version="chunks-v4",
    )

    assert abi.abi_id == "openai/text-embedding-3-large@2026.03#chunks-v4"


def test_embedding_abi_rejects_invalid_metric() -> None:
    with pytest.raises(ValueError):
        EmbeddingABI(
            model_id="model",
            provider="provider",
            version="v1",
            dimensions=8,
            distance_metric="manhattan",
        )


def test_migration_plan_rejects_same_source_and_target() -> None:
    with pytest.raises(ValueError):
        MigrationPlan(
            source_abi_id="same",
            target_abi_id="same",
            alias_name="active",
        )


def test_migration_plan_transition_rules() -> None:
    plan = MigrationPlan(
        source_abi_id="old",
        target_abi_id="new",
        alias_name="active",
    )

    plan.transition(MigrationState.PROVISIONED)
    plan.transition(MigrationState.BACKFILLING)
    plan.transition(MigrationState.SHADOW_EVAL)
    plan.transition(MigrationState.READY_TO_CUTOVER)

    assert plan.state == MigrationState.READY_TO_CUTOVER

    with pytest.raises(ValueError):
        plan.transition(MigrationState.PROVISIONED)


def test_migration_event_round_trip() -> None:
    event = MigrationEvent(
        plan_id="plan-1",
        event_type="STATE_TRANSITION",
        actor="tester",
        reason="validate audit trail",
        from_state="DRAFT",
        to_state="PROVISIONED",
        details={"documents": 5},
        event_id=7,
    )

    restored = MigrationEvent.from_dict(event.to_dict())

    assert restored == event
