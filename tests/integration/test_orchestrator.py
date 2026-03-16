from __future__ import annotations

from pathlib import Path

from vectormigrate.compat import OrthogonalProcrustesAdapter
from vectormigrate.embedder import DeterministicHashEmbedder
from vectormigrate.models import (
    Document,
    EmbeddingABI,
    MigrationPlan,
    MigrationState,
    QueryCase,
)
from vectormigrate.orchestrator import MigrationOrchestrator
from vectormigrate.registry import SQLiteRegistry
from vectormigrate.vector_store import InMemoryVectorBackend


def make_environment(
    tmp_path: Path,
) -> tuple[
    MigrationOrchestrator,
    EmbeddingABI,
    EmbeddingABI,
    DeterministicHashEmbedder,
    DeterministicHashEmbedder,
]:
    registry = SQLiteRegistry(tmp_path / "state.sqlite")
    backend = InMemoryVectorBackend()
    orchestrator = MigrationOrchestrator(registry, backend)
    legacy_abi = EmbeddingABI(
        model_id="legacy",
        provider="vectormigrate",
        version="2026.03",
        dimensions=24,
        chunker_version="v1",
    )
    new_abi = EmbeddingABI(
        model_id="upgrade",
        provider="vectormigrate",
        version="2026.04",
        dimensions=24,
        chunker_version="v1",
    )
    orchestrator.register_abi(legacy_abi)
    orchestrator.register_abi(new_abi)
    legacy_embedder = DeterministicHashEmbedder(legacy_abi, semantic_salt="shared", rotation_seed=5)
    new_embedder = DeterministicHashEmbedder(new_abi, semantic_salt="shared", rotation_seed=19)
    return orchestrator, legacy_abi, new_abi, legacy_embedder, new_embedder


def test_orchestrator_end_to_end_cutover(tmp_path: Path) -> None:
    orchestrator, legacy_abi, new_abi, legacy_embedder, new_embedder = make_environment(tmp_path)
    plan = orchestrator.create_plan(
        MigrationPlan(
            source_abi_id=legacy_abi.abi_id or "",
            target_abi_id=new_abi.abi_id or "",
            alias_name="active",
        )
    )
    orchestrator.provision_plan(plan.plan_id)

    documents = [
        Document("doc-1", "Embedding ABI versioning prevents mixed vector spaces."),
        Document("doc-2", "Blue green migrations rely on alias cutovers."),
        Document("doc-3", "Shadow evaluation compares recall and ranking quality."),
    ]
    orchestrator.index_documents(legacy_abi.abi_id or "", documents, legacy_embedder)
    orchestrator.enable_dual_write(plan.plan_id)
    orchestrator.backfill(plan.plan_id, documents, new_embedder)

    metrics = orchestrator.evaluate_plan(
        plan.plan_id,
        [
            QueryCase("q1", "blue green alias cutovers", {"doc-2": 3.0}),
            QueryCase("q2", "ranking quality evaluation", {"doc-3": 3.0}),
        ],
        legacy_embedder,
        new_embedder,
        top_k=2,
    )
    assert metrics.passes is True

    cutover = orchestrator.cutover(plan.plan_id)
    hits = orchestrator.search_namespace(
        "active",
        "blue green alias cutovers",
        new_embedder,
        top_k=2,
    )

    assert cutover.state == MigrationState.CUTOVER
    assert hits[0].doc_id == "doc-2"


def test_orchestrator_adapter_search_on_legacy_namespace(tmp_path: Path) -> None:
    orchestrator, legacy_abi, _new_abi, legacy_embedder, new_embedder = make_environment(tmp_path)
    documents = [
        Document("doc-1", "Query time adapters map new queries back to legacy spaces."),
        Document("doc-2", "Full backfills still matter for durable migrations."),
    ]
    orchestrator.index_documents(legacy_abi.abi_id or "", documents, legacy_embedder)

    adapter = OrthogonalProcrustesAdapter().fit(
        new_embedder.embed([document.text for document in documents]),
        legacy_embedder.embed([document.text for document in documents]),
    )
    hits = orchestrator.adapter_search(
        legacy_abi.abi_id or "",
        "map new queries to legacy spaces",
        new_embedder,
        adapter,
        top_k=2,
    )

    assert hits[0].doc_id == "doc-1"


def test_orchestrator_rollback_and_confidence_gated_search(tmp_path: Path) -> None:
    orchestrator, legacy_abi, new_abi, legacy_embedder, new_embedder = make_environment(tmp_path)
    plan = orchestrator.create_plan(
        MigrationPlan(
            source_abi_id=legacy_abi.abi_id or "",
            target_abi_id=new_abi.abi_id or "",
            alias_name="active",
        )
    )
    orchestrator.provision_plan(plan.plan_id, actor="tester")
    documents = [
        Document("doc-1", "Alias swaps should remain reversible in holdover."),
        Document("doc-2", "Dual read fallback protects uncertain adapter traffic."),
    ]
    orchestrator.index_documents(legacy_abi.abi_id or "", documents, legacy_embedder)
    orchestrator.enable_dual_write(plan.plan_id, actor="tester")
    orchestrator.backfill(plan.plan_id, documents, new_embedder, actor="tester")
    orchestrator.evaluate_plan(
        plan.plan_id,
        [QueryCase("q1", "reversible alias swaps", {"doc-1": 3.0})],
        legacy_embedder,
        new_embedder,
        top_k=1,
        actor="tester",
    )
    orchestrator.cutover(plan.plan_id, actor="tester")
    rolled_back = orchestrator.rollback(plan.plan_id, actor="tester", reason="bad latency")

    assert rolled_back.state == MigrationState.ROLLED_BACK
    assert orchestrator.vector_backend.alias_target("active") == legacy_abi.abi_id

    adapter = OrthogonalProcrustesAdapter().fit(
        new_embedder.embed([document.text for document in documents]),
        legacy_embedder.embed([document.text for document in documents]),
    )
    decision = orchestrator.confidence_gated_search(
        plan.plan_id,
        "adapter traffic fallback",
        legacy_embedder,
        new_embedder,
        adapter,
        threshold=0.99,
        top_k=2,
    )
    events = orchestrator.registry.list_events(plan.plan_id)

    assert decision.mode == "dual_read"
    assert events[-1].to_state == "ROLLED_BACK"
