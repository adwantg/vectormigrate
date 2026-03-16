from __future__ import annotations

from collections.abc import Sequence

from vectormigrate.compat import CompatibilityAdapter
from vectormigrate.embedder import Embedder
from vectormigrate.evaluation import compare_search_paths
from vectormigrate.fusion import reciprocal_rank_fusion
from vectormigrate.models import (
    Document,
    EmbeddingABI,
    MigrationPlan,
    MigrationState,
    QueryCase,
    SearchHit,
    ShadowMetrics,
)
from vectormigrate.registry import SQLiteRegistry
from vectormigrate.routing import ConfidenceGatedSearchRouter, RoutingDecision
from vectormigrate.vector_store import InMemoryVectorBackend


class MigrationOrchestrator:
    """Coordinates ABIs, plans, vector spaces, evaluation, and cutover."""

    def __init__(self, registry: SQLiteRegistry, vector_backend: InMemoryVectorBackend) -> None:
        self.registry = registry
        self.vector_backend = vector_backend

    def register_abi(self, abi: EmbeddingABI) -> EmbeddingABI:
        return self.registry.register_abi(abi)

    def create_plan(self, plan: MigrationPlan) -> MigrationPlan:
        self.registry.require_abi(plan.source_abi_id)
        self.registry.require_abi(plan.target_abi_id)
        return self.registry.create_plan(plan)

    def provision_plan(self, plan_id: str, actor: str = "system") -> MigrationPlan:
        plan = self.registry.require_plan(plan_id)
        source_abi = self.registry.require_abi(plan.source_abi_id)
        target_abi = self.registry.require_abi(plan.target_abi_id)
        self.vector_backend.create_namespace(source_abi.abi_id or "", source_abi)
        self.vector_backend.create_namespace(target_abi.abi_id or "", target_abi)
        if self.vector_backend.alias_target(plan.alias_name) is None:
            self.vector_backend.set_alias(plan.alias_name, source_abi.abi_id or "")
        return self.registry.transition_plan(
            plan_id,
            MigrationState.PROVISIONED,
            actor=actor,
            reason="provision vector namespaces",
        )

    def index_documents(
        self,
        abi_id: str,
        documents: Sequence[Document],
        embedder: Embedder,
    ) -> None:
        abi = self.registry.require_abi(abi_id)
        if abi.dimensions != embedder.abi.dimensions:
            raise ValueError("embedder dimensions do not match ABI dimensions")
        self.vector_backend.create_namespace(abi_id, abi)
        vectors = embedder.embed([document.text for document in documents])
        self.vector_backend.upsert(abi_id, documents, vectors)

    def enable_dual_write(self, plan_id: str, actor: str = "system") -> MigrationPlan:
        return self.registry.transition_plan(
            plan_id,
            MigrationState.DUAL_WRITE,
            actor=actor,
            reason="enable dual write",
        )

    def backfill(
        self,
        plan_id: str,
        documents: Sequence[Document],
        target_embedder: Embedder,
        actor: str = "system",
    ) -> MigrationPlan:
        plan = self.registry.require_plan(plan_id)
        if plan.state not in {
            MigrationState.PROVISIONED,
            MigrationState.DUAL_WRITE,
            MigrationState.SHADOW_EVAL,
        }:
            raise ValueError(f"cannot backfill from state {plan.state}")
        if plan.state != MigrationState.BACKFILLING:
            self.registry.transition_plan(
                plan_id,
                MigrationState.BACKFILLING,
                actor=actor,
                reason="start backfill",
                details={"document_count": len(documents)},
            )
        self.index_documents(plan.target_abi_id, documents, target_embedder)
        self.registry.append_event(
            plan_id=plan_id,
            event_type="BACKFILL_COMPLETED",
            actor=actor,
            details={"document_count": len(documents)},
        )
        return self.registry.require_plan(plan_id)

    def search_namespace(
        self,
        namespace_or_alias: str,
        query_text: str,
        embedder: Embedder,
        top_k: int = 5,
    ) -> list[SearchHit]:
        query_vector = embedder.embed([query_text])[0]
        return self.vector_backend.search(namespace_or_alias, query_vector, top_k=top_k)

    def dual_read_search(
        self,
        plan_id: str,
        query_text: str,
        source_embedder: Embedder,
        target_embedder: Embedder,
        top_k: int = 5,
    ) -> list[SearchHit]:
        plan = self.registry.require_plan(plan_id)
        source_hits = self.search_namespace(
            plan.source_abi_id,
            query_text,
            source_embedder,
            top_k=top_k,
        )
        target_hits = self.search_namespace(
            plan.target_abi_id,
            query_text,
            target_embedder,
            top_k=top_k,
        )
        return reciprocal_rank_fusion([source_hits, target_hits], top_k=top_k)

    def adapter_search(
        self,
        legacy_abi_id: str,
        query_text: str,
        new_embedder: Embedder,
        adapter: CompatibilityAdapter,
        top_k: int = 5,
    ) -> list[SearchHit]:
        query_vector = new_embedder.embed([query_text])
        adapted_query = adapter.transform(query_vector)[0]
        return self.vector_backend.search(legacy_abi_id, adapted_query, top_k=top_k)

    def evaluate_plan(
        self,
        plan_id: str,
        query_cases: Sequence[QueryCase],
        source_embedder: Embedder,
        target_embedder: Embedder,
        top_k: int = 5,
        ndcg_gate: float = -0.02,
        recall_gate: float = -0.01,
        actor: str = "system",
    ) -> ShadowMetrics:
        plan = self.registry.require_plan(plan_id)
        if plan.state not in {
            MigrationState.PROVISIONED,
            MigrationState.DUAL_WRITE,
            MigrationState.BACKFILLING,
            MigrationState.SHADOW_EVAL,
        }:
            raise ValueError(f"cannot evaluate plan from state {plan.state}")
        if plan.state != MigrationState.SHADOW_EVAL:
            plan = self.registry.transition_plan(
                plan_id,
                MigrationState.SHADOW_EVAL,
                actor=actor,
                reason="run shadow evaluation",
            )
        metrics = compare_search_paths(
            query_cases=query_cases,
            baseline_search=lambda text, k: self.search_namespace(
                plan.source_abi_id,
                text,
                source_embedder,
                k,
            ),
            candidate_search=lambda text, k: self.search_namespace(
                plan.target_abi_id,
                text,
                target_embedder,
                k,
            ),
            top_k=top_k,
            ndcg_gate=ndcg_gate,
            recall_gate=recall_gate,
        )
        latest = self.registry.require_plan(plan_id)
        latest.gate_report = metrics.to_dict()
        latest.updated_at = latest.updated_at
        if metrics.passes:
            latest.transition(MigrationState.READY_TO_CUTOVER)
        self.registry.save_plan(latest)
        self.registry.append_event(
            plan_id=plan_id,
            event_type="EVALUATION_RECORDED",
            actor=actor,
            details={"passes": metrics.passes, "top_k": top_k},
        )
        return metrics

    def cutover(self, plan_id: str, actor: str = "system") -> MigrationPlan:
        plan = self.registry.require_plan(plan_id)
        if plan.state != MigrationState.READY_TO_CUTOVER:
            raise ValueError("plan must be READY_TO_CUTOVER before cutover")
        self.vector_backend.set_alias(plan.alias_name, plan.target_abi_id)
        return self.registry.transition_plan(
            plan_id,
            MigrationState.CUTOVER,
            actor=actor,
            reason="cut over alias to target abi",
            details={"alias_name": plan.alias_name, "target_abi_id": plan.target_abi_id},
        )

    def enter_holdover(self, plan_id: str, actor: str = "system") -> MigrationPlan:
        return self.registry.transition_plan(
            plan_id,
            MigrationState.HOLDOVER,
            actor=actor,
            reason="enter holdover window",
        )

    def rollback(
        self,
        plan_id: str,
        actor: str = "system",
        reason: str = "rollback requested",
    ) -> MigrationPlan:
        plan = self.registry.require_plan(plan_id)
        if plan.state not in {MigrationState.CUTOVER, MigrationState.HOLDOVER}:
            raise ValueError("plan must be CUTOVER or HOLDOVER before rollback")
        self.vector_backend.set_alias(plan.alias_name, plan.source_abi_id)
        return self.registry.transition_plan(
            plan_id,
            MigrationState.ROLLED_BACK,
            actor=actor,
            reason=reason,
            details={"alias_name": plan.alias_name, "restored_abi_id": plan.source_abi_id},
        )

    def decommission(self, plan_id: str, actor: str = "system") -> MigrationPlan:
        return self.registry.transition_plan(
            plan_id,
            MigrationState.DECOMMISSIONED,
            actor=actor,
            reason="decommission migration plan",
        )

    def confidence_gated_search(
        self,
        plan_id: str,
        query_text: str,
        source_embedder: Embedder,
        target_embedder: Embedder,
        adapter: CompatibilityAdapter,
        threshold: float = 0.6,
        top_k: int = 5,
    ) -> RoutingDecision:
        router = ConfidenceGatedSearchRouter(threshold=threshold)
        query_vector = target_embedder.embed([query_text])
        confidence = adapter.confidence(query_vector)
        plan = self.registry.require_plan(plan_id)
        return router.route(
            query_text=query_text,
            adapter_confidence=lambda _query: confidence,
            adapter_search=lambda text, k: self.adapter_search(
                plan.source_abi_id,
                text,
                target_embedder,
                adapter,
                top_k=k,
            ),
            fallback_search=lambda text, k: self.dual_read_search(
                plan_id,
                text,
                source_embedder,
                target_embedder,
                top_k=k,
            ),
            top_k=top_k,
        )
