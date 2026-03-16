from __future__ import annotations

from pathlib import Path

from vectormigrate.compat import OrthogonalProcrustesAdapter
from vectormigrate.embedder import DeterministicHashEmbedder
from vectormigrate.models import Document, EmbeddingABI, MigrationPlan, QueryCase
from vectormigrate.orchestrator import MigrationOrchestrator
from vectormigrate.registry import SQLiteRegistry
from vectormigrate.vector_store import InMemoryVectorBackend


def run_demo(db_path: str | Path) -> dict[str, object]:
    registry = SQLiteRegistry(db_path)
    backend = InMemoryVectorBackend()
    orchestrator = MigrationOrchestrator(registry, backend)

    legacy_abi = EmbeddingABI(
        model_id="demo-legacy",
        provider="vectormigrate",
        version="2026.03",
        dimensions=32,
        chunker_version="demo-v1",
        embedding_scope="rag_chunk",
    )
    new_abi = EmbeddingABI(
        model_id="demo-upgrade",
        provider="vectormigrate",
        version="2026.04",
        dimensions=32,
        chunker_version="demo-v1",
        embedding_scope="rag_chunk",
    )
    orchestrator.register_abi(legacy_abi)
    orchestrator.register_abi(new_abi)

    plan = orchestrator.create_plan(
        MigrationPlan(
            source_abi_id=legacy_abi.abi_id or "",
            target_abi_id=new_abi.abi_id or "",
            alias_name="knowledge_active",
            strategy="blue_green",
            shadow_percent=25.0,
        )
    )
    orchestrator.provision_plan(plan.plan_id)

    documents = [
        Document(
            doc_id="doc-1",
            text="Embedding model upgrades require versioned vector spaces and clean cutovers.",
            metadata={"topic": "migration"},
        ),
        Document(
            doc_id="doc-2",
            text="OpenSearch aliases support zero downtime swaps during reindexing migrations.",
            metadata={"topic": "opensearch"},
        ),
        Document(
            doc_id="doc-3",
            text=(
                "Weaviate collection aliases keep production traffic stable during "
                "vectorizer migrations."
            ),
            metadata={"topic": "weaviate"},
        ),
        Document(
            doc_id="doc-4",
            text="Qdrant named vectors let one record carry multiple embedding representations.",
            metadata={"topic": "qdrant"},
        ),
        Document(
            doc_id="doc-5",
            text="Shadow evaluation compares recall and nDCG before an embedding cutover.",
            metadata={"topic": "evaluation"},
        ),
    ]

    legacy_embedder = DeterministicHashEmbedder(
        legacy_abi,
        semantic_salt="demo-space",
        rotation_seed=3,
    )
    new_embedder = DeterministicHashEmbedder(
        new_abi,
        semantic_salt="demo-space",
        rotation_seed=17,
    )

    orchestrator.index_documents(legacy_abi.abi_id or "", documents, legacy_embedder)
    before_cutover = orchestrator.search_namespace(
        "knowledge_active",
        "opensearch aliases zero downtime swaps",
        legacy_embedder,
        top_k=3,
    )

    orchestrator.enable_dual_write(plan.plan_id)
    orchestrator.backfill(plan.plan_id, documents, new_embedder)

    query_cases = [
        QueryCase("q1", "opensearch aliases zero downtime swaps", {"doc-2": 3.0}),
        QueryCase("q2", "embedding cutover evaluation", {"doc-1": 2.0, "doc-5": 3.0}),
        QueryCase("q3", "multiple vectors in one record", {"doc-4": 3.0}),
    ]
    shadow_metrics = orchestrator.evaluate_plan(
        plan.plan_id,
        query_cases,
        legacy_embedder,
        new_embedder,
        top_k=3,
    )
    cutover_plan = orchestrator.cutover(plan.plan_id)
    after_cutover = orchestrator.search_namespace(
        "knowledge_active",
        "opensearch aliases zero downtime swaps",
        new_embedder,
        top_k=3,
    )

    paired_texts = [document.text for document in documents]
    adapter = OrthogonalProcrustesAdapter().fit(
        new_embedder.embed(paired_texts),
        legacy_embedder.embed(paired_texts),
    )
    adapter_hits = orchestrator.adapter_search(
        legacy_abi.abi_id or "",
        "embedding migration aliases",
        new_embedder,
        adapter,
        top_k=3,
    )

    return {
        "legacy_abi_id": legacy_abi.abi_id,
        "new_abi_id": new_abi.abi_id,
        "plan_id": cutover_plan.plan_id,
        "plan_state": cutover_plan.state.value,
        "shadow_metrics": shadow_metrics.to_dict(),
        "before_cutover_top_ids": [hit.doc_id for hit in before_cutover],
        "after_cutover_top_ids": [hit.doc_id for hit in after_cutover],
        "adapter_top_ids": [hit.doc_id for hit in adapter_hits],
        "adapter_mean_cosine": adapter.mean_cosine_similarity(
            new_embedder.embed(paired_texts),
            legacy_embedder.embed(paired_texts),
        ),
    }
