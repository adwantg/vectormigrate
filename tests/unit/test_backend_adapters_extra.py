from __future__ import annotations

from typing import Any

import pytest

from vectormigrate.backends.pgvector import namespace_name_for_abi, partial_index_sql, search_sql
from vectormigrate.backends.qdrant import QdrantAdapter
from vectormigrate.backends.weaviate import WeaviateAdapter
from vectormigrate.errors import ValidationError
from vectormigrate.models import EmbeddingABI, MigrationPlan


class FakeTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls.append((method, path, body))
        return {"ok": True, "path": path}


def test_weaviate_adapter_builds_collection_and_alias_calls() -> None:
    transport = FakeTransport()
    adapter = WeaviateAdapter(transport)
    abi = EmbeddingABI("model", "provider", "v1", dimensions=16)

    adapter.create_collection("TargetCollection", abi)
    adapter.swap_alias("active", "TargetCollection")

    assert transport.calls[0][0:2] == ("POST", "/v1/schema")
    assert transport.calls[1][0:2] == ("PUT", "/v1/aliases/active")


def test_qdrant_adapter_builds_collection_and_search_calls() -> None:
    transport = FakeTransport()
    adapter = QdrantAdapter(transport)
    abi = EmbeddingABI("model", "provider", "v1", dimensions=16)

    adapter.create_collection("target_collection", abi)
    adapter.search("target_collection", "default_vector", [0.1, 0.2], limit=2)

    assert transport.calls[0][0:2] == ("PUT", "/collections/target_collection")
    assert transport.calls[1][0:2] == ("POST", "/collections/target_collection/points/query")


def test_pgvector_helpers_build_sql() -> None:
    abi = EmbeddingABI("model", "provider", "v1", dimensions=16)
    index_sql = partial_index_sql("chunks", "embedding", "embedding_abi_id", abi.abi_id or "")
    query_sql = search_sql("chunks", "embedding", "embedding_abi_id", abi.abi_id or "", 16)
    namespace = namespace_name_for_abi("retrieval", abi)

    assert "CREATE INDEX" in index_sql
    assert "vector_cosine_ops" in index_sql
    assert "ORDER BY embedding <=> %s::vector(16)" in query_sql
    assert namespace.startswith("retrieval_")


def test_pgvector_namespace_rejects_bad_prefix() -> None:
    abi = EmbeddingABI("model", "provider", "v1", dimensions=16)
    with pytest.raises(ValidationError):
        namespace_name_for_abi(" bad prefix", abi)


def test_weaviate_and_qdrant_compile_plan() -> None:
    plan = MigrationPlan(source_abi_id="old", target_abi_id="new", alias_name="active")
    weaviate_ops = WeaviateAdapter(FakeTransport()).compile_plan(plan, "TargetCollection")
    qdrant_ops = QdrantAdapter(FakeTransport()).compile_plan(plan, "target_collection")

    assert weaviate_ops[0].name == "create_target_collection"
    assert qdrant_ops[0].name == "create_named_vector_collection"
