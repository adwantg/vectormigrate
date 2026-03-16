from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from vectormigrate.backends.base import BackendCapabilities, BackendOperation
from vectormigrate.models import EmbeddingABI, MigrationPlan


class OpenSearchTransport(Protocol):
    def request(
        self,
        method: str,
        path: str,
        body: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]: ...


class OpenSearchAdapter:
    """Small, testable OpenSearch adapter around index, reindex, alias, and search operations."""

    def __init__(self, transport: OpenSearchTransport) -> None:
        self.transport = transport

    @staticmethod
    def capabilities() -> BackendCapabilities:
        return BackendCapabilities(
            supports_alias_swap=True,
            supports_reindex=True,
            supports_named_vectors=False,
            supports_server_side_rrf=False,
        )

    def create_index(
        self,
        index_name: str,
        abi: EmbeddingABI,
        vector_field: str = "embedding",
    ) -> Mapping[str, Any]:
        body = {
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    vector_field: {
                        "type": "knn_vector",
                        "dimension": abi.dimensions,
                        "method": {
                            "name": "hnsw",
                            "space_type": self._space_type(abi.distance_metric),
                            "engine": "lucene",
                        },
                    }
                }
            },
        }
        return self.transport.request("PUT", f"/{index_name}", body=body)

    def reindex(
        self,
        source_index: str,
        target_index: str,
    ) -> Mapping[str, Any]:
        body = {"source": {"index": source_index}, "dest": {"index": target_index}}
        return self.transport.request("POST", "/_reindex", body=body)

    def swap_alias(
        self,
        alias_name: str,
        target_index: str,
        source_index: str | None = None,
    ) -> Mapping[str, Any]:
        actions: list[dict[str, Any]] = []
        if source_index is not None:
            actions.append({"remove": {"index": source_index, "alias": alias_name}})
        actions.append({"add": {"index": target_index, "alias": alias_name}})
        return self.transport.request("POST", "/_aliases", body={"actions": actions})

    def search(
        self,
        index_name: str,
        query_vector: list[float],
        vector_field: str = "embedding",
        size: int = 5,
    ) -> Mapping[str, Any]:
        body = {
            "size": size,
            "query": {
                "knn": {
                    vector_field: {
                        "vector": query_vector,
                        "k": size,
                    }
                }
            },
        }
        return self.transport.request("POST", f"/{index_name}/_search", body=body)

    def compile_plan(
        self,
        plan: MigrationPlan,
        source_index: str,
        target_index: str,
        vector_field: str = "embedding",
    ) -> list[BackendOperation]:
        return [
            BackendOperation(
                name="create_target_index",
                method="PUT",
                path=f"/{target_index}",
                body={"vector_field": vector_field, "target_abi_id": plan.target_abi_id},
            ),
            BackendOperation(
                name="reindex_corpus",
                method="POST",
                path="/_reindex",
                body={"source_index": source_index, "target_index": target_index},
            ),
            BackendOperation(
                name="swap_alias",
                method="POST",
                path="/_aliases",
                body={
                    "alias_name": plan.alias_name,
                    "source_index": source_index,
                    "target_index": target_index,
                },
            ),
        ]

    @staticmethod
    def _space_type(distance_metric: str) -> str:
        metric_to_space = {
            "cosine": "cosinesimil",
            "dot": "innerproduct",
            "l2": "l2",
        }
        return metric_to_space[distance_metric]
