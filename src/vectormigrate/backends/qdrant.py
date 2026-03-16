from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from vectormigrate.backends.base import BackendCapabilities, BackendOperation
from vectormigrate.models import EmbeddingABI, MigrationPlan
from vectormigrate.validation import validate_vector_field_name


class QdrantTransport(Protocol):
    def request(
        self,
        method: str,
        path: str,
        body: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]: ...


class QdrantAdapter:
    def __init__(self, transport: QdrantTransport) -> None:
        self.transport = transport

    @staticmethod
    def capabilities() -> BackendCapabilities:
        return BackendCapabilities(
            supports_alias_swap=False,
            supports_reindex=False,
            supports_named_vectors=True,
            supports_server_side_rrf=False,
        )

    def create_collection(
        self,
        collection_name: str,
        abi: EmbeddingABI,
        vector_name: str = "default_vector",
    ) -> Mapping[str, Any]:
        validate_vector_field_name(vector_name)
        body = {
            "vectors": {
                vector_name: {
                    "size": abi.dimensions,
                    "distance": self._distance(abi.distance_metric),
                }
            }
        }
        return self.transport.request("PUT", f"/collections/{collection_name}", body=body)

    def search(
        self,
        collection_name: str,
        vector_name: str,
        query_vector: list[float],
        limit: int = 5,
    ) -> Mapping[str, Any]:
        validate_vector_field_name(vector_name)
        body = {
            "query": query_vector,
            "using": vector_name,
            "limit": limit,
            "with_payload": True,
        }
        return self.transport.request(
            "POST",
            f"/collections/{collection_name}/points/query",
            body=body,
        )

    def compile_plan(
        self,
        plan: MigrationPlan,
        collection_name: str,
        vector_name: str = "default_vector",
    ) -> list[BackendOperation]:
        validate_vector_field_name(vector_name)
        return [
            BackendOperation(
                name="create_named_vector_collection",
                method="PUT",
                path=f"/collections/{collection_name}",
                body={
                    "collection_name": collection_name,
                    "vector_name": vector_name,
                    "target_abi_id": plan.target_abi_id,
                },
            )
        ]

    @staticmethod
    def _distance(distance_metric: str) -> str:
        metric_to_distance = {
            "cosine": "Cosine",
            "dot": "Dot",
            "l2": "Euclid",
        }
        return metric_to_distance[distance_metric]
