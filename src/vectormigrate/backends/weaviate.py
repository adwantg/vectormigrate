from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from vectormigrate.backends.base import BackendCapabilities, BackendOperation
from vectormigrate.models import EmbeddingABI, MigrationPlan
from vectormigrate.validation import validate_vector_field_name


class WeaviateTransport(Protocol):
    def request(
        self,
        method: str,
        path: str,
        body: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]: ...


class WeaviateAdapter:
    def __init__(self, transport: WeaviateTransport) -> None:
        self.transport = transport

    @staticmethod
    def capabilities() -> BackendCapabilities:
        return BackendCapabilities(
            supports_alias_swap=True,
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
            "class": collection_name,
            "vectorConfig": {
                vector_name: {
                    "vectorizer": {"none": {}},
                    "vectorIndexType": "hnsw",
                    "vectorIndexConfig": {"distance": abi.distance_metric},
                }
            },
        }
        return self.transport.request("POST", "/v1/schema", body=body)

    def swap_alias(self, alias_name: str, target_collection: str) -> Mapping[str, Any]:
        body = {"class": target_collection}
        return self.transport.request("PUT", f"/v1/aliases/{alias_name}", body=body)

    def compile_plan(
        self,
        plan: MigrationPlan,
        target_collection: str,
        vector_name: str = "default_vector",
    ) -> list[BackendOperation]:
        validate_vector_field_name(vector_name)
        return [
            BackendOperation(
                name="create_target_collection",
                method="POST",
                path="/v1/schema",
                body={
                    "target_collection": target_collection,
                    "vector_name": vector_name,
                    "target_abi_id": plan.target_abi_id,
                },
            ),
            BackendOperation(
                name="swap_collection_alias",
                method="PUT",
                path=f"/v1/aliases/{plan.alias_name}",
                body={"alias_name": plan.alias_name, "target_collection": target_collection},
            ),
        ]
