from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from vectormigrate.models import Document, EmbeddingABI, SearchHit


@dataclass
class StoredVector:
    doc_id: str
    text: str
    vector: NDArray[np.float64]
    metadata: dict[str, object]


@dataclass
class NamespaceConfig:
    dimensions: int
    distance_metric: str


class InMemoryVectorBackend:
    """Minimal backend that models namespaces and aliases."""

    def __init__(self) -> None:
        self._configs: dict[str, NamespaceConfig] = {}
        self._namespaces: dict[str, dict[str, StoredVector]] = {}
        self._aliases: dict[str, str] = {}

    def create_namespace(self, namespace: str, abi: EmbeddingABI) -> None:
        if namespace in self._configs:
            existing = self._configs[namespace]
            if (
                existing.dimensions != abi.dimensions
                or existing.distance_metric != abi.distance_metric
            ):
                raise ValueError(f"namespace {namespace} already exists with incompatible config")
            return
        self._configs[namespace] = NamespaceConfig(
            dimensions=abi.dimensions,
            distance_metric=abi.distance_metric,
        )
        self._namespaces[namespace] = {}

    def upsert(
        self,
        namespace: str,
        documents: Iterable[Document],
        embeddings: NDArray[np.float64],
    ) -> None:
        if namespace not in self._configs:
            raise KeyError(f"unknown namespace: {namespace}")
        docs = list(documents)
        if len(docs) != len(embeddings):
            raise ValueError("documents and embeddings must have equal length")
        config = self._configs[namespace]
        for document, vector in zip(docs, embeddings, strict=True):
            if vector.shape != (config.dimensions,):
                raise ValueError(
                    f"embedding for {document.doc_id} has shape {vector.shape}, "
                    f"expected {(config.dimensions,)}"
                )
            self._namespaces[namespace][document.doc_id] = StoredVector(
                doc_id=document.doc_id,
                text=document.text,
                vector=np.asarray(vector, dtype=np.float64),
                metadata=dict(document.metadata),
            )

    def search(
        self,
        namespace_or_alias: str,
        query_vector: NDArray[np.float64],
        top_k: int = 5,
    ) -> list[SearchHit]:
        namespace = self.resolve_namespace(namespace_or_alias)
        config = self._configs[namespace]
        if query_vector.shape != (config.dimensions,):
            raise ValueError(
                f"query vector shape {query_vector.shape} does not match namespace dimensions "
                f"{config.dimensions}"
            )

        hits: list[SearchHit] = []
        for record in self._namespaces[namespace].values():
            score = self._score(query_vector, record.vector, config.distance_metric)
            metadata = {"text": record.text, **record.metadata}
            hits.append(
                SearchHit(
                    doc_id=record.doc_id,
                    score=score,
                    namespace=namespace,
                    metadata=metadata,
                )
            )
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]

    def set_alias(self, alias_name: str, namespace: str) -> None:
        if namespace not in self._configs:
            raise KeyError(f"unknown namespace: {namespace}")
        self._aliases[alias_name] = namespace

    def resolve_namespace(self, namespace_or_alias: str) -> str:
        if namespace_or_alias in self._configs:
            return namespace_or_alias
        if namespace_or_alias in self._aliases:
            return self._aliases[namespace_or_alias]
        raise KeyError(f"unknown namespace or alias: {namespace_or_alias}")

    def alias_target(self, alias_name: str) -> str | None:
        return self._aliases.get(alias_name)

    @staticmethod
    def _score(
        query_vector: NDArray[np.float64],
        document_vector: NDArray[np.float64],
        metric: str,
    ) -> float:
        if metric == "dot":
            return float(np.dot(query_vector, document_vector))
        if metric == "l2":
            return -float(np.linalg.norm(query_vector - document_vector))
        query_norm = np.linalg.norm(query_vector)
        doc_norm = np.linalg.norm(document_vector)
        if query_norm == 0 or doc_norm == 0:
            return 0.0
        return float(np.dot(query_vector, document_vector) / (query_norm * doc_norm))
