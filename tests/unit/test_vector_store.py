from __future__ import annotations

import numpy as np
import pytest

from vectormigrate.models import Document, EmbeddingABI
from vectormigrate.vector_store import InMemoryVectorBackend


def test_in_memory_vector_backend_search_and_alias() -> None:
    abi = EmbeddingABI(
        model_id="legacy",
        provider="vectormigrate",
        version="v1",
        dimensions=3,
    )
    backend = InMemoryVectorBackend()
    backend.create_namespace(abi.abi_id or "", abi)
    backend.upsert(
        abi.abi_id or "",
        [
            Document("a", "first"),
            Document("b", "second"),
        ],
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    backend.set_alias("active", abi.abi_id or "")

    hits = backend.search("active", np.array([0.9, 0.1, 0.0]), top_k=2)

    assert [hit.doc_id for hit in hits] == ["a", "b"]


def test_in_memory_vector_backend_rejects_wrong_dimensions() -> None:
    abi = EmbeddingABI(
        model_id="legacy",
        provider="vectormigrate",
        version="v1",
        dimensions=3,
    )
    backend = InMemoryVectorBackend()
    backend.create_namespace(abi.abi_id or "", abi)

    with pytest.raises(ValueError):
        backend.upsert(
            abi.abi_id or "",
            [Document("a", "first")],
            np.array([[1.0, 0.0]]),
        )
