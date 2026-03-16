from __future__ import annotations

from typing import Any

from vectormigrate.backends.opensearch import OpenSearchAdapter
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
        return {"acknowledged": True, "path": path}


def test_opensearch_adapter_builds_requests() -> None:
    transport = FakeTransport()
    adapter = OpenSearchAdapter(transport)
    abi = EmbeddingABI("upgrade", "vectormigrate", "v2", dimensions=16)

    adapter.create_index("target-index", abi)
    adapter.reindex("source-index", "target-index")
    adapter.swap_alias("active", "target-index", source_index="source-index")
    adapter.search("target-index", [0.1, 0.2], size=2)

    assert transport.calls[0][0:2] == ("PUT", "/target-index")
    assert transport.calls[1][0:2] == ("POST", "/_reindex")
    assert transport.calls[2][0:2] == ("POST", "/_aliases")
    assert transport.calls[3][0:2] == ("POST", "/target-index/_search")


def test_opensearch_adapter_compiles_plan() -> None:
    adapter = OpenSearchAdapter(FakeTransport())
    plan = MigrationPlan(
        source_abi_id="old",
        target_abi_id="new",
        alias_name="active",
    )

    operations = adapter.compile_plan(plan, "source-index", "target-index")

    assert [operation.name for operation in operations] == [
        "create_target_index",
        "reindex_corpus",
        "swap_alias",
    ]
