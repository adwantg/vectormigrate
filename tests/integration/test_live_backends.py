from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any, cast

import psycopg
import pytest

from vectormigrate.backends.opensearch import OpenSearchAdapter
from vectormigrate.backends.pgvector import partial_index_sql, search_sql
from vectormigrate.backends.qdrant import QdrantAdapter
from vectormigrate.backends.weaviate import WeaviateAdapter
from vectormigrate.models import EmbeddingABI

pytestmark = pytest.mark.live_backend


def _live_enabled() -> bool:
    return os.getenv("VECTORMIGRATE_RUN_LIVE_BACKENDS") == "1"


if not _live_enabled():
    pytest.skip(
        "live backend tests require VECTORMIGRATE_RUN_LIVE_BACKENDS=1",
        allow_module_level=True,
    )


class JsonHttpTransport:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def request(
        self,
        method: str,
        path: str,
        body: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        return _json_request(self.base_url, method, path, body)


def _json_request(
    base_url: str,
    method: str,
    path: str,
    body: Mapping[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    allowed_statuses: set[int] | None = None,
) -> Mapping[str, Any]:
    payload = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=payload,
        method=method,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            content = response.read()
            if not content:
                return {}
            return cast(Mapping[str, Any], json.loads(content.decode("utf-8")))
    except urllib.error.HTTPError as exc:  # pragma: no cover - exercised only when failures occur
        if allowed_statuses is not None and exc.code in allowed_statuses:
            return {}
        detail = exc.read().decode("utf-8", errors="replace")
        raise AssertionError(f"{method} {path} failed with {exc.code}: {detail}") from exc


def _raw_request(
    base_url: str,
    method: str,
    path: str,
    data: bytes,
    content_type: str,
) -> str:
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        method=method,
        headers={"Content-Type": content_type},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return cast(str, response.read().decode("utf-8"))


def _wait_http(url: str, timeout_seconds: int = 180) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status < 500:
                    return
        except urllib.error.URLError:
            time.sleep(1)
            continue
        time.sleep(1)
    raise AssertionError(f"Timed out waiting for {url}")


@pytest.fixture(scope="module")
def opensearch_url() -> str:
    url = os.environ["VECTORMIGRATE_OPENSEARCH_URL"]
    _wait_http(f"{url}/_cluster/health")
    return url


@pytest.fixture(scope="module")
def weaviate_url() -> str:
    url = os.environ["VECTORMIGRATE_WEAVIATE_URL"]
    _wait_http(f"{url}/v1/.well-known/ready")
    return url


@pytest.fixture(scope="module")
def qdrant_url() -> str:
    url = os.environ["VECTORMIGRATE_QDRANT_URL"]
    _wait_http(f"{url}/readyz")
    return url


@pytest.fixture(scope="module")
def pgvector_dsn() -> str:
    return os.environ["VECTORMIGRATE_PGVECTOR_DSN"]


def test_opensearch_adapter_hits_live_cluster(opensearch_url: str) -> None:
    adapter = OpenSearchAdapter(JsonHttpTransport(opensearch_url))
    abi = EmbeddingABI("live-upgrade", "vectormigrate", "v1", dimensions=4)
    source_index = "vectormigrate-live-source"
    target_index = "vectormigrate-live-target"
    alias_name = "vectormigrate-live-alias"

    _json_request(opensearch_url, "DELETE", f"/{source_index}", allowed_statuses={404})
    _json_request(opensearch_url, "DELETE", f"/{target_index}", allowed_statuses={404})

    adapter.create_index(source_index, abi)
    adapter.swap_alias(alias_name, source_index)

    bulk_lines = [
        json.dumps({"index": {"_index": source_index}}),
        json.dumps({"doc_id": "doc-1", "embedding": [1.0, 0.0, 0.0, 0.0]}),
        json.dumps({"index": {"_index": source_index}}),
        json.dumps({"doc_id": "doc-2", "embedding": [0.0, 1.0, 0.0, 0.0]}),
        "",
    ]
    _raw_request(
        opensearch_url,
        "POST",
        "/_bulk?refresh=true",
        "\n".join(bulk_lines).encode("utf-8"),
        "application/x-ndjson",
    )

    adapter.create_index(target_index, abi)
    reindex_payload = adapter.reindex(source_index, target_index)
    assert reindex_payload["total"] >= 2
    _json_request(opensearch_url, "POST", f"/{target_index}/_refresh")
    adapter.swap_alias(alias_name, target_index, source_index=source_index)

    search_payload = adapter.search(alias_name, [1.0, 0.0, 0.0, 0.0], size=2)
    hit_ids = [hit["_source"]["doc_id"] for hit in search_payload["hits"]["hits"]]
    assert hit_ids[0] == "doc-1"


def test_weaviate_collection_and_alias_work_live(weaviate_url: str) -> None:
    adapter = WeaviateAdapter(JsonHttpTransport(weaviate_url))
    abi = EmbeddingABI("live-upgrade", "vectormigrate", "v1", dimensions=4)
    collection_name = "VectorMigrateLiveCollection"
    alias_name = "VectorMigrateLiveAlias"

    schema = _json_request(weaviate_url, "GET", "/v1/schema")
    classes = {entry["class"] for entry in schema.get("classes", [])}
    if collection_name in classes:
        _json_request(weaviate_url, "DELETE", f"/v1/schema/{collection_name}")

    adapter.create_collection(collection_name, abi)
    _json_request(
        weaviate_url,
        "POST",
        "/v1/aliases",
        {"alias": alias_name, "class": collection_name},
        allowed_statuses={422},
    )
    alias_response = adapter.swap_alias(alias_name, collection_name)
    assert alias_response

    _json_request(
        weaviate_url,
        "POST",
        "/v1/objects",
        {
            "class": collection_name,
            "properties": {"doc_id": "doc-1"},
            "vectors": {"default_vector": [1.0, 0.0, 0.0, 0.0]},
        },
    )
    graphql_payload = _json_request(
        weaviate_url,
        "POST",
        "/v1/graphql",
        {"query": "{Get{VectorMigrateLiveAlias(nearVector:{vector:[1,0,0,0]}, limit:1){doc_id}}}"},
    )
    alias_hits = graphql_payload["data"]["Get"][alias_name]
    assert alias_hits[0]["doc_id"] == "doc-1"


def test_qdrant_collection_and_search_work_live(qdrant_url: str) -> None:
    adapter = QdrantAdapter(JsonHttpTransport(qdrant_url))
    abi = EmbeddingABI("live-upgrade", "vectormigrate", "v1", dimensions=4)
    collection_name = "vectormigrate_live_collection"
    vector_name = "default_vector"

    _json_request(qdrant_url, "DELETE", f"/collections/{collection_name}", allowed_statuses={404})
    adapter.create_collection(collection_name, abi, vector_name=vector_name)

    _json_request(
        qdrant_url,
        "PUT",
        f"/collections/{collection_name}/points?wait=true",
        {
            "points": [
                {
                    "id": 1,
                    "vector": {vector_name: [1.0, 0.0, 0.0, 0.0]},
                    "payload": {"doc_id": "doc-1"},
                },
                {
                    "id": 2,
                    "vector": {vector_name: [0.0, 1.0, 0.0, 0.0]},
                    "payload": {"doc_id": "doc-2"},
                },
            ]
        },
    )

    search_payload = adapter.search(collection_name, vector_name, [1.0, 0.0, 0.0, 0.0], limit=2)
    points = search_payload["result"]["points"]
    assert points[0]["payload"]["doc_id"] == "doc-1"


def test_pgvector_helpers_execute_against_live_database(pgvector_dsn: str) -> None:
    abi = EmbeddingABI("live-upgrade", "vectormigrate", "v1", dimensions=4)

    with psycopg.connect(pgvector_dsn) as connection:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        connection.execute("DROP TABLE IF EXISTS chunks")
        connection.execute(
            """
            CREATE TABLE chunks (
                id TEXT PRIMARY KEY,
                embedding_abi_id TEXT NOT NULL,
                embedding vector(4) NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO chunks (id, embedding_abi_id, embedding)
            VALUES
                ('doc-1', %s, '[1,0,0,0]'),
                ('doc-2', %s, '[0,1,0,0]')
            """,
            (abi.abi_id or "", abi.abi_id or ""),
        )

        connection.execute(
            partial_index_sql("chunks", "embedding", "embedding_abi_id", abi.abi_id or "")
        )
        rows = connection.execute(
            search_sql("chunks", "embedding", "embedding_abi_id", abi.abi_id or "", 4),
            ("[1,0,0,0]",),
        ).fetchall()
        explain_rows = connection.execute(
            "EXPLAIN " + search_sql("chunks", "embedding", "embedding_abi_id", abi.abi_id or "", 4),
            ("[1,0,0,0]",),
        ).fetchall()

    assert rows[0][0] == "doc-1"
    explain_text = "\n".join(row[0] for row in explain_rows)
    assert "chunks" in explain_text
