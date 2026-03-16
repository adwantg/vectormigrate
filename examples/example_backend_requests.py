from __future__ import annotations

import json

from vectormigrate import EmbeddingABI, MigrationPlan
from vectormigrate.backends import OpenSearchAdapter, QdrantAdapter, WeaviateAdapter


class FakeTransport:
    def request(self, method, path, body=None):
        return {"method": method, "path": path, "body": body}


def main() -> None:
    abi = EmbeddingABI("demo-model", "vectormigrate", "v1", dimensions=16)
    plan = MigrationPlan(source_abi_id="old", target_abi_id="new", alias_name="retrieval_active")

    opensearch = OpenSearchAdapter(FakeTransport()).create_index("target-index", abi)
    weaviate_ops = WeaviateAdapter(FakeTransport()).compile_plan(plan, "TargetCollection")
    qdrant_ops = QdrantAdapter(FakeTransport()).compile_plan(plan, "target_collection")

    print(
        json.dumps(
            {
                "opensearch_path": opensearch["path"],
                "weaviate_ops": [operation.name for operation in weaviate_ops],
                "qdrant_ops": [operation.name for operation in qdrant_ops],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
