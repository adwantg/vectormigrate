from __future__ import annotations

from vectormigrate.models import EmbeddingABI
from vectormigrate.validation import validate_alias_name, validate_vector_field_name


def partial_index_sql(
    table_name: str,
    vector_column: str,
    abi_column: str,
    abi_id: str,
    index_name: str | None = None,
    method: str = "hnsw",
    distance_metric: str = "cosine",
) -> str:
    validate_vector_field_name(vector_column)
    validate_vector_field_name(abi_column)
    index = index_name or f"{table_name}_{vector_column}_{method}_{abs(hash(abi_id)) % 100000}"
    opclass = {
        "cosine": "vector_cosine_ops",
        "dot": "vector_ip_ops",
        "l2": "vector_l2_ops",
    }[distance_metric]
    return (
        f"CREATE INDEX {index} ON {table_name} USING {method} ({vector_column} {opclass}) "
        f"WHERE {abi_column} = '{abi_id}';"
    )


def search_sql(
    table_name: str,
    vector_column: str,
    abi_column: str,
    abi_id: str,
    dimensions: int,
    limit: int = 5,
) -> str:
    validate_vector_field_name(vector_column)
    validate_vector_field_name(abi_column)
    return (
        f"SELECT * FROM {table_name} "
        f"WHERE {abi_column} = '{abi_id}' "
        f"ORDER BY {vector_column} <=> %s::vector({dimensions}) "
        f"LIMIT {limit};"
    )


def namespace_name_for_abi(prefix: str, abi: EmbeddingABI) -> str:
    validate_alias_name(prefix)
    return f"{prefix}_{abi.provider}_{abi.model_id}_{abi.version}".replace("/", "_")
