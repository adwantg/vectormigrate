from __future__ import annotations

import csv
import json
from pathlib import Path

from vectormigrate.models import Document, QueryCase


def load_documents(path: str | Path) -> list[Document]:
    input_path = Path(path)
    if input_path.suffix == ".jsonl":
        return _load_documents_jsonl(input_path)
    if input_path.suffix == ".csv":
        return _load_documents_csv(input_path)
    raise ValueError(f"unsupported document file format: {input_path.suffix}")


def load_query_cases(path: str | Path) -> list[QueryCase]:
    input_path = Path(path)
    if input_path.suffix == ".jsonl":
        return _load_query_cases_jsonl(input_path)
    if input_path.suffix == ".csv":
        return _load_query_cases_csv(input_path)
    raise ValueError(f"unsupported query file format: {input_path.suffix}")


def _load_documents_jsonl(path: Path) -> list[Document]:
    documents: list[Document] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        documents.append(
            Document(
                doc_id=str(payload["doc_id"]),
                text=str(payload["text"]),
                metadata=payload.get("metadata", {}),
            )
        )
    return documents


def _load_documents_csv(path: Path) -> list[Document]:
    documents: list[Document] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metadata = json.loads(row["metadata"]) if row.get("metadata") else {}
            documents.append(
                Document(
                    doc_id=str(row["doc_id"]),
                    text=str(row["text"]),
                    metadata=metadata,
                )
            )
    return documents


def _load_query_cases_jsonl(path: Path) -> list[QueryCase]:
    queries: list[QueryCase] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        relevance = {str(key): float(value) for key, value in payload["relevance"].items()}
        queries.append(
            QueryCase(
                query_id=str(payload["query_id"]),
                text=str(payload["text"]),
                relevance=relevance,
            )
        )
    return queries


def _load_query_cases_csv(path: Path) -> list[QueryCase]:
    queries: list[QueryCase] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            relevance_payload = json.loads(row["relevance"])
            relevance = {str(key): float(value) for key, value in relevance_payload.items()}
            queries.append(
                QueryCase(
                    query_id=str(row["query_id"]),
                    text=str(row["text"]),
                    relevance=relevance,
                )
            )
    return queries
