from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vectormigrate.datasets import PairedVectorDataset
from vectormigrate.io import load_documents, load_query_cases


def test_load_documents_from_jsonl_and_csv(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "docs.jsonl"
    jsonl_path.write_text(
        json.dumps({"doc_id": "a", "text": "first", "metadata": {"topic": "x"}}) + "\n",
        encoding="utf-8",
    )
    csv_path = tmp_path / "docs.csv"
    csv_path.write_text(
        'doc_id,text,metadata\nb,second,"{""topic"": ""y""}"\n',
        encoding="utf-8",
    )

    docs_jsonl = load_documents(jsonl_path)
    docs_csv = load_documents(csv_path)

    assert docs_jsonl[0].metadata["topic"] == "x"
    assert docs_csv[0].doc_id == "b"


def test_load_query_cases_from_jsonl_and_csv(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "queries.jsonl"
    jsonl_path.write_text(
        json.dumps({"query_id": "q1", "text": "question", "relevance": {"doc-1": 3}}) + "\n",
        encoding="utf-8",
    )
    csv_path = tmp_path / "queries.csv"
    csv_path.write_text(
        'query_id,text,relevance\nq2,another,"{""doc-2"": 2}"\n',
        encoding="utf-8",
    )

    queries_jsonl = load_query_cases(jsonl_path)
    queries_csv = load_query_cases(csv_path)

    assert queries_jsonl[0].relevance["doc-1"] == 3.0
    assert queries_csv[0].query_id == "q2"


def test_paired_vector_dataset_round_trip(tmp_path: Path) -> None:
    dataset = PairedVectorDataset(
        [
            {"record_id": "1", "source_vector": [1.0, 2.0], "target_vector": [3.0, 4.0]},
            {"record_id": "2", "source_vector": [5.0, 6.0], "target_vector": [7.0, 8.0]},
        ]
    )

    path = dataset.save(tmp_path / "pairs.jsonl")
    loaded = PairedVectorDataset.load(path)
    source, target = loaded.to_matrices()

    assert np.array_equal(source, np.array([[1.0, 2.0], [5.0, 6.0]]))
    assert np.array_equal(target, np.array([[3.0, 4.0], [7.0, 8.0]]))
