from __future__ import annotations

from pathlib import Path

from vectormigrate.datasets import PairedVectorDataset
from vectormigrate.io import load_documents, load_query_cases
from vectormigrate.notebook import execute_notebook_smoke, execute_notebook_subprocess_smoke

ROOT = Path(__file__).resolve().parents[2]


def test_example_jsonl_and_csv_assets_load() -> None:
    docs_jsonl = load_documents(ROOT / "examples/sample_documents.jsonl")
    docs_csv = load_documents(ROOT / "examples/sample_documents.csv")
    queries_jsonl = load_query_cases(ROOT / "examples/sample_queries.jsonl")
    queries_csv = load_query_cases(ROOT / "examples/sample_queries.csv")
    pairs = PairedVectorDataset.load(ROOT / "examples/sample_paired_vectors.jsonl")

    assert len(docs_jsonl) == 2
    assert len(docs_csv) == 2
    assert len(queries_jsonl) == 2
    assert len(queries_csv) == 2
    assert pairs.to_matrices()[0].shape == (2, 3)


def test_notebook_smoke_executes_example_cells() -> None:
    globals_dict = execute_notebook_smoke(
        ROOT / "notebooks/vectormigrate_walkthrough.ipynb",
        ROOT,
    )

    assert "docs" in globals_dict
    assert "queries" in globals_dict
    assert len(globals_dict["docs"]) == 2
    assert len(globals_dict["queries"]) == 2


def test_notebook_subprocess_smoke_executes_example_cells() -> None:
    payload = execute_notebook_subprocess_smoke(
        ROOT / "notebooks/vectormigrate_walkthrough.ipynb",
        ROOT,
    )

    assert payload["executed_code_cells"] > 0
    assert "docs" in payload["defined_names"]
    assert "queries" in payload["defined_names"]
