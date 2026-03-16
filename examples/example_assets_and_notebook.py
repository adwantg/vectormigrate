from __future__ import annotations

import json
from pathlib import Path

from vectormigrate import (
    PairedVectorDataset,
    execute_notebook_smoke,
    load_documents,
    load_query_cases,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    documents = load_documents(root / "examples/sample_documents.jsonl")
    queries = load_query_cases(root / "examples/sample_queries.csv")
    pairs = PairedVectorDataset.load(root / "examples/sample_paired_vectors.jsonl")
    globals_dict = execute_notebook_smoke(root / "notebooks/vectormigrate_walkthrough.ipynb", root)

    print(
        json.dumps(
            {
                "documents": len(documents),
                "queries": len(queries),
                "paired_shape": pairs.to_matrices()[0].shape,
                "notebook_queries": len(globals_dict["queries"]),
            },
            indent=2,
            sort_keys=True,
            default=list,
        )
    )


if __name__ == "__main__":
    main()
