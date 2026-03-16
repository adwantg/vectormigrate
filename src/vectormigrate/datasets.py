from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


class PairedVectorDataset:
    """Small JSONL-backed paired-vector dataset for compatibility training."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(record, sort_keys=True) for record in self.records]
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> PairedVectorDataset:
        input_path = Path(path)
        records: list[dict[str, Any]] = []
        for raw_line in input_path.read_text(encoding="utf-8").splitlines():
            if raw_line.strip():
                records.append(json.loads(raw_line))
        return cls(records)

    def to_matrices(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if not self.records:
            return np.zeros((0, 0), dtype=np.float64), np.zeros((0, 0), dtype=np.float64)
        source = np.asarray([record["source_vector"] for record in self.records], dtype=np.float64)
        target = np.asarray([record["target_vector"] for record in self.records], dtype=np.float64)
        return source, target
