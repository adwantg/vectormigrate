from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Protocol

from vectormigrate.evaluation import ndcg_at_k, recall_at_k


@dataclass(frozen=True)
class TelemetryEvent:
    event_type: str
    payload: Mapping[str, Any]


class TelemetrySink(Protocol):
    def emit(self, event_type: str, payload: Mapping[str, Any]) -> None: ...


class InMemoryTelemetrySink:
    def __init__(self) -> None:
        self.events: list[TelemetryEvent] = []

    def emit(self, event_type: str, payload: Mapping[str, Any]) -> None:
        self.events.append(TelemetryEvent(event_type=event_type, payload=dict(payload)))


class JsonlTelemetrySink:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event_type: str, payload: Mapping[str, Any]) -> None:
        record = {"event_type": event_type, "payload": dict(payload)}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


class OpenTelemetryLikeCollector(Protocol):
    def record_event(self, name: str, attributes: Mapping[str, Any]) -> None: ...


class OpenTelemetryBridgeSink:
    def __init__(self, collector: OpenTelemetryLikeCollector) -> None:
        self.collector = collector

    def emit(self, event_type: str, payload: Mapping[str, Any]) -> None:
        self.collector.record_event(event_type, dict(payload))


@dataclass
class OnlineShadowEvaluator:
    sink: TelemetrySink
    top_k: int = 5
    _records: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        query_id: str,
        baseline_ids: Sequence[str],
        candidate_ids: Sequence[str],
        relevance: Mapping[str, float],
    ) -> None:
        relevance_dict = {str(key): float(value) for key, value in relevance.items()}
        record = {
            "query_id": query_id,
            "baseline_ids": list(baseline_ids[: self.top_k]),
            "candidate_ids": list(candidate_ids[: self.top_k]),
            "relevance": relevance_dict,
        }
        self._records.append(record)
        self.sink.emit("shadow_query_recorded", record)

    def summary(self) -> dict[str, Any]:
        if not self._records:
            return {
                "query_count": 0,
                "avg_baseline_ndcg_at_k": 0.0,
                "avg_candidate_ndcg_at_k": 0.0,
                "avg_baseline_recall_at_k": 0.0,
                "avg_candidate_recall_at_k": 0.0,
            }
        baseline_ndcgs: list[float] = []
        candidate_ndcgs: list[float] = []
        baseline_recalls: list[float] = []
        candidate_recalls: list[float] = []
        for record in self._records:
            relevance = dict(record["relevance"])
            relevant_ids = set(relevance)
            baseline_ids = list(record["baseline_ids"])
            candidate_ids = list(record["candidate_ids"])
            baseline_ndcgs.append(ndcg_at_k(baseline_ids, relevance, self.top_k))
            candidate_ndcgs.append(ndcg_at_k(candidate_ids, relevance, self.top_k))
            baseline_recalls.append(recall_at_k(baseline_ids, relevant_ids, self.top_k))
            candidate_recalls.append(recall_at_k(candidate_ids, relevant_ids, self.top_k))
        summary = {
            "query_count": len(self._records),
            "avg_baseline_ndcg_at_k": mean(baseline_ndcgs),
            "avg_candidate_ndcg_at_k": mean(candidate_ndcgs),
            "avg_baseline_recall_at_k": mean(baseline_recalls),
            "avg_candidate_recall_at_k": mean(candidate_recalls),
        }
        self.sink.emit("shadow_summary_computed", summary)
        return summary
