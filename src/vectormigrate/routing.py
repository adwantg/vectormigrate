from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from vectormigrate.models import SearchHit


@dataclass(frozen=True)
class RoutingDecision:
    mode: str
    confidence: float
    hits: tuple[SearchHit, ...]
    reason: str


class ConfidenceGatedSearchRouter:
    def __init__(self, threshold: float = 0.6) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = threshold

    def route(
        self,
        query_text: str,
        adapter_confidence: Callable[[str], float],
        adapter_search: Callable[[str, int], Sequence[SearchHit]],
        fallback_search: Callable[[str, int], Sequence[SearchHit]],
        top_k: int = 5,
    ) -> RoutingDecision:
        confidence = float(adapter_confidence(query_text))
        if confidence >= self.threshold:
            hits = tuple(adapter_search(query_text, top_k))
            return RoutingDecision(
                mode="adapter",
                confidence=confidence,
                hits=hits,
                reason="adapter confidence meets threshold",
            )
        hits = tuple(fallback_search(query_text, top_k))
        return RoutingDecision(
            mode="dual_read",
            confidence=confidence,
            hits=hits,
            reason="adapter confidence below threshold",
        )
