from __future__ import annotations

from vectormigrate.models import SearchHit
from vectormigrate.routing import ConfidenceGatedSearchRouter


def test_confidence_router_uses_adapter_when_confident() -> None:
    router = ConfidenceGatedSearchRouter(threshold=0.6)

    decision = router.route(
        query_text="query",
        adapter_confidence=lambda _query: 0.9,
        adapter_search=lambda _query, _top_k: [SearchHit("adapter-doc", 1.0, "legacy")],
        fallback_search=lambda _query, _top_k: [SearchHit("fallback-doc", 0.8, "dual")],
    )

    assert decision.mode == "adapter"
    assert decision.hits[0].doc_id == "adapter-doc"


def test_confidence_router_falls_back_when_not_confident() -> None:
    router = ConfidenceGatedSearchRouter(threshold=0.6)

    decision = router.route(
        query_text="query",
        adapter_confidence=lambda _query: 0.2,
        adapter_search=lambda _query, _top_k: [SearchHit("adapter-doc", 1.0, "legacy")],
        fallback_search=lambda _query, _top_k: [SearchHit("fallback-doc", 0.8, "dual")],
    )

    assert decision.mode == "dual_read"
    assert decision.hits[0].doc_id == "fallback-doc"
