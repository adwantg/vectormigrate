from __future__ import annotations

from vectormigrate.evaluation import compare_search_paths, ndcg_at_k, recall_at_k
from vectormigrate.fusion import reciprocal_rank_fusion
from vectormigrate.models import QueryCase, SearchHit


def test_reciprocal_rank_fusion_deduplicates_and_merges() -> None:
    fused = reciprocal_rank_fusion(
        [
            [
                SearchHit("doc-1", 0.9, "legacy"),
                SearchHit("doc-2", 0.8, "legacy"),
            ],
            [
                SearchHit("doc-2", 0.95, "new"),
                SearchHit("doc-3", 0.7, "new"),
            ],
        ],
        top_k=3,
    )

    assert [hit.doc_id for hit in fused] == ["doc-2", "doc-1", "doc-3"]


def test_ranking_metrics_handle_partial_relevance() -> None:
    ranked = ["doc-1", "doc-2", "doc-3"]
    relevance = {"doc-1": 3.0, "doc-3": 1.0}

    assert recall_at_k(ranked, set(relevance), top_k=2) == 0.5
    assert ndcg_at_k(ranked, relevance, top_k=3) > 0.9


def test_compare_search_paths_flags_bad_candidate() -> None:
    queries = [QueryCase("q1", "query", {"good-doc": 3.0})]

    metrics = compare_search_paths(
        query_cases=queries,
        baseline_search=lambda _text, _k: [SearchHit("good-doc", 1.0, "legacy")],
        candidate_search=lambda _text, _k: [SearchHit("bad-doc", 1.0, "new")],
        top_k=1,
        ndcg_gate=-0.02,
        recall_gate=-0.01,
    )

    assert metrics.passes is False
    assert metrics.delta_ndcg_at_k == -1.0
    assert metrics.delta_recall_at_k == -1.0
