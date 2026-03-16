from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from statistics import mean

from vectormigrate.models import QueryCase, SearchHit, ShadowMetrics

SearchFn = Callable[[str, int], Sequence[SearchHit]]


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], top_k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(1 for doc_id in ranked_ids[:top_k] if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def ndcg_at_k(ranked_ids: Sequence[str], relevance: dict[str, float], top_k: int) -> float:
    if not relevance:
        return 0.0
    gains = []
    for doc_id in ranked_ids[:top_k]:
        rel = relevance.get(doc_id, 0.0)
        gains.append(2**rel - 1.0)
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))

    ideal_gains = sorted((2**score - 1.0 for score in relevance.values()), reverse=True)[:top_k]
    ideal_dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(ideal_gains))
    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


def compare_search_paths(
    query_cases: Sequence[QueryCase],
    baseline_search: SearchFn,
    candidate_search: SearchFn,
    top_k: int = 5,
    ndcg_gate: float = -0.02,
    recall_gate: float = -0.01,
) -> ShadowMetrics:
    per_query: list[dict[str, object]] = []
    baseline_ndcgs: list[float] = []
    candidate_ndcgs: list[float] = []
    baseline_recalls: list[float] = []
    candidate_recalls: list[float] = []

    for case in query_cases:
        baseline_hits = baseline_search(case.text, top_k)
        candidate_hits = candidate_search(case.text, top_k)
        baseline_ids = [hit.doc_id for hit in baseline_hits]
        candidate_ids = [hit.doc_id for hit in candidate_hits]
        relevance = {doc_id: float(score) for doc_id, score in case.relevance.items()}
        relevant_ids = set(relevance)

        baseline_ndcg = ndcg_at_k(baseline_ids, relevance, top_k)
        candidate_ndcg = ndcg_at_k(candidate_ids, relevance, top_k)
        baseline_recall = recall_at_k(baseline_ids, relevant_ids, top_k)
        candidate_recall = recall_at_k(candidate_ids, relevant_ids, top_k)

        baseline_ndcgs.append(baseline_ndcg)
        candidate_ndcgs.append(candidate_ndcg)
        baseline_recalls.append(baseline_recall)
        candidate_recalls.append(candidate_recall)
        per_query.append(
            {
                "query_id": case.query_id,
                "baseline_top_ids": baseline_ids,
                "candidate_top_ids": candidate_ids,
                "baseline_ndcg_at_k": baseline_ndcg,
                "candidate_ndcg_at_k": candidate_ndcg,
                "baseline_recall_at_k": baseline_recall,
                "candidate_recall_at_k": candidate_recall,
            }
        )

    baseline_ndcg_avg = mean(baseline_ndcgs) if baseline_ndcgs else 0.0
    candidate_ndcg_avg = mean(candidate_ndcgs) if candidate_ndcgs else 0.0
    baseline_recall_avg = mean(baseline_recalls) if baseline_recalls else 0.0
    candidate_recall_avg = mean(candidate_recalls) if candidate_recalls else 0.0

    delta_ndcg = candidate_ndcg_avg - baseline_ndcg_avg
    delta_recall = candidate_recall_avg - baseline_recall_avg
    return ShadowMetrics(
        baseline_ndcg_at_k=baseline_ndcg_avg,
        candidate_ndcg_at_k=candidate_ndcg_avg,
        delta_ndcg_at_k=delta_ndcg,
        baseline_recall_at_k=baseline_recall_avg,
        candidate_recall_at_k=candidate_recall_avg,
        delta_recall_at_k=delta_recall,
        passes=delta_ndcg >= ndcg_gate and delta_recall >= recall_gate,
        top_k=top_k,
        per_query=tuple(per_query),
    )
