from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from vectormigrate.models import SearchHit


def reciprocal_rank_fusion(
    result_sets: Sequence[Sequence[SearchHit]],
    top_k: int = 5,
    k: int = 60,
) -> list[SearchHit]:
    fused_scores: dict[str, float] = defaultdict(float)
    exemplars: dict[str, SearchHit] = {}

    for result_set in result_sets:
        for rank, hit in enumerate(result_set, start=1):
            fused_scores[hit.doc_id] += 1.0 / (k + rank)
            exemplars.setdefault(hit.doc_id, hit)

    fused_hits = [
        SearchHit(
            doc_id=doc_id,
            score=score,
            namespace=exemplars[doc_id].namespace,
            metadata=exemplars[doc_id].metadata,
        )
        for doc_id, score in fused_scores.items()
    ]
    fused_hits.sort(key=lambda item: item.score, reverse=True)
    return fused_hits[:top_k]
