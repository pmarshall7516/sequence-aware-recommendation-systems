"""Ranking metric implementations for next-item prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MetricBundle:
    """Container for standard ranking metrics at one cutoff K."""

    hit_rate: float
    recall: float
    mrr: float
    ndcg: float


def rank_of_target(recommended_items: list[int], target_item: int) -> int | None:
    """Return 1-based rank position of target in recommendation list, or None."""

    for idx, item in enumerate(recommended_items, start=1):
        if item == target_item:
            return idx
    return None


def metric_at_k(recommended_items: list[int], target_item: int, k: int) -> MetricBundle:
    """Compute HitRate, Recall, MRR, and NDCG at cutoff K for one example."""

    top_k = recommended_items[:k]
    target_rank = rank_of_target(top_k, target_item)

    if target_rank is None:
        return MetricBundle(hit_rate=0.0, recall=0.0, mrr=0.0, ndcg=0.0)

    # For single-target next-item prediction, hit rate and recall coincide.
    hit = 1.0
    recall = 1.0
    mrr = 1.0 / target_rank
    ndcg = 1.0 / math.log2(target_rank + 1)
    return MetricBundle(hit_rate=hit, recall=recall, mrr=mrr, ndcg=ndcg)


def aggregate_metric_bundles(bundles: list[MetricBundle]) -> MetricBundle:
    """Average metric bundles over multiple prediction examples."""

    if not bundles:
        return MetricBundle(hit_rate=0.0, recall=0.0, mrr=0.0, ndcg=0.0)

    n = float(len(bundles))
    return MetricBundle(
        hit_rate=sum(b.hit_rate for b in bundles) / n,
        recall=sum(b.recall for b in bundles) / n,
        mrr=sum(b.mrr for b in bundles) / n,
        ndcg=sum(b.ndcg for b in bundles) / n,
    )
