"""Shared model evaluation logic for next-item ranking experiments."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.evaluation.metrics import aggregate_metric_bundles, metric_at_k


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration options for ranking evaluation."""

    k_values: tuple[int, ...] = (5, 10, 20)
    split_name: str = "test"


def _prefix_bucket(prefix_length: int) -> str:
    """Map a prefix length to one of the predefined reporting buckets."""

    if prefix_length == 1:
        return "prefix_1"
    if prefix_length == 2:
        return "prefix_2"
    if 3 <= prefix_length <= 5:
        return "prefix_3_5"
    return "prefix_6_plus"


def _session_length_bucket(session_length: int) -> str:
    """Map full session length to short/medium/long reporting buckets."""

    if 2 <= session_length <= 3:
        return "session_short_2_3"
    if 4 <= session_length <= 6:
        return "session_medium_4_6"
    return "session_long_7_plus"


def evaluate_model(
    model: Any,
    examples_df: pd.DataFrame,
    candidate_items: list[int],
    config: EvaluationConfig,
) -> dict[str, Any]:
    """Evaluate a recommender model on prefix-target examples.

    Parameters
    ----------
    model:
        Any object exposing a `recommend(context, candidate_items, k)` method.
    examples_df:
        DataFrame with columns: context (list[int]), target_item (int), prefix_length,
        full_session_length.
    candidate_items:
        Item universe used for ranking.
    config:
        Evaluation parameters (K values and split name).

    Returns
    -------
    dict containing overall metrics, segmented metrics, and runtime details.
    """

    start_time = time.time()

    if examples_df.empty:
        raise ValueError("Cannot evaluate on an empty example set.")

    max_k = max(config.k_values)

    # Store per-k metric bundles for overall and segment-level summaries.
    overall_per_k: dict[int, list[Any]] = {k: [] for k in config.k_values}
    by_prefix_per_k: dict[str, dict[int, list[Any]]] = defaultdict(
        lambda: {k: [] for k in config.k_values}
    )
    by_session_len_per_k: dict[str, dict[int, list[Any]]] = defaultdict(
        lambda: {k: [] for k in config.k_values}
    )

    for row in examples_df.itertuples(index=False):
        context = list(row.context)
        target_item = int(row.target_item)

        recommended = model.recommend(context=context, candidate_items=candidate_items, k=max_k)

        prefix_bucket = _prefix_bucket(int(row.prefix_length))
        session_bucket = _session_length_bucket(int(row.full_session_length))

        for k in config.k_values:
            bundle = metric_at_k(recommended_items=recommended, target_item=target_item, k=k)
            overall_per_k[k].append(bundle)
            by_prefix_per_k[prefix_bucket][k].append(bundle)
            by_session_len_per_k[session_bucket][k].append(bundle)

    # Aggregate overall metrics.
    overall_metrics: dict[str, float] = {}
    for k in config.k_values:
        agg = aggregate_metric_bundles(overall_per_k[k])
        overall_metrics[f"HitRate@{k}"] = agg.hit_rate
        overall_metrics[f"Recall@{k}"] = agg.recall
        overall_metrics[f"MRR@{k}"] = agg.mrr
        overall_metrics[f"NDCG@{k}"] = agg.ndcg

    # Aggregate segment metrics.
    by_prefix_metrics: dict[str, dict[str, float]] = {}
    for bucket, per_k in by_prefix_per_k.items():
        bucket_metrics: dict[str, float] = {}
        for k in config.k_values:
            agg = aggregate_metric_bundles(per_k[k])
            bucket_metrics[f"HitRate@{k}"] = agg.hit_rate
            bucket_metrics[f"Recall@{k}"] = agg.recall
            bucket_metrics[f"MRR@{k}"] = agg.mrr
            bucket_metrics[f"NDCG@{k}"] = agg.ndcg
        by_prefix_metrics[bucket] = bucket_metrics

    by_session_len_metrics: dict[str, dict[str, float]] = {}
    for bucket, per_k in by_session_len_per_k.items():
        bucket_metrics = {}
        for k in config.k_values:
            agg = aggregate_metric_bundles(per_k[k])
            bucket_metrics[f"HitRate@{k}"] = agg.hit_rate
            bucket_metrics[f"Recall@{k}"] = agg.recall
            bucket_metrics[f"MRR@{k}"] = agg.mrr
            bucket_metrics[f"NDCG@{k}"] = agg.ndcg
        by_session_len_metrics[bucket] = bucket_metrics

    runtime_seconds = time.time() - start_time

    return {
        "split": config.split_name,
        "num_examples": int(len(examples_df)),
        "candidate_count": int(len(candidate_items)),
        "overall": overall_metrics,
        "by_prefix_length": by_prefix_metrics,
        "by_session_length": by_session_len_metrics,
        "runtime_seconds": runtime_seconds,
    }
