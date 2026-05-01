"""Global popularity baseline recommender."""

from __future__ import annotations

from collections import Counter

from src.models.base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """Recommend globally popular items regardless of session order.

    This is a non-personalized unordered baseline.
    """

    def __init__(self, exclude_seen: bool = True) -> None:
        self.exclude_seen = exclude_seen
        self.item_counts: Counter[int] = Counter()

    def fit(self, session_sequences: list[list[int]]) -> None:
        """Count item frequencies across all training interactions."""

        counts: Counter[int] = Counter()
        for seq in session_sequences:
            counts.update(seq)
        self.item_counts = counts

    def score_candidates(self, context: list[int], candidate_items: list[int]) -> dict[int, float]:
        """Score each candidate by global interaction count."""

        seen = set(context) if self.exclude_seen else set()
        scores: dict[int, float] = {}
        for item in candidate_items:
            if item in seen:
                continue
            scores[item] = float(self.item_counts.get(item, 0))
        return scores
