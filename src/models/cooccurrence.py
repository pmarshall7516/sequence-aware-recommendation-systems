"""Unordered item co-occurrence recommender."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations

from src.models.base import BaseRecommender


class CooccurrenceRecommender(BaseRecommender):
    """Recommend items that co-occur in sessions regardless of order."""

    def __init__(self, similarity: str = "cosine", exclude_seen: bool = True) -> None:
        if similarity not in {"raw", "cosine"}:
            raise ValueError("similarity must be one of: raw, cosine")
        self.similarity = similarity
        self.exclude_seen = exclude_seen
        self.co_counts: dict[int, Counter[int]] = defaultdict(Counter)
        self.item_counts: Counter[int] = Counter()

    def fit(self, session_sequences: list[list[int]]) -> None:
        """Build pairwise item co-occurrence statistics from sessions.

        Each session contributes one co-occurrence per unique item pair.
        """

        co_counts: dict[int, Counter[int]] = defaultdict(Counter)
        item_counts: Counter[int] = Counter()

        for seq in session_sequences:
            unique_items = list(dict.fromkeys(seq))
            item_counts.update(unique_items)
            for i, j in combinations(unique_items, 2):
                co_counts[i][j] += 1
                co_counts[j][i] += 1

        self.co_counts = co_counts
        self.item_counts = item_counts

    def _pair_score(self, i: int, j: int) -> float:
        """Compute pair similarity score from stored co-occurrence stats."""

        raw = self.co_counts.get(i, Counter()).get(j, 0)
        if raw == 0:
            return 0.0

        if self.similarity == "raw":
            return float(raw)

        denom = (self.item_counts.get(i, 1) * self.item_counts.get(j, 1)) ** 0.5
        return float(raw / denom) if denom > 0 else 0.0

    def score_candidates(self, context: list[int], candidate_items: list[int]) -> dict[int, float]:
        """Aggregate item similarities from all context items (unordered)."""

        seen = set(context) if self.exclude_seen else set()
        context_unique = list(dict.fromkeys(context))

        scores: dict[int, float] = {}
        for candidate in candidate_items:
            if candidate in seen:
                continue
            total = 0.0
            for ctx_item in context_unique:
                total += self._pair_score(ctx_item, candidate)
            scores[candidate] = total

        return scores
