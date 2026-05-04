"""Recency-weighted Markov recommender.

This model combines transition distributions from multiple context items while
assigning larger weight to more recent interactions.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from src.models.base import BaseRecommender


class WeightedMarkovRecommender(BaseRecommender):
    """Markov-style model with recency-weighted multi-step context."""

    def __init__(self, decay: float = 0.7, alpha: float = 0.0, exclude_seen: bool = True) -> None:
        if not 0 < decay <= 1:
            raise ValueError("decay must be in (0, 1]")
        self.decay = decay
        self.alpha = alpha
        self.exclude_seen = exclude_seen
        self.transitions: dict[int, Counter[int]] = defaultdict(Counter)
        self.global_counts: Counter[int] = Counter()

    def fit(self, session_sequences: list[list[int]]) -> None:
        """Collect first-order transitions for each observed item."""

        transitions: dict[int, Counter[int]] = defaultdict(Counter)
        global_counts: Counter[int] = Counter()

        for seq in session_sequences:
            global_counts.update(seq)
            for prev_item, next_item in zip(seq[:-1], seq[1:]):
                transitions[prev_item][next_item] += 1

        self.transitions = transitions
        self.global_counts = global_counts

    def score_candidates(self, context: list[int], candidate_items: list[int]) -> dict[int, float]:
        """Score candidates using a recency-decayed sum of transition evidence."""

        seen = set(context) if self.exclude_seen else set()
        scores: dict[int, float] = {item: 0.0 for item in candidate_items if item not in seen}

        if not context:
            for item in scores:
                scores[item] = float(self.global_counts.get(item, 0))
            return scores

        reversed_context = list(reversed(context))
        for distance, ctx_item in enumerate(reversed_context):
            weight = self.decay ** distance
            next_counts = self.transitions.get(ctx_item)
            if not next_counts:
                continue

            total = sum(next_counts.values())
            vocab_size = max(len(scores), 1)

            for candidate in scores:
                count = next_counts.get(candidate, 0)
                if self.alpha > 0:
                    score = (count + self.alpha) / (total + self.alpha * vocab_size)
                else:
                    score = float(count)
                scores[candidate] += weight * score

        # If no transitions contributed anything, use global popularity fallback.
        if all(value == 0.0 for value in scores.values()):
            for item in scores:
                scores[item] = float(self.global_counts.get(item, 0))

        return scores
