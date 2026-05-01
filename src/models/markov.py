"""First-order Markov next-item recommender."""

from __future__ import annotations

from collections import Counter, defaultdict

from src.models.base import BaseRecommender


class MarkovRecommender(BaseRecommender):
    """Model next-item probabilities from adjacent item transitions."""

    def __init__(self, alpha: float = 0.0, exclude_seen: bool = True) -> None:
        self.alpha = alpha
        self.exclude_seen = exclude_seen
        self.transitions: dict[int, Counter[int]] = defaultdict(Counter)
        self.global_counts: Counter[int] = Counter()

    def fit(self, session_sequences: list[list[int]]) -> None:
        """Estimate transition counts i_t -> i_(t+1)."""

        transitions: dict[int, Counter[int]] = defaultdict(Counter)
        global_counts: Counter[int] = Counter()

        for seq in session_sequences:
            global_counts.update(seq)
            for prev_item, next_item in zip(seq[:-1], seq[1:]):
                transitions[prev_item][next_item] += 1

        self.transitions = transitions
        self.global_counts = global_counts

    def score_candidates(self, context: list[int], candidate_items: list[int]) -> dict[int, float]:
        """Score candidates from transitions out of the last context item.

        Falls back to popularity when last-item transition data is missing.
        """

        seen = set(context) if self.exclude_seen else set()
        last_item = context[-1] if context else None

        scores: dict[int, float] = {}
        if last_item is None or last_item not in self.transitions:
            for item in candidate_items:
                if item in seen:
                    continue
                scores[item] = float(self.global_counts.get(item, 0))
            return scores

        next_counts = self.transitions[last_item]
        total = sum(next_counts.values())
        vocab_size = max(len(candidate_items), 1)

        for candidate in candidate_items:
            if candidate in seen:
                continue
            count = next_counts.get(candidate, 0)
            if self.alpha > 0:
                prob = (count + self.alpha) / (total + self.alpha * vocab_size)
                scores[candidate] = float(prob)
            else:
                scores[candidate] = float(count)

        return scores
