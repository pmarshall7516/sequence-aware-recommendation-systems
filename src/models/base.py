"""Base interface for recommender models used in this project."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseRecommender(ABC):
    """Abstract base recommender interface.

    Every model implementation is expected to:
    - `fit` on training sequences.
    - `score_candidates` for a given context and candidate set.
    - `recommend` top-k items using candidate scores.
    """

    @abstractmethod
    def fit(self, session_sequences: list[list[int]]) -> None:
        """Fit model state from training session sequences."""

    @abstractmethod
    def score_candidates(self, context: list[int], candidate_items: list[int]) -> dict[int, float]:
        """Return model scores for candidate items given the context."""

    def recommend(self, context: list[int], candidate_items: list[int], k: int) -> list[int]:
        """Return top-k candidate items ranked by descending score.

        This default implementation relies on `score_candidates`. Models can override
        if they need custom behavior.
        """

        scores = self.score_candidates(context=context, candidate_items=candidate_items)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [item for item, _ in ranked[:k]]
