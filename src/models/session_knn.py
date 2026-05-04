"""Session-based k-nearest-neighbor recommenders."""

from __future__ import annotations

from collections import Counter, defaultdict

from src.models.base import BaseRecommender


class SessionKNNRecommender(BaseRecommender):
    """Session-kNN recommender with optional recency-aware scoring.

    The model compares the current prefix with historical sessions and aggregates item
    evidence from similar sessions.
    """

    def __init__(
        self,
        neighbors: int = 100,
        max_candidate_sessions: int = 10_000,
        recency_decay: float | None = None,
        exclude_seen: bool = True,
    ) -> None:
        self.neighbors = neighbors
        self.max_candidate_sessions = max_candidate_sessions
        self.recency_decay = recency_decay
        self.exclude_seen = exclude_seen

        self.sessions: list[list[int]] = []
        self.item_to_sessions: dict[int, list[int]] = defaultdict(list)

    def fit(self, session_sequences: list[list[int]]) -> None:
        """Store training sessions and build inverted index item -> session IDs."""

        self.sessions = [list(seq) for seq in session_sequences]
        item_to_sessions: dict[int, list[int]] = defaultdict(list)

        for sid, seq in enumerate(self.sessions):
            unique_items = set(seq)
            for item in unique_items:
                item_to_sessions[item].append(sid)

        self.item_to_sessions = item_to_sessions

    def _candidate_neighbor_session_ids(self, context_items: list[int]) -> list[int]:
        """Collect historical session IDs sharing at least one context item."""

        candidates: set[int] = set()
        for item in set(context_items):
            for sid in self.item_to_sessions.get(item, []):
                candidates.add(sid)
                if len(candidates) >= self.max_candidate_sessions:
                    break
            if len(candidates) >= self.max_candidate_sessions:
                break
        return list(candidates)

    def _session_similarity(self, context_items: list[int], neighbor_items: list[int]) -> float:
        """Compute similarity between current context and a neighbor session.

        Uses Jaccard-like overlap by default and optionally applies recency weighting
        on context items.
        """

        ctx_set = set(context_items)
        neighbor_set = set(neighbor_items)

        if not ctx_set or not neighbor_set:
            return 0.0

        if self.recency_decay is None:
            intersection = len(ctx_set.intersection(neighbor_set))
            union = len(ctx_set.union(neighbor_set))
            return float(intersection / union) if union > 0 else 0.0

        # Recency-weighted overlap: recent context items get larger weights.
        weighted_overlap = 0.0
        total_weight = 0.0
        reversed_context = list(reversed(context_items))
        for distance, item in enumerate(reversed_context):
            weight = self.recency_decay ** distance
            total_weight += weight
            if item in neighbor_set:
                weighted_overlap += weight

        if total_weight == 0:
            return 0.0

        return float(weighted_overlap / total_weight)

    def score_candidates(self, context: list[int], candidate_items: list[int]) -> dict[int, float]:
        """Score candidates using weighted evidence from nearest sessions."""

        seen = set(context) if self.exclude_seen else set()
        neighbor_ids = self._candidate_neighbor_session_ids(context)

        scored_neighbors: list[tuple[int, float]] = []
        for sid in neighbor_ids:
            sim = self._session_similarity(context_items=context, neighbor_items=self.sessions[sid])
            if sim > 0:
                scored_neighbors.append((sid, sim))

        scored_neighbors.sort(key=lambda x: x[1], reverse=True)
        top_neighbors = scored_neighbors[: self.neighbors]

        # Aggregate candidate scores from neighbor sessions.
        scores = Counter()  # type: ignore[var-annotated]
        candidate_set = set(candidate_items)
        for sid, sim in top_neighbors:
            seq = self.sessions[sid]

            # Weight item contribution by both neighbor similarity and item frequency in session.
            item_counts = Counter(seq)
            for item, count in item_counts.items():
                if item in seen:
                    continue
                if item not in candidate_set:
                    continue
                scores[item] += sim * count

        return {item: float(scores.get(item, 0.0)) for item in candidate_items if item not in seen}
