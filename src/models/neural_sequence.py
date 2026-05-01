"""GRU-based neural sequence recommender for next-item prediction."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from src.models.base import BaseRecommender


@dataclass
class NeuralSequenceConfig:
    """Configuration for the GRU recommender."""

    embedding_dim: int = 64
    hidden_dim: int = 128
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 5
    max_seq_len: int = 20
    random_seed: int = 42


class NeuralSequenceRecommender(BaseRecommender):
    """Simple GRU next-item model with a familiar recommender interface.

    Notes:
    - Uses compact `item_idx` integers as inputs.
    - Internally shifts all item IDs by +1 to reserve token `0` for padding.
    - Trains on all prefix-target pairs generated from training sessions.
    """

    def __init__(self, config: NeuralSequenceConfig | None = None, exclude_seen: bool = True) -> None:
        self.config = config or NeuralSequenceConfig()
        self.exclude_seen = exclude_seen

        # Fitted-state attributes.
        self._torch = None
        self._model = None
        self._device = None
        self._num_items = 0
        self._global_counts: Counter[int] = Counter()

    def _lazy_import_torch(self) -> None:
        """Import torch only when needed so non-neural workflows remain lightweight."""

        if self._torch is not None:
            return

        try:
            import torch  # type: ignore
            import torch.nn as nn  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for NeuralSequenceRecommender. Install torch first."
            ) from exc

        self._torch = torch
        self._nn = nn

    def _build_examples(self, session_sequences: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
        """Construct fixed-length prefix-target training arrays."""

        max_len = self.config.max_seq_len
        features: list[list[int]] = []
        targets: list[int] = []

        for seq in session_sequences:
            for target_pos in range(1, len(seq)):
                prefix = seq[:target_pos]
                target = seq[target_pos]

                # Shift IDs by +1 so 0 can be used as PAD.
                prefix_shifted = [item + 1 for item in prefix[-max_len:]]
                target_shifted = target + 1

                padded = [0] * (max_len - len(prefix_shifted)) + prefix_shifted
                features.append(padded)
                targets.append(target_shifted)

        if not features:
            return np.zeros((0, max_len), dtype=np.int64), np.zeros((0,), dtype=np.int64)

        x = np.asarray(features, dtype=np.int64)
        y = np.asarray(targets, dtype=np.int64)
        return x, y

    def fit(self, session_sequences: list[list[int]]) -> None:
        """Train the GRU next-item model on session prefix-target examples."""

        self._lazy_import_torch()
        torch = self._torch
        nn = self._nn

        # Store fallback popularity for robustness in cold/degenerate cases.
        for seq in session_sequences:
            self._global_counts.update(seq)

        if not session_sequences:
            return

        self._num_items = max((max(seq) for seq in session_sequences if seq), default=-1) + 1
        if self._num_items <= 0:
            return

        x_np, y_np = self._build_examples(session_sequences)
        if len(x_np) == 0:
            return

        class GRUNextItemModel(nn.Module):
            """Minimal GRU architecture for next-item classification."""

            def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int) -> None:
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
                self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
                self.output = nn.Linear(hidden_dim, vocab_size)

            def forward(self, x):  # type: ignore[no-untyped-def]
                emb = self.embedding(x)
                _, h_n = self.gru(emb)
                logits = self.output(h_n[-1])
                return logits

        vocab_size = self._num_items + 1  # +1 for padding index 0

        # Reproducibility and device configuration.
        torch.manual_seed(self.config.random_seed)
        self._device = torch.device("cpu")

        model = GRUNextItemModel(
            vocab_size=vocab_size,
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
        ).to(self._device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        x_tensor = torch.tensor(x_np, dtype=torch.long)
        y_tensor = torch.tensor(y_np, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        model.train()
        for _epoch in range(self.config.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self._device)
                batch_y = batch_y.to(self._device)

                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        self._model = model.eval()

    def _score_with_model(self, context: list[int], candidate_items: list[int]) -> dict[int, float]:
        """Score candidates from neural logits for one context."""

        torch = self._torch
        if self._model is None or self._num_items <= 0:
            return {}

        max_len = self.config.max_seq_len
        context_shifted = [item + 1 for item in context[-max_len:]]
        padded = [0] * (max_len - len(context_shifted)) + context_shifted

        x = torch.tensor([padded], dtype=torch.long, device=self._device)
        with torch.no_grad():
            logits = self._model(x)[0]

        seen = set(context) if self.exclude_seen else set()
        scores: dict[int, float] = {}
        for item in candidate_items:
            if item in seen:
                continue

            # Items outside known vocabulary are assigned a small fallback score.
            shifted = item + 1
            if 0 <= shifted < logits.shape[0]:
                scores[item] = float(logits[shifted].item())
            else:
                scores[item] = float(self._global_counts.get(item, 0))

        return scores

    def score_candidates(self, context: list[int], candidate_items: list[int]) -> dict[int, float]:
        """Score candidate items for one prediction context."""

        if self._model is None:
            # Fallback to popularity if the model is not trainable in current env.
            seen = set(context) if self.exclude_seen else set()
            return {
                item: float(self._global_counts.get(item, 0))
                for item in candidate_items
                if item not in seen
            }

        return self._score_with_model(context=context, candidate_items=candidate_items)
