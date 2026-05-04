#!/usr/bin/env python3
"""Run the full core experiment suite and write metrics artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging

from src.models.cooccurrence import CooccurrenceRecommender
from src.models.markov import MarkovRecommender
from src.models.neural_sequence import NeuralSequenceConfig, NeuralSequenceRecommender
from src.models.popularity import PopularityRecommender
from src.models.session_knn import SessionKNNRecommender
from src.models.weighted_markov import WeightedMarkovRecommender
from src.utils.config import METRICS_DIR
from src.utils.experiments import flatten_result_for_summary, save_summary_table, train_and_evaluate_model
from src.utils.logging_utils import configure_logging
from src.utils.script_helpers import resolve_eval_split_path, resolve_train_path

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the full run script."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument(
        "--include-neural",
        action="store_true",
        help="Include GRU model run (requires PyTorch).",
    )
    parser.add_argument(
        "--warm-start-only",
        action="store_true",
        help="Evaluate only examples with targets present in train split.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute core model suite and persist per-model + summary outputs."""

    args = parse_args()
    configure_logging()

    train_path = resolve_train_path()
    eval_path = resolve_eval_split_path(args.split)

    experiment_specs = [
        (
            "popularity",
            PopularityRecommender(exclude_seen=True),
            {"exclude_seen": True},
        ),
        (
            "cooccurrence",
            CooccurrenceRecommender(similarity="cosine", exclude_seen=True),
            {"similarity": "cosine", "exclude_seen": True},
        ),
        (
            "markov",
            MarkovRecommender(alpha=0.0, exclude_seen=True),
            {"alpha": 0.0, "exclude_seen": True},
        ),
        (
            "weighted_markov",
            WeightedMarkovRecommender(decay=0.7, alpha=0.0, exclude_seen=True),
            {"decay": 0.7, "alpha": 0.0, "exclude_seen": True},
        ),
        (
            "session_knn_unordered",
            SessionKNNRecommender(
                neighbors=100,
                max_candidate_sessions=10_000,
                recency_decay=None,
                exclude_seen=True,
            ),
            {
                "neighbors": 100,
                "max_candidate_sessions": 10000,
                "recency_decay": None,
                "exclude_seen": True,
            },
        ),
        (
            "session_knn_sequence",
            SessionKNNRecommender(
                neighbors=100,
                max_candidate_sessions=10_000,
                recency_decay=0.7,
                exclude_seen=True,
            ),
            {
                "neighbors": 100,
                "max_candidate_sessions": 10000,
                "recency_decay": 0.7,
                "exclude_seen": True,
            },
        ),
    ]

    if args.include_neural:
        neural_model = NeuralSequenceRecommender(
            config=NeuralSequenceConfig(
                embedding_dim=64,
                hidden_dim=128,
                learning_rate=1e-3,
                batch_size=256,
                epochs=5,
                max_seq_len=20,
                random_seed=42,
            ),
            exclude_seen=True,
        )
        experiment_specs.append(
            (
                "neural_sequence",
                neural_model,
                {
                    "embedding_dim": 64,
                    "hidden_dim": 128,
                    "learning_rate": 1e-3,
                    "batch_size": 256,
                    "epochs": 5,
                    "max_seq_len": 20,
                    "random_seed": 42,
                    "exclude_seen": True,
                },
            )
        )

    summary_rows = []

    for model_name, model, model_config in experiment_specs:
        output_path = METRICS_DIR / f"{model_name}_{args.split}.json"
        LOGGER.info("Running experiment: %s", model_name)
        try:
            result = train_and_evaluate_model(
                model=model,
                model_name=model_name,
                model_config=model_config,
                train_path=train_path,
                eval_path=eval_path,
                output_metrics_path=output_path,
                split_name=args.split,
                warm_start_only=args.warm_start_only,
            )
            summary_rows.append(flatten_result_for_summary(result))
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Experiment failed for %s: %s", model_name, exc)

    if summary_rows:
        summary_path = METRICS_DIR / f"summary_{args.split}.csv"
        save_summary_table(summary_rows, summary_path)
        LOGGER.info("Wrote summary table to %s", summary_path)
    else:
        LOGGER.warning("No successful experiment results to summarize.")


if __name__ == "__main__":
    main()
