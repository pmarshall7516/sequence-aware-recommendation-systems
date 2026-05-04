#!/usr/bin/env python3
"""Train and evaluate GRU-based neural sequence recommender."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from src.models.neural_sequence import NeuralSequenceConfig, NeuralSequenceRecommender
from src.utils.config import METRICS_DIR
from src.utils.experiments import train_and_evaluate_model
from src.utils.logging_utils import configure_logging
from src.utils.script_helpers import resolve_eval_split_path, resolve_train_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for neural sequence experiment."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-seq-len", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--include-seen", action="store_true")
    parser.add_argument("--warm-start-only", action="store_true", default=False)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Run neural sequence model training + evaluation."""

    args = parse_args()
    configure_logging()

    train_path = resolve_train_path()
    eval_path = resolve_eval_split_path(args.split)
    output_path = args.output or (METRICS_DIR / f"neural_sequence_{args.split}.json")

    config = NeuralSequenceConfig(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_seq_len=args.max_seq_len,
        random_seed=args.random_seed,
    )

    model = NeuralSequenceRecommender(config=config, exclude_seen=not args.include_seen)

    train_and_evaluate_model(
        model=model,
        model_name="neural_sequence_gru",
        model_config={
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "max_seq_len": args.max_seq_len,
            "random_seed": args.random_seed,
            "exclude_seen": not args.include_seen,
        },
        train_path=train_path,
        eval_path=eval_path,
        output_metrics_path=output_path,
        split_name=args.split,
        warm_start_only=args.warm_start_only,
    )


if __name__ == "__main__":
    main()
