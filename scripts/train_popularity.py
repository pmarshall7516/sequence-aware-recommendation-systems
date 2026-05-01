#!/usr/bin/env python3
"""Train and evaluate the global popularity baseline."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from src.models.popularity import PopularityRecommender
from src.utils.config import METRICS_DIR
from src.utils.experiments import train_and_evaluate_model
from src.utils.logging_utils import configure_logging
from src.utils.script_helpers import resolve_eval_split_path, resolve_train_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for popularity experiment."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--include-seen", action="store_true")
    parser.add_argument("--warm-start-only", action="store_true", default=False)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Run popularity model training + evaluation."""

    args = parse_args()
    configure_logging()

    train_path = resolve_train_path()
    eval_path = resolve_eval_split_path(args.split)
    output_path = args.output or (METRICS_DIR / f"popularity_{args.split}.json")

    model = PopularityRecommender(exclude_seen=not args.include_seen)

    train_and_evaluate_model(
        model=model,
        model_name="popularity",
        model_config={"exclude_seen": not args.include_seen},
        train_path=train_path,
        eval_path=eval_path,
        output_metrics_path=output_path,
        split_name=args.split,
        warm_start_only=args.warm_start_only,
    )


if __name__ == "__main__":
    main()
