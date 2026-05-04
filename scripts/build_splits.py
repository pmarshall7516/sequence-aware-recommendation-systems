#!/usr/bin/env python3
"""Build chronological session-level train/validation/test splits."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging

import pandas as pd

from src.data.splits import SplitConfig, split_sessions_chronologically, summarize_splits
from src.utils.config import SESSIONS_PARQUET, TEST_PARQUET, TRAIN_PARQUET, VALIDATION_PARQUET
from src.utils.io import write_parquet
from src.utils.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for split generation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions-path", type=Path, default=SESSIONS_PARQUET)
    parser.add_argument("--train-output", type=Path, default=TRAIN_PARQUET)
    parser.add_argument("--validation-output", type=Path, default=VALIDATION_PARQUET)
    parser.add_argument("--test-output", type=Path, default=TEST_PARQUET)

    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    """Run chronological splitting and write output files."""

    args = parse_args()
    configure_logging()

    LOGGER.info("Reading processed sessions from %s", args.sessions_path)
    sessions_df = pd.read_parquet(args.sessions_path)

    split_config = SplitConfig(
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
    )

    LOGGER.info("Creating chronological session splits")
    train_df, val_df, test_df = split_sessions_chronologically(sessions_df=sessions_df, config=split_config)

    LOGGER.info("Writing train split to %s", args.train_output)
    write_parquet(train_df, args.train_output)

    LOGGER.info("Writing validation split to %s", args.validation_output)
    write_parquet(val_df, args.validation_output)

    LOGGER.info("Writing test split to %s", args.test_output)
    write_parquet(test_df, args.test_output)

    summary = summarize_splits(train_df, val_df, test_df)
    LOGGER.info("Split summary: %s", summary)


if __name__ == "__main__":
    main()
