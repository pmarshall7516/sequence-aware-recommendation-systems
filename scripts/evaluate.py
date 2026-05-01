#!/usr/bin/env python3
"""Aggregate saved per-model JSON metrics into a summary CSV."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from src.utils.config import METRICS_DIR
from src.utils.experiments import flatten_result_for_summary, save_summary_table
from src.utils.io import read_json


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for metrics aggregation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, default=METRICS_DIR)
    parser.add_argument("--glob", type=str, default="*.json")
    parser.add_argument("--output", type=Path, default=METRICS_DIR / "summary.csv")
    return parser.parse_args()


def main() -> None:
    """Load metrics JSON files and write flattened summary CSV."""

    args = parse_args()

    rows = []
    for path in sorted(args.metrics_dir.glob(args.glob)):
        payload = read_json(path)
        rows.append(flatten_result_for_summary(payload))

    save_summary_table(rows, args.output)
    print(f"Wrote summary table: {args.output}")


if __name__ == "__main__":
    main()
