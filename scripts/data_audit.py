#!/usr/bin/env python3
"""Generate a lightweight audit report for processed session data."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

import pandas as pd

from src.utils.config import EVENTS_CSV, SESSIONS_PARQUET


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for data auditing."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-path", type=Path, default=EVENTS_CSV)
    parser.add_argument("--sessions-path", type=Path, default=SESSIONS_PARQUET)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "tables" / "data_audit.json")
    return parser.parse_args()


def main() -> None:
    """Compute key dataset statistics and write them as JSON."""

    args = parse_args()

    raw = pd.read_csv(args.events_path)
    sessions = pd.read_parquet(args.sessions_path)

    event_counts = raw["event"].value_counts(dropna=False).to_dict()

    summary = {
        "raw_rows": int(len(raw)),
        "raw_unique_visitors": int(raw["visitorid"].nunique()),
        "raw_unique_items": int(raw["itemid"].nunique()),
        "raw_event_counts": {str(k): int(v) for k, v in event_counts.items()},
        "processed_rows": int(len(sessions)),
        "processed_sessions": int(sessions["session_id"].nunique()),
        "processed_unique_items": int(sessions["item_id"].nunique()),
        "processed_avg_session_length": float(sessions.groupby("session_id").size().mean()),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)

    print(f"Wrote audit summary: {args.output}")


if __name__ == "__main__":
    main()
