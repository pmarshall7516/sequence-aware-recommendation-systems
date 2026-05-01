#!/usr/bin/env python3
"""Preprocess Retailrocket events into clean, sessionized parquet files.

This script is the first executable stage of the project pipeline.

Outputs:
- data/processed/sessions.parquet
- data/processed/item_id_map.parquet
- data/processed/item_metadata.parquet (optional)
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging

from src.data.preprocessing import build_item_metadata, preprocess_events
from src.utils.config import (
    EVENTS_CSV,
    ITEM_ID_MAP_PARQUET,
    ITEM_METADATA_PARQUET,
    ITEM_PROPERTIES_PART1_CSV,
    ITEM_PROPERTIES_PART2_CSV,
    SESSIONS_PARQUET,
)
from src.utils.io import write_parquet
from src.utils.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Build command-line arguments for preprocessing."""

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--events-csv", type=Path, default=EVENTS_CSV)
    parser.add_argument(
        "--event-types",
        nargs="+",
        default=["view"],
        help="Event types to keep (e.g., view addtocart transaction).",
    )
    parser.add_argument("--session-gap-minutes", type=int, default=30)
    parser.add_argument("--min-session-length", type=int, default=2)

    parser.add_argument("--sessions-output", type=Path, default=SESSIONS_PARQUET)
    parser.add_argument("--item-map-output", type=Path, default=ITEM_ID_MAP_PARQUET)

    parser.add_argument(
        "--build-item-metadata",
        action="store_true",
        help="If set, extract category/availability metadata from item property files.",
    )
    parser.add_argument("--item-properties-part1", type=Path, default=ITEM_PROPERTIES_PART1_CSV)
    parser.add_argument("--item-properties-part2", type=Path, default=ITEM_PROPERTIES_PART2_CSV)
    parser.add_argument("--item-metadata-output", type=Path, default=ITEM_METADATA_PARQUET)

    return parser.parse_args()


def main() -> None:
    """Execute the preprocessing pipeline and persist outputs."""

    args = parse_args()
    configure_logging()

    LOGGER.info("Starting preprocessing pipeline")

    processed_df, item_map = preprocess_events(
        events_csv_path=args.events_csv,
        event_types=args.event_types,
        inactivity_minutes=args.session_gap_minutes,
        min_session_length=args.min_session_length,
    )

    LOGGER.info("Writing sessionized events to %s", args.sessions_output)
    write_parquet(processed_df, args.sessions_output)

    LOGGER.info("Writing item index mapping to %s", args.item_map_output)
    write_parquet(item_map, args.item_map_output)

    if args.build_item_metadata:
        LOGGER.info("Building item metadata (category + availability)")
        metadata_df = build_item_metadata(
            part1_csv_path=args.item_properties_part1,
            part2_csv_path=args.item_properties_part2,
        )
        LOGGER.info("Writing item metadata to %s", args.item_metadata_output)
        write_parquet(metadata_df, args.item_metadata_output)

    LOGGER.info("Preprocessing completed successfully")


if __name__ == "__main__":
    main()
