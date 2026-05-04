"""High-level preprocessing pipeline for Retailrocket data."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data.schemas import ITEM_ID_COL
from src.data.sessionization import (
    add_item_index,
    add_session_features,
    assign_sessions,
    drop_consecutive_duplicate_items,
    filter_events_by_type,
    filter_min_session_length,
    standardize_events,
)

LOGGER = logging.getLogger(__name__)


def preprocess_events(
    events_csv_path: Path,
    event_types: Iterable[str],
    inactivity_minutes: int,
    min_session_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full events preprocessing pipeline.

    Parameters
    ----------
    events_csv_path:
        Path to raw events CSV.
    event_types:
        Event types to keep (e.g., ["view"]).
    inactivity_minutes:
        Threshold for starting a new session.
    min_session_length:
        Minimum number of events required to keep a session.

    Returns
    -------
    processed_events:
        Fully cleaned and sessionized event rows.
    item_map:
        Item id to item index mapping.
    """

    LOGGER.info("Reading raw events from %s", events_csv_path)
    raw_df = pd.read_csv(events_csv_path)

    LOGGER.info("Standardizing raw event schema")
    df = standardize_events(raw_df)

    LOGGER.info("Filtering event types to: %s", sorted(set(event_types)))
    df = filter_events_by_type(df, event_types)

    LOGGER.info("Assigning session IDs using %s-minute inactivity threshold", inactivity_minutes)
    df = assign_sessions(df, inactivity_minutes=inactivity_minutes)

    LOGGER.info("Dropping immediate duplicate item interactions")
    df = drop_consecutive_duplicate_items(df)

    LOGGER.info("Filtering sessions shorter than %s interactions", min_session_length)
    df = filter_min_session_length(df, min_session_length=min_session_length)

    LOGGER.info("Adding session feature columns")
    df = add_session_features(df)

    LOGGER.info("Building compact item index mapping")
    df, item_map = add_item_index(df)

    LOGGER.info(
        "Preprocessing complete: %s events, %s sessions, %s unique items",
        len(df),
        df["session_id"].nunique(),
        df[ITEM_ID_COL].nunique(),
    )
    return df, item_map


def build_item_metadata(
    part1_csv_path: Path,
    part2_csv_path: Path,
    chunk_size: int = 1_000_000,
) -> pd.DataFrame:
    """Extract lightweight item metadata from large properties files.

    This function keeps only two interpretable properties for now:
    - `categoryid`
    - `available`

    For each `(item, property)`, the most recent value by timestamp is retained.
    """

    keep_properties = {"categoryid", "available"}
    selected_chunks: list[pd.DataFrame] = []

    for csv_path in (part1_csv_path, part2_csv_path):
        LOGGER.info("Scanning item properties from %s", csv_path)
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            filtered = chunk[chunk["property"].isin(keep_properties)].copy()
            if filtered.empty:
                continue
            selected_chunks.append(filtered)

    if not selected_chunks:
        LOGGER.warning("No selected item metadata rows found.")
        return pd.DataFrame(columns=[ITEM_ID_COL, "category_id", "available_flag"])

    meta = pd.concat(selected_chunks, ignore_index=True)

    # Convert timestamps for correct latest-value selection.
    meta["timestamp"] = pd.to_datetime(meta["timestamp"], unit="ms", utc=True)
    meta = meta.sort_values(["itemid", "property", "timestamp"])

    latest = meta.groupby(["itemid", "property"], as_index=False).tail(1)

    # Pivot properties so each item has one row.
    pivot = latest.pivot(index="itemid", columns="property", values="value").reset_index()
    pivot = pivot.rename(columns={"itemid": ITEM_ID_COL, "categoryid": "category_id"})

    if "available" in pivot.columns:
        pivot["available_flag"] = pivot["available"].astype(str).str.strip().map(
            {"1": 1, "0": 0, "true": 1, "false": 0}
        )
        pivot = pivot.drop(columns=["available"])
    else:
        pivot["available_flag"] = pd.NA

    return pivot
