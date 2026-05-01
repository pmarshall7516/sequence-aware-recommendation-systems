"""Functions for transforming event streams into cleaned sessions."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.data.schemas import (
    EVENT_TYPE_COL,
    ITEM_ID_COL,
    ITEM_IDX_COL,
    POSITION_COL,
    RAW_EVENT_COL,
    RAW_ITEM_COL,
    RAW_TIMESTAMP_COL,
    RAW_VISITOR_COL,
    SESSION_END_COL,
    SESSION_ID_COL,
    SESSION_LENGTH_COL,
    SESSION_START_COL,
    TIMESTAMP_COL,
    VISITOR_ID_COL,
)


def standardize_events(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw Retailrocket events into a shared internal schema.

    Steps:
    - Rename columns to consistent snake_case names.
    - Convert timestamp from unix milliseconds to timezone-aware datetime.
    - Sort by visitor and timestamp to preserve event chronology.
    """

    required_cols = {RAW_TIMESTAMP_COL, RAW_VISITOR_COL, RAW_EVENT_COL, RAW_ITEM_COL}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in events file: {sorted(missing)}")

    out = df.rename(
        columns={
            RAW_TIMESTAMP_COL: TIMESTAMP_COL,
            RAW_VISITOR_COL: VISITOR_ID_COL,
            RAW_EVENT_COL: EVENT_TYPE_COL,
            RAW_ITEM_COL: ITEM_ID_COL,
        }
    ).copy()

    # Retailrocket stores timestamps in Unix milliseconds.
    out[TIMESTAMP_COL] = pd.to_datetime(out[TIMESTAMP_COL], unit="ms", utc=True)

    out = out.sort_values([VISITOR_ID_COL, TIMESTAMP_COL]).reset_index(drop=True)
    return out


def filter_events_by_type(df: pd.DataFrame, event_types: Iterable[str]) -> pd.DataFrame:
    """Filter events to selected event types (e.g., only view events)."""

    selected = {e.strip() for e in event_types if e.strip()}
    if not selected:
        raise ValueError("At least one event type is required.")

    return df[df[EVENT_TYPE_COL].isin(selected)].copy()


def assign_sessions(df: pd.DataFrame, inactivity_minutes: int) -> pd.DataFrame:
    """Assign session IDs based on inactivity gaps within each visitor stream.

    A new session starts when:
    - It is the first event for the visitor, or
    - Time gap from previous event exceeds inactivity threshold.
    """

    out = df.sort_values([VISITOR_ID_COL, TIMESTAMP_COL]).copy()

    # Compute time gap to the previous event for each visitor.
    prev_ts = out.groupby(VISITOR_ID_COL)[TIMESTAMP_COL].shift(1)
    gap_seconds = (out[TIMESTAMP_COL] - prev_ts).dt.total_seconds()

    is_new_session = prev_ts.isna() | (gap_seconds > inactivity_minutes * 60)

    # Cumulative session index per visitor.
    out["_visitor_session_index"] = is_new_session.groupby(out[VISITOR_ID_COL]).cumsum()

    # Build a stable composite session key then convert to compact integer ID.
    out["_session_key"] = (
        out[VISITOR_ID_COL].astype(str) + "_" + out["_visitor_session_index"].astype(str)
    )

    out[SESSION_ID_COL] = pd.factorize(out["_session_key"], sort=False)[0].astype("int64")

    out = out.drop(columns=["_visitor_session_index", "_session_key"])
    return out


def drop_consecutive_duplicate_items(df: pd.DataFrame) -> pd.DataFrame:
    """Remove immediate repeated item interactions within the same session.

    Example:
    [A, A, B, B, B, C] becomes [A, B, C]

    This keeps sequence signal but removes noisy repeated refresh/click events.
    """

    out = df.sort_values([SESSION_ID_COL, TIMESTAMP_COL]).copy()
    prev_item = out.groupby(SESSION_ID_COL)[ITEM_ID_COL].shift(1)
    keep_mask = prev_item.isna() | (out[ITEM_ID_COL] != prev_item)
    return out[keep_mask].copy()


def filter_min_session_length(df: pd.DataFrame, min_session_length: int) -> pd.DataFrame:
    """Remove sessions shorter than a required minimum length."""

    lengths = df.groupby(SESSION_ID_COL)[ITEM_ID_COL].transform("size")
    out = df[lengths >= min_session_length].copy()

    # Re-factorize session IDs so downstream processing sees compact IDs.
    out[SESSION_ID_COL] = pd.factorize(out[SESSION_ID_COL], sort=False)[0].astype("int64")
    return out


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-event position and per-session metadata columns."""

    out = df.sort_values([SESSION_ID_COL, TIMESTAMP_COL]).copy()

    out[POSITION_COL] = out.groupby(SESSION_ID_COL).cumcount().astype("int32")
    out[SESSION_LENGTH_COL] = out.groupby(SESSION_ID_COL)[ITEM_ID_COL].transform("size").astype(
        "int32"
    )
    out[SESSION_START_COL] = out.groupby(SESSION_ID_COL)[TIMESTAMP_COL].transform("min")
    out[SESSION_END_COL] = out.groupby(SESSION_ID_COL)[TIMESTAMP_COL].transform("max")

    return out


def add_item_index(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a compact integer item index mapping.

    Returns:
    - Processed events with `item_idx` column.
    - Mapping DataFrame from original item_id to item_idx.
    """

    out = df.copy()
    out[ITEM_IDX_COL], uniques = pd.factorize(out[ITEM_ID_COL], sort=False)
    out[ITEM_IDX_COL] = out[ITEM_IDX_COL].astype("int64")

    item_map = pd.DataFrame({ITEM_ID_COL: uniques})
    item_map[ITEM_IDX_COL] = item_map.index.astype("int64")
    return out, item_map
