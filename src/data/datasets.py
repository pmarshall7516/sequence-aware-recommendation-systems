"""Dataset loading helpers for experiment scripts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.splits import generate_prefix_target_examples


def load_split_events(path: Path) -> pd.DataFrame:
    """Load split event rows from parquet."""

    return pd.read_parquet(path)


def load_examples_from_split(path: Path) -> pd.DataFrame:
    """Load split events and convert to prefix-target examples."""

    events = load_split_events(path)
    return generate_prefix_target_examples(events)


def load_train_item_universe(train_events: pd.DataFrame) -> list[int]:
    """Extract sorted item universe from training split."""

    return sorted(train_events["item_idx"].unique().tolist())


def load_session_sequences(events_df: pd.DataFrame) -> list[list[int]]:
    """Convert event rows to ordered list-of-lists session sequences."""

    grouped = (
        events_df.sort_values(["session_id", "position"]).groupby("session_id")["item_idx"].apply(list)
    )
    return grouped.tolist()
