"""Train/validation/test splitting and example-generation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.schemas import (
    ITEM_IDX_COL,
    POSITION_COL,
    SESSION_ID_COL,
    SESSION_LENGTH_COL,
    SESSION_START_COL,
)


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for chronological session-level splitting."""

    train_ratio: float = 0.7
    validation_ratio: float = 0.1
    test_ratio: float = 0.2


def validate_split_config(config: SplitConfig) -> None:
    """Validate split ratios to guard against accidental misconfiguration."""

    total = config.train_ratio + config.validation_ratio + config.test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    if min(config.train_ratio, config.validation_ratio, config.test_ratio) <= 0:
        raise ValueError("All split ratios must be strictly positive")


def split_sessions_chronologically(
    sessions_df: pd.DataFrame,
    config: SplitConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split sessions into train/validation/test by session start timestamp."""

    validate_split_config(config)

    session_starts = (
        sessions_df[[SESSION_ID_COL, SESSION_START_COL]]
        .drop_duplicates(SESSION_ID_COL)
        .sort_values(SESSION_START_COL)
        .reset_index(drop=True)
    )

    n_sessions = len(session_starts)
    train_end = int(n_sessions * config.train_ratio)
    val_end = train_end + int(n_sessions * config.validation_ratio)

    train_ids = set(session_starts.iloc[:train_end][SESSION_ID_COL].tolist())
    val_ids = set(session_starts.iloc[train_end:val_end][SESSION_ID_COL].tolist())
    test_ids = set(session_starts.iloc[val_end:][SESSION_ID_COL].tolist())

    train_df = sessions_df[sessions_df[SESSION_ID_COL].isin(train_ids)].copy()
    val_df = sessions_df[sessions_df[SESSION_ID_COL].isin(val_ids)].copy()
    test_df = sessions_df[sessions_df[SESSION_ID_COL].isin(test_ids)].copy()

    return train_df, val_df, test_df


def build_session_sequences(events_df: pd.DataFrame) -> dict[int, list[int]]:
    """Convert event rows into ordered session->item_idx sequences."""

    sorted_df = events_df.sort_values([SESSION_ID_COL, POSITION_COL])
    grouped = sorted_df.groupby(SESSION_ID_COL)[ITEM_IDX_COL].apply(list)
    return {int(session_id): items for session_id, items in grouped.items()}


def generate_prefix_target_examples(events_df: pd.DataFrame) -> pd.DataFrame:
    """Generate next-item prediction examples from session sequences.

    For a session [i1, i2, i3], generates:
    - context=[i1], target=i2
    - context=[i1, i2], target=i3
    """

    sequences = build_session_sequences(events_df)

    rows: list[dict[str, object]] = []
    for session_id, items in sequences.items():
        full_len = len(items)
        for target_pos in range(1, full_len):
            rows.append(
                {
                    "session_id": session_id,
                    "context": items[:target_pos],
                    "target_item": items[target_pos],
                    "prefix_length": target_pos,
                    "full_session_length": full_len,
                }
            )

    return pd.DataFrame(rows)


def warm_start_filter(examples_df: pd.DataFrame, train_item_universe: set[int]) -> pd.DataFrame:
    """Keep only examples whose target item appeared in training."""

    return examples_df[examples_df["target_item"].isin(train_item_universe)].copy()


def summarize_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, int]:
    """Return basic row and session counts for split diagnostics."""

    return {
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "train_sessions": train_df[SESSION_ID_COL].nunique(),
        "validation_sessions": val_df[SESSION_ID_COL].nunique(),
        "test_sessions": test_df[SESSION_ID_COL].nunique(),
    }
