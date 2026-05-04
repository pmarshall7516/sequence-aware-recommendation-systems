"""Small helper functions used by command-line scripts."""

from __future__ import annotations

from pathlib import Path

from src.utils.config import TEST_PARQUET, TRAIN_PARQUET, VALIDATION_PARQUET


def resolve_eval_split_path(split_name: str) -> Path:
    """Map split name string to the corresponding default parquet path."""

    split_key = split_name.strip().lower()
    if split_key == "validation":
        return VALIDATION_PARQUET
    if split_key == "test":
        return TEST_PARQUET
    raise ValueError("split must be either 'validation' or 'test'")


def resolve_train_path() -> Path:
    """Return canonical train split path."""

    return TRAIN_PARQUET
