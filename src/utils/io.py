"""Shared I/O helpers for reading/writing tabular artifacts.

These helpers keep serialization consistent across scripts and avoid repetitive
filesystem boilerplate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_directory(path: Path) -> None:
    """Create a directory (and parents) if it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)


def ensure_parent_dir(file_path: Path) -> None:
    """Create the parent directory for a file path if needed."""

    ensure_directory(file_path.parent)


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV file through pandas with caller-provided options."""

    return pd.read_csv(path, **kwargs)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to Parquet and ensure the parent directory exists."""

    ensure_parent_dir(path)
    df.to_parquet(path, index=False)


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a parquet file into a DataFrame."""

    return pd.read_parquet(path)


def write_json(data: dict[str, Any], path: Path) -> None:
    """Write JSON with deterministic formatting for easier diffs."""

    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, sort_keys=True)


def read_json(path: Path) -> dict[str, Any]:
    """Read JSON file and return a dictionary."""

    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)
