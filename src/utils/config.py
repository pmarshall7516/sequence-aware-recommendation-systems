"""Project-wide configuration constants and default paths.

This module centralizes common paths and defaults used by multiple scripts so the
pipeline remains consistent and easier to maintain.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved filesystem paths used across the project."""

    project_root: Path
    data_dir: Path
    processed_dir: Path
    outputs_dir: Path
    metrics_dir: Path
    figures_dir: Path
    tables_dir: Path


# Infer the repository root from this file location.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Canonical directories.
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

# Main raw file locations.
EVENTS_CSV = DATA_DIR / "events.csv"
ITEM_PROPERTIES_PART1_CSV = DATA_DIR / "item_properties_part1.csv"
ITEM_PROPERTIES_PART2_CSV = DATA_DIR / "item_properties_part2.csv"
CATEGORY_TREE_CSV = DATA_DIR / "category_tree.csv"

# Main processed file locations.
SESSIONS_PARQUET = PROCESSED_DIR / "sessions.parquet"
TRAIN_PARQUET = PROCESSED_DIR / "train.parquet"
VALIDATION_PARQUET = PROCESSED_DIR / "validation.parquet"
TEST_PARQUET = PROCESSED_DIR / "test.parquet"
ITEM_METADATA_PARQUET = PROCESSED_DIR / "item_metadata.parquet"
ITEM_ID_MAP_PARQUET = PROCESSED_DIR / "item_id_map.parquet"

# Reproducibility defaults.
DEFAULT_RANDOM_SEED = 42

# Sessionization defaults.
DEFAULT_SESSION_GAP_MINUTES = 30
DEFAULT_MIN_SESSION_LENGTH = 2


def get_paths() -> ProjectPaths:
    """Return a structured bundle of commonly used paths."""

    return ProjectPaths(
        project_root=PROJECT_ROOT,
        data_dir=DATA_DIR,
        processed_dir=PROCESSED_DIR,
        outputs_dir=OUTPUTS_DIR,
        metrics_dir=METRICS_DIR,
        figures_dir=FIGURES_DIR,
        tables_dir=TABLES_DIR,
    )
