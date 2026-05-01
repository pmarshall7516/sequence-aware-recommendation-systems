"""Utility functions to configure consistent logging across scripts."""

from __future__ import annotations

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once for command-line scripts.

    The chosen format includes timestamp, level, and message for reproducible logs.
    """

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
