"""Helpers for deterministic experiments."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set pseudo-random seeds for standard Python and NumPy.

    Torch seeding is handled in the neural script to avoid a hard dependency.
    """

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
