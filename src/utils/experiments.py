"""Reusable experiment orchestration helpers for model scripts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.datasets import (
    load_examples_from_split,
    load_session_sequences,
    load_split_events,
    load_train_item_universe,
)
from src.data.splits import warm_start_filter
from src.evaluation.evaluator import EvaluationConfig, evaluate_model
from src.utils.io import write_json

LOGGER = logging.getLogger(__name__)


def train_and_evaluate_model(
    *,
    model: Any,
    model_name: str,
    model_config: dict[str, Any],
    train_path: Path,
    eval_path: Path,
    output_metrics_path: Path,
    split_name: str,
    k_values: tuple[int, ...] = (5, 10, 20),
    warm_start_only: bool = True,
) -> dict[str, Any]:
    """Train one model and evaluate it on one split.

    This helper is used by each model-specific training script to keep behavior
    standardized.
    """

    LOGGER.info("Loading train events from %s", train_path)
    train_events = load_split_events(train_path)

    LOGGER.info("Loading evaluation examples from %s", eval_path)
    eval_examples = load_examples_from_split(eval_path)

    LOGGER.info("Building training sequences")
    train_sequences = load_session_sequences(train_events)

    candidate_items = load_train_item_universe(train_events)
    train_item_set = set(candidate_items)

    if warm_start_only:
        LOGGER.info("Applying warm-start filter (targets must exist in train)")
        eval_examples = warm_start_filter(eval_examples, train_item_set)

    LOGGER.info("Fitting model: %s", model_name)
    model.fit(train_sequences)

    LOGGER.info("Evaluating model: %s on %s split", model_name, split_name)
    eval_result = evaluate_model(
        model=model,
        examples_df=eval_examples,
        candidate_items=candidate_items,
        config=EvaluationConfig(k_values=k_values, split_name=split_name),
    )

    result = {
        "model_name": model_name,
        "model_config": model_config,
        "split": split_name,
        "warm_start_only": warm_start_only,
        "num_train_rows": int(len(train_events)),
        "num_eval_examples": int(len(eval_examples)),
        "candidate_protocol": "train_item_universe",
        "metrics": eval_result,
    }

    write_json(result, output_metrics_path)
    LOGGER.info("Saved metrics to %s", output_metrics_path)

    return result


def flatten_result_for_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested metrics dict into one tabular row for summary CSV."""

    row = {
        "model_name": result["model_name"],
        "split": result["split"],
        "warm_start_only": result["warm_start_only"],
        "num_train_rows": result["num_train_rows"],
        "num_eval_examples": result["num_eval_examples"],
        "candidate_protocol": result["candidate_protocol"],
        "runtime_seconds": result["metrics"]["runtime_seconds"],
    }

    overall = result["metrics"]["overall"]
    row.update(overall)

    for key, value in result["model_config"].items():
        row[f"config_{key}"] = value

    return row


def save_summary_table(rows: list[dict[str, Any]], path: Path) -> None:
    """Write a flattened summary table to CSV for quick model comparison."""

    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
