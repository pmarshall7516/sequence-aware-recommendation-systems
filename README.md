# Sequence-Aware Recommendation Systems

## 1. Introduction

This project builds and validates a complete experimental pipeline to answer one core research question:

> Do sequence-aware recommendation approaches outperform unordered recommendation methods for next-item prediction within the same session?

The implementation follows a reproducible workflow on the Retailrocket e-commerce dataset. It includes data preprocessing, sessionization, chronological splitting, model training, and ranking-based evaluation. The code compares unordered baselines (popularity, co-occurrence) against sequence-aware models (Markov, session-kNN, GRU neural sequence model), then reports both overall and segment-level performance.

## 2. Quick Start

### 2.1 Environment and dependencies

1. Create and activate a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Confirm dataset files exist in `data/`.

Required:

- `data/events.csv`
- `data/item_properties_part1.csv`
- `data/item_properties_part2.csv`
- `data/category_tree.csv`

### 2.2 Run data pipeline and save processed data

1. Run preprocessing (default: `view` events, 30-minute session gap, min session length 2).

```bash
python scripts/preprocess_data.py --event-types view --session-gap-minutes 30 --min-session-length 2
```

2. Build chronological train/validation/test splits.

```bash
python scripts/build_splits.py
```

3. Optional: generate audit summary.

```bash
python scripts/data_audit.py --output outputs/tables/data_audit.json
```

Main processed outputs:

- `data/processed/sessions.parquet`
- `data/processed/item_id_map.parquet`
- `data/processed/train.parquet`
- `data/processed/validation.parquet`
- `data/processed/test.parquet`

Optional metadata output:

- `data/processed/item_metadata.parquet` (if `--build-item-metadata` is used)

### 2.3 Train models and run predictions/evaluation

Run all core experiments:

```bash
python scripts/run_all_experiments.py --split test
```

Run an individual model (example):

```bash
python scripts/train_markov.py --split test
```

Aggregate per-model metric JSON into one table:

```bash
python scripts/evaluate.py --metrics-dir outputs/metrics --glob "*_test.json" --output outputs/metrics/summary_test.csv
```

### 2.4 Model list and what each model tests

- `popularity`: global frequency baseline with no session context.
- `cooccurrence`: unordered session-context baseline using item co-occurrence.
- `markov`: first-order sequence model using immediate previous item.
- `weighted_markov`: sequence model using recency-weighted context transitions.
- `session_knn_unordered`: nearest-neighbor session method without recency weighting.
- `session_knn_sequence`: nearest-neighbor session method with recency weighting.
- `neural_sequence` (optional): GRU-based next-item sequence model.

### 2.5 Interpret results and complete project deliverables

1. Open `outputs/metrics/summary_test.csv` for model-level comparison.
2. Inspect per-model JSON files in `outputs/metrics/` for segment-level metrics.
3. Fill `documentation/3-results-summary.md` with final findings.
4. Use notebooks in `notebooks/` for analysis visuals and report narrative.

## 3. Files

### 3.1 Data pipeline scripts

These scripts build the reproducible data foundation used by every model.

- `scripts/preprocess_data.py`: end-to-end preprocessing entrypoint. Reads raw events, filters event types, sessionizes events, removes immediate duplicates, adds sequence/session features, and saves processed parquet.
- `scripts/build_splits.py`: creates chronological session-level train/validation/test splits to avoid temporal leakage.
- `scripts/data_audit.py`: computes summary statistics for raw and processed data to validate data quality and experiment readiness.

### 3.2 Training and experiment scripts

These scripts train recommender models and evaluate next-item ranking performance.

- `scripts/train_popularity.py`: trains/evaluates popularity baseline.
- `scripts/train_cooccurrence.py`: trains/evaluates unordered co-occurrence baseline.
- `scripts/train_markov.py`: trains/evaluates first-order Markov model.
- `scripts/train_weighted_markov.py`: trains/evaluates recency-weighted Markov model.
- `scripts/train_session_knn.py`: trains/evaluates unordered or sequence-aware session-kNN.
- `scripts/train_neural_sequence.py`: trains/evaluates GRU neural model (requires PyTorch).
- `scripts/run_all_experiments.py`: orchestrates the core experiment suite and writes all per-model metric artifacts.
- `scripts/evaluate.py`: combines saved per-model JSON metrics into a single summary CSV.

### 3.3 Model files (`src/models`)

These modules implement model logic and scoring behavior for next-item ranking.

- `src/models/base.py`: abstract recommender interface.
- `src/models/popularity.py`: global popularity recommender.
- `src/models/cooccurrence.py`: unordered item co-occurrence recommender.
- `src/models/markov.py`: first-order transition recommender.
- `src/models/weighted_markov.py`: recency-weighted transition recommender.
- `src/models/session_knn.py`: nearest-neighbor session recommender variants.
- `src/models/neural_sequence.py`: GRU sequence recommender.

### 3.4 Data and evaluation core modules

These modules provide reusable infrastructure used across scripts and models.

- `src/data/schemas.py`: canonical column names for raw/processed data.
- `src/data/sessionization.py`: event standardization, session creation, duplicate handling, feature columns, item indexing.
- `src/data/preprocessing.py`: high-level preprocessing orchestration and item metadata extraction.
- `src/data/splits.py`: chronological split logic and prefix-target example generation.
- `src/data/datasets.py`: helpers for loading split datasets and sequences.
- `src/evaluation/metrics.py`: HitRate/Recall/MRR/NDCG implementations.
- `src/evaluation/evaluator.py`: model evaluation with overall + bucketed segment metrics.
- `src/utils/experiments.py`: shared train-and-evaluate orchestration.
- `src/utils/config.py`: project paths and default constants.
- `src/utils/io.py`: file IO utilities.
- `src/utils/logging_utils.py`: logging configuration helper.
- `src/utils/randomness.py`: reproducibility seed helper.
- `src/utils/script_helpers.py`: split-path resolution helpers for scripts.

### 3.5 Documentation files

These files define planning, validation, and reporting structure.

- `documentation/0-initial-plan.md`: detailed implementation and experiment roadmap.
- `documentation/1-data-audit.md`: audit scope, commands, and quality checks.
- `documentation/2-experiment-design.md`: finalized experiment protocol and model matrix.
- `documentation/3-results-summary.md`: final-results template and reporting checklist.
- `documentation/4-validation-report.md`: proposal-to-code alignment validation report.

### 3.6 Notebooks

These notebooks are used to synthesize findings and present results.

- `notebooks/01_data_audit.ipynb`: exploratory audit and distribution checks.
- `notebooks/02_baseline_results.ipynb`: baseline model comparison notebook.
- `notebooks/03_result_analysis.ipynb`: segment-level analysis and final interpretation notebook.

## Recommended full run commands

```bash
python scripts/preprocess_data.py --event-types view --session-gap-minutes 30 --min-session-length 2
python scripts/build_splits.py
python scripts/run_all_experiments.py --split test
python scripts/evaluate.py --metrics-dir outputs/metrics --glob "*_test.json" --output outputs/metrics/summary_test.csv
```

## Optional commands

Run metadata extraction:

```bash
python scripts/preprocess_data.py --event-types view --build-item-metadata
```

Include neural model in orchestrated run:

```bash
python scripts/run_all_experiments.py --split test --include-neural
```

## 4. Interpret Results

This section explains how to interpret ranking metrics in `outputs/metrics/*.json` and `outputs/metrics/summary_test.csv`.

### 4.1 HitRate@K

Definition:

- `HitRate@K` is the fraction of prediction examples where the true next item appears anywhere in the top `K` recommendations.

Formula intuition:

- For each example: `1` if target is in top `K`, else `0`.
- Average over all examples.

Why it matters:

- It measures whether the model can place the right item in the visible shortlist.
- In product recommendation interfaces, this is a practical success signal because users often inspect only a small set of recommendations.

How to read it:

- Higher is better.
- `HitRate@10` is a strong headline metric for this project.

### 4.2 Recall@K

Definition:

- `Recall@K` is the proportion of relevant items retrieved in top `K`.

In this project:

- Each example has one true target item, so `Recall@K` is numerically equivalent to `HitRate@K`.

Why it matters:

- It is standard in recommender evaluation and keeps your results comparable to literature.
- Even when equal to hit rate here, it remains useful for consistency and reporting clarity.

How to read it:

- Higher is better.
- Expect `Recall@20 >= Recall@10 >= Recall@5` because larger `K` is less strict.

### 4.3 MRR@K (Mean Reciprocal Rank)

Definition:

- `MRR@K` rewards models that place the correct item higher in the ranked list.

Formula intuition:

- If target rank is `r` in top `K`, score is `1/r`.
- If target is missing from top `K`, score is `0`.
- Average over all examples.

Why it matters:

- Two models can have similar `HitRate@K`, but one may rank the true item much earlier.
- `MRR@K` captures this rank quality directly.

How to read it:

- Higher is better.
- Larger gap in MRR than HitRate usually means better ranking precision near the top.

### 4.4 NDCG@K (Normalized Discounted Cumulative Gain)

Definition:

- `NDCG@K` is a rank-sensitive metric that gives larger credit when the correct item appears earlier.

Formula intuition:

- Gain is discounted by rank using a logarithmic penalty.
- Earlier correct predictions contribute more than later ones.

Why it matters:

- It balances retrieval success with position quality.
- It is robust for evaluating top-heavy recommendation experiences.

How to read it:

- Higher is better.
- If `NDCG@K` improves while HitRate changes little, ranking order has improved.

### 4.5 How to compare models in this project

Use this interpretation order:

1. Compare `HitRate@10` (main success metric).
2. Check `MRR@20` and `NDCG@20` to verify rank quality, not just retrieval.
3. Confirm consistency with `Recall@20`.
4. Inspect segment metrics (`by_prefix_length`, `by_session_length`) to test whether sequence-aware gains grow with more context.

### 4.6 Expected proposal-aligned patterns

When sequence modeling is helpful, you should often see:

1. Sequence-aware models (`markov`, `weighted_markov`, `session_knn_sequence`, `neural_sequence`) outperform unordered baselines on `HitRate@10`.
2. Larger improvements in `MRR@20` and `NDCG@20`, indicating better top-rank ordering.
3. Bigger gains in longer-context segments (`prefix_3_5`, `prefix_6_plus`, `session_long_7_plus`).

### 4.7 Practical caveats while interpreting

1. Popularity effects can make a simple baseline look strong in e-commerce data.
2. If warm-start filtering is disabled, cold-start targets can lower all metrics.
3. Runtime should be considered alongside metric gains when selecting a deployable model.
