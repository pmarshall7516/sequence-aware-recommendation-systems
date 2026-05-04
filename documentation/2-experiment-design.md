# Experiment Design

## Research Question

Do sequence-aware recommendation models outperform unordered recommendation models for next-item prediction in the same session?

## Dataset

Retailrocket e-commerce behavior data from:

- `data/events.csv`
- `data/item_properties_part1.csv`
- `data/item_properties_part2.csv`
- `data/category_tree.csv`

## Task Definition

Given a session prefix, rank candidate items and predict the next item interaction.

Example from one session sequence `[i1, i2, i3, i4]`:

- context `[i1]` -> target `i2`
- context `[i1, i2]` -> target `i3`
- context `[i1, i2, i3]` -> target `i4`

## Splitting Strategy

Chronological session-level split:

- Train: earliest 70% sessions
- Validation: next 10% sessions
- Test: latest 20% sessions

Implemented in `scripts/build_splits.py` and `src/data/splits.py`.

## Model Families

Unordered baselines:

1. Global Popularity (`src/models/popularity.py`)
2. Item Co-occurrence (`src/models/cooccurrence.py`)

Sequence-aware models:

1. First-order Markov (`src/models/markov.py`)
2. Weighted Markov (`src/models/weighted_markov.py`)
3. Session-kNN sequence-aware variant (`src/models/session_knn.py`)
4. GRU neural sequence model (`src/models/neural_sequence.py`)

## Evaluation Metrics

For K in `{5, 10, 20}`:

- HitRate@K
- Recall@K
- MRR@K
- NDCG@K

Implemented in:

- `src/evaluation/metrics.py`
- `src/evaluation/evaluator.py`

## Segment Analysis

Metrics are also reported by:

1. Prefix length bucket: `1`, `2`, `3-5`, `6+`
2. Full session length bucket: `2-3`, `4-6`, `7+`

This supports the proposal requirement to compare short vs long session behavior.

## Candidate Ranking Protocol

Default protocol ranks over all training items.

- Candidate universe: unique `item_idx` in training split.
- Optional warm-start-only evaluation: keep only examples whose targets appear in train.

## Run Plan

1. Preprocess data.
2. Build splits.
3. Run baseline and sequence-aware models.
4. Aggregate metrics to summary table.
5. Compare performance and analyze segment-level gaps.

Core command:

```bash
python scripts/run_all_experiments.py --split test
python scripts/evaluate.py --metrics-dir outputs/metrics --glob "*_test.json" --output outputs/metrics/summary_test.csv
```

## Expected Outcome (Hypothesis)

Sequence-aware models are expected to outperform unordered baselines, with larger gains for longer-context sessions where event order carries stronger intent.
