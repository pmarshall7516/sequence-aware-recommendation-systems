# Results Summary

## Status

Use this document to record final experimental outcomes after running the full dataset pipeline.

## Required Artifacts

1. Per-model metrics JSON files in `outputs/metrics/`.
2. Combined summary CSV in `outputs/metrics/summary_test.csv`.
3. Optional analysis tables/figures in `outputs/tables/` and `outputs/figures/`.

## Main Comparison Table

Fill in from `summary_test.csv`:

| Model | HitRate@10 | Recall@20 | MRR@20 | NDCG@20 | Notes |
|---|---:|---:|---:|---:|---|
| popularity | TBD | TBD | TBD | TBD | |
| cooccurrence | TBD | TBD | TBD | TBD | |
| markov | TBD | TBD | TBD | TBD | |
| weighted_markov | TBD | TBD | TBD | TBD | |
| session_knn_unordered | TBD | TBD | TBD | TBD | |
| session_knn_sequence | TBD | TBD | TBD | TBD | |
| neural_sequence | TBD | TBD | TBD | TBD | Optional run |

## Segment Findings

Record whether sequence-aware gains increase with context:

1. Prefix-length buckets (`1`, `2`, `3-5`, `6+`).
2. Full-session buckets (`2-3`, `4-6`, `7+`).

## Proposal Answer

Document a direct answer to the proposal question:

- Did sequence-aware models outperform unordered baselines?
- By how much on headline metrics?
- Was the effect stronger for longer sessions?

## Practical Interpretation

Summarize what was learned for real-world e-commerce recommendation:

1. Whether simple sequence-aware models are enough.
2. Whether neural models provide additional gains worth complexity.
3. Where the current pipeline still needs improvement.
