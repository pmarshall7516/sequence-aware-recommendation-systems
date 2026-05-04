# Data Audit

## Purpose

This document defines and records the data audit for the Retailrocket dataset used in this project. It ensures the data pipeline is aligned with the research question:

> Do sequence-aware recommendation approaches outperform unordered recommendation methods for next-item prediction within a session?

## Data Sources

The pipeline uses the proposal-specified dataset files:

- `data/events.csv`
- `data/item_properties_part1.csv`
- `data/item_properties_part2.csv`
- `data/category_tree.csv`

## Audit Scope

The audit checks:

1. Raw event schema and event-type distribution.
2. Timestamp conversion correctness (Unix ms to UTC datetime).
3. Sessionization behavior under inactivity threshold.
4. Session length distribution after filtering.
5. Item coverage and sparsity implications.
6. Processed split readiness for next-item prediction.

## Commands

Run preprocessing first:

```bash
python scripts/preprocess_data.py --event-types view --session-gap-minutes 30 --min-session-length 2
```

Then generate the audit summary:

```bash
python scripts/data_audit.py --output outputs/tables/data_audit.json
```

## Current Audit Output Format

`outputs/tables/data_audit.json` includes:

- `raw_rows`
- `raw_unique_visitors`
- `raw_unique_items`
- `raw_event_counts`
- `processed_rows`
- `processed_sessions`
- `processed_unique_items`
- `processed_avg_session_length`

## Sessionization Standard

This project uses:

- 30-minute inactivity threshold for starting a new session.
- Minimum session length of 2 events.
- Optional removal of immediate repeated item clicks within session.

These settings are implemented in `src/data/sessionization.py` and driven through `scripts/preprocess_data.py`.

## Proposal Alignment

This audit process directly supports the proposal by validating that:

1. Events are sequenced reliably by timestamp.
2. Sessions are explicitly constructed for within-session next-item prediction.
3. The processed output can be split chronologically for leakage-safe evaluation.

## Update Procedure

After every major preprocessing change, regenerate:

1. `data/processed/sessions.parquet`
2. `outputs/tables/data_audit.json`

Then update this file with any parameter changes and observed data caveats.
