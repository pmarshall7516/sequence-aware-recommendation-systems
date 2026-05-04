# Initial Experiment Development Plan

## Project Goal

Answer the research question:

> Do sequence-aware recommendation approaches outperform unordered recommendation methods when the task is to predict the next item a user will interact with within the same session?

The project should compare recommender models that use different amounts of ordering information on the Retailrocket e-commerce behavior dataset. The central experimental claim is that preserving event order within a session improves next-item prediction compared with methods that treat previous interactions as an unordered set.

## Guiding Experimental Principles

1. Use the same cleaned session dataset for every model.
2. Use the same train, validation, and test split for every model.
3. Evaluate every model on exactly the same next-item prediction examples.
4. Separate tuning from final testing.
5. Report both overall performance and performance by session length.
6. Include simple baselines first so gains from sequence-aware models are meaningful.
7. Keep the first implementation reproducible and simple before adding neural models.

## Proposed Repository Structure

A coding agent should build toward this structure:

```text
sequence-aware-recommendation-systems/
├── data/
│   ├── events.csv
│   ├── item_properties_part1.csv
│   ├── item_properties_part2.csv
│   ├── category_tree.csv
│   └── processed/
│       ├── sessions.parquet
│       ├── train.parquet
│       ├── validation.parquet
│       ├── test.parquet
│       ├── item_metadata.parquet
│       └── negative_samples.parquet
├── documentation/
│   ├── 0-initial-plan.md
│   ├── 1-data-audit.md
│   ├── 2-experiment-design.md
│   └── 3-results-summary.md
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_baseline_results.ipynb
│   └── 03_result_analysis.ipynb
├── outputs/
│   ├── metrics/
│   ├── figures/
│   └── tables/
├── scripts/
│   ├── preprocess_data.py
│   ├── build_splits.py
│   ├── train_popularity.py
│   ├── train_cooccurrence.py
│   ├── train_markov.py
│   ├── train_session_knn.py
│   ├── train_neural_sequence.py
│   ├── evaluate.py
│   └── run_all_experiments.py
├── src/
│   ├── data/
│   ├── models/
│   ├── evaluation/
│   └── utils/
├── requirements.txt
└── README.md
```

If time is limited, prioritize `scripts/`, `src/`, `outputs/`, and `documentation/` over notebooks.

## Phase 1: Data Audit

### Objective

Understand the Retailrocket dataset enough to define clean sessions, valid prediction examples, and feasible model inputs.

### Input Files

The available raw files are:

```text
data/events.csv
data/item_properties_part1.csv
data/item_properties_part2.csv
data/category_tree.csv
```

### Data Audit Tasks

1. Inspect `events.csv` columns, dtypes, missing values, duplicate rows, and event counts.
2. Confirm timestamp unit and convert timestamps into UTC datetimes.
3. Count unique visitors, items, event types, and total events.
4. Count events by `event` type, likely including `view`, `addtocart`, and `transaction`.
5. Determine whether next-item prediction should use all event types or only `view` events for the first experiment.
6. Compute item popularity distribution and identify long-tail sparsity.
7. Compute user activity distribution.
8. Identify whether `visitorid` can be used directly as a session proxy or whether sessions must be segmented by inactivity gaps.
9. Inspect item metadata from `item_properties_part1.csv`, `item_properties_part2.csv`, and `category_tree.csv`.
10. Decide whether metadata will be included in the main experiments or reserved for analysis only.

### Recommended Session Definition

Use timestamp-ordered events grouped by `visitorid`, then split into sessions using an inactivity threshold.

Default threshold:

```text
30 minutes of inactivity starts a new session
```

Rationale:

- The proposal asks about within-session next-item prediction.
- Retailrocket provides visitor IDs and timestamps but does not provide explicit session IDs.
- A 30-minute inactivity window is a common pragmatic sessionization rule for web behavior.

### Event Filtering Decision

Recommended primary experiment:

```text
Use view events only.
```

Rationale:

- View events provide the largest number of sequential interactions.
- Add-to-cart and transaction events are much sparser and may create too few valid sessions.
- The research question is about next interacted item, and views are a valid e-commerce interaction.

Recommended secondary experiment if time permits:

```text
Repeat evaluation on mixed events with event-type weighting or separate event-type subsets.
```

Possible event handling variants:

1. `view_only`: only item views.
2. `all_events`: views, add-to-cart, and transactions treated as item interactions.
3. `cart_transaction_only`: sparse high-intent interactions, used only if enough sessions remain.

### Minimum Session Criteria

After sessionization, keep sessions satisfying:

```text
session_length >= 2
```

For model training robustness, also create an optional stricter subset:

```text
session_length >= 3
```

A session of length 2 gives one prediction example. A session of length 3 or more allows stronger tests of context length.

### Data Audit Outputs

Create `documentation/1-data-audit.md` with:

1. Dataset row counts.
2. Event type counts.
3. Number of unique visitors and items.
4. Sessionization rule used.
5. Number of sessions before and after filtering.
6. Session length distribution.
7. Train, validation, and test split sizes.
8. Any important caveats.

Create figures in `outputs/figures/`:

1. Event type count bar chart.
2. Session length histogram.
3. Item popularity log-log plot.
4. User activity log-log plot.

## Phase 2: Preprocessing Pipeline

### Objective

Build reproducible processed files that every model can consume.

### Script

Create:

```text
scripts/preprocess_data.py
```

### Required Steps

1. Read `data/events.csv`.
2. Convert timestamps to datetime.
3. Sort events by `visitorid` and timestamp.
4. Filter to selected event types, starting with `view`.
5. Sessionize each visitor using the 30-minute inactivity rule.
6. Remove duplicate consecutive interactions with the same item inside a session if they occur immediately back-to-back.
7. Remove sessions shorter than 2 interactions.
8. Re-index items to compact integer IDs for efficient modeling.
9. Re-index sessions to compact session IDs.
10. Save processed sessions to `data/processed/sessions.parquet`.

### Processed Session Schema

The processed sessions file should contain one row per event:

```text
session_id: int
visitor_id: int or string
item_id: original item id
item_idx: compact integer item id
timestamp: datetime
event_type: string
position: zero-based position within session
session_length: total length of session
```

### Item Metadata Processing

Create metadata if feasible:

```text
data/processed/item_metadata.parquet
```

Recommended minimal metadata fields:

```text
item_id
category_id
available_flag
```

Do not make metadata a dependency for the first set of recommender experiments. Use it later for qualitative analysis, such as whether sequence-aware models recommend items from more relevant categories.

## Phase 3: Train, Validation, and Test Splits

### Objective

Create leakage-safe session-based splits.

### Script

Create:

```text
scripts/build_splits.py
```

### Recommended Split Strategy

Use chronological session-level splitting by session start time:

```text
train: earliest 70% of sessions
validation: next 10% of sessions
test: latest 20% of sessions
```

Rationale:

- Recommender systems should be evaluated on future interactions.
- Random splitting can leak future item popularity and user behavior patterns into training.
- Session-level splitting prevents the same session from being split across train and test.

### Alternative Split for Robustness

If chronological splits create severe item cold-start issues, run an additional filtered evaluation:

```text
warm-start test set: remove test examples where the target item never appears in training
```

Report both:

1. `all_test`: all valid test examples.
2. `warm_test`: only examples with target items observed in training.

The main comparison should use `warm_test` if many models cannot score unseen items.

### Next-Item Prediction Examples

For each session with item sequence:

```text
[i1, i2, i3, ..., in]
```

Generate prediction examples using prefix-target pairs:

```text
context=[i1] target=i2
context=[i1, i2] target=i3
context=[i1, i2, i3] target=i4
...
context=[i1, ..., i(n-1)] target=in
```

This creates more examples than using only the final item as target and better tests how models use increasing context.

### Optional Final-Step Evaluation

Also compute a stricter session completion evaluation:

```text
context=[i1, ..., i(n-1)] target=in
```

Use this as a secondary metric table, not the primary experiment.

### Split Outputs

Save:

```text
data/processed/train.parquet
data/processed/validation.parquet
data/processed/test.parquet
```

Each split should contain event rows, not just examples. The evaluation script can generate prefix-target examples dynamically.

## Phase 4: Candidate Generation and Evaluation Protocol

### Objective

Evaluate all models under a consistent ranking task.

### Ranking Task

For each prediction example, a model receives the session prefix and returns a ranked list of candidate items.

Primary ranking size:

```text
K values: 5, 10, 20
```

### Candidate Item Universe

Use this default candidate universe:

```text
All items observed in the training set
```

Rationale:

- This gives a true ranking task over known items.
- It avoids unstable metrics caused by random negative sampling.

If full ranking over all items is too slow, use sampled evaluation as a fallback:

```text
1 positive target item + 100 sampled negative items
```

If sampled evaluation is used, generate one fixed negative sample file and reuse it for all models:

```text
data/processed/negative_samples.parquet
```

### Primary Metrics

Use ranking metrics:

```text
HitRate@K
Recall@K
MRR@K
NDCG@K
```

Recommended reporting:

1. `HitRate@10` as the main headline metric.
2. `Recall@20` as a broader retrieval metric.
3. `MRR@20` to reward higher-ranked correct predictions.
4. `NDCG@20` to reward rank-sensitive relevance.

For single-target next-item prediction, `HitRate@K` and `Recall@K` are equivalent if each example has exactly one relevant target. The report should state this clearly.

### Metric Definitions

For each example with one target item:

```text
HitRate@K = 1 if target appears in top K, otherwise 0
Recall@K = 1 if target appears in top K, otherwise 0
MRR@K = 1 / rank if target appears in top K, otherwise 0
NDCG@K = 1 / log2(rank + 1) if target appears in top K, otherwise 0
```

Average each metric across examples.

### Segment-Level Evaluation

Report all metrics by session prefix length:

```text
prefix length = 1
prefix length = 2
prefix length = 3-5
prefix length = 6+
```

Also report by original full session length:

```text
short sessions: length 2-3
medium sessions: length 4-6
long sessions: length 7+
```

This directly addresses whether sequence modeling helps more when more behavioral context is available.

## Phase 5: Models to Implement

Implement models in increasing complexity. Stop after the Markov and session-kNN models if time is limited; those are enough to answer the core research question.

## Model 1: Global Popularity Baseline

### Type

Unordered baseline with no personalization and no sequence information.

### Training

Count item frequencies in the training set.

Possible variants:

1. `popularity_all`: count all training interactions.
2. `popularity_recent`: count interactions with time decay, if time permits.

### Prediction

For every session prefix, recommend the globally most popular items not already in the session prefix.

### Purpose

This is the minimum baseline. Any useful recommender should beat this.

## Model 2: Unordered Item Co-Occurrence Recommender

### Type

Unordered session-based baseline.

### Training

Build an item-item co-occurrence matrix from training sessions. Two items co-occur if they appear in the same session, regardless of order.

Possible weighting:

```text
cooccurrence_score(i, j) = number of sessions containing both i and j
```

Better weighting if time permits:

```text
cosine_similarity(i, j) = cooccurrence(i, j) / sqrt(freq(i) * freq(j))
```

### Prediction

Given a session prefix, score candidate item `j` by aggregating similarities from all items in the prefix:

```text
score(j) = sum(similarity(i, j) for i in prefix_items)
```

Do not use the order of prefix items.

### Purpose

This is the strongest unordered baseline. It tests whether sequence-aware methods outperform a method that knows what items appeared in the session but not the order.

## Model 3: First-Order Markov Chain

### Type

Sequence-aware baseline.

### Training

Count ordered adjacent transitions in training sessions:

```text
i_t -> i_(t+1)
```

Estimate transition scores:

```text
P(next=j | current=i) = count(i -> j) / count(i -> *)
```

### Prediction

Given a session prefix, use only the most recent item:

```text
last_item = prefix[-1]
score(j) = P(j | last_item)
```

Fallback behavior:

1. If `last_item` has known transitions, use Markov scores.
2. If not, fall back to global popularity.

### Purpose

This tests whether the immediately previous item gives useful next-item signal.

## Model 4: Higher-Order or Weighted Markov Variant

### Type

Sequence-aware model using more than the most recent item.

### Training

Reuse first-order transition counts.

### Prediction

Score candidates using a recency-weighted sum over prefix items:

```text
score(j) = sum(decay^(distance_from_end) * P(j | item_i))
```

Recommended decay values to tune on validation:

```text
0.3, 0.5, 0.7, 0.9
```

### Purpose

This tests whether more ordered context helps beyond the last item.

## Model 5: Session-Based k-Nearest Neighbors

### Type

Sequence-aware or semi-sequence-aware nearest-neighbor method.

### Training

Store training sessions as item sequences.

### Similarity Options

Start with unordered Jaccard similarity:

```text
similarity(current_session, historical_session) = |intersection| / |union|
```

Then add a sequence-aware variant:

```text
Give larger weight to matches with recent prefix items.
```

Recommended sequence-aware weighting:

```text
weight(item at prefix position p) = decay^(last_position - p)
```

### Prediction

1. Find top `N` similar historical sessions.
2. Score items that occur after matched context items in those sessions.
3. Rank by weighted neighbor similarity.

Recommended hyperparameters:

```text
neighbors: 50, 100, 200
sample_size: 5000, 10000, 20000 historical sessions if full search is slow
decay: 0.5, 0.7, 0.9
```

### Purpose

Session-kNN is a strong non-neural session recommendation method and provides a meaningful comparison against Markov models.

## Model 6: Simple Neural Sequence Model

### Type

Sequence-aware neural model.

### Recommended Model

Use a small GRU-based next-item model if time and compute permit.

Architecture:

```text
item embedding -> GRU -> linear layer over item vocabulary
```

Training examples:

```text
input prefix: [i1, ..., i(t-1)]
target: i_t
```

Practical constraints:

1. Limit to the top `N` most frequent items if full vocabulary is too large.
2. Map rare items to an unknown token or filter sessions to frequent items.
3. Keep the first neural model intentionally small.

Suggested hyperparameters:

```text
embedding_dim: 64
hidden_dim: 64 or 128
batch_size: 128 or 256
max_sequence_length: 20
epochs: 5-10
early_stopping_metric: validation MRR@20
optimizer: Adam
learning_rate: 0.001
```

### Purpose

This tests whether a learned sequence model improves over simpler transition and neighbor methods. It is useful but not mandatory for the core final project.

## Phase 6: Hyperparameter Tuning

### Objective

Tune only on validation data, then evaluate once on test data.

### Hyperparameters by Model

Global popularity:

```text
exclude_seen: true
optional time_decay: none, 0.99, 0.95
```

Co-occurrence:

```text
similarity: raw_count, cosine
aggregation: sum, max
exclude_seen: true
```

Markov:

```text
smoothing: none, add_alpha
alpha: 0.01, 0.1, 1.0
fallback: popularity
```

Weighted Markov:

```text
decay: 0.3, 0.5, 0.7, 0.9
smoothing: none, add_alpha
```

Session-kNN:

```text
neighbors: 50, 100, 200
decay: none, 0.5, 0.7, 0.9
similarity: jaccard, recency_weighted_overlap
```

Neural sequence model:

```text
embedding_dim: 64, 128
hidden_dim: 64, 128
max_sequence_length: 10, 20
learning_rate: 0.001
```

### Selection Rule

Select the best variant for each model family using:

```text
validation MRR@20
```

Then report final performance on the test set.

## Phase 7: Main Experiment Matrix

The minimum publishable experiment matrix should include:

| Model | Uses Session Items? | Uses Order? | Uses Recency? | Required? |
|---|---:|---:|---:|---:|
| Global popularity | No | No | No | Yes |
| Unordered co-occurrence | Yes | No | No | Yes |
| First-order Markov | Yes | Yes | Last item only | Yes |
| Weighted Markov | Yes | Yes | Yes | Recommended |
| Session-kNN unordered | Yes | No | Optional | Recommended |
| Session-kNN sequence-aware | Yes | Yes | Yes | Recommended |
| GRU sequence model | Yes | Yes | Yes | Optional |

The core research question can be answered if the project implements at least:

1. Global popularity.
2. Unordered co-occurrence.
3. First-order Markov.
4. One stronger sequence-aware method, preferably weighted Markov or session-kNN.

## Phase 8: Statistical and Practical Comparison

### Objective

Determine whether observed improvements are meaningful, not just numerically different.

### Recommended Comparison

For each pair of models, compare per-example metric outcomes on the same test examples.

Useful paired comparisons:

```text
First-order Markov vs unordered co-occurrence
Weighted Markov vs unordered co-occurrence
Sequence-aware session-kNN vs unordered session-kNN
GRU vs best non-neural baseline
```

### Significance Testing

Use bootstrap confidence intervals:

1. Sample test examples with replacement.
2. Compute metric difference between two models.
3. Repeat 1,000 times.
4. Report 95% confidence interval.

If the confidence interval for the difference excludes zero, treat the improvement as statistically meaningful.

### Output Table

Create a table like:

| Comparison | Metric | Mean Difference | 95% CI Low | 95% CI High | Interpretation |
|---|---:|---:|---:|---:|---|
| Markov - Cooccurrence | HitRate@10 | TBD | TBD | TBD | TBD |

## Phase 9: Analysis Questions to Answer

The final report should answer these questions directly:

1. Which model performs best overall?
2. Does any sequence-aware model outperform the unordered co-occurrence baseline?
3. Is the improvement larger for longer sessions?
4. Is the first-order Markov model enough, or do longer-prefix models help?
5. Are gains consistent across `HitRate@K`, `MRR@K`, and `NDCG@K`?
6. How much of the performance is explained by global item popularity?
7. Are sequence-aware methods still useful for short sessions with only one or two context items?
8. What are the main failure cases?

## Phase 10: Result Artifacts

### Metrics Files

Save machine-readable metrics:

```text
outputs/metrics/popularity.json
outputs/metrics/cooccurrence.json
outputs/metrics/markov.json
outputs/metrics/weighted_markov.json
outputs/metrics/session_knn.json
outputs/metrics/neural_sequence.json
outputs/metrics/summary.csv
```

Each metrics file should include:

```text
model_name
model_config
split
candidate_protocol
num_examples
HitRate@5
HitRate@10
HitRate@20
Recall@5
Recall@10
Recall@20
MRR@20
NDCG@20
runtime_seconds
```

### Figures

Create:

```text
outputs/figures/main_metric_comparison.png
outputs/figures/metrics_by_prefix_length.png
outputs/figures/metrics_by_session_length.png
outputs/figures/bootstrap_confidence_intervals.png
```

### Tables

Create:

```text
outputs/tables/main_results.md
outputs/tables/results_by_prefix_length.md
outputs/tables/results_by_session_length.md
outputs/tables/statistical_comparisons.md
```

## Phase 11: Implementation Order for a Coding Agent

A coding agent should execute the project in this order:

1. Create base project structure.
2. Add `requirements.txt` with core dependencies.
3. Implement data audit script or notebook.
4. Write `documentation/1-data-audit.md` from the audit output.
5. Implement `scripts/preprocess_data.py`.
6. Validate processed sessions manually with small printed samples.
7. Implement `scripts/build_splits.py`.
8. Implement metric functions in `src/evaluation/metrics.py`.
9. Implement a shared evaluator in `src/evaluation/evaluator.py`.
10. Implement popularity baseline.
11. Implement unordered co-occurrence baseline.
12. Implement first-order Markov model.
13. Run validation evaluation for the three initial models.
14. Fix performance and correctness issues before adding more models.
15. Implement weighted Markov.
16. Implement session-kNN if runtime is feasible.
17. Add neural model only after non-neural experiments are complete.
18. Run final test evaluation once per selected model.
19. Generate result tables and figures.
20. Write final report around the actual evidence.

## Phase 12: Recommended Python Dependencies

Use a minimal dependency set:

```text
pandas
numpy
scipy
scikit-learn
pyarrow
matplotlib
seaborn
tqdm
```

Optional neural dependencies:

```text
torch
```

Optional experiment tracking:

```text
mlflow
```

Avoid adding heavyweight frameworks until the baseline pipeline is correct.

## Phase 13: Correctness Checks

Before trusting results, verify these conditions:

1. No test session events appear in training data.
2. No validation session events appear in training data.
3. The target item is not included in the context prefix.
4. Items already seen in the prefix are excluded from recommendations unless explicitly testing repeat behavior.
5. Every model receives the same prediction examples.
6. Every model uses the same candidate item universe.
7. Hyperparameters are selected on validation only.
8. Test metrics are computed once after model selection.
9. Cold-start target items are either handled consistently or filtered into a clearly named warm-start evaluation.
10. Random seeds are fixed for negative sampling, neural training, and any sampled session-kNN search.

## Phase 14: Performance Considerations

The Retailrocket files are large, especially item properties. Use these constraints:

1. Process `events.csv` first because it is the core experiment file.
2. Read item property files only after the main event pipeline works.
3. Use Parquet for processed data to avoid repeatedly parsing CSV.
4. Avoid dense item-item matrices; use sparse dictionaries or scipy sparse matrices.
5. For co-occurrence, cap very long sessions or use efficient pair generation.
6. For session-kNN, begin with a sampled candidate pool or inverted index from item to sessions.
7. For full ranking, precompute top recommendations where possible.
8. Profile runtime before implementing neural models.

## Phase 15: Final Report Outline

The final written report should use this structure:

1. Introduction and motivation.
2. Research question and hypothesis.
3. Dataset description.
4. Sessionization and preprocessing.
5. Experimental design.
6. Models compared.
7. Evaluation metrics.
8. Results.
9. Analysis by session length.
10. Statistical comparison.
11. Discussion of findings.
12. Limitations.
13. Future work.
14. Conclusion.

## Expected Outcome

The working hypothesis is:

```text
Sequence-aware methods will outperform unordered baselines for next-item prediction, especially for medium and long sessions where recent interaction order gives stronger intent signals.
```

The most likely result pattern is:

1. Popularity performs surprisingly well because e-commerce item popularity is highly skewed.
2. Co-occurrence beats popularity because session context matters.
3. First-order Markov beats or matches co-occurrence when the next item is strongly related to the immediately previous item.
4. Weighted Markov or sequence-aware session-kNN performs best among simple non-neural methods.
5. Neural sequence modeling may improve results, but only if enough data remains after filtering and the item vocabulary is handled carefully.

## Minimum Viable Final Project

If time becomes constrained, the project should still produce a complete answer with:

1. Clean sessionized Retailrocket view-event data.
2. Chronological train, validation, and test splits.
3. Popularity baseline.
4. Unordered co-occurrence baseline.
5. First-order Markov sequence-aware model.
6. Weighted Markov sequence-aware model.
7. HitRate@10, Recall@20, MRR@20, and NDCG@20.
8. Results by short, medium, and long sessions.
9. A clear conclusion about whether order helped.

This minimum version is enough to answer the proposed research question rigorously without overextending implementation scope.
