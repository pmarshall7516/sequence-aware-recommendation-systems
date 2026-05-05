# Results Review

## Executive Summary

This project tests the question:

> Do sequence-aware recommendation models outperform unordered recommendation models for next-item prediction within the same e-commerce session?

The answer from the current test outputs is: **partially yes, but with an important nuance**.

Across model families, the average sequence-aware result is better than the average unordered result. Sequence-aware models average **15.88% HitRate@10**, while the unordered family averages **10.68% HitRate@10**, a **+5.20 percentage-point** absolute gain and **+48.7% relative gain**. This supports the broad hypothesis that session order carries useful signal.

However, the strongest individual model is **session_knn_unordered**, not a sequence-aware model. It reaches **17.46% HitRate@10**, slightly ahead of **session_knn_sequence** at **17.15% HitRate@10**. This means the hypothesis should not be stated as "every sequence-aware model beats every unordered model." A more accurate conclusion is:

> Sequence information improves next-item recommendation when compared against simple unordered baselines and is especially useful in long-context sessions, but a strong unordered nearest-neighbor session model remains highly competitive overall.

The clearest evidence for the sequence hypothesis comes from the segment analysis:

- **Weighted Markov beats co-occurrence in every session-length bucket** and its gains grow as sessions get longer.
- **Session-kNN sequence beats session-kNN unordered in long sessions**: **11.68% vs. 10.48% HitRate@10** for sessions with length 7+.
- **Weighted Markov is the best model for prefix length 6+**, reaching **8.93% HitRate@10**, just ahead of sequence-aware session-kNN at **8.83%**.

## Data and Experiment Setup

### Dataset

The experiment uses the Retailrocket e-commerce behavior dataset.

Raw data summary:

| Quantity | Value |
|---|---:|
| Raw events | 2,756,101 |
| Unique visitors | 1,407,580 |
| Unique items | 235,061 |
| View events | 2,664,312 |
| Add-to-cart events | 69,332 |
| Transaction events | 22,457 |

The main experiment uses **view events only**. This is reasonable because views are the densest signal and produce enough sequential examples for next-item prediction.

### Session Construction

Sessions were created by sorting events by visitor and timestamp, then splitting visitor histories after **30 minutes of inactivity**. Sessions shorter than two events were removed because they cannot create a next-item prediction example.

Processed data summary:

| Quantity | Value |
|---|---:|
| Processed interaction rows | 952,622 |
| Processed sessions | 260,955 |
| Processed unique items | 121,344 |
| Average session length | 3.65 |
| Median session length | 2 |
| 95th percentile session length | 9 |

This tells us the dataset is large but sparse. Most sessions are short, and the item catalog is large. That combination makes global recommendation difficult and explains why popularity performs poorly.

### Chronological Split

The split is chronological at the session level, which prevents future sessions from leaking into training.

| Split | Rows | Sessions | Unique items | Date range |
|---|---:|---:|---:|---|
| Train | 679,881 | 182,668 | 100,472 | 2015-05-03 to 2015-07-31 |
| Validation | 91,349 | 26,095 | 33,073 | 2015-07-31 to 2015-08-17 |
| Test | 181,392 | 52,192 | 50,188 | 2015-08-17 to 2015-09-18 |

The test set produces **129,200 prefix-target examples**. Models rank over the **100,472 training items**. The current run uses `warm_start_only=False`, so cold-start test targets are included even if they do not appear in the training candidate universe. This makes the task harder and more realistic, but it also depresses absolute metric values.

### Evaluation Task

Each session is converted into next-item examples:

| Session | Prediction examples |
|---|---|
| `[i1, i2, i3, i4]` | `[i1] -> i2`, `[i1, i2] -> i3`, `[i1, i2, i3] -> i4` |

Metrics are computed at K = 5, 10, and 20:

- **HitRate@K**: whether the true next item appears in the top K recommendations.
- **Recall@K**: same value as HitRate@K here because each example has one target item.
- **MRR@K**: rewards placing the true item near the top of the ranked list.
- **NDCG@K**: also rewards higher-ranked hits, with a logarithmic rank discount.

The most presentation-friendly headline metric is **HitRate@10** because it answers: "Was the true next item in the top 10 recommendations?"

## Models Evaluated

### Popularity Baseline

**What it does:** Recommends globally popular items from the training set, ignoring the current session.

**What it tests:** Whether broad item popularity alone is enough to predict the next click.

**Result:** Popularity performs extremely poorly: **0.56% HitRate@10** and **0.80% HitRate@20**.

**Interpretation:** A generic "most viewed items" list does not work for next-item prediction in this dataset. The item catalog is too large and user intent is too session-specific. This baseline is still useful because it establishes that the task cannot be solved by popularity alone.

### Co-occurrence Baseline

**What it does:** Treats the session prefix as an unordered set of items and recommends items that commonly appeared in the same training sessions. It uses cosine-normalized co-occurrence.

**What it tests:** Whether item relatedness helps without using event order.

**Result:** Co-occurrence reaches **14.02% HitRate@10** and **18.05% HitRate@20**.

**Interpretation:** Session context matters a lot. Moving from popularity to co-occurrence adds **+13.45 percentage points** of HitRate@10. This shows that previous items in the session are highly informative, even before modeling order.

### First-Order Markov

**What it does:** Uses adjacent item transitions from training sessions. At prediction time, it mainly uses the last item in the prefix and recommends items that often followed that item.

**What it tests:** Whether the immediately previous item is enough sequence information to improve next-item ranking.

**Result:** Markov reaches **14.45% HitRate@10**, narrowly beating co-occurrence by **+0.43 percentage points**.

**Interpretation:** The immediate last item contains useful signal, but the first-order model is too limited. It improves over co-occurrence overall and for prefix length 1 and 6+, but it underperforms co-occurrence for prefix length 2 and 3-5. This suggests that using only the last click can throw away useful broader session context.

### Weighted Markov

**What it does:** Uses transition evidence from multiple context items, with recent items weighted more heavily through a decay factor of 0.7.

**What it tests:** Whether preserving order and recency across the full prefix improves over both unordered context and last-item-only sequence modeling.

**Result:** Weighted Markov reaches **16.05% HitRate@10**, **18.82% HitRate@20**, **0.0853 MRR@20**, and **0.1087 NDCG@20**.

**Interpretation:** This is the strongest transition-based sequence model. It beats co-occurrence by **+2.03 percentage points HitRate@10** (**+14.5% relative**) and beats first-order Markov by **+1.61 percentage points HitRate@10**. This is strong evidence that recency-weighted sequence context is useful.

### Session-kNN Unordered

**What it does:** Finds historical sessions that share items with the current prefix using set-style overlap, then recommends items from the nearest sessions. It does not apply recency weighting to the prefix.

**What it tests:** Whether nearest-neighbor session similarity can capture user intent without explicit ordering.

**Result:** This is the best overall model: **17.46% HitRate@10**, **21.67% HitRate@20**, **0.0870 MRR@20**, and **0.1161 NDCG@20**.

**Interpretation:** Similar historical sessions are very powerful. The unordered kNN model beats co-occurrence by **+3.44 percentage points HitRate@10** (**+24.5% relative**) and slightly beats the sequence-aware session-kNN variant overall. This is the main result that complicates the hypothesis.

### Session-kNN Sequence

**What it does:** Uses the same nearest-neighbor session idea as session-kNN unordered, but applies recency weighting to the current prefix so recent context items matter more.

**What it tests:** Whether adding order sensitivity to a strong nearest-neighbor session model improves recommendation quality.

**Result:** It reaches **17.15% HitRate@10**, **21.26% HitRate@20**, **0.0860 MRR@20**, and **0.1144 NDCG@20**.

**Interpretation:** It is the second-best overall model and only **0.31 percentage points** behind unordered session-kNN at HitRate@10. It loses overall because most sessions are short, but it wins in long sessions. For sessions of length 7+, sequence-aware session-kNN gets **11.68% HitRate@10**, compared with **10.48%** for unordered session-kNN. This supports the more specific hypothesis that order matters more when there is enough context to model.

### Neural Sequence Model

**What it does:** The codebase includes a GRU-based next-item model that learns item embeddings and a recurrent sequence representation.

**Current output status:** No `neural_sequence_test.json` output is present in the current `outputs/metrics/` directory, so it was not part of the final test results reviewed here.

**Presentation guidance:** Mention it as implemented but optional/not included in the final run unless a completed neural output is generated before the presentation. Do not claim neural results without an artifact.

## Headline Results

Sorted by HitRate@10:

| Model | HitRate@5 | HitRate@10 | HitRate@20 | MRR@20 | NDCG@20 | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| session_knn_unordered | 13.11% | 17.46% | 21.67% | 0.0870 | 0.1161 | 44.4 min |
| session_knn_sequence | 12.97% | 17.15% | 21.26% | 0.0860 | 0.1144 | 42.3 min |
| weighted_markov | 12.63% | 16.05% | 18.82% | 0.0853 | 0.1087 | 144.8 min |
| markov | 11.76% | 14.45% | 16.15% | 0.0795 | 0.0985 | 49.7 min |
| cooccurrence | 10.48% | 14.02% | 18.05% | 0.0697 | 0.0944 | 465.9 min |
| popularity | 0.22% | 0.56% | 0.80% | 0.0012 | 0.0027 | 65.7 min |

Key observations:

1. **Popularity is not competitive.** It confirms that session context is necessary.
2. **Co-occurrence is a strong simple baseline.** It captures item relatedness without order.
3. **Markov only slightly beats co-occurrence overall.** Last-item transitions are useful but incomplete.
4. **Weighted Markov is the best pure sequence transition model.** Multi-item recency weighting matters.
5. **Session-kNN models are strongest overall.** Historical session similarity is the most useful modeling strategy in this run.
6. **The sequence-aware kNN variant does not beat unordered kNN overall.** The difference is small, but it matters for the conclusion.

## Family-Level Comparison

Using the project's model grouping:

| Metric | Unordered mean | Sequence-aware mean | Absolute gain | Relative gain |
|---|---:|---:|---:|---:|
| HitRate@10 | 10.68% | 15.88% | +5.20 pts | +48.7% |
| Recall@20 | 13.51% | 18.74% | +5.23 pts | +38.8% |
| MRR@20 | 0.0526 | 0.0836 | +0.0309 | +58.8% |
| NDCG@20 | 0.0711 | 0.1072 | +0.0361 | +50.8% |

This supports the broad project hypothesis, but the unordered family includes the very weak popularity baseline. If popularity is removed and only contextual unordered models are compared against sequence-aware models, the conclusion is more nuanced:

| Metric | Contextual unordered mean | Sequence-aware mean | Interpretation |
|---|---:|---:|---|
| HitRate@10 | 15.74% | 15.88% | Sequence-aware is only slightly higher |
| HitRate@20 | 19.86% | 18.74% | Contextual unordered is higher |
| MRR@20 | 0.0783 | 0.0836 | Sequence-aware ranks hits higher |
| NDCG@20 | 0.1052 | 0.1072 | Sequence-aware is slightly higher |

The best interpretation is that sequence awareness improves ranking precision and rank quality, but strong session-neighbor methods can capture much of the same signal without explicit sequence modeling.

## Segment Analysis by Prefix Length

HitRate@10 by prefix length:

| Model | Prefix 1 | Prefix 2 | Prefix 3-5 | Prefix 6+ |
|---|---:|---:|---:|---:|
| popularity | 0.72% | 0.40% | 0.45% | 0.51% |
| cooccurrence | 20.04% | 11.43% | 12.60% | 6.01% |
| markov | 21.66% | 10.91% | 11.00% | 6.96% |
| weighted_markov | 21.66% | 13.29% | 14.72% | 8.93% |
| session_knn_unordered | 25.35% | 14.74% | 15.33% | 6.64% |
| session_knn_sequence | 23.79% | 14.03% | 15.31% | 8.83% |

Interpretation:

- For **prefix 1**, the unordered session-kNN model is best. With only one item of context, there is little sequence structure to exploit.
- For **prefix 2 and 3-5**, session-kNN unordered remains slightly best, but weighted Markov is much stronger than plain Markov.
- For **prefix 6+**, weighted Markov and sequence-aware session-kNN become the best models. This is the clearest prefix-level evidence for the hypothesis.
- Co-occurrence drops from **20.04%** at prefix 1 to **6.01%** at prefix 6+, while sequence-aware methods retain more value in the longest prefixes.

The expected pattern appears in long prefixes: as more ordered context becomes available, explicit sequence models become more useful.

## Segment Analysis by Full Session Length

HitRate@10 by full session length:

| Model | Session 2-3 | Session 4-6 | Session 7+ |
|---|---:|---:|---:|
| popularity | 0.57% | 0.51% | 0.59% |
| cooccurrence | 18.66% | 14.42% | 8.82% |
| markov | 19.60% | 14.19% | 9.15% |
| weighted_markov | 20.13% | 16.69% | 11.29% |
| session_knn_unordered | 23.49% | 18.32% | 10.48% |
| session_knn_sequence | 22.08% | 17.50% | 11.68% |

Interpretation:

- Short sessions are easiest because their next item is often closely tied to one or two initial clicks.
- Performance declines for longer sessions across all models, which suggests long sessions are more diverse and harder to predict.
- The sequence-aware advantage is strongest in long sessions:
  - Weighted Markov beats co-occurrence by **+2.47 percentage points** in session length 7+.
  - Sequence-aware session-kNN beats unordered session-kNN by **+1.20 percentage points** in session length 7+.
- This directly supports the hypothesis that order helps most when the session contains enough behavior for order to matter.

## Hypothesis Verdict

### Original hypothesis

Sequence-aware recommendation approaches should outperform unordered recommendation methods for next-item prediction, especially for longer sessions where event order carries stronger intent.

### Verdict

**Mostly supported, with nuance.**

Supported:

- Sequence-aware models beat unordered models on family averages.
- Weighted Markov clearly beats the unordered co-occurrence baseline.
- Sequence-aware models are strongest in long-prefix and long-session segments.
- Sequence-aware models have better rank-sensitive averages than contextual unordered models, especially MRR@20.

Not fully supported:

- The best single model is unordered session-kNN.
- Sequence-aware session-kNN does not beat unordered session-kNN overall.
- Plain first-order Markov only slightly improves over co-occurrence and loses in some prefix buckets.

Final presentation claim:

> The results show that sequence matters, but the amount of benefit depends on the model and the amount of available session context. Recency-weighted sequence models are most valuable for longer sessions, while nearest-neighbor session similarity is the strongest overall approach on this dataset.

## Visuals to Use

Use these existing artifacts in the presentation:

1. `outputs/figures/audit_event_type_counts.png`
   - Use when explaining why the experiment focuses on view events.
   - Key point: views dominate the dataset.

2. `outputs/figures/audit_session_length_and_item_popularity.png`
   - Use for background.
   - Key point: sessions are short and item popularity is sparse/long-tailed.

3. `outputs/figures/baseline_metric_comparison.png`
   - Use for the main model comparison slide.
   - Key point: session-kNN models lead overall; popularity is far behind.

4. `outputs/figures/analysis_hitrate10_by_prefix_bucket.png`
   - Use for hypothesis testing.
   - Key point: sequence-aware methods become stronger in the longest prefix bucket.

5. `outputs/figures/analysis_hitrate10_by_session_bucket.png`
   - Use for discussion.
   - Key point: long-session results provide the strongest evidence for order-aware modeling.

## Practical Significance

For an e-commerce recommender, the results suggest:

- A generic popularity recommender is not enough for within-session recommendations.
- Item-to-item relatedness is a strong minimum viable baseline.
- Simple sequence models can improve ranking quality without neural complexity.
- Weighted Markov is a good practical compromise because it is interpretable and directly supports the hypothesis.
- Session-kNN is the strongest overall approach in this experiment, but the sequence-aware version is better for long sessions.
- The best production strategy may be hybrid: use unordered session-neighbor similarity for short sessions and sequence-aware recency weighting for longer sessions.

## Limitations and Caveats

1. **No statistical significance testing is included.** Differences are clear in some comparisons, but bootstrap confidence intervals would make the conclusion stronger.
2. **Cold-start targets are included.** Because ranking is over the training item universe and `warm_start_only=False`, some test targets may be impossible to recommend. This lowers absolute scores.
3. **Neural results are not present.** The GRU model exists in code but does not have a final output artifact in the reviewed run.
4. **Only view events are used.** Add-to-cart and transaction events are much sparser and were not part of the primary experiment.
5. **Runtime numbers are evaluation-pipeline dependent.** They are useful for relative discussion, but not optimized benchmark claims.

## Presentation-Ready Conclusion

The experiment successfully answers the research question. Session context is essential: popularity achieves only **0.56% HitRate@10**, while context-aware models reach **14-17%**. Sequence awareness improves results over simple unordered baselines and is most useful when sessions are long enough for order to provide meaningful intent. The strongest overall model is unordered session-kNN, so the final conclusion should be nuanced: sequence-aware models are valuable, especially for long sessions, but strong unordered session similarity remains a very competitive approach for this dataset.

