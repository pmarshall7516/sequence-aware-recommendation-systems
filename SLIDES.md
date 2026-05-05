# Sequence-Aware Recommendation Systems

- Research question: Do sequence-aware models outperform unordered models for within-session next-item prediction?
- Dataset: Retailrocket e-commerce behavior data.
- Task: Given a session prefix, rank items and predict the next item.
- Main conclusion: Sequence matters most in longer sessions, but unordered session-kNN is the strongest overall model.

# Problem and Hypothesis

- Problem: Recommend the next item a visitor will interact with during the same browsing session.
- Hypothesis: Preserving item order should improve next-item prediction over unordered context methods.
- Expected effect: sequence-aware gains should be larger for longer prefixes and longer sessions.
- Why it matters: better next-item ranking can improve product discovery and session-level personalization.

# Background: Why Sessions Matter

- Retailrocket provides timestamped visitor-item interactions, not explicit sessions.
- Sessions were constructed using a 30-minute inactivity threshold.
- Only sessions with at least two view events were kept.
- View events were used because they dominate the dataset and provide enough sequential behavior.
- Visual: `outputs/figures/audit_event_type_counts.png`.

# Data Summary

- Raw events: 2,756,101.
- Raw visitors: 1,407,580.
- Raw unique items: 235,061.
- Processed sessions: 260,955.
- Processed interactions: 952,622.
- Average session length: 3.65; median session length: 2.
- Visual: `outputs/figures/audit_session_length_and_item_popularity.png`.

# Experimental Design

- Chronological session-level split prevents temporal leakage.
- Train: earliest 70% of sessions.
- Validation: next 10% of sessions.
- Test: latest 20% of sessions.
- Test set: 52,192 sessions and 129,200 next-item prediction examples.
- Candidate universe: 100,472 training items.

# Evaluation Framework

- Each session creates prefix-target examples.
- Example: `[i1, i2, i3]` creates `[i1] -> i2` and `[i1, i2] -> i3`.
- Metrics: HitRate@5/10/20, Recall@5/10/20, MRR@5/10/20, NDCG@5/10/20.
- Headline metric: HitRate@10.
- Segment analysis: prefix length and full session length buckets.

# Models Compared

- Popularity: recommends globally frequent items.
- Co-occurrence: recommends items that appear in similar sessions, ignoring order.
- Markov: recommends items that commonly follow the last item.
- Weighted Markov: uses multiple previous items with recency weighting.
- Session-kNN unordered: finds similar historical sessions by item overlap.
- Session-kNN sequence: session-kNN with recency-weighted context.
- Neural GRU model exists in code, but no final neural test artifact is present.

# Headline Results

- Best overall: session-kNN unordered with 17.46% HitRate@10.
- Second: session-kNN sequence with 17.15% HitRate@10.
- Best transition model: weighted Markov with 16.05% HitRate@10.
- Co-occurrence baseline: 14.02% HitRate@10.
- Popularity baseline: 0.56% HitRate@10.
- Visual: `outputs/figures/baseline_metric_comparison.png`.

# Results Table

- session-kNN unordered: 17.46% HitRate@10, 21.67% HitRate@20, 0.0870 MRR@20.
- session-kNN sequence: 17.15% HitRate@10, 21.26% HitRate@20, 0.0860 MRR@20.
- weighted Markov: 16.05% HitRate@10, 18.82% HitRate@20, 0.0853 MRR@20.
- Markov: 14.45% HitRate@10, 16.15% HitRate@20, 0.0795 MRR@20.
- Co-occurrence: 14.02% HitRate@10, 18.05% HitRate@20, 0.0697 MRR@20.
- Popularity: 0.56% HitRate@10, 0.80% HitRate@20, 0.0012 MRR@20.

# Family-Level Finding

- Unordered model mean: 10.68% HitRate@10.
- Sequence-aware model mean: 15.88% HitRate@10.
- Absolute gain: +5.20 percentage points.
- Relative gain: +48.7%.
- Interpretation: the broad sequence-aware hypothesis is supported at the family level.
- Caveat: the weak popularity baseline makes the unordered family average lower.

# Prefix-Length Findings

- Prefix 1 winner: session-kNN unordered at 25.35% HitRate@10.
- Prefix 2 winner: session-kNN unordered at 14.74% HitRate@10.
- Prefix 3-5 winner: session-kNN unordered at 15.33% HitRate@10.
- Prefix 6+ winner: weighted Markov at 8.93% HitRate@10.
- Interpretation: order-aware models become most valuable when longer ordered context exists.
- Visual: `outputs/figures/analysis_hitrate10_by_prefix_bucket.png`.

# Session-Length Findings

- Short sessions 2-3: session-kNN unordered wins with 23.49% HitRate@10.
- Medium sessions 4-6: session-kNN unordered wins with 18.32% HitRate@10.
- Long sessions 7+: session-kNN sequence wins with 11.68% HitRate@10.
- Weighted Markov also improves strongly in long sessions at 11.29% HitRate@10.
- Interpretation: long sessions provide the clearest evidence that sequence order matters.
- Visual: `outputs/figures/analysis_hitrate10_by_session_bucket.png`.

# Model Interpretation

- Popularity fails because the catalog is large and session intent is specific.
- Co-occurrence shows that session context is necessary.
- Markov shows that the previous item helps, but last-item-only context is limited.
- Weighted Markov shows that recency-weighted sequence context improves ranking.
- Session-kNN unordered shows that historical session similarity is very strong.
- Session-kNN sequence shows that order helps most in longer sessions.

# Did Results Match the Hypothesis?

- Yes: sequence-aware models outperform unordered models on family averages.
- Yes: weighted Markov beats co-occurrence overall and in long-session segments.
- Yes: sequence-aware session-kNN beats unordered session-kNN for sessions length 7+.
- No: sequence-aware session-kNN does not beat unordered session-kNN overall.
- Final verdict: mostly supported, but the strongest claim is segment-specific rather than universal.

# Practical Takeaways

- Session context is much more useful than global popularity.
- Simple, interpretable models perform well on this task.
- Recency weighting is useful for longer sessions.
- A hybrid recommender would make sense: unordered session-kNN for short sessions, sequence-aware scoring for longer sessions.
- Neural models should only be discussed as future work unless final neural results are generated.

# Limitations

- No bootstrap confidence intervals or significance tests were run.
- Cold-start test targets are included, so some targets may be impossible to recommend from the training candidate set.
- The primary experiment uses view events only.
- Runtime numbers are not optimized production benchmarks.
- Neural sequence results are not included in the current output artifacts.

# Final Conclusion

- The experiment answers the research question with a nuanced result.
- Sequence-aware methods improve over simple unordered baselines and help most in longer sessions.
- The best overall model is unordered session-kNN, showing that strong session similarity can capture much of the intent signal.
- The most defensible conclusion: sequence information is valuable, but its benefit depends on model design and session length.

# Delivery Checklist

- Start with the research question and hypothesis.
- Explain sessionization before model results.
- Use HitRate@10 as the main metric and define it simply.
- Point directly to the best model and the hypothesis nuance.
- Spend extra time on the prefix/session segment visuals.
- End with what was learned, limitations, and future work.

