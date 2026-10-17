# 0002. Reciprocal rank fusion for hybrid retrieval
Status: accepted
Date: 2026-10-17

## Context
BM25 and dense retrieval fail on different queries: BM25 on paraphrases, dense on rare exact terms. Combining them needs a fusion rule, and their scores live on unrelated scales.

## Options
1. Weighted sum of normalised scores. Can be tuned well. Needs per-corpus calibration; min-max normalisation is sensitive to outliers.
2. Learned re-ranker (cross-encoder). Best quality. A second model to run on CPU per query.
3. Weighted reciprocal rank fusion: score = sum of w / (k + rank). Uses ranks only, no calibration.

## Decision
Option 3. It needs no calibration data and adds microseconds, which fits a CPU-only, zero-download pipeline.

## Consequences
- Tests check RRF scores against the formula and that hybrid is never worse than either component on the bundled set.
- On the bundled 24 queries all three retrievers tie, so the benchmark does not show hybrid winning; that needs a harder query set.
- A cross-encoder re-ranking stage can be added after fusion if a deployment can afford it.
