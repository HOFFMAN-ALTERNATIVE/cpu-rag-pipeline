# 0003. Exact flat search with optional int8 quantisation
Status: accepted
Date: 2026-10-17

## Context
The vector store must be correct, simple to persist, and small enough for low-memory VPSes. Typical self-hosted corpora (team docs, runbooks) are thousands to tens of thousands of chunks.

## Options
1. An ANN index (HNSW, IVF). Sub-linear queries at scale. Approximate results, build cost, and another moving part to evaluate.
2. Exact inner-product search over a NumPy matrix, plus an `Int8Index` with per-vector scalar quantisation for about 4x less memory.

## Decision
Option 2. Exact results make retrieval metrics meaningful, and at this corpus size a matrix product is fast enough on one core.

## Consequences
- Query cost grows linearly with corpus size. Large corpora should swap in an ANN index behind the same store interface.
- Tests assert flat search equals brute-force NumPy and int8 keeps recall@10 at or above 0.95 against exact.
- Persistence is two files (`index.json`, `vectors.npz`), loaded with `allow_pickle=False`.
