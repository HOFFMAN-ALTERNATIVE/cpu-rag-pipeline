# 0001. Feature hashing as the default embedder
Status: accepted
Date: 2026-10-17

## Context
The pipeline targets CPU-only servers and must be testable in CI with exact, reproducible results. Neural embedding models need a download of hundreds of megabytes, a runtime (ONNX, PyTorch or a server), and produce floating-point results that drift across library versions.

## Options
1. Bundle a small neural model (for example a MiniLM via ONNX Runtime). Better semantic matching. Heavy dependencies, model download in CI, non-bit-exact results.
2. Signed feature hashing of stemmed unigrams, bigrams and character trigrams with sublinear TF and corpus IDF as the default, plus an `Embedder` protocol and an OpenAI-compatible `/v1/embeddings` client for real models.

## Decision
Option 2. A zero-download deterministic baseline keeps the evaluation reproducible bit for bit, and the protocol lets users plug in a neural model without code changes.

## Consequences
- Paraphrases without shared vocabulary are missed by the default embedder; BM25 and the hybrid ranker partly compensate.
- CI can gate on exact retrieval metrics because nothing in the pipeline is random.
- Teams that need semantic recall pass an `OpenAICompatibleEmbedder` (any local `/v1/embeddings` server) to the pipeline from Python and re-run the same evaluation to compare. The CLI does not expose this yet.
