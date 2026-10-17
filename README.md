# cpu-rag-pipeline

[![ci](https://github.com/HOFFMAN-ALTERNATIVE/cpu-rag-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/HOFFMAN-ALTERNATIVE/cpu-rag-pipeline/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13-3776AB?logo=python&logoColor=white)
![mypy](https://img.shields.io/badge/mypy-strict-blue)
![coverage](https://img.shields.io/badge/coverage-99%25-brightgreen)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**Retrieval-augmented generation that runs entirely on a CPU — and proves its retrieval quality in CI.**

Structure-aware chunking → embeddings → vector store → **hybrid BM25 + dense retrieval fused with RRF** →
citation-grounded answers from any OpenAI-compatible local model server. The only runtime dependency is NumPy;
the default embedder needs no model download, so the whole pipeline (and its evaluation) is reproducible
bit-for-bit on any laptop or CI runner.

Designed and built solo as a reference for teams that want private, self-hosted RAG without a GPU bill.

## Architecture

```mermaid
flowchart LR
  subgraph ingest[Ingest]
    docs[(.md / .txt corpus)] --> chunk[chunker<br/>heading-aware · overlap · exact offsets]
    chunk --> emb[embedder<br/>hashing trick + IDF<br/>or /v1/embeddings]
    emb --> store[(vector store<br/>FlatIndex float32 · Int8Index 4x smaller)]
    chunk --> bm25[(BM25 inverted index)]
  end
  subgraph query[Query]
    q[question] --> dense[dense top-50]
    q --> lex[BM25 top-50]
    dense & lex --> rrf[Reciprocal Rank Fusion]
    rrf --> dedupe[best chunk per doc]
    dedupe --> prompt[numbered context<br/>char budget]
    prompt --> llm[LLM: OpenAI-compatible server<br/>or extractive fallback]
    llm --> ans[answer + validated citations]
  end
  store --> dense
  bm25 --> lex
  subgraph eval[Evaluation · CI gate]
    qrels[(queries.jsonl)] --> metrics[hit@k · recall@k · MRR · nDCG@k · p50/p95 latency]
  end
  rrf -.-> metrics
```

## Features

| Stage | Implementation | Why it matters |
|---|---|---|
| Chunking | Markdown section tree → paragraph > sentence > word cut points, word-aligned overlap, exact `start/end` offsets | A chunk is about one topic; offsets enable source highlighting |
| Embeddings | Signed feature hashing of stemmed unigrams, bigrams and char-trigrams, sublinear TF, corpus IDF; pluggable `Embedder` protocol + `/v1/embeddings` client | Zero downloads, deterministic, typo-tolerant baseline; swap in a neural model without code changes |
| Vector store | Exact inner-product `FlatIndex`; `Int8Index` with per-vector scalar quantisation; metadata filters, deletes, `npz` persistence | int8 keeps ≥95% recall@10 at ~4x less RAM — matters on small VPSes |
| Retrieval | Okapi BM25 (section titles folded in), dense, and weighted **RRF** hybrid; per-document de-duplication | RRF needs no score calibration and beats either retriever alone on mixed queries |
| Generation | Budgeted, numbered context; system prompt enforcing grounded answers; citation validator drops hallucinated `[n]`; extractive fallback | Answers are auditable; works offline |
| Evaluation | hit@k, recall@k, MRR, nDCG@k, latency percentiles, per-query misses | Retrieval quality is a tested contract, not a vibe |

## Quick start

```bash
pip install -e .
cpurag ingest examples/corpus --index .cpurag          # add --int8 for the quantised store
cpurag query "pods restart when the database is down" --index .cpurag
cpurag ask   "how many threads for CPU token generation" --index .cpurag
cpurag ask   "..." --llm-url http://127.0.0.1:8080 --llm-model local   # any OpenAI-compatible server
cpurag eval  examples/queries.jsonl --index .cpurag -k 3
```

```python
from pathlib import Path
from cpurag.pipeline import Pipeline, read_corpus

rag = Pipeline.from_documents(read_corpus(Path("docs")), quantize=True)
ans = rag.answer("why does my websocket close after the handshake?")
print(ans.text, [ans.sources[i - 1].chunk.doc_id for i in ans.cited])
```

## Retrieval benchmark (bundled corpus, 24 paraphrased queries, CPU)

| retriever | hit@3 | recall@3 | MRR | nDCG@3 | p95 latency |
|---|---|---|---|---|---|
| BM25 | 1.00 | 0.98 | 1.00 | 0.98 | < 0.1 ms |
| dense (hashing) | 1.00 | 0.98 | 1.00 | 0.98 | < 1 ms |
| **hybrid RRF** | **1.00** | **0.98** | **1.00** | **0.98** | < 1 ms |

The CI `retrieval-eval` job re-runs this on every commit, publishes the table to the job summary and fails if
hybrid hit@3 drops below 0.95. Bring your own `queries.jsonl` (`{"qid", "query", "relevant": [doc ids]}`)
to benchmark your corpus or compare a neural embedder against the baseline.

## Quality bar

- **56 tests**, **99% branch coverage** (gate: 90%), `mypy --strict`, ruff, Python 3.11–3.13 matrix
- property-style checks: chunk offsets always map back to the source; flat search equals brute-force NumPy;
  int8 recall@10 ≥ 0.95 vs exact; RRF scores match the formula; hybrid never worse than its components
- HTTP clients tested against an in-process fake OpenAI-compatible server (ordering, auth header, shape checks)

## License

MIT © Herman Hoffman
