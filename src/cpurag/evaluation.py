"""Retrieval evaluation: recall@k, hit-rate@k, MRR, nDCG@k and latency percentiles."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from cpurag.retrieval import Hit, Retriever


@dataclass(frozen=True, slots=True)
class Query:
    qid: str
    text: str
    relevant: frozenset[str]  # relevant doc ids


def load_queries(path: Path) -> list[Query]:
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            out.append(Query(d["qid"], d["query"], frozenset(d["relevant"])))
    return out


def _ranked_docs(hits: Sequence[Hit]) -> list[str]:
    seen: list[str] = []
    for h in hits:
        if h.chunk.doc_id not in seen:
            seen.append(h.chunk.doc_id)
    return seen


def recall_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def hit_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: frozenset[str]) -> float:
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, math.ceil(p / 100 * len(s)) - 1))
    return s[idx]


@dataclass
class Report:
    k: int
    n: int
    recall: float
    hit_rate: float
    mrr: float
    ndcg: float
    p50_ms: float
    p95_ms: float
    misses: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, float | int | list[str]]:
        return {
            "k": self.k,
            "queries": self.n,
            f"recall@{self.k}": round(self.recall, 4),
            f"hit@{self.k}": round(self.hit_rate, 4),
            "mrr": round(self.mrr, 4),
            f"ndcg@{self.k}": round(self.ndcg, 4),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "misses": self.misses,
        }


def evaluate(retriever: Retriever, queries: Sequence[Query], k: int = 5, depth: int = 20) -> Report:
    """Run every query, score at document level (any chunk of a relevant doc counts)."""
    rec, hit, rr, nd, lat = [], [], [], [], []
    misses = []
    for q in queries:
        t0 = time.perf_counter()
        hits = retriever.search(q.text, depth)
        lat.append((time.perf_counter() - t0) * 1000)
        ranked = _ranked_docs(hits)
        rec.append(recall_at_k(ranked, q.relevant, k))
        hit.append(hit_at_k(ranked, q.relevant, k))
        rr.append(reciprocal_rank(ranked, q.relevant))
        nd.append(ndcg_at_k(ranked, q.relevant, k))
        if not hit[-1]:
            misses.append(q.qid)
    n = len(queries) or 1
    return Report(
        k=k,
        n=len(queries),
        recall=sum(rec) / n,
        hit_rate=sum(hit) / n,
        mrr=sum(rr) / n,
        ndcg=sum(nd) / n,
        p50_ms=percentile(lat, 50),
        p95_ms=percentile(lat, 95),
        misses=misses,
    )
