"""Lexical (BM25), dense and hybrid (Reciprocal Rank Fusion) retrievers."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from cpurag.chunking import Chunk
from cpurag.embeddings import Embedder
from cpurag.store import Filter, FlatIndex
from cpurag.text import terms


@dataclass(frozen=True, slots=True)
class Hit:
    chunk: Chunk
    score: float
    rank: int


class Retriever(Protocol):
    def search(self, query: str, k: int = 10) -> list[Hit]: ...


def _hits(chunks: dict[str, Chunk], scored: Sequence[tuple[str, float]]) -> list[Hit]:
    return [Hit(chunks[cid], s, r) for r, (cid, s) in enumerate(scored, start=1)]


class BM25Retriever:
    """Okapi BM25 over stemmed terms, with the section heading folded into each chunk."""

    def __init__(self, chunks: Sequence[Chunk], *, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.chunks = {c.chunk_id: c for c in chunks}
        self._tf: dict[str, Counter[str]] = {}
        self._len: dict[str, int] = {}
        self._postings: dict[str, list[str]] = defaultdict(list)
        for c in chunks:
            tf = Counter(terms(f"{c.section} {c.text}"))
            self._tf[c.chunk_id] = tf
            self._len[c.chunk_id] = sum(tf.values())
            for t in tf:
                self._postings[t].append(c.chunk_id)
        n = len(chunks)
        self._avg = (sum(self._len.values()) / n) if n else 0.0
        self._idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self._postings.items()}

    def search(self, query: str, k: int = 10) -> list[Hit]:
        scores: dict[str, float] = defaultdict(float)
        for t in set(terms(query)):
            idf = self._idf.get(t)
            if idf is None:
                continue
            for cid in self._postings[t]:
                tf = self._tf[cid][t]
                norm = self.k1 * (1 - self.b + self.b * self._len[cid] / (self._avg or 1))
                scores[cid] += idf * tf * (self.k1 + 1) / (tf + norm)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return _hits(self.chunks, ranked)


class DenseRetriever:
    """Embedding search over a vector index."""

    def __init__(self, embedder: Embedder, index: FlatIndex, chunks: Sequence[Chunk]) -> None:
        self.embedder, self.index = embedder, index
        self.chunks = {c.chunk_id: c for c in chunks}

    @staticmethod
    def passage(c: Chunk) -> str:
        return f"{c.section}\n{c.text}" if c.section else c.text

    @classmethod
    def build(
        cls, embedder: Embedder, chunks: Sequence[Chunk], index: FlatIndex | None = None
    ) -> DenseRetriever:
        index = index if index is not None else FlatIndex(embedder.dim)
        if chunks:
            vecs = embedder.embed([cls.passage(c) for c in chunks])
            index.add(
                [c.chunk_id for c in chunks], vecs, [{"doc_id": c.doc_id, **c.metadata} for c in chunks]
            )
        return cls(embedder, index, chunks)

    def search(self, query: str, k: int = 10, where: Filter | None = None) -> list[Hit]:
        q = self.embedder.embed([query])[0]
        return _hits(self.chunks, self.index.search(q, k, where))


class HybridRetriever:
    """Reciprocal Rank Fusion of several retrievers: score = Σ wᵢ / (rrf_k + rankᵢ).

    RRF needs no score calibration between BM25 and cosine, which is why it is the
    default fusion in most production search stacks.
    """

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        *,
        weights: Sequence[float] | None = None,
        rrf_k: int = 60,
        depth: int = 50,
    ) -> None:
        if weights is not None and len(weights) != len(retrievers):
            raise ValueError("one weight per retriever")
        self.retrievers = list(retrievers)
        self.weights = list(weights) if weights is not None else [1.0] * len(retrievers)
        self.rrf_k, self.depth = rrf_k, depth

    def search(self, query: str, k: int = 10) -> list[Hit]:
        fused: dict[str, float] = defaultdict(float)
        chunks: dict[str, Chunk] = {}
        for w, r in zip(self.weights, self.retrievers, strict=True):
            for h in r.search(query, self.depth):
                fused[h.chunk.chunk_id] += w / (self.rrf_k + h.rank)
                chunks[h.chunk.chunk_id] = h.chunk
        ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return _hits(chunks, ranked)


def dedupe_by_doc(hits: Sequence[Hit], k: int) -> list[Hit]:
    """Keep the best chunk per document (diversifies the context window)."""
    seen: set[str] = set()
    out: list[Hit] = []
    for h in hits:
        if h.chunk.doc_id in seen:
            continue
        seen.add(h.chunk.doc_id)
        out.append(Hit(h.chunk, h.score, len(out) + 1))
        if len(out) == k:
            break
    return out
