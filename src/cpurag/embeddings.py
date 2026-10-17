"""Embedders. The default is dependency-free and deterministic; real models plug in via the protocol."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import urllib.request
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from cpurag._net import http_base_url
from cpurag.text import terms

Matrix = npt.NDArray[np.float32]


@runtime_checkable
class Embedder(Protocol):
    """Anything that maps texts to L2-normalised float32 vectors of a fixed dimension."""

    @property
    def dim(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> Matrix: ...


def l2_normalize(m: Matrix) -> Matrix:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out: Matrix = (m / norms).astype(np.float32, copy=False)
    return out


def _h(feature: str) -> int:
    # stable across processes (Python's hash() is salted), fast enough for CPU ingest
    return int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")


class HashingEmbedder:
    """Feature-hashed sparse-to-dense embedder (the "hashing trick").

    Features: stemmed unigrams, word bigrams and character trigrams of each term (robust to
    typos and inflections). Signed hashing keeps collisions unbiased, sublinear TF dampens
    repetition, and an optional IDF fitted on the corpus down-weights ubiquitous terms.
    No model download, no GPU, bit-for-bit reproducible — ideal as a baseline and for CI.
    """

    def __init__(self, dim: int = 2048, *, char_ngrams: bool = True) -> None:
        if dim < 64 or dim & (dim - 1):
            raise ValueError("dim must be a power of two >= 64")
        self._dim = dim
        self.char_ngrams = char_ngrams
        self.idf: dict[str, float] = {}

    @property
    def dim(self) -> int:
        return self._dim

    def features(self, text: str) -> Counter[str]:
        ts = terms(text)
        feats: Counter[str] = Counter(f"w:{t}" for t in ts)
        feats.update(f"b:{a}_{b}" for a, b in itertools.pairwise(ts))
        if self.char_ngrams:
            for t in ts:
                padded = f"<{t}>"
                feats.update(f"c:{padded[i : i + 3]}" for i in range(len(padded) - 2))
        return feats

    def fit(self, corpus: Iterable[str]) -> HashingEmbedder:
        """Learn smoothed IDF weights from the corpus (call once at ingest time)."""
        df: Counter[str] = Counter()
        n = 0
        for doc in corpus:
            n += 1
            df.update(set(self.features(doc)))
        self.idf = {f: math.log((1 + n) / (1 + c)) + 1.0 for f, c in df.items()}
        return self

    def embed(self, texts: Sequence[str]) -> Matrix:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        mask = self._dim - 1
        for row, text in enumerate(texts):
            for feat, tf in self.features(text).items():
                h = _h(feat)
                weight = (1.0 + math.log(tf)) * self.idf.get(feat, 1.0)
                if feat.startswith("c:"):
                    weight *= 0.35  # subword evidence is supportive, not decisive
                out[row, h & mask] += weight if (h >> 63) & 1 else -weight
        return l2_normalize(out)


class OpenAICompatibleEmbedder:
    """Client for any ``/v1/embeddings`` endpoint (llama.cpp server, text-embeddings-inference, etc.)."""

    def __init__(
        self, base_url: str, model: str, dim: int, *, api_key: str = "", timeout: float = 30
    ) -> None:
        self.base_url = http_base_url(base_url)
        self.model = model
        self._dim = dim
        self.api_key = api_key
        self.timeout = timeout

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> Matrix:
        body = json.dumps({"model": self.model, "input": list(texts)}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/v1/embeddings", data=body, headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.load(resp)
        rows = sorted(payload["data"], key=lambda d: d["index"])
        m = np.asarray([r["embedding"] for r in rows], dtype=np.float32)
        if m.shape != (len(texts), self._dim):
            raise ValueError(f"endpoint returned shape {m.shape}, expected {(len(texts), self._dim)}")
        return l2_normalize(m)
