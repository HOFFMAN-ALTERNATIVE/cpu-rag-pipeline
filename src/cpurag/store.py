"""In-process vector stores: exact float32 and memory-lean int8 (4x smaller)."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

Matrix = npt.NDArray[np.float32]
Filter = Callable[[dict[str, str]], bool]


class FlatIndex:
    """Exact cosine search over L2-normalised vectors (inner product)."""

    kind = "flat"

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.ids: list[str] = []
        self.meta: list[dict[str, str]] = []
        self._vecs = np.zeros((0, dim), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.ids)

    def _check(self, vecs: Matrix, ids: Sequence[str]) -> None:
        if vecs.ndim != 2 or vecs.shape[1] != self.dim:
            raise ValueError(f"expected (n, {self.dim}) vectors, got {vecs.shape}")
        if len(ids) != vecs.shape[0]:
            raise ValueError("ids and vectors length mismatch")
        dup = set(ids) & set(self.ids)
        if dup or len(set(ids)) != len(ids):
            raise ValueError(f"duplicate ids: {sorted(dup)[:3]}")

    def add(self, ids: Sequence[str], vecs: Matrix, meta: Sequence[dict[str, str]] | None = None) -> None:
        self._check(vecs, ids)
        self.ids.extend(ids)
        self.meta.extend(meta or [{} for _ in ids])
        self._store(vecs.astype(np.float32, copy=False))

    def _store(self, vecs: Matrix) -> None:
        self._vecs = np.vstack([self._vecs, vecs])

    def _scores(self, q: Matrix) -> npt.NDArray[np.float32]:
        return self._vecs @ q

    def search(self, query: Matrix, k: int = 10, where: Filter | None = None) -> list[tuple[str, float]]:
        """Top-k (id, score) by cosine similarity, optionally restricted by a metadata predicate."""
        if len(self) == 0 or k <= 0:
            return []
        q = query.reshape(-1).astype(np.float32)
        scores = self._scores(q).astype(np.float32)
        if where is not None:
            allowed = np.fromiter((where(m) for m in self.meta), dtype=bool, count=len(self))
            scores = np.where(allowed, scores, -np.inf).astype(np.float32)
        k = min(k, len(self))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(self.ids[i], float(scores[i])) for i in top if np.isfinite(scores[i])]

    def delete(self, ids: Sequence[str]) -> int:
        """Remove ids; returns how many were removed."""
        drop = set(ids)
        keep = [i for i, x in enumerate(self.ids) if x not in drop]
        removed = len(self.ids) - len(keep)
        self.ids = [self.ids[i] for i in keep]
        self.meta = [self.meta[i] for i in keep]
        self._take(np.asarray(keep, dtype=np.int64))
        return removed

    def _take(self, keep: npt.NDArray[np.int64]) -> None:
        self._vecs = self._vecs[keep] if len(keep) else np.zeros((0, self.dim), dtype=np.float32)

    # -- persistence -------------------------------------------------------
    def _arrays(self) -> dict[str, npt.NDArray[np.generic]]:
        return {"vecs": self._vecs}

    def _load_arrays(self, data: dict[str, npt.NDArray[np.generic]]) -> None:
        self._vecs = data["vecs"].astype(np.float32)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path / "vectors.npz", **self._arrays())  # type: ignore[arg-type]
        (path / "index.json").write_text(
            json.dumps({"kind": self.kind, "dim": self.dim, "ids": self.ids, "meta": self.meta})
        )

    @staticmethod
    def load(path: Path) -> FlatIndex:
        info = json.loads((path / "index.json").read_text())
        cls: type[FlatIndex] = {"flat": FlatIndex, "int8": Int8Index}[info["kind"]]
        idx = cls(info["dim"])
        idx.ids, idx.meta = info["ids"], info["meta"]
        with np.load(path / "vectors.npz", allow_pickle=False) as data:
            idx._load_arrays(dict(data))
        return idx


class Int8Index(FlatIndex):
    """Symmetric per-vector int8 scalar quantisation: 4x less RAM, ~same ranking quality.

    Each vector is stored as ``round(v / s * 127)`` with ``s = max|v|``; scores are computed
    in float32 from the int8 codes and rescaled, so no float copy of the corpus is kept.
    """

    kind = "int8"

    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self._codes = np.zeros((0, dim), dtype=np.int8)
        self._scale = np.zeros((0,), dtype=np.float32)

    def _store(self, vecs: Matrix) -> None:
        scale = np.abs(vecs).max(axis=1)
        scale[scale == 0] = 1.0
        codes = np.clip(np.rint(vecs / scale[:, None] * 127), -127, 127).astype(np.int8)
        self._codes = np.vstack([self._codes, codes])
        self._scale = np.concatenate([self._scale, (scale / 127).astype(np.float32)])

    def _scores(self, q: Matrix) -> npt.NDArray[np.float32]:
        return ((self._codes.astype(np.float32) @ q) * self._scale).astype(np.float32)

    def _take(self, keep: npt.NDArray[np.int64]) -> None:
        self._codes = self._codes[keep]
        self._scale = self._scale[keep]

    def _arrays(self) -> dict[str, npt.NDArray[np.generic]]:
        return {"codes": self._codes, "scale": self._scale}

    def _load_arrays(self, data: dict[str, npt.NDArray[np.generic]]) -> None:
        self._codes = data["codes"].astype(np.int8)
        self._scale = data["scale"].astype(np.float32)

    @property
    def nbytes(self) -> int:
        return int(self._codes.nbytes + self._scale.nbytes)
