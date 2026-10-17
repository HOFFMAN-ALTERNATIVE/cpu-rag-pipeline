from pathlib import Path

import numpy as np
import pytest

from cpurag.embeddings import l2_normalize
from cpurag.store import FlatIndex, Int8Index

rng = np.random.default_rng(7)


def random_unit(n: int, d: int) -> np.ndarray:
    return l2_normalize(rng.standard_normal((n, d)).astype(np.float32))


def test_flat_search_matches_brute_force() -> None:
    vecs = random_unit(500, 64)
    idx = FlatIndex(64)
    idx.add([f"v{i}" for i in range(500)], vecs)
    q = random_unit(1, 64)[0]
    expected = np.argsort(-(vecs @ q))[:10]
    got = idx.search(q, 10)
    assert [g[0] for g in got] == [f"v{i}" for i in expected]
    assert got[0][1] >= got[-1][1]


def test_int8_keeps_ranking_quality_with_4x_less_memory() -> None:
    vecs = random_unit(2000, 128)
    ids = [f"v{i}" for i in range(2000)]
    flat, q8 = FlatIndex(128), Int8Index(128)
    flat.add(ids, vecs)
    q8.add(ids, vecs)
    recalls = []
    for q in random_unit(50, 128):
        exact = {i for i, _ in flat.search(q, 10)}
        approx = {i for i, _ in q8.search(q, 10)}
        recalls.append(len(exact & approx) / 10)
    assert np.mean(recalls) >= 0.95
    assert q8.nbytes <= vecs.nbytes / 3.8  # 1 byte/dim + one float scale per vector


def test_metadata_filter() -> None:
    idx = FlatIndex(8)
    v = random_unit(4, 8)
    idx.add(["a", "b", "c", "d"], v, [{"lang": "en"}, {"lang": "de"}, {"lang": "en"}, {"lang": "de"}])
    hits = idx.search(v[1], 4, where=lambda m: m["lang"] == "en")
    assert {h[0] for h in hits} == {"a", "c"}


def test_delete_and_k_bounds() -> None:
    for cls in (FlatIndex, Int8Index):
        idx = cls(8)
        v = random_unit(3, 8)
        idx.add(["a", "b", "c"], v)
        assert idx.delete(["b", "zzz"]) == 1
        assert len(idx) == 2
        assert idx.search(v[2], 10)[0][0] == "c"
        assert idx.search(v[0], 0) == []
        idx.delete(["a", "c"])
        assert idx.search(v[0], 3) == []


def test_rejects_bad_input() -> None:
    idx = FlatIndex(8)
    idx.add(["a"], random_unit(1, 8))
    with pytest.raises(ValueError, match="duplicate"):
        idx.add(["a"], random_unit(1, 8))
    with pytest.raises(ValueError, match="duplicate"):
        idx.add(["x", "x"], random_unit(2, 8))
    with pytest.raises(ValueError, match="expected"):
        idx.add(["b"], random_unit(1, 16))
    with pytest.raises(ValueError, match="mismatch"):
        idx.add(["b", "c"], random_unit(1, 8))


@pytest.mark.parametrize("cls", [FlatIndex, Int8Index])
def test_save_load_round_trip(tmp_path: Path, cls: type[FlatIndex]) -> None:
    v = random_unit(20, 16)
    idx = cls(16)
    idx.add([f"v{i}" for i in range(20)], v, [{"i": str(i)} for i in range(20)])
    idx.save(tmp_path)
    back = FlatIndex.load(tmp_path)
    assert type(back) is cls and back.meta[3] == {"i": "3"}
    assert back.search(v[5], 3) == idx.search(v[5], 3)
