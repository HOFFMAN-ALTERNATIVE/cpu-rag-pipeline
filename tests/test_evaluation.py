from pathlib import Path

import pytest

from cpurag.evaluation import (
    hit_at_k,
    load_queries,
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
)

REL = frozenset({"a", "c"})


def test_recall_and_hit() -> None:
    assert recall_at_k(["a", "b", "c"], REL, 2) == 0.5
    assert recall_at_k(["a", "b", "c"], REL, 3) == 1.0
    assert hit_at_k(["b", "x"], REL, 2) == 0.0
    assert recall_at_k(["a"], frozenset(), 3) == 0.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["b", "c", "a"], REL) == pytest.approx(0.5)
    assert reciprocal_rank(["x"], REL) == 0.0


def test_ndcg_perfect_and_partial() -> None:
    assert ndcg_at_k(["a", "c", "b"], REL, 3) == pytest.approx(1.0)
    # relevant at ranks 2 and 3: (1/log2(3) + 1/log2(4)) / (1 + 1/log2(3))
    assert ndcg_at_k(["b", "a", "c"], REL, 3) == pytest.approx((0.6309 + 0.5) / 1.6309, abs=1e-3)
    assert ndcg_at_k(["a"], frozenset(), 3) == 0.0


def test_percentile_nearest_rank() -> None:
    assert percentile([5, 1, 3, 2, 4], 50) == 3
    assert percentile(list(range(1, 101)), 95) == 95
    assert percentile([], 95) == 0.0


def test_load_queries(tmp_path: Path) -> None:
    p = tmp_path / "q.jsonl"
    p.write_text('{"qid":"1","query":"x","relevant":["a","b"]}\n\n')
    (q,) = load_queries(p)
    assert q.relevant == frozenset({"a", "b"}) and q.text == "x"
