"""Retrieval quality gates on the bundled corpus — a regression here fails CI."""

import pytest

from cpurag.evaluation import Query, evaluate
from cpurag.pipeline import Pipeline


@pytest.mark.parametrize(
    ("mode", "min_hit", "min_mrr"), [("bm25", 0.9, 0.85), ("dense", 0.9, 0.85), ("hybrid", 0.95, 0.9)]
)
def test_quality_gates(
    pipeline: Pipeline, queries: list[Query], mode: str, min_hit: float, min_mrr: float
) -> None:
    r = evaluate(getattr(pipeline, mode), queries, k=3)
    assert r.n == len(queries) >= 20
    assert r.hit_rate >= min_hit, r.as_dict()
    assert r.mrr >= min_mrr, r.as_dict()
    assert r.ndcg >= min_mrr - 0.05, r.as_dict()


def test_hybrid_is_never_worse_than_either_component(pipeline: Pipeline, queries: list[Query]) -> None:
    rep = {m: evaluate(getattr(pipeline, m), queries, k=3) for m in ("bm25", "dense", "hybrid")}
    assert rep["hybrid"].mrr >= min(rep["bm25"].mrr, rep["dense"].mrr)
    assert rep["hybrid"].recall >= max(rep["bm25"].recall, rep["dense"].recall) - 0.05


def test_int8_index_matches_float_quality(
    docs: list[tuple[str, str]], queries: list[Query], pipeline: Pipeline
) -> None:
    q8 = Pipeline.from_documents(docs, quantize=True)
    assert q8.index.kind == "int8"
    assert abs(evaluate(q8.dense, queries, k=3).mrr - evaluate(pipeline.dense, queries, k=3).mrr) <= 0.03


def test_typo_and_paraphrase_robustness(pipeline: Pipeline) -> None:
    assert (
        pipeline.retrieve("kubernets readines probe", 1, mode="dense")[0].chunk.doc_id == "kubernetes-probes"
    )
    assert pipeline.retrieve("keep three copies offsite", 1)[0].chunk.doc_id == "backups"


def test_cpu_latency_budget(pipeline: Pipeline, queries: list[Query]) -> None:
    r = evaluate(pipeline.hybrid, queries, k=3)
    assert r.p95_ms < 50, f"hybrid p95 {r.p95_ms:.2f} ms on CPU"
