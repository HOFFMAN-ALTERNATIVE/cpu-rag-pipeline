import pytest

from cpurag.chunking import Chunk
from cpurag.embeddings import HashingEmbedder
from cpurag.retrieval import BM25Retriever, DenseRetriever, Hit, HybridRetriever, dedupe_by_doc


def ch(doc: str, i: int, text: str, section: str = "") -> Chunk:
    return Chunk(doc, f"{doc}#{i}", text, 0, len(text), section)


CHUNKS = [
    ch("nginx", 0, "proxy_buffering off for streaming responses", "Timeouts"),
    ch("nginx", 1, "websocket upgrade headers for the proxy", "WebSockets"),
    ch("tls", 0, "renew certificates automatically with acme", "Issuance"),
    ch("ssh", 0, "disable password authentication and use keys", "Keys"),
]


class Fixed:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids

    def search(self, query: str, k: int = 10) -> list[Hit]:
        by_id = {c.chunk_id: c for c in CHUNKS}
        return [Hit(by_id[i], 1.0, r) for r, i in enumerate(self.ids[:k], start=1)]


def test_bm25_ranks_matching_terms_and_uses_sections() -> None:
    bm = BM25Retriever(CHUNKS)
    assert bm.search("acme certificate renewal")[0].chunk.doc_id == "tls"
    assert bm.search("websockets")[0].chunk.chunk_id == "nginx#1"  # matched via section title + stem
    assert bm.search("kubernetes") == []


def test_bm25_prefers_shorter_docs_for_equal_tf() -> None:
    short = ch("a", 0, "redis cluster")
    long = ch("b", 0, "redis " + "filler words " * 30)
    assert BM25Retriever([long, short]).search("redis")[0].chunk.doc_id == "a"


def test_dense_retriever_and_filter() -> None:
    d = DenseRetriever.build(HashingEmbedder(512), CHUNKS)
    assert d.search("password login keys")[0].chunk.doc_id == "ssh"
    only_nginx = d.search("certificates", 4, where=lambda m: m["doc_id"] == "nginx")
    assert {h.chunk.doc_id for h in only_nginx} == {"nginx"}


def test_rrf_fusion_math() -> None:
    h = HybridRetriever([Fixed(["tls#0", "ssh#0"]), Fixed(["ssh#0", "nginx#0"])], rrf_k=60)
    res = h.search("q", 3)
    assert [r.chunk.chunk_id for r in res] == ["ssh#0", "tls#0", "nginx#0"]
    assert res[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert [r.rank for r in res] == [1, 2, 3]


def test_rrf_weights() -> None:
    h = HybridRetriever([Fixed(["tls#0"]), Fixed(["ssh#0"])], weights=[1.0, 3.0])
    assert h.search("q")[0].chunk.chunk_id == "ssh#0"
    with pytest.raises(ValueError):
        HybridRetriever([Fixed([])], weights=[1.0, 2.0])


def test_dedupe_by_doc_keeps_best_chunk_per_doc() -> None:
    hits = Fixed(["nginx#1", "nginx#0", "tls#0", "ssh#0"]).search("q")
    out = dedupe_by_doc(hits, 2)
    assert [h.chunk.chunk_id for h in out] == ["nginx#1", "tls#0"]
    assert [h.rank for h in out] == [1, 2]
