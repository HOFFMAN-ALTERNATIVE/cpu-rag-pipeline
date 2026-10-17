"""End-to-end pipeline: ingest → chunk → embed → index → hybrid retrieve → grounded answer."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from cpurag.chunking import Chunk, chunk_text
from cpurag.embeddings import HashingEmbedder
from cpurag.generation import LLM, SYSTEM_PROMPT, ExtractiveLLM, build_prompt, citations
from cpurag.retrieval import BM25Retriever, DenseRetriever, Hit, HybridRetriever, Retriever, dedupe_by_doc
from cpurag.store import FlatIndex, Int8Index

SUFFIXES = (".md", ".markdown", ".txt", ".rst")


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[Hit]
    cited: list[int]


def read_corpus(root: Path) -> list[tuple[str, str]]:
    """(doc_id, text) for every text file under root; doc_id is the relative path without suffix."""
    docs = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUFFIXES:
            docs.append((p.relative_to(root).with_suffix("").as_posix(), p.read_text(encoding="utf-8")))
    return docs


class Pipeline:
    def __init__(self, chunks: Sequence[Chunk], embedder: HashingEmbedder, index: FlatIndex) -> None:
        self.chunks = list(chunks)
        self.embedder = embedder
        self.index = index
        self.dense = DenseRetriever(embedder, index, self.chunks)
        self.bm25 = BM25Retriever(self.chunks)
        self.hybrid = HybridRetriever([self.bm25, self.dense])

    @classmethod
    def from_documents(
        cls,
        docs: Iterable[tuple[str, str]],
        *,
        dim: int = 2048,
        quantize: bool = False,
        max_chars: int = 800,
        overlap: int = 120,
    ) -> Pipeline:
        chunks = [
            c for doc_id, text in docs for c in chunk_text(doc_id, text, max_chars=max_chars, overlap=overlap)
        ]
        embedder = HashingEmbedder(dim).fit(DenseRetriever.passage(c) for c in chunks)
        index: FlatIndex = Int8Index(dim) if quantize else FlatIndex(dim)
        DenseRetriever.build(embedder, chunks, index)
        return cls(chunks, embedder, index)

    def retrieve(self, question: str, k: int = 5, *, mode: str = "hybrid") -> list[Hit]:
        table: dict[str, Retriever] = {"hybrid": self.hybrid, "bm25": self.bm25, "dense": self.dense}
        retriever = table[mode]
        return dedupe_by_doc(retriever.search(question, k * 4), k)

    def answer(self, question: str, llm: LLM | None = None, k: int = 4) -> Answer:
        hits = self.retrieve(question, k)
        text = (llm or ExtractiveLLM()).complete(SYSTEM_PROMPT, build_prompt(question, hits))
        return Answer(text=text, sources=hits, cited=citations(text, len(hits)))

    # -- persistence -------------------------------------------------------
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.index.save(path)
        with (path / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(asdict(c)) + "\n")
        (path / "embedder.json").write_text(json.dumps({"dim": self.embedder.dim, "idf": self.embedder.idf}))

    @classmethod
    def load(cls, path: Path) -> Pipeline:
        chunks = [Chunk(**json.loads(line)) for line in (path / "chunks.jsonl").read_text().splitlines()]
        cfg = json.loads((path / "embedder.json").read_text())
        emb = HashingEmbedder(cfg["dim"])
        emb.idf = cfg["idf"]
        return cls(chunks, emb, FlatIndex.load(path))
