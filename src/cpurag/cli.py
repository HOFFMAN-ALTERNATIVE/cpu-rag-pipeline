"""``cpurag`` command line: ingest, query, ask, eval."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from cpurag.evaluation import evaluate, load_queries
from cpurag.generation import OpenAICompatibleLLM
from cpurag.pipeline import Pipeline, read_corpus


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cpurag", description="Local CPU-only RAG pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="chunk + embed a folder of .md/.txt into an index")
    ing.add_argument("corpus", type=Path)
    ing.add_argument("--index", type=Path, default=Path(".cpurag"))
    ing.add_argument("--dim", type=int, default=2048)
    ing.add_argument("--int8", action="store_true", help="int8-quantised vectors (4x less RAM)")

    q = sub.add_parser("query", help="retrieve top passages")
    q.add_argument("question")
    q.add_argument("--index", type=Path, default=Path(".cpurag"))
    q.add_argument("-k", type=int, default=5)
    q.add_argument("--mode", choices=["hybrid", "bm25", "dense"], default="hybrid")

    a = sub.add_parser("ask", help="retrieve + generate a cited answer")
    a.add_argument("question")
    a.add_argument("--index", type=Path, default=Path(".cpurag"))
    a.add_argument("--llm-url", default="", help="OpenAI-compatible server; empty = extractive answer")
    a.add_argument("--llm-model", default="local")

    e = sub.add_parser("eval", help="retrieval metrics against a qrels JSONL")
    e.add_argument("queries", type=Path)
    e.add_argument("--index", type=Path, default=Path(".cpurag"))
    e.add_argument("-k", type=int, default=5)

    args = ap.parse_args(argv)
    if args.cmd == "ingest":
        docs = read_corpus(args.corpus)
        if not docs:
            print(f"no documents found under {args.corpus}", file=sys.stderr)
            return 1
        p = Pipeline.from_documents(docs, dim=args.dim, quantize=args.int8)
        p.save(args.index)
        print(
            json.dumps(
                {
                    "documents": len(docs),
                    "chunks": len(p.chunks),
                    "index": str(args.index),
                    "kind": p.index.kind,
                }
            )
        )
        return 0

    p = Pipeline.load(args.index)
    if args.cmd == "query":
        for h in p.retrieve(args.question, args.k, mode=args.mode):
            print(f"{h.rank:>2}. {h.chunk.doc_id} — {h.chunk.section or '(top)'}  [{h.score:.4f}]")
        return 0
    if args.cmd == "ask":
        llm = OpenAICompatibleLLM(args.llm_url, args.llm_model) if args.llm_url else None
        ans = p.answer(args.question, llm)
        print(ans.text)
        for i, h in enumerate(ans.sources, start=1):
            print(f"  [{i}] {h.chunk.doc_id}{' — ' + h.chunk.section if h.chunk.section else ''}")
        return 0
    queries = load_queries(args.queries)
    report = {
        m: evaluate(r, queries, k=args.k).as_dict()
        for m, r in (("bm25", p.bm25), ("dense", p.dense), ("hybrid", p.hybrid))
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
