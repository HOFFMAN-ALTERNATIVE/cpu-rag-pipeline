from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from cpurag.evaluation import Query, load_queries
from cpurag.pipeline import Pipeline, read_corpus

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "examples" / "corpus"
QUERIES = ROOT / "examples" / "queries.jsonl"


@pytest.fixture(scope="session")
def docs() -> list[tuple[str, str]]:
    return read_corpus(CORPUS)


@pytest.fixture(scope="session")
def pipeline(docs: list[tuple[str, str]]) -> Pipeline:
    return Pipeline.from_documents(docs)


@pytest.fixture(scope="session")
def queries() -> list[Query]:
    return load_queries(QUERIES)


class FakeOpenAI(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible server: /v1/embeddings and /v1/chat/completions."""

    requests: list[dict[str, Any]] = []  # noqa: RUF012

    def log_message(self, *args: object) -> None:  # silence test output
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOpenAI.requests.append({"path": self.path, "auth": self.headers.get("Authorization"), **body})
        if self.path == "/v1/embeddings":
            data = [
                {"index": i, "embedding": [float(len(t)), 1.0, 0.0, 0.0]} for i, t in enumerate(body["input"])
            ]
            payload: dict[str, Any] = {"data": list(reversed(data))}  # out of order on purpose
        else:
            payload = {"choices": [{"message": {"content": " Use proxy_http_version 1.1 [1] [7] "}}]}
        out = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture()
def fake_openai() -> Iterator[str]:
    FakeOpenAI.requests = []
    srv = HTTPServer(("127.0.0.1", 0), FakeOpenAI)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()
    srv.server_close()
