import json
from pathlib import Path

import pytest
from conftest import CORPUS, QUERIES

from cpurag.cli import main
from cpurag.pipeline import Pipeline


def test_save_load_round_trip(tmp_path: Path, pipeline: Pipeline) -> None:
    pipeline.save(tmp_path)
    back = Pipeline.load(tmp_path)
    q = "rate limit shared across replicas"
    assert [h.chunk.chunk_id for h in back.retrieve(q)] == [h.chunk.chunk_id for h in pipeline.retrieve(q)]
    assert back.chunks == pipeline.chunks


def test_cli_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    idx = tmp_path / "idx"
    assert main(["ingest", str(CORPUS), "--index", str(idx), "--int8"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["documents"] == 12 and info["kind"] == "int8" and info["chunks"] > 12

    assert main(["query", "free certificates renewal", "--index", str(idx), "-k", "2"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith(" 1. tls-certificates") and len(out) == 2

    assert main(["ask", "how many threads for CPU token generation", "--index", str(idx)]) == 0
    out = capsys.readouterr().out
    assert "physical cores" in out and "[1] cpu-inference" in out

    assert main(["eval", str(QUERIES), "--index", str(idx), "-k", "3"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert set(rep) == {"bm25", "dense", "hybrid"} and rep["hybrid"]["hit@3"] >= 0.95


def test_cli_query_modes(tmp_path: Path, capsys: pytest.CaptureFixture[str], pipeline: Pipeline) -> None:
    pipeline.save(tmp_path)
    for mode in ("bm25", "dense"):
        assert main(["query", "ban ips after failed logins", "--index", str(tmp_path), "--mode", mode]) == 0
        assert "ssh-hardening" in capsys.readouterr().out.splitlines()[0]


def test_cli_ask_with_llm_server(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], pipeline: Pipeline, fake_openai: str
) -> None:
    pipeline.save(tmp_path)
    assert main(["ask", "websocket", "--index", str(tmp_path), "--llm-url", fake_openai]) == 0
    assert "proxy_http_version" in capsys.readouterr().out


def test_cli_empty_corpus(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ingest", str(tmp_path), "--index", str(tmp_path / "i")]) == 1
    assert "no documents" in capsys.readouterr().err
