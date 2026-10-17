import pytest
from conftest import FakeOpenAI

from cpurag.chunking import Chunk
from cpurag.embeddings import OpenAICompatibleEmbedder
from cpurag.generation import (
    SYSTEM_PROMPT,
    ExtractiveLLM,
    OpenAICompatibleLLM,
    build_prompt,
    citations,
)
from cpurag.pipeline import Pipeline
from cpurag.retrieval import Hit


def hit(doc: str, text: str, rank: int, section: str = "") -> Hit:
    return Hit(Chunk(doc, f"{doc}#0", text, 0, len(text), section), 1.0, rank)


def test_prompt_numbers_passages_and_respects_budget() -> None:
    hits = [hit("a", "alpha " * 50, 1, "Intro"), hit("b", "beta " * 50, 2), hit("c", "gamma " * 50, 3)]
    p = build_prompt("what?", hits, max_context_chars=700)
    assert "[1] a — Intro" in p and "[2] b" in p and "[3] c" not in p
    assert p.rstrip().endswith("Answer with citations:")
    assert "[1] a" in build_prompt("q", hits[:1], max_context_chars=10), "always keep at least one passage"


def test_citations_drop_hallucinated_numbers() -> None:
    assert citations("x [2] y [1][2] z [9] [0]", 3) == [2, 1]


def test_extractive_answer_is_grounded_and_cited() -> None:
    hits = [
        hit("nginx", "Use proxy_http_version 1.1 for websockets. Unrelated sentence here.", 1),
        hit("tls", "Certificates expire after ninety days.", 2),
    ]
    out = ExtractiveLLM(max_sentences=1).complete(
        SYSTEM_PROMPT, build_prompt("websockets proxy version", hits)
    )
    assert out == "Use proxy_http_version 1.1 for websockets. [1]"


def test_extractive_says_unknown_without_overlap() -> None:
    out = ExtractiveLLM().complete(
        SYSTEM_PROMPT, build_prompt("quantum chromodynamics", [hit("a", "bread.", 1)])
    )
    assert "don't know" in out


def test_pipeline_answer_cites_the_right_source(pipeline: Pipeline) -> None:
    ans = pipeline.answer("why does my websocket close right after the handshake behind nginx")
    assert ans.cited and ans.sources[ans.cited[0] - 1].chunk.doc_id == "nginx-reverse-proxy"
    assert "proxy_http_version" in ans.text


def test_openai_compatible_llm(fake_openai: str, pipeline: Pipeline) -> None:
    ans = pipeline.answer("websocket handshake", OpenAICompatibleLLM(fake_openai, "local-chat", api_key="t"))
    assert ans.text == "Use proxy_http_version 1.1 [1] [7]"
    assert ans.cited == [1]  # [7] does not exist and is discarded
    req = FakeOpenAI.requests[0]
    assert req["path"] == "/v1/chat/completions" and req["messages"][0]["content"] == SYSTEM_PROMPT
    assert req["auth"] == "Bearer t"


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://host/", "localhost:8080", "http://"])
def test_clients_reject_non_http_base_urls(url: str) -> None:
    with pytest.raises(ValueError, match="http"):
        OpenAICompatibleLLM(url, "m")
    with pytest.raises(ValueError, match="http"):
        OpenAICompatibleEmbedder(url, "m", 8)
