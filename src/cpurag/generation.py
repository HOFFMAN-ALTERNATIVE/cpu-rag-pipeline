"""Grounded answer generation: citation-numbered prompts and pluggable LLM backends."""

from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Sequence
from typing import Protocol

from cpurag._net import http_base_url
from cpurag.retrieval import Hit
from cpurag.text import terms

SYSTEM_PROMPT = (
    "You answer strictly from the numbered context passages. Cite passages like [1] or [2][3]. "
    "If the context does not contain the answer, say you don't know. Never invent sources."
)


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


def build_prompt(question: str, hits: Sequence[Hit], *, max_context_chars: int = 6000) -> str:
    """Pack the best passages into a budget, numbered for citation."""
    parts: list[str] = []
    used = 0
    for i, h in enumerate(hits, start=1):
        header = f"[{i}] {h.chunk.doc_id}" + (f" — {h.chunk.section}" if h.chunk.section else "")
        block = f"{header}\n{h.chunk.text}\n"
        if used + len(block) > max_context_chars and parts:
            break
        parts.append(block)
        used += len(block)
    return "Context:\n" + "\n".join(parts) + f"\nQuestion: {question}\nAnswer with citations:"


_CITATION = re.compile(r"\[(\d+)\]")


def citations(answer: str, n_passages: int) -> list[int]:
    """Citation numbers used in an answer that point at real passages (hallucinated ones dropped)."""
    seen: list[int] = []
    for m in _CITATION.finditer(answer):
        i = int(m.group(1))
        if 1 <= i <= n_passages and i not in seen:
            seen.append(i)
    return seen


class ExtractiveLLM:
    """Model-free fallback: returns the context sentences that best overlap the question.

    Useful offline, in tests, and as a "does retrieval alone answer this?" baseline.
    """

    def __init__(self, max_sentences: int = 2) -> None:
        self.max_sentences = max_sentences

    def complete(self, system: str, user: str) -> str:
        question = user.rsplit("Question:", 1)[-1]
        q = set(terms(question))
        scored: list[tuple[float, int, str]] = []
        for m in re.finditer(r"^\[(\d+)\][^\n]*\n(.*?)(?=^\[\d+\]|\nQuestion:)", user, re.S | re.M):
            idx = int(m.group(1))
            body = "\n".join(ln for ln in m.group(2).splitlines() if not ln.lstrip().startswith("#"))
            for sent in re.split(r"(?<=[.!?])\s+", body.strip()):
                overlap = len(q & set(terms(sent)))
                if overlap:
                    scored.append((overlap / (len(set(terms(sent))) ** 0.5 or 1), idx, sent.strip()))
        if not scored:
            return "I don't know based on the provided context."
        scored.sort(key=lambda t: (-t[0], t[1]))
        return " ".join(f"{s} [{i}]" for _, i, s in scored[: self.max_sentences])


class OpenAICompatibleLLM:
    """Chat client for any ``/v1/chat/completions`` server (llama.cpp, vLLM, LocalAI, …)."""

    def __init__(
        self, base_url: str, model: str, *, api_key: str = "", temperature: float = 0.1, timeout: float = 120
    ) -> None:
        self.base_url, self.model = http_base_url(base_url), model
        self.api_key, self.temperature, self.timeout = api_key, temperature, timeout

    def complete(self, system: str, user: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "temperature": self.temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            }
        ).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/v1/chat/completions", data=body, headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.load(resp)
        return str(payload["choices"][0]["message"]["content"]).strip()
