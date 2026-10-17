"""Structure-aware chunking with overlap and exact source offsets."""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable unit of text. ``start``/``end`` index into the original document."""

    doc_id: str
    chunk_id: str
    text: str
    start: int
    end: int
    section: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


def _sections(text: str) -> list[tuple[str, int, int]]:
    """Split markdown into (heading path, start, end) spans; plain text is one section."""
    marks = list(_HEADING.finditer(text))
    if not marks:
        return [("", 0, len(text))]
    spans: list[tuple[str, int, int]] = []
    if marks[0].start() > 0:
        spans.append(("", 0, marks[0].start()))
    stack: list[tuple[int, str]] = []
    for i, m in enumerate(marks):
        level, title = len(m.group(1)), m.group(2)
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        spans.append((" > ".join(t for _, t in stack), m.start(), end))
    return spans


def _split_points(text: str, start: int, end: int, max_chars: int) -> list[int]:
    """Choose cut points inside [start, end): prefer paragraph, then sentence, then word boundaries."""
    cuts = [start]
    pos = start
    while end - pos > max_chars:
        window = text[pos : pos + max_chars]
        cut = window.rfind("\n\n")
        if cut < max_chars // 2:
            ends = [m.end() for m in _SENTENCE_END.finditer(window)]
            cut = ends[-1] if ends and ends[-1] >= max_chars // 2 else -1
        if cut < max_chars // 2:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = max_chars  # pathological: one giant token
        pos += cut
        cuts.append(pos)
    cuts.append(end)
    return cuts


def chunk_text(
    doc_id: str,
    text: str,
    *,
    max_chars: int = 800,
    overlap: int = 120,
    min_chars: int = 40,
    metadata: dict[str, str] | None = None,
) -> list[Chunk]:
    """Chunk a document by markdown section, then by size with overlap.

    * never crosses a section boundary, so a chunk is about one topic;
    * prefers paragraph > sentence > word boundaries when it must cut;
    * each chunk is prefixed by nothing — the section path is kept in ``Chunk.section``
      and offsets let callers highlight the exact source span.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap must be in [0, max_chars)")
    meta = dict(metadata or {})
    chunks: list[Chunk] = []
    for section, s_start, s_end in _sections(text):
        cuts = _split_points(text, s_start, s_end, max_chars - overlap if overlap else max_chars)
        for a, b in itertools.pairwise(cuts):
            lo = max(s_start, a - overlap) if chunks and a != s_start else a
            # extend the left edge to a word boundary so overlap never starts mid-word
            if lo != a:
                sp = text.find(" ", lo, a)
                lo = sp + 1 if sp != -1 else a
            raw = text[lo:b]
            stripped = raw.strip()
            if len(stripped) < min_chars:
                continue
            lead = len(raw) - len(raw.lstrip())
            start = lo + lead
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}#{len(chunks)}",
                    text=stripped,
                    start=start,
                    end=start + len(stripped),
                    section=section,
                    metadata=meta,
                )
            )
    return chunks
