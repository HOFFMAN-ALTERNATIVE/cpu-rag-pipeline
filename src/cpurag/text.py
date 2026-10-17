"""Tokenisation shared by the lexical and hashing components."""

from __future__ import annotations

import re
import unicodedata

_TOKEN = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")

# small, domain-neutral English stop list — enough to keep BM25 / hashing focused
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
        "we",
        "our",
        "not",
        "no",
        "yes",
        "if",
        "else",
        "than",
        "so",
        "such",
    ]
)


def normalize(text: str) -> str:
    """NFKC-normalise and lowercase."""
    return unicodedata.normalize("NFKC", text).lower()


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Split into lowercase word tokens; keeps dotted/hyphenated identifiers (e.g. ``nginx.conf``)."""
    toks = _TOKEN.findall(normalize(text))
    if drop_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return toks


def stem(token: str) -> str:
    """Tiny suffix stripper (not Porter) — cheap recall boost for plurals and -ing/-ed forms."""
    for suf in ("ingly", "edly", "ing", "ies", "ed", "es", "s"):
        if len(token) > len(suf) + 2 and token.endswith(suf):
            if suf == "ies":
                return token[: -len(suf)] + "y"
            return token[: -len(suf)]
    return token


def terms(text: str) -> list[str]:
    """Tokens after stop-word removal and stemming."""
    return [stem(t) for t in tokenize(text)]
