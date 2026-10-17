"""Shared checks for the HTTP clients."""

from __future__ import annotations

from urllib.parse import urlsplit


def http_base_url(url: str) -> str:
    """Return ``url`` without a trailing slash, or raise if it is not plain http(s).

    ``urllib`` also opens ``file://`` and ``ftp://`` URLs, so an unchecked base URL from a
    config file or CLI flag could read local files instead of calling a model server.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError(f"base URL must be http:// or https:// with a host, got {url!r}")
    return url.rstrip("/")
