"""
HTTP GET tool — fetch a public URL and return truncated text.

Educational network I/O with guardrails: timeout, size cap, http(s) only.
Not a full browser — HTML is returned mostly raw so the model can skim it.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import requests

from core.tool_registry import Tool

_MAX_BYTES = 50_000
_TIMEOUT = 15


def http_get(url: str, max_chars: int = 4000) -> str:
    """GET a URL and return status + truncated response body.

    Parameters
    ----------
    url:
        Absolute ``http://`` or ``https://`` URL.
    max_chars:
        How much of the body to keep in the observation (default 4000).
    """
    url = (url or "").strip()
    if not url:
        return "ERROR: empty url"

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "ERROR: url must be absolute http(s), e.g. https://example.com"

    max_chars = max(200, min(int(max_chars), 12_000))

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "VisualAgentHarness/0.1 (educational fetch)"},
            timeout=_TIMEOUT,
            stream=True,
            allow_redirects=True,
        )
        # Read at most _MAX_BYTES to avoid huge downloads
        chunks: list[bytes] = []
        total = 0
        for chunk in resp.iter_content(chunk_size=4096):
            if not chunk:
                continue
            chunks.append(chunk)
            total += len(chunk)
            if total >= _MAX_BYTES:
                break
        raw = b"".join(chunks)
        # Prefer utf-8; fall back so binary-ish pages still stringify
        text = raw.decode(resp.encoding or "utf-8", errors="replace")
    except requests.RequestException as exc:
        return f"ERROR: request failed: {exc}"

    final_url = str(resp.url)
    content_type = resp.headers.get("Content-Type", "unknown")
    body = text[:max_chars]
    truncated = len(text) > max_chars or total >= _MAX_BYTES

    lines = [
        f"GET {url}",
        f"Status: {resp.status_code} {resp.reason}",
        f"Final URL: {final_url}",
        f"Content-Type: {content_type}",
        f"Bytes read: {total}{' (capped)' if total >= _MAX_BYTES else ''}",
        "",
        body if body else "(empty body)",
    ]
    if truncated:
        lines.append("")
        lines.append("… truncated for the agent context window …")
    return "\n".join(lines)


HTTP_GET_TOOL = Tool(
    name="http_get",
    description=(
        "Fetch a public http(s) URL and return status + truncated text body. "
        "Use to read a specific page the user names. Not a search engine — "
        "prefer `search` or `wikipedia` when you only have a topic, not a URL."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Absolute URL, e.g. 'https://example.com'.",
            },
            "max_chars": {
                "type": "integer",
                "description": "Max characters of body to return (200-12000). Default 4000.",
            },
        },
        "required": ["url"],
    },
    func=http_get,
)


def run(**kwargs: Any) -> str:
    return http_get(**kwargs)
