"""
Wikipedia tool — free encyclopedic lookup (no API key).

Uses the public Wikipedia REST summary endpoint so agents can ground
answers in real articles during demos without Tavily.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import requests

from core.tool_registry import Tool

_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
_SEARCH_URL = "https://en.wikipedia.org/w/api.php"


def wikipedia(query: str) -> str:
    """Look up a topic on English Wikipedia and return a short summary.

    Parameters
    ----------
    query:
        Article title or search phrase, e.g. ``"Large language model"``.
    """
    query = (query or "").strip()
    if not query:
        return "ERROR: empty query"

    headers = {
        # Wikipedia asks clients to identify themselves
        "User-Agent": "VisualAgentHarness/0.1 (educational; local agent demo)",
        "Accept": "application/json",
    }

    try:
        # 1) Try the query as a page title first
        title = query.replace(" ", "_")
        resp = requests.get(
            _SUMMARY_URL.format(title=quote(title, safe="_")),
            headers=headers,
            timeout=15,
        )

        # 2) If missing, search for the best match then fetch summary
        if resp.status_code == 404:
            search = requests.get(
                _SEARCH_URL,
                params={
                    "action": "opensearch",
                    "search": query,
                    "limit": 1,
                    "namespace": 0,
                    "format": "json",
                },
                headers=headers,
                timeout=15,
            )
            search.raise_for_status()
            data = search.json()
            matches = data[1] if isinstance(data, list) and len(data) > 1 else []
            if not matches:
                return f"No Wikipedia article found for {query!r}."
            title = str(matches[0]).replace(" ", "_")
            resp = requests.get(
                _SUMMARY_URL.format(title=quote(title, safe="_")),
                headers=headers,
                timeout=15,
            )

        if resp.status_code == 404:
            return f"No Wikipedia article found for {query!r}."
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as exc:
        return f"ERROR: Wikipedia request failed: {exc}"

    page_title = payload.get("title") or query
    extract = (payload.get("extract") or "").strip()
    url = (payload.get("content_urls") or {}).get("desktop", {}).get("page") or payload.get(
        "url", ""
    )
    description = (payload.get("description") or "").strip()

    if not extract:
        return f"Wikipedia page '{page_title}' has no summary text."

    lines = [f"Wikipedia: {page_title}"]
    if description:
        lines.append(f"({description})")
    lines.append("")
    # Keep observations short so the context window stays readable
    if len(extract) > 1200:
        extract = extract[:1200].rstrip() + "…"
    lines.append(extract)
    if url:
        lines.append("")
        lines.append(f"URL: {url}")
    return "\n".join(lines)


WIKIPEDIA_TOOL = Tool(
    name="wikipedia",
    description=(
        "Look up a topic on English Wikipedia and return a short summary. "
        "Use for definitions, history, biographies, and background facts. "
        "Prefer this over guessing when the user asks 'what is…'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Topic or article title, e.g. 'Ollama' or 'Paris'.",
            }
        },
        "required": ["query"],
    },
    func=wikipedia,
)


def run(**kwargs: Any) -> str:
    return wikipedia(**kwargs)
