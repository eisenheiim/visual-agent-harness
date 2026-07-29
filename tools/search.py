"""
Search tool — live web search by default (no API key).

Backend priority:
  1. Tavily — if ``TAVILY_API_KEY`` is set (best quality)
  2. DuckDuckGo — free HTML search via ``requests`` (default)
  3. Mock corpus — offline / forced with ``VAH_SEARCH_BACKEND=mock``

Set ``VAH_SEARCH_BACKEND`` to ``tavily``, ``duckduckgo``, or ``mock`` to pin a backend.
"""

from __future__ import annotations

import os
import re
from html import unescape
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import requests

from core.tool_registry import Tool


_STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her",
    "was", "one", "our", "out", "day", "get", "has", "him", "his", "how", "man",
    "new", "now", "old", "see", "two", "way", "who", "did", "its", "let", "put",
    "say", "she", "too", "use", "what", "when", "with", "about", "into", "from",
    "that", "this", "have", "will", "your", "there", "their", "which", "would",
    "make", "like", "just", "over", "such", "take", "than", "them", "then",
    "some", "could", "other", "right", "tell", "please", "using", "search",
}

# Offline corpus — used only when live search is unavailable or forced
_MOCK_CORPUS: list[dict[str, str]] = [
    {
        "title": "Weather in Paris",
        "url": "https://example.com/weather/paris",
        "snippet": (
            "Current conditions in Paris, France: 18°C, partly cloudy, "
            "light breeze from the west at 12 km/h. Humidity 62%."
        ),
        "keywords": "weather forecast temperature paris france celsius cloudy",
    },
    {
        "title": "Weather in Tokyo",
        "url": "https://example.com/weather/tokyo",
        "snippet": (
            "Tokyo, Japan right now: 24°C, humid with scattered showers. "
            "Wind 8 km/h easterly. Expect evening clearing."
        ),
        "keywords": "weather forecast temperature tokyo japan humid showers",
    },
    {
        "title": "Weather in New York",
        "url": "https://example.com/weather/new-york",
        "snippet": (
            "New York City: 22°C, clear skies. UV index moderate. "
            "Overnight low around 15°C."
        ),
        "keywords": "weather forecast temperature new york city nyc clear",
    },
    {
        "title": "What is an LLM agent?",
        "url": "https://example.com/docs/llm-agents",
        "snippet": (
            "An LLM agent is a loop: the model thinks, optionally calls tools, "
            "observes results, and repeats until it produces a final answer. "
            "Memory and tool registries turn a chatbot into an agent."
        ),
        "keywords": "llm agent agents harness tools memory think act observe loop",
    },
    {
        "title": "Ollama local models",
        "url": "https://example.com/docs/ollama",
        "snippet": (
            "Ollama runs open-weight models locally (Llama, Mistral, Qwen, …). "
            "Install the app, pull a model with `ollama pull llama3.2`, then chat "
            "via the ollama Python package — no cloud API key required."
        ),
        "keywords": "ollama local model llama mistral qwen install pull",
    },
    {
        "title": "Python math module highlights",
        "url": "https://example.com/python/math",
        "snippet": (
            "The Python math module provides sqrt, sin, cos, log, factorial, and more. "
            "For exact arithmetic prefer the calculator tool or integer ops."
        ),
        "keywords": "python math module sqrt sin cos log factorial calculator",
    },
    {
        "title": "Visual Agent Harness",
        "url": "https://example.com/visual-agent-harness",
        "snippet": (
            "Visual Agent Harness is an educational Python project that visualizes "
            "the think → act → observe agent loop with Streamlit, Ollama, and simple tools."
        ),
        "keywords": "visual agent harness streamlit ollama education prototype github",
    },
    {
        "title": "Large language models (overview)",
        "url": "https://example.com/docs/llm-overview",
        "snippet": (
            "A large language model (LLM) predicts tokens from context. "
            "Popular families include Llama, GPT, Mistral, and Qwen. "
            "Agents wrap LLMs with tools and memory."
        ),
        "keywords": "large language model llm gpt tokens transformers neural",
    },
    {
        "title": "Istanbul city facts",
        "url": "https://example.com/places/istanbul",
        "snippet": (
            "Istanbul spans Europe and Asia across the Bosphorus. "
            "It is Turkey's largest city and a historic trade hub "
            "(Byzantium, Constantinople)."
        ),
        "keywords": "istanbul turkey bosphorus city europe asia constantinople unique",
    },
    {
        "title": "Tokyo city facts",
        "url": "https://example.com/places/tokyo",
        "snippet": (
            "Tokyo is Japan's capital and the world's largest metro area. "
            "What sets it apart: hyper-efficient transit, dense neon districts "
            "(Shibuya, Shinjuku), quiet residential lanes beside skyscrapers, "
            "and a blend of centuries-old temples with cutting-edge tech culture."
        ),
        "keywords": (
            "tokyo japan capital city unique apart distinctive shibuya shinjuku "
            "transit temples tech metro facts"
        ),
    },
    {
        "title": "Coffee brewing basics",
        "url": "https://example.com/food/coffee",
        "snippet": (
            "Common brew methods: pour-over, espresso, French press, AeroPress. "
            "A typical starting ratio is about 1:16 coffee to water by weight."
        ),
        "keywords": "coffee brew espresso pour-over french press ratio",
    },
]


def _terms(text: str) -> set[str]:
    raw = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in raw if len(t) > 2 and t not in _STOPWORDS}


def _mock_search(query: str, max_results: int = 3) -> list[dict[str, str]]:
    """Score corpus entries by term overlap. Never invents unrelated hits."""
    q_terms = _terms(query)
    if not q_terms:
        return []

    weather_query = bool(
        q_terms & {"weather", "forecast", "temperature", "humidity", "rain", "sunny"}
    )

    scored: list[tuple[float, dict[str, str]]] = []
    for doc in _MOCK_CORPUS:
        blob = f"{doc['title']} {doc.get('keywords', '')} {doc['snippet']}".lower()
        overlap = sum(1 for t in q_terms if t in blob)
        if not overlap:
            continue
        is_weather_doc = "weather" in doc["title"].lower() or "weather" in doc.get(
            "keywords", ""
        )
        if is_weather_doc and not weather_query:
            continue
        score = overlap + (0.5 if doc["title"].lower() in query.lower() else 0.0)
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:max_results]]


def _unwrap_ddg_url(href: str) -> str:
    """DuckDuckGo wraps outbound links; unwrap when possible."""
    if "uddg=" in href:
        qs = parse_qs(urlparse(href).query)
        if "uddg" in qs and qs["uddg"]:
            return unquote(qs["uddg"][0])
    return href


def _strip_tags(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html)
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _duckduckgo_search(query: str, max_results: int = 3) -> list[dict[str, str]]:
    """Live web search via DuckDuckGo HTML (no API key)."""
    resp = requests.post(
        "https://html.duckduckgo.com/html/",
        data={"q": query},
        headers={
            "User-Agent": (
                "VisualAgentHarness/0.1 (+https://github.com/visual-agent-harness; educational)"
            )
        },
        timeout=20,
    )
    resp.raise_for_status()
    html = resp.text

    # Each result block: result__a (title/url) + result__snippet
    blocks = re.findall(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>'
        r'.*?(?:class="result__snippet"[^>]*>(.*?)</(?:a|td)|$)',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    results: list[dict[str, str]] = []
    for href, title_html, snippet_html in blocks:
        title = _strip_tags(title_html)
        snippet = _strip_tags(snippet_html) if snippet_html else ""
        url = _unwrap_ddg_url(unescape(href))
        if not title:
            continue
        results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= max_results:
            break

    if results:
        return results

    # Fallback: DuckDuckGo Instant Answer API (sparser, but key-free)
    ia = requests.get(
        "https://api.duckduckgo.com/",
        params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
        headers={"User-Agent": "VisualAgentHarness/0.1"},
        timeout=15,
    )
    ia.raise_for_status()
    data = ia.json()
    if data.get("AbstractText"):
        results.append(
            {
                "title": data.get("Heading") or query,
                "url": data.get("AbstractURL") or "",
                "snippet": data["AbstractText"],
            }
        )
    for topic in data.get("RelatedTopics") or []:
        if len(results) >= max_results:
            break
        if isinstance(topic, dict) and topic.get("Text"):
            results.append(
                {
                    "title": (topic.get("Text") or "")[:80],
                    "url": topic.get("FirstURL") or "",
                    "snippet": topic.get("Text") or "",
                }
            )
    return results[:max_results]


def _tavily_search(query: str, max_results: int = 3) -> list[dict[str, str]]:
    """Real web search via Tavily (https://tavily.com)."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not set")

    resp = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "include_answer": False,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    results = []
    for item in data.get("results", [])[:max_results]:
        results.append(
            {
                "title": item.get("title") or "(no title)",
                "url": item.get("url") or "",
                "snippet": item.get("content") or "",
            }
        )
    return results


def _resolve_backend() -> str:
    """Pick search backend: mock | tavily | duckduckgo."""
    forced = (os.environ.get("VAH_SEARCH_BACKEND") or "").strip().lower()
    if forced in {"mock", "tavily", "duckduckgo", "ddg"}:
        return "duckduckgo" if forced == "ddg" else forced
    if os.environ.get("TAVILY_API_KEY"):
        return "tavily"
    return "duckduckgo"


def search(query: str, max_results: int = 3) -> str:
    """Search the live web (DuckDuckGo / Tavily) or the offline mock corpus."""
    query = (query or "").strip()
    if not query:
        return "ERROR: empty query"

    max_results = max(1, min(int(max_results), 5))
    backend = _resolve_backend()
    source = backend
    hits: list[dict[str, str]] = []
    errors: list[str] = []

    try:
        if backend == "mock":
            hits = _mock_search(query, max_results=max_results)
        elif backend == "tavily":
            hits = _tavily_search(query, max_results=max_results)
        else:
            hits = _duckduckgo_search(query, max_results=max_results)
            source = "duckduckgo"
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{backend}: {exc}")
        # Prefer the other live backend before dropping to mock
        if backend != "tavily" and os.environ.get("TAVILY_API_KEY"):
            try:
                hits = _tavily_search(query, max_results=max_results)
                source = "tavily (fallback)"
            except Exception as exc2:  # noqa: BLE001
                errors.append(f"tavily: {exc2}")
        if not hits and backend != "duckduckgo":
            try:
                hits = _duckduckgo_search(query, max_results=max_results)
                source = "duckduckgo (fallback)"
            except Exception as exc3:  # noqa: BLE001
                errors.append(f"duckduckgo: {exc3}")
        if not hits:
            hits = _mock_search(query, max_results=max_results)
            source = "mock (offline fallback)"
            if errors:
                source += f" after: {errors[0]}"

    if not hits:
        hint = (
            " No hits. If you're offline, the mock corpus only covers a few demo topics "
            "(weather Paris/Tokyo/NYC, LLM agents, Ollama, Istanbul, Tokyo facts, coffee). "
            "Or set TAVILY_API_KEY for Tavily. Encyclopedia topics: prefer `wikipedia`."
        )
        return f"No results for: {query!r}.{hint}"

    lines = [f"Search results for {query!r} [{source}]:", ""]
    for i, hit in enumerate(hits, 1):
        lines.append(f"{i}. {hit['title']}")
        if hit.get("url"):
            lines.append(f"   URL: {hit['url']}")
        lines.append(f"   {hit['snippet']}")
        lines.append("")
    return "\n".join(lines).strip()


SEARCH_TOOL = Tool(
    name="search",
    description=(
        "Search the live web for factual information (news, weather, docs, places). "
        "Default backend is DuckDuckGo (no API key). "
        "Uses Tavily when TAVILY_API_KEY is set. "
        "For weather, query like 'weather in Paris' — there is no separate weather tool."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query, e.g. 'weather in Paris' or 'what is an LLM agent'.",
            },
            "max_results": {
                "type": "integer",
                "description": "How many results to return (1-5). Default 3.",
            },
        },
        "required": ["query"],
    },
    func=search,
)


def run(**kwargs: Any) -> str:
    return search(**kwargs)
