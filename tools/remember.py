"""
Remember tool — let the agent write into long-term (RAG) memory.

Short-term chat is automatic. Long-term only grows when something
explicitly calls ``Memory.remember`` — this tool is that hook.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from core.tool_registry import Tool

# Wired at register time so the tool can reach the active Memory instance
_remember_fn: Optional[Callable[..., Any]] = None


def bind_memory(remember_fn: Callable[..., Any]) -> None:
    """Attach ``memory.remember`` (or harness.remember) before the agent runs."""
    global _remember_fn
    _remember_fn = remember_fn


def remember(text: str, tags: str = "") -> str:
    """Store a fact in long-term memory for later RAG recall.

    Parameters
    ----------
    text:
        The fact to remember, e.g. ``"User's favorite city is Tokyo"``.
    tags:
        Optional comma-separated tags, e.g. ``"preference,profile"``.
    """
    text = (text or "").strip()
    if not text:
        return "ERROR: empty text — nothing stored."
    if _remember_fn is None:
        return (
            "ERROR: long-term memory is not bound. "
            "In the UI this should not happen; in scripts call bind_memory(...)."
        )
    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
    item = _remember_fn(text, tags=tag_list or None)
    item_id = getattr(item, "id", None) or "ok"
    return f"Stored in long-term memory as `{item_id}`: {text}"


REMEMBER_TOOL = Tool(
    name="remember",
    description=(
        "ONLY when the user explicitly asks to remember/save/keep/hatırla a fact. "
        "Do not call this for unrelated tasks (math, search, code). "
        "Writes into long-term RAG memory. Saying 'I will remember' in plain text "
        "does NOT store anything — emit this tool call JSON first. "
        'Example: {"tool":"remember","arguments":{"text":"User favorite city is Tokyo","tags":"preference"}}'
    ),
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Fact to store, e.g. 'User favorite city is Tokyo'.",
            },
            "tags": {
                "type": "string",
                "description": "Optional comma-separated tags, e.g. 'preference,profile'.",
            },
        },
        "required": ["text"],
    },
    func=remember,
)


def run(**kwargs: Any) -> str:
    return remember(**kwargs)
