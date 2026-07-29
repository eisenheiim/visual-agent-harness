"""
Memory — short-term conversation history + long-term RAG store.

Two layers, deliberately simple so you can *see* what agents remember:

  Short-term (working memory)
    → Ordered list of chat messages for the current session.
    → Truncated when it grows past ``max_short_term`` turns.

  Long-term (RAG memory)
    → Bag of text snippets with naive bag-of-words similarity.
    → No vector DB required — pure Python, zero extra deps.
    → Good enough to teach "retrieve relevant past facts" without
      drowning beginners in embeddings infrastructure.

Swap the scorer for real embeddings (Ollama / OpenAI) later; the
harness only depends on the Memory public API.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9_]+", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens — intentionally dumb and inspectable."""
    return _TOKEN.findall(text.lower())


def _bag(text: str) -> dict[str, float]:
    """Term-frequency bag (L2-normalized) for cosine similarity."""
    counts: dict[str, float] = {}
    for tok in _tokenize(text):
        counts[tok] = counts.get(tok, 0.0) + 1.0
    norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
    return {k: v / norm for k, v in counts.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    # Iterate the smaller bag for a tiny speed win
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """One turn in the short-term conversation."""

    role: str  # "system" | "user" | "assistant" | "tool" | "observation"
    content: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {"role": self.role, "content": self.content}
        if self.meta:
            d["meta"] = self.meta
        return d


@dataclass
class MemoryItem:
    """One long-term memory snippet."""

    id: str
    text: str
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    # Precomputed bag kept off the public to_dict for cleaner dumps
    _bag: dict[str, float] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "tags": self.tags,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class Memory:
    """Short-term chat buffer + long-term RAG store.

    Example::

        mem = Memory(max_short_term=20)
        mem.add_user("What is 2+2?")
        mem.add_assistant("4")
        mem.remember("User likes concise answers.", tags=["preference"])
        hits = mem.recall("How should I answer?", top_k=3)
    """

    def __init__(
        self,
        max_short_term: int = 40,
        store_path: Optional[str | Path] = None,
    ) -> None:
        self.max_short_term = max_short_term
        self.short_term: list[Message] = []
        self.long_term: list[MemoryItem] = []
        self._store_path = Path(store_path) if store_path else None
        self._id_counter = 0

        if self._store_path and self._store_path.exists():
            self.load(self._store_path)

    # -- short-term ---------------------------------------------------------

    def add(self, role: str, content: str, **meta: Any) -> Message:
        """Append a message and enforce the turn budget."""
        msg = Message(role=role, content=content, meta=meta)
        self.short_term.append(msg)
        self._trim_short_term()
        return msg

    def add_system(self, content: str, **meta: Any) -> Message:
        return self.add("system", content, **meta)

    def add_user(self, content: str, **meta: Any) -> Message:
        return self.add("user", content, **meta)

    def add_assistant(self, content: str, **meta: Any) -> Message:
        return self.add("assistant", content, **meta)

    def add_observation(self, tool_name: str, content: str, **meta: Any) -> Message:
        """Record a tool result as an observation the model can read next turn."""
        meta = {"tool": tool_name, **meta}
        return self.add("observation", content, **meta)

    def _trim_short_term(self) -> None:
        """Keep system messages + the most recent non-system turns."""
        if len(self.short_term) <= self.max_short_term:
            return
        system = [m for m in self.short_term if m.role == "system"]
        rest = [m for m in self.short_term if m.role != "system"]
        keep = self.max_short_term - len(system)
        self.short_term = system + rest[-max(keep, 0) :]

    def clear_short_term(self, keep_system: bool = True) -> None:
        if keep_system:
            self.short_term = [m for m in self.short_term if m.role == "system"]
        else:
            self.short_term = []

    def as_chat_messages(self) -> list[dict[str, str]]:
        """Format for Ollama / OpenAI chat APIs.

        Observations are folded into ``role=user`` messages labeled clearly,
        because not every backend supports a custom ``tool`` role.
        """
        out: list[dict[str, str]] = []
        for m in self.short_term:
            if m.role == "observation":
                tool = m.meta.get("tool", "tool")
                out.append(
                    {
                        "role": "user",
                        "content": f"[OBSERVATION from tool `{tool}`]\n{m.content}",
                    }
                )
            else:
                # Map unknown roles to user to stay API-safe
                role = m.role if m.role in {"system", "user", "assistant"} else "user"
                out.append({"role": role, "content": m.content})
        return out

    # -- long-term (RAG) ----------------------------------------------------

    def remember(
        self,
        text: str,
        tags: Optional[Iterable[str]] = None,
        item_id: Optional[str] = None,
    ) -> MemoryItem:
        """Store a fact/snippet for later recall."""
        self._id_counter += 1
        item = MemoryItem(
            id=item_id or f"mem_{self._id_counter}",
            text=text.strip(),
            tags=list(tags or []),
            _bag=_bag(text),
        )
        self.long_term.append(item)
        return item

    def forget(self, item_id: str) -> bool:
        before = len(self.long_term)
        self.long_term = [i for i in self.long_term if i.id != item_id]
        return len(self.long_term) < before

    def recall(self, query: str, top_k: int = 3, min_score: float = 0.01) -> list[tuple[MemoryItem, float]]:
        """Return the top-k most similar long-term items (cosine over TF bags).

        Falls back to any token-overlap match if nothing clears ``min_score`` —
        educational demos often use short pins + short questions.
        """
        if not self.long_term:
            return []

        q = _bag(query)
        scored = [(_cosine(q, item._bag), item) for item in self.long_term]
        strong = [(s, i) for s, i in scored if s >= min_score]
        strong.sort(key=lambda x: x[0], reverse=True)
        if strong:
            return [(item, score) for score, item in strong[:top_k]]

        # Soft fallback: keep anything with any overlap, or last-pinned items
        soft = [(s, i) for s, i in scored if s > 0]
        soft.sort(key=lambda x: x[0], reverse=True)
        if soft:
            return [(item, score) for score, item in soft[:top_k]]

        # Last resort: most recent memories (still better than silent miss)
        recent = list(reversed(self.long_term))[:top_k]
        return [(item, 0.0) for item in recent]

    def recall_text(self, query: str, top_k: int = 3) -> str:
        """Pretty block to inject into the system / user prompt."""
        hits = self.recall(query, top_k=top_k)
        if not hits:
            return ""
        lines = ["Relevant long-term memories:"]
        for item, score in hits:
            tag_str = f" [{', '.join(item.tags)}]" if item.tags else ""
            lines.append(f"- ({score:.2f}){tag_str} {item.text}")
        return "\n".join(lines)

    # -- persistence --------------------------------------------------------

    def save(self, path: Optional[str | Path] = None) -> Path:
        """Persist short-term + long-term to JSON (educational, not encrypted)."""
        target = Path(path) if path else self._store_path
        if target is None:
            raise ValueError("No store_path configured; pass path= explicitly.")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "short_term": [m.to_dict() for m in self.short_term],
            "long_term": [i.to_dict() for i in self.long_term],
            "id_counter": self._id_counter,
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    def load(self, path: Optional[str | Path] = None) -> None:
        source = Path(path) if path else self._store_path
        if source is None or not source.exists():
            return
        data = json.loads(source.read_text(encoding="utf-8"))
        self.short_term = [
            Message(role=m["role"], content=m["content"], meta=m.get("meta", {}))
            for m in data.get("short_term", [])
        ]
        self.long_term = []
        for item in data.get("long_term", []):
            mem = MemoryItem(
                id=item["id"],
                text=item["text"],
                tags=item.get("tags", []),
                created_at=item.get("created_at", time.time()),
                _bag=_bag(item["text"]),
            )
            self.long_term.append(mem)
        self._id_counter = int(data.get("id_counter", len(self.long_term)))

    # -- introspection (great for the Streamlit UI) -------------------------

    def snapshot(self) -> dict[str, Any]:
        """Serializable view of current memory state for visualization."""
        return {
            "short_term_count": len(self.short_term),
            "long_term_count": len(self.long_term),
            "short_term": [m.to_dict() for m in self.short_term],
            "long_term_clean": [i.to_dict() for i in self.long_term],
        }
