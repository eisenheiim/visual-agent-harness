"""
Tracing — capture a full agent run as inspectable JSON.

Why this exists
---------------
The Streamlit timeline is great live. Traces are for *after*:
compare two models, file a bug with a reproducible artifact, or
feed an eval runner.

A trace is just the ordered list of :class:`AgentStep` dicts plus
run metadata — no proprietary format.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from .harness import AgentEvent, AgentStep


@dataclass
class RunTrace:
    """One complete agent run, ready to save or score."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    backend: str = ""
    model: str = ""
    user_message: str = ""
    final_answer: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    # -- builders -----------------------------------------------------------

    @classmethod
    def from_steps(
        cls,
        steps: list[AgentStep] | list[dict[str, Any]],
        *,
        backend: str = "",
        model: str = "",
        user_message: str = "",
        final_answer: str = "",
        meta: Optional[dict[str, Any]] = None,
    ) -> "RunTrace":
        normalized: list[dict[str, Any]] = []
        for step in steps:
            if isinstance(step, AgentStep):
                normalized.append(step.to_dict())
            else:
                normalized.append(dict(step))

        # Infer user message / answer from steps when not provided
        msg = user_message
        answer = final_answer
        for s in normalized:
            if s.get("event") == AgentEvent.RUN_START.value and not msg:
                msg = s.get("content") or ""
                m = s.get("meta") or {}
                backend = backend or str(m.get("backend") or "")
                model = model or str(m.get("model") or "")
            if s.get("event") == AgentEvent.FINAL.value:
                answer = s.get("content") or answer

        return cls(
            backend=backend,
            model=model,
            user_message=msg,
            final_answer=answer,
            steps=normalized,
            meta=meta or {},
        )

    # -- queries (handy for evals) ------------------------------------------

    def tool_names(self) -> list[str]:
        return [
            s["tool_name"]
            for s in self.steps
            if s.get("event") == AgentEvent.TOOL_CALL.value and s.get("tool_name")
        ]

    def has_error(self) -> bool:
        return any(s.get("event") == AgentEvent.ERROR.value for s in self.steps)

    def think_count(self) -> int:
        return sum(1 for s in self.steps if s.get("event") == AgentEvent.THINK_END.value)

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "backend": self.backend,
            "model": self.model,
            "user_message": self.user_message[:200],
            "final_answer": (self.final_answer or "")[:300],
            "tools_called": self.tool_names(),
            "think_count": self.think_count(),
            "step_count": len(self.steps),
            "has_error": self.has_error(),
        }

    # -- persistence --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "RunTrace":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


def build_trace_from_harness(harness: Any) -> RunTrace:
    """Convenience: snapshot :class:`AgentHarness` after a run."""
    return RunTrace.from_steps(
        harness.last_steps,
        backend=getattr(harness, "backend", ""),
        model=getattr(harness, "model", ""),
        final_answer=getattr(harness, "last_answer", "") or "",
        meta={
            "system_prompt_chars": len(getattr(harness, "last_system_prompt", "") or ""),
            "rag_hits": getattr(harness, "last_rag_hits", []),
        },
    )
