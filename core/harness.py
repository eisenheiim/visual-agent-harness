"""
AgentHarness — the educational agentic loop.

The loop every LLM agent secretly runs:

    ┌─────────┐
    │  THINK  │  ← LLM reads history + tools, decides next action
    └────┬────┘
         │
         ▼
    ┌─────────┐     tool call?     ┌─────────┐
    │   ACT   │ ─────────────────► │ OBSERVE │ ──┐
    └────┬────┘                    └─────────┘   │
         │ final answer                          │
         ▼                                       │
       DONE ◄────────────────────────────────────┘
              (observation appended → think again)

This module makes that loop explicit, instrumented, and visualizable.
Every step emits an :class:`AgentEvent` so the Streamlit UI (or your own
logger) can watch the agent reason in real time.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterator, Literal, Optional

from .memory import Memory
from .tool_registry import ToolCall, ToolRegistry


def _looks_like_remember_request(text: str) -> bool:
    """Heuristic: user is asking the agent to persist a fact."""
    lower = (text or "").lower()
    triggers = (
        "remember that",
        "remember this",
        "please remember",
        "don't forget",
        "do not forget",
        "save this",
        "save that",
        "keep in mind",
        "store this",
        "note that",
        "hatırla",
        "unutma",
        "kaydet",
    )
    return any(t in lower for t in triggers)


def _extract_remember_fact(text: str) -> str:
    """Best-effort pull of the fact the user asked to store."""
    patterns = (
        r"remember\s+(?:that\s+|this\s+)?(.+?)(?:\.|$)",
        r"(?:don't|do not)\s+forget\s+(?:that\s+)?(.+?)(?:\.|$)",
        r"(?:save|store|note)\s+(?:this|that)?\s*:?\s*(.+?)(?:\.|$)",
        r"hatırla(?:\s+ki)?\s*:?\s*(.+?)(?:\.|$)",
        r"kaydet\s*:?\s*(.+?)(?:\.|$)",
    )
    for pat in patterns:
        match = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            fact = match.group(1).strip()
            # Cut trailing task clauses: "Tokyo. Then search..."
            fact = re.split(
                r"\b(?:then|and then|after that|sonra)\b", fact, maxsplit=1, flags=re.I
            )[0]
            fact = fact.strip(" \n\t.,;:")
            if len(fact) >= 3:
                return fact
    # Fallback: whole message, truncated
    cleaned = " ".join((text or "").split())
    return cleaned[:240] if cleaned else ""


def _format_think_payload(
    messages: list[dict[str, str]],
    rag_hits: list[dict[str, Any]],
    preview_chars: int = 280,
) -> str:
    """Human-readable dump of what is about to be sent to the LLM."""
    lines = [f"Calling the model with {len(messages)} message(s):", ""]
    if rag_hits:
        lines.append("RAG hits in this turn:")
        for hit in rag_hits:
            tags = ", ".join(hit.get("tags") or []) or "—"
            lines.append(
                f"  • [{hit.get('score', 0):.2f}] ({tags}) {hit.get('text', '')}"
            )
        lines.append("")
    else:
        lines.append("RAG hits: (none for this call)")
        lines.append("")

    for i, msg in enumerate(messages, 1):
        role = msg.get("role", "?")
        content = (msg.get("content") or "").strip()
        # Keep system short in the timeline; full text is in learner mode
        limit = 160 if role == "system" else preview_chars
        preview = content if len(content) <= limit else content[:limit].rstrip() + "…"
        lines.append(f"── [{i}] {role} ({len(content)} chars) ──")
        lines.append(preview)
        lines.append("")
    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Events & steps — the visualization contract
# ---------------------------------------------------------------------------

class AgentEvent(str, Enum):
    """Lifecycle signals the UI / examples can subscribe to."""

    RUN_START = "run_start"
    THINK_START = "think_start"
    THINK_END = "think_end"
    TOOL_CALL = "tool_call"
    OBSERVATION = "observation"
    FINAL = "final"
    ERROR = "error"
    RUN_END = "run_end"


@dataclass
class AgentStep:
    """One discrete beat of the agent loop (perfect for timeline UIs)."""

    event: AgentEvent
    iteration: int = 0
    content: str = ""
    tool_name: Optional[str] = None
    tool_args: Optional[dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event.value,
            "iteration": self.iteration,
            "content": self.content,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "timestamp": self.timestamp,
            "meta": self.meta,
        }


EventCallback = Callable[[AgentStep], None]
BackendName = Literal["ollama", "openai"]


# ---------------------------------------------------------------------------
# LLM backends (kept tiny and side-by-side for comparison)
# ---------------------------------------------------------------------------

def _chat_ollama(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
) -> str:
    """Call a local Ollama model. Requires `ollama serve` running."""
    import ollama  # lazy import so openai-only users don't need it installed to import the module

    response = ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": temperature},
    )
    # ollama-python returns either an object or a dict depending on version
    if isinstance(response, dict):
        return response["message"]["content"]
    return response.message.content


def _chat_openai(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
) -> str:
    """Call OpenAI Chat Completions. Needs OPENAI_API_KEY in the environment."""
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM = """You are a careful, transparent AI agent running inside Visual Agent Harness.
Your job is to solve the user's request using step-by-step reasoning and tools when needed.

Rules:
1. Prefer tools over guessing when a tool can give a precise answer.
2. Never invent tool results — wait for the [OBSERVATION] message.
3. Keep intermediate reasoning short; put the user-facing answer in {"final": "..."}.
4. If a tool errors, try a different approach or explain the failure in your final answer.
5. MEMORY IS A TOOL — saying "I'll remember" in text does NOTHING.
   When the user asks you to remember / save / keep a fact or preference,
   you MUST call the `remember` tool with that fact BEFORE {"final": "..."}.
   Example:
     {"tool": "remember", "arguments": {"text": "User favorite city is Tokyo", "tags": "preference"}}
"""


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------

class AgentHarness:
    """Minimal, inspectable agent loop.

    Parameters
    ----------
    backend:
        ``"ollama"`` (default, local) or ``"openai"``.
    model:
        Model id, e.g. ``"llama3.2"`` or ``"gpt-4o-mini"``.
    tools:
        Optional :class:`ToolRegistry`. Created empty if omitted.
    memory:
        Optional :class:`Memory`. Created fresh if omitted.
    max_iterations:
        Safety cap on think→act cycles per ``run()`` call.
    temperature:
        Sampling temperature forwarded to the backend.
    system_prompt:
        Base system instructions (tool catalog is appended automatically).
    on_event:
        Optional callback invoked for every :class:`AgentStep`.
    """

    def __init__(
        self,
        backend: BackendName = "ollama",
        model: str = "llama3.2",
        tools: Optional[ToolRegistry] = None,
        memory: Optional[Memory] = None,
        max_iterations: int = 8,
        temperature: float = 0.2,
        system_prompt: str = DEFAULT_SYSTEM,
        on_event: Optional[EventCallback] = None,
    ) -> None:
        if backend not in {"ollama", "openai"}:
            raise ValueError(f"Unsupported backend: {backend!r} (use 'ollama' or 'openai')")

        self.backend = backend
        self.model = model
        self.tools = tools or ToolRegistry()
        self.memory = memory or Memory()
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.on_event = on_event

        # Full timeline of the last run — UI reads this
        self.last_steps: list[AgentStep] = []
        self.last_answer: Optional[str] = None
        self.last_system_prompt: str = ""
        self.last_tool_schemas: list[dict[str, Any]] = []
        self.last_rag_hits: list[dict[str, Any]] = []

        self._ensure_system_message()

    # -- wiring -------------------------------------------------------------

    def _ensure_system_message(self) -> None:
        """Refresh the system message so tool changes are always visible."""
        # Drop old system messages, then re-insert a single up-to-date one
        self.memory.short_term = [m for m in self.memory.short_term if m.role != "system"]
        block = f"{self.system_prompt.strip()}\n\n{self.tools.prompt_block()}"
        self.memory.add_system(block)

    def _messages_for_llm(self) -> list[dict[str, str]]:
        """Chat messages for the backend — always starts with a system message."""
        self._ensure_system_message()
        messages = self.memory.as_chat_messages()
        sys_text = self.last_system_prompt or self.system_prompt
        # Guarantee messages[0] is system even if memory was weirdly cleared
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": sys_text}] + [
                m for m in messages if m.get("role") != "system"
            ]
        elif not (messages[0].get("content") or "").strip():
            messages[0] = {"role": "system", "content": sys_text}
        self.last_system_prompt = messages[0]["content"]
        return messages

    def register_builtin_tools(self) -> None:
        """Attach the full built-in tool kit (math, search, wiki, remember, …)."""
        from tools import register_builtin_tools

        register_builtin_tools(self.tools, memory=self.memory)
        self._ensure_system_message()

    def set_event_callback(self, callback: Optional[EventCallback]) -> None:
        self.on_event = callback

    # -- LLM call -----------------------------------------------------------

    def _llm(self, messages: list[dict[str, str]]) -> str:
        if self.backend == "ollama":
            return _chat_ollama(messages, self.model, self.temperature)
        return _chat_openai(messages, self.model, self.temperature)

    # -- events -------------------------------------------------------------

    def _emit(self, step: AgentStep) -> AgentStep:
        self.last_steps.append(step)
        if self.on_event:
            self.on_event(step)
        return step

    # -- main loop ----------------------------------------------------------

    def run(self, user_message: str, use_rag: bool = True) -> str:
        """Run the agentic loop to completion. Returns the final answer string."""
        steps = list(self.stream(user_message, use_rag=use_rag))
        # Prefer an explicit FINAL event; fall back to last assistant text
        for step in reversed(steps):
            if step.event == AgentEvent.FINAL:
                return step.content
        return self.last_answer or ""

    def last_trace(self):
        """Build a :class:`RunTrace` from ``last_steps`` (for export / evals)."""
        from .tracing import build_trace_from_harness

        return build_trace_from_harness(self)

    def stream(self, user_message: str, use_rag: bool = True) -> Iterator[AgentStep]:
        """Yield :class:`AgentStep` objects as the loop progresses.

        This is what the Streamlit UI consumes for live visualization.
        """
        self.last_steps = []
        self.last_answer = None
        self._ensure_system_message()

        # Snapshot system prompt AFTER ensure — used by UI hover / learner mode
        system_msgs = [m.content for m in self.memory.short_term if m.role == "system"]
        self.last_system_prompt = system_msgs[0] if system_msgs else self.system_prompt
        self.last_tool_schemas = self.tools.schemas()

        # Optionally inject RAG context beside the user message
        prompt = user_message
        rag_hits: list[dict[str, Any]] = []
        rag_block = ""
        if use_rag and self.memory.long_term:
            scored = self.memory.recall(user_message, top_k=3)
            rag_hits = [
                {
                    "id": item.id,
                    "text": item.text,
                    "tags": list(item.tags),
                    "score": round(float(score), 4),
                }
                for item, score in scored
            ]
            rag_block = self.memory.recall_text(user_message, top_k=3)
            if rag_block:
                prompt = f"{rag_block}\n\nUser request:\n{user_message}"

        self.last_rag_hits = rag_hits

        # Small local models often "promise" to remember without calling the tool.
        # Nudge them hard when the user clearly asks to store something.
        if _looks_like_remember_request(user_message) and "remember" in self.tools:
            prompt = (
                "[REQUIRED] The user asked you to store something in long-term memory.\n"
                "Your FIRST action must be a `remember` tool call JSON, e.g.\n"
                '{"tool": "remember", "arguments": {"text": "<the fact>", "tags": "preference"}}\n'
                "Only after you receive the observation may you call other tools or "
                'respond with {"final": "..."}.\n\n'
                f"{prompt}"
            )

        self.memory.add_user(prompt)

        # Re-read system after add_user (still present; trim keeps it)
        system_msgs = [m.content for m in self.memory.short_term if m.role == "system"]
        if system_msgs:
            self.last_system_prompt = system_msgs[0]

        yield self._emit(
            AgentStep(
                event=AgentEvent.RUN_START,
                content=user_message,
                meta={
                    "backend": self.backend,
                    "model": self.model,
                    "use_rag": use_rag,
                    "long_term_count": len(self.memory.long_term),
                    "rag_hits": rag_hits,
                    "rag_injected": bool(rag_block),
                    "prompt_sent_to_model": prompt,
                    "system_prompt": self.last_system_prompt,
                    "tool_schemas": self.last_tool_schemas,
                },
            )
        )

        try:
            used_remember = False
            wants_remember = _looks_like_remember_request(user_message) and "remember" in self.tools

            for iteration in range(1, self.max_iterations + 1):
                # THINK -----------------------------------------------------
                messages = self._messages_for_llm()
                sys_text = self.last_system_prompt or ""
                for msg in messages:
                    if msg.get("role") == "system" and msg.get("content"):
                        sys_text = msg["content"]
                        break
                self.last_system_prompt = sys_text

                yield self._emit(
                    AgentStep(
                        event=AgentEvent.THINK_START,
                        iteration=iteration,
                        content=_format_think_payload(
                            messages, rag_hits if iteration == 1 else []
                        ),
                        meta={
                            "messages": messages,
                            "message_count": len(messages),
                            "rag_hits": list(rag_hits),
                            "rag_injected": bool(rag_block),
                            "long_term_count": len(self.memory.long_term),
                            "system_prompt": sys_text,
                            "tool_schemas": self.last_tool_schemas,
                        },
                    )
                )

                raw = self._llm(messages)
                self.memory.add_assistant(raw)

                yield self._emit(
                    AgentStep(
                        event=AgentEvent.THINK_END,
                        iteration=iteration,
                        content=raw,
                        meta={
                            "messages": messages,
                            "message_count": len(messages),
                            "system_prompt": sys_text,
                            "model_reply": raw,
                            "tool_schemas": self.last_tool_schemas,
                            "rag_hits": list(rag_hits),
                        },
                    )
                )

                # Parse decision --------------------------------------------
                call: Optional[ToolCall] = self.tools.parse_tool_call(raw)

                def _ensure_remember_saved(iteration_n: int):
                    """Safety net when the model skips the remember tool."""
                    nonlocal used_remember
                    if not wants_remember or used_remember:
                        return
                    fact = _extract_remember_fact(user_message)
                    if not fact:
                        return
                    forced = ToolCall(
                        name="remember",
                        arguments={"text": fact, "tags": "preference"},
                        raw='{"tool":"remember","arguments":{"text":%r}}' % fact,
                    )
                    yield self._emit(
                        AgentStep(
                            event=AgentEvent.TOOL_CALL,
                            iteration=iteration_n,
                            content=forced.raw,
                            tool_name="remember",
                            tool_args=forced.arguments,
                            meta={"auto": True, "reason": "model_skipped_remember_tool"},
                        )
                    )
                    observation = self.tools.execute_call(forced)
                    used_remember = True
                    self.memory.add_observation("remember", observation)
                    yield self._emit(
                        AgentStep(
                            event=AgentEvent.OBSERVATION,
                            iteration=iteration_n,
                            content=observation,
                            tool_name="remember",
                            meta={"auto": True},
                        )
                    )

                # No structured call → treat entire reply as the final answer
                if call is None:
                    yield from _ensure_remember_saved(iteration)
                    self.last_answer = raw.strip()
                    yield self._emit(
                        AgentStep(
                            event=AgentEvent.FINAL,
                            iteration=iteration,
                            content=self.last_answer,
                            meta={"reason": "unstructured_reply"},
                        )
                    )
                    break

                # Explicit final --------------------------------------------
                if call.name == "__final__":
                    yield from _ensure_remember_saved(iteration)
                    self.last_answer = str(call.arguments.get("answer", "")).strip()
                    yield self._emit(
                        AgentStep(
                            event=AgentEvent.FINAL,
                            iteration=iteration,
                            content=self.last_answer,
                        )
                    )
                    break

                # ACT → OBSERVE ---------------------------------------------
                resolved_name = self.tools.resolve_name(call.name)
                if resolved_name == "remember":
                    used_remember = True
                yield self._emit(
                    AgentStep(
                        event=AgentEvent.TOOL_CALL,
                        iteration=iteration,
                        content=call.raw,
                        tool_name=resolved_name,
                        tool_args=call.arguments,
                        meta={"requested_tool": call.name}
                        if resolved_name != call.name
                        else {},
                    )
                )

                observation = self.tools.execute_call(call)
                self.memory.add_observation(resolved_name, observation)

                yield self._emit(
                    AgentStep(
                        event=AgentEvent.OBSERVATION,
                        iteration=iteration,
                        content=observation,
                        tool_name=resolved_name,
                    )
                )
            else:
                # for-else: loop exhausted without break
                msg = (
                    f"Stopped after {self.max_iterations} iterations without a final answer. "
                    "Try raising max_iterations or simplifying the task."
                )
                self.last_answer = msg
                yield self._emit(AgentStep(event=AgentEvent.ERROR, content=msg))

        except Exception as exc:  # noqa: BLE001 — surface backend/runtime errors as events
            err = f"{type(exc).__name__}: {exc}"
            self.last_answer = err
            yield self._emit(AgentStep(event=AgentEvent.ERROR, content=err))

        yield self._emit(
            AgentStep(
                event=AgentEvent.RUN_END,
                content=self.last_answer or "",
                meta={"steps": len(self.last_steps)},
            )
        )

    # -- convenience --------------------------------------------------------

    def remember(self, text: str, tags: Optional[list[str]] = None) -> None:
        """Store a long-term memory (thin wrapper for demos / UI)."""
        self.memory.remember(text, tags=tags)

    def reset_conversation(self, keep_long_term: bool = True) -> None:
        """Clear short-term chat; optionally wipe RAG memory too."""
        self.memory.clear_short_term(keep_system=False)
        if not keep_long_term:
            self.memory.long_term.clear()
        self.last_steps = []
        self.last_answer = None
        self._ensure_system_message()
