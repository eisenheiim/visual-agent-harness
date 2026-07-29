"""
Visual Agent Harness — Streamlit UI

Run from the repo root:

    streamlit run ui/app.py

This app is the "wow" surface: every think → act → observe beat of the
agent loop is rendered as a live timeline so newcomers can *see* agents.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import streamlit as st

# Make `core` / `tools` importable when launched via `streamlit run ui/app.py`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Streamlit keeps imported modules in sys.modules across reruns — reload core so
# harness/RAG/parser fixes apply without a manual server restart.
import importlib

import core.env as _core_env
import core.memory as _core_memory
import core.tool_registry as _core_tools
import core.tracing as _core_tracing
import core.harness as _core_harness

importlib.reload(_core_env)
importlib.reload(_core_memory)
importlib.reload(_core_tools)
importlib.reload(_core_tracing)
importlib.reload(_core_harness)

from core.env import load_env  # noqa: E402
from core.harness import AgentEvent, AgentHarness  # noqa: E402
from core.memory import Memory  # noqa: E402
from core.tracing import build_trace_from_harness  # noqa: E402
from tools import register_builtin_tools  # noqa: E402

load_env()


# ---------------------------------------------------------------------------
# Page chrome
# ---------------------------------------------------------------------------

_logo = ROOT / "assets" / "logo.png"
st.set_page_config(
    page_title="Visual Agent Harness",
    page_icon=str(_logo) if _logo.exists() else "🔁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom theme polish — teal/coral to match the logo, not the usual purple AI look.
# Keep text dark on the light main canvas; only use light text on solid teal sidebar chrome.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap');

    :root {
        --vah-teal: #0D7377;
        --vah-teal-deep: #085456;
        --vah-coral: #FF6B4A;
        --vah-sand: #F7F3EE;
        --vah-ink: #1A1F24;
        --vah-muted: #3D474F;
    }

    html, body, .stApp {
        font-family: 'DM Sans', sans-serif;
        color: var(--vah-ink) !important;
    }

    .stApp {
        background:
            radial-gradient(1200px 600px at 10% -10%, rgba(13,115,119,0.12), transparent 60%),
            radial-gradient(900px 500px at 100% 0%, rgba(255,107,74,0.10), transparent 55%),
            linear-gradient(180deg, #FBF9F6 0%, #F3EEE7 100%);
    }

    /* Main pane: force readable dark text (overrides Streamlit dark-mode white text) */
    section.main, section.main p, section.main span, section.main label,
    section.main li, section.main .stMarkdown, section.main [data-testid="stWidgetLabel"] {
        color: var(--vah-ink) !important;
    }
    section.main h1, section.main h2, section.main h3 {
        font-family: 'DM Sans', sans-serif;
        letter-spacing: -0.02em;
        color: var(--vah-ink) !important;
    }
    section.main .stCaption, section.main small {
        color: var(--vah-muted) !important;
    }

    .vah-kicker {
        color: var(--vah-teal) !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.75rem;
        margin: 0;
    }
    .vah-title {
        font-size: 2rem; font-weight: 700; margin: 0.15rem 0 0.35rem 0;
        color: var(--vah-ink) !important;
    }
    .vah-sub {
        color: var(--vah-muted) !important; margin: 0; max-width: 42rem; line-height: 1.5;
    }
    .vah-sub strong { color: var(--vah-ink) !important; }

    .loop-rail {
        display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 1.25rem 0 0.5rem;
    }
    .loop-chip {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        border: 1px solid rgba(13,115,119,0.25);
        background: #ffffff;
        color: var(--vah-teal-deep) !important;
    }
    .loop-chip.active {
        background: var(--vah-teal);
        color: #ffffff !important;
        border-color: var(--vah-teal);
    }
    .loop-chip.coral {
        background: var(--vah-coral);
        color: #ffffff !important;
        border-color: var(--vah-coral);
    }

    .step-card {
        background: #ffffff;
        border: 1px solid rgba(26,31,36,0.08);
        border-left: 4px solid var(--vah-teal);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 8px 24px rgba(26,31,36,0.04);
        color: var(--vah-ink) !important;
    }
    .step-card.tool { border-left-color: var(--vah-coral); }
    .step-card.obs { border-left-color: #2A9D8F; }
    .step-card.final { border-left-color: #C9A227; background: #FFF9EB; }
    .step-card.error { border-left-color: #C1121F; background: #FFF1F0; }
    .step-card.think-in { border-left-color: #4A6FA5; background: #F3F6FB; position: relative; cursor: help; overflow: visible; z-index: 1; }
    .step-card.think-out { border-left-color: #6B5B95; background: #F7F4FB; position: relative; cursor: help; overflow: visible; z-index: 1; }
    .step-card.think-in:hover,
    .step-card.think-out:hover { z-index: 50; }
    .step-card.think-in .step-label::after,
    .step-card.think-out .step-label::after {
        content: " · hover for details";
        text-transform: none;
        letter-spacing: 0;
        font-weight: 500;
        opacity: 0.65;
    }
    .step-hover {
        display: none;
        position: absolute;
        left: 0;
        right: 0;
        top: calc(100% - 4px);
        z-index: 60;
        max-height: min(70vh, 560px);
        overflow: auto;
        padding: 0.9rem 1rem;
        border-radius: 12px;
        border: 1px solid rgba(74, 111, 165, 0.35);
        background: #ffffff;
        box-shadow: 0 16px 40px rgba(26, 31, 36, 0.18);
        color: #1A1F24 !important;
    }
    .step-card.think-out .step-hover {
        border-color: rgba(107, 91, 149, 0.4);
    }
    .step-card.think-in:hover .step-hover,
    .step-card.think-in:focus-within .step-hover,
    .step-card.think-out:hover .step-hover,
    .step-card.think-out:focus-within .step-hover {
        display: block;
    }
    .step-hover h4 {
        margin: 0.75rem 0 0.35rem 0;
        font-size: 0.72rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #4A6FA5 !important;
        font-family: 'JetBrains Mono', monospace;
    }
    .step-hover h4:first-of-type { margin-top: 0.25rem; }
    .step-hover pre {
        margin: 0 0 0.5rem 0;
        white-space: pre-wrap;
        word-break: break-word;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        line-height: 1.45;
        color: #1A1F24 !important;
        background: #F7F9FC;
        border-radius: 8px;
        padding: 0.65rem 0.75rem;
    }
    .step-hover .hint {
        font-size: 0.75rem;
        color: #5C6770 !important;
        margin-bottom: 0.5rem;
    }
    .step-hover .sys-box {
        border: 1px solid rgba(13, 115, 119, 0.35);
        background: #F0FAFA;
        border-radius: 8px;
        padding: 0.65rem 0.75rem;
        margin-bottom: 0.6rem;
    }
    .step-hover .sys-box pre {
        background: transparent;
        padding: 0;
        margin: 0;
        max-height: 240px;
        overflow: auto;
    }

    .step-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--vah-muted) !important;
        margin-bottom: 0.35rem;
    }
    .step-body {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        white-space: pre-wrap;
        line-height: 1.45;
        color: var(--vah-ink) !important;
    }

    /* Sidebar: teal chrome — light labels, but dark text inside light inputs */
    div[data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #0D7377 0%, #085456 100%);
    }
    div[data-testid="stSidebar"] h1,
    div[data-testid="stSidebar"] h2,
    div[data-testid="stSidebar"] h3,
    div[data-testid="stSidebar"] .stMarkdown p,
    div[data-testid="stSidebar"] .stMarkdown,
    div[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    div[data-testid="stSidebar"] label,
    div[data-testid="stSidebar"] .stCaption {
        color: #F7F3EE !important;
    }
    div[data-testid="stSidebar"] input,
    div[data-testid="stSidebar"] textarea,
    div[data-testid="stSidebar"] select,
    div[data-testid="stSidebar"] [data-baseweb="select"] > div,
    div[data-testid="stSidebar"] [data-baseweb="input"],
    div[data-testid="stSidebar"] [data-baseweb="base-input"] {
        background-color: #ffffff !important;
        color: #1A1F24 !important;
    }
    div[data-testid="stSidebar"] [data-baseweb="select"] span,
    div[data-testid="stSidebar"] input,
    div[data-testid="stSidebar"] textarea {
        color: #1A1F24 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "steps": [],
        "answer": None,
        "running": False,
        "memory": Memory(max_short_term=40, store_path=str(ROOT / ".sessions" / "ui_memory.json")),
        "history": [],  # past Q&A for the chat column
        "system_prompt": "",
        "tool_schemas": [],
        "rag_hits": [],
        "prompt_sent": "",
        "rag_injected": False,
        "think_messages": [],  # [{iteration, messages}] for learner mode
        "learner_mode": False,
        "last_trace_json": "",
        "last_trace_id": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    logo_path = ROOT / "assets" / "logo.png"
    if logo_path.exists():
        st.image(str(logo_path), width=72)
    st.markdown("### Controls")
    backend = st.selectbox("Backend", ["ollama", "openai"], index=0)
    default_model = "llama3.2" if backend == "ollama" else "gpt-4o-mini"
    model = st.text_input("Model", value=default_model)
    max_iterations = st.slider("Max iterations", 1, 16, 8)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    use_rag = st.checkbox("Use long-term memory (RAG)", value=True)
    learner_mode = st.checkbox(
        "Learner mode (show internals)",
        value=st.session_state.learner_mode,
        help="Opt-in: anatomy of the loop + raw messages sent to the model each THINK.",
    )
    st.session_state.learner_mode = learner_mode
    st.markdown("---")
    st.caption("Ollama: run `ollama serve` + `ollama pull llama3.2`")
    st.caption("OpenAI: export `OPENAI_API_KEY`")
    st.caption("Search: DuckDuckGo by default · optional `TAVILY_API_KEY`")
    st.caption("Force mock: `VAH_SEARCH_BACKEND=mock`")

    if st.button("Reset conversation", use_container_width=True):
        st.session_state.memory.clear_short_term(keep_system=False)
        st.session_state.steps = []
        st.session_state.answer = None
        st.session_state.history = []
        st.session_state.system_prompt = ""
        st.session_state.tool_schemas = []
        st.session_state.rag_hits = []
        st.session_state.prompt_sent = ""
        st.session_state.rag_injected = False
        st.session_state.think_messages = []
        st.rerun()

    if st.button("Save memory to disk", use_container_width=True):
        path = st.session_state.memory.save()
        st.success(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

logo_html = ""
if (ROOT / "assets" / "logo.png").exists():
    # Streamlit serves local images better via st.image; keep text hero separate
    pass

col_hero, col_loop = st.columns([1.4, 1])
with col_hero:
    st.markdown(
        """
        <p class="vah-kicker">Educational · Local-first · Inspectable</p>
        <h1 class="vah-title">Visual Agent Harness</h1>
        <p class="vah-sub">
          Watch an LLM agent <strong>think</strong>, <strong>call tools</strong>,
          and <strong>observe</strong> results — step by step. Built for learning
          and rapid prototyping, not enterprise black boxes.
        </p>
        """,
        unsafe_allow_html=True,
    )
with col_loop:
    st.markdown(
        """
        <div class="loop-rail">
          <span class="loop-chip active">1 · THINK</span>
          <span class="loop-chip coral">2 · ACT</span>
          <span class="loop-chip">3 · OBSERVE</span>
          <span class="loop-chip">4 · FINAL</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("The same loop every agent framework hides behind abstractions.")


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

EXAMPLES = {
    "Reasoning only": "Explain the think → act → observe loop in two sentences.",
    "Calculator": "What is (144 ** 0.5) * 7 + 3?",
    "Weather (search)": "What's the weather in Paris right now?",
    "Wikipedia": "Use wikipedia to summarize what an LLM agent is.",
    "Datetime": "What day of the week is it right now in Europe/Istanbul?",
    "Unit convert": "Convert 26 celsius to fahrenheit using unit_convert.",
    "HTTP fetch": "Fetch https://example.com with http_get and tell me the page title if you can spot it.",
    "Python scratchpad": "Use python_exec to sum the squares of numbers 1 through 20.",
    "Memory": (
        "Remember that my favorite city is Tokyo. "
        "You must use the remember tool. "
        "Then search for a fun fact about my favorite city."
    ),
}

pick = st.selectbox("Try an example prompt", list(EXAMPLES.keys()))
user_prompt = st.text_area(
    "Your message",
    value=EXAMPLES[pick],
    height=100,
    placeholder="Ask the agent something…",
)

st.markdown("##### Long-term memory")
if st.session_state.pop("_clear_pin", False):
    st.session_state.pin_memory_input = ""

pin_col, pin_btn_col = st.columns([3, 1])
with pin_col:
    st.text_input(
        "Pin a fact (saved immediately — no Run needed)",
        placeholder="e.g. User's favorite city is Tokyo",
        key="pin_memory_input",
        label_visibility="collapsed",
    )
with pin_btn_col:
    pin_clicked = st.button("Pin", use_container_width=True, type="secondary")

if pin_clicked:
    note = (st.session_state.get("pin_memory_input") or "").strip()
    if not note:
        st.warning("Type something to pin first.")
    else:
        item = st.session_state.memory.remember(note, tags=["ui", "pinned"])
        try:
            path = st.session_state.memory.save()
            st.success(f"Pinned `{item.id}` and saved → {path.name}")
        except Exception:
            st.success(f"Pinned `{item.id}` (in-session). Use sidebar to save to disk.")
        st.session_state._clear_pin = True
        st.rerun()

# Show current pins above the run button so feedback is obvious
_pinned = st.session_state.memory.long_term
if _pinned:
    st.caption(
        "Long-term store: "
        + " · ".join(
            f"`{i.id}` {i.text[:40]}{'…' if len(i.text) > 40 else ''}" for i in _pinned[-5:]
        )
    )
else:
    st.caption("Long-term store is empty — pin a fact above, then ask the agent about it.")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def _esc(text: str) -> str:
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _step_event(step) -> str:
    if isinstance(step, dict):
        return str(step.get("event") or "")
    return step.event.value if hasattr(step.event, "value") else str(step.event)


def _step_meta(step) -> dict:
    if isinstance(step, dict):
        return step.get("meta") or {}
    return step.meta or {}


def _step_field(step, name: str, default=None):
    if isinstance(step, dict):
        return step.get(name, default)
    return getattr(step, name, default)


def _classify_llm_message(msg: dict) -> tuple[str, str]:
    """Return (display_role, tool_name) for a chat message sent to the model."""
    role = msg.get("role") or "?"
    content = msg.get("content") or ""
    if role == "system":
        return "system", ""
    # Observations are folded into user role with a clear label (see Memory.as_chat_messages)
    if content.startswith("[OBSERVATION from tool"):
        tool = ""
        start = content.find("`")
        end = content.find("`", start + 1) if start >= 0 else -1
        if start >= 0 and end > start:
            tool = content[start + 1 : end]
        return "observation", tool
    return role, ""


def _compact_system_prompt_for_display(text: str, tool_schemas: list | None = None) -> str:
    """Keep rules; replace bulky tool catalog with names only."""
    text = text or ""
    names: list[str] = []
    if tool_schemas:
        names = [str(t.get("name") or "") for t in tool_schemas if t.get("name")]
    if not names:
        names = re.findall(r"^###\s+(\S+)", text, flags=re.MULTILINE)

    marker = "Available tools:"
    if marker in text:
        head = text.split(marker, 1)[0].rstrip()
    else:
        # Fallback: cut at first ### tool heading
        m = re.search(r"\n###\s+\S+", text)
        head = text[: m.start()].rstrip() if m else text.rstrip()

    if names:
        listing = ", ".join(f"`{n}`" for n in names)
        return f"{head}\n\nAvailable tools (names only): {listing}"
    return head or text


def _think_hover_html(step, *, include_reply: bool = False) -> str:
    """Hover panel — system + tool observations + other messages sent to the model."""
    del include_reply  # kept for call-site compatibility
    meta = _step_meta(step)
    messages = list(meta.get("messages") or [])
    rag_hits = meta.get("rag_hits") or []
    rag_injected = bool(meta.get("rag_injected")) and bool(rag_hits)
    tool_schemas = meta.get("tool_schemas") or st.session_state.get("tool_schemas") or []

    system_msgs = []
    observation_msgs = []
    other_msgs = []
    for msg in messages:
        kind, tool = _classify_llm_message(msg)
        if kind == "system":
            system_msgs.append(msg)
        elif kind == "observation":
            observation_msgs.append((msg, tool))
        else:
            other_msgs.append(msg)

    parts: list[str] = [
        '<div class="step-hover" tabindex="0">',
        '<div class="hint">Hover: what the model receives this THINK</div>',
    ]

    if rag_injected:
        lt = meta.get("long_term_count")
        parts.append(
            f"<h4>RAG hits · store size {lt if lt is not None else '?'}</h4>"
        )
        lines = [
            f"[{hit.get('score', 0):.2f}] ({', '.join(hit.get('tags') or []) or '—'}) {hit.get('text', '')}"
            for hit in rag_hits
        ]
        parts.append(f"<pre>{_esc(chr(10).join(lines))}</pre>")

    # System prompt — rules + tool names only (no descriptions / schemas)
    parts.append("<h4>System prompt</h4>")
    parts.append('<div class="sys-box">')
    if system_msgs:
        raw = system_msgs[0].get("content") or ""
        shown = _compact_system_prompt_for_display(raw, tool_schemas)
        parts.append(f"<pre>{_esc(shown)}</pre>")
    else:
        parts.append("<pre>(missing)</pre>")
    parts.append("</div>")

    # Tool observations next to system — only when present (after a tool turn)
    if observation_msgs:
        parts.append(f"<h4>Tool observations sent to model ({len(observation_msgs)})</h4>")
        blocks = []
        for msg, tool in observation_msgs:
            content = msg.get("content") or ""
            label = f"observation · {tool}" if tool else "observation"
            shown = content if len(content) <= 2000 else content[:2000] + f"\n… ({len(content)} chars)"
            blocks.append(f"── {label} ({len(content)} chars) ──\n{shown}")
        parts.append(f"<pre>{_esc(chr(10).join(blocks))}</pre>")

    parts.append(f"<h4>Other messages ({len(other_msgs)})</h4>")
    if other_msgs:
        msg_blocks = []
        for i, msg in enumerate(other_msgs, 1):
            role = msg.get("role", "?")
            content = msg.get("content") or ""
            shown = content if len(content) <= 1200 else content[:1200] + f"\n… ({len(content)} chars)"
            msg_blocks.append(f"── [{i}] {role} ({len(content)} chars) ──\n{shown}")
        parts.append(f"<pre>{_esc(chr(10).join(msg_blocks))}</pre>")
    else:
        parts.append("<pre>(none)</pre>")

    parts.append("</div>")
    return "".join(parts)


def _render_step(step) -> str:
    """Return HTML for one timeline card (accepts AgentStep or dict)."""
    kind = "step-card"
    ev = _step_event(step)
    meta = _step_meta(step)
    body = _step_field(step, "content") or ""
    iteration = _step_field(step, "iteration") or 0
    tool_name = _step_field(step, "tool_name")
    tool_args = _step_field(step, "tool_args")
    label = ev.upper().replace("_", " ")
    hover = ""

    if ev == AgentEvent.TOOL_CALL.value:
        kind += " tool"
        label = f"ACT · {tool_name}"
        body = f"args = {tool_args}\n\n{body}"
    elif ev == AgentEvent.OBSERVATION.value:
        kind += " obs"
        label = f"OBSERVE · {tool_name}"
    elif ev == AgentEvent.FINAL.value:
        kind += " final"
        label = "FINAL ANSWER"
    elif ev == AgentEvent.ERROR.value:
        kind += " error"
    elif ev == AgentEvent.THINK_START.value:
        kind += " think-in"
        n = meta.get("message_count") or len(meta.get("messages") or [])
        rag_hits = meta.get("rag_hits") or []
        rag_n = len(rag_hits) if meta.get("rag_injected") else 0
        messages = meta.get("messages") or []
        obs = []
        sys_len = 0
        for msg in messages:
            kind_m, tool = _classify_llm_message(msg)
            if kind_m == "system":
                sys_len = len(msg.get("content") or "")
            elif kind_m == "observation":
                content = msg.get("content") or ""
                preview = content if len(content) <= 220 else content[:220].rstrip() + "…"
                obs.append((tool or "tool", preview, len(content)))

        bits = [f"Calling the model with {n} message(s)"]
        if sys_len:
            bits.append(f"system {sys_len} chars")
        if rag_n:
            bits.append(f"rag {rag_n}")
        if obs:
            bits.append(f"{len(obs)} tool observation(s)")
        label = f"THINK START · iter {iteration} · " + " · ".join(
            [f"{n} msgs"] + ([f"rag {rag_n}"] if rag_n else []) + ([f"obs {len(obs)}"] if obs else [])
        )
        body_lines = [" · ".join(bits), ""]
        if sys_len:
            body_lines.append(f"── system ({sys_len} chars) ──")
            body_lines.append("(full text on hover)")
            body_lines.append("")
        for tool, preview, length in obs:
            body_lines.append(f"── observation · {tool} ({length} chars) ──")
            body_lines.append(preview)
            body_lines.append("")
        body = "\n".join(body_lines).rstrip()
        hover = _think_hover_html(step, include_reply=False)
    elif ev == AgentEvent.THINK_END.value:
        kind += " think-out"
        label = f"THINK END · iteration {iteration}"
        n = len(str(body))
        body = f"Model responded ({n} chars)"
        hover = _think_hover_html(step, include_reply=True)
    elif ev == AgentEvent.RUN_START.value:
        hits = meta.get("rag_hits") or []
        if meta.get("rag_injected") and hits:
            lt = meta.get("long_term_count", "?")
            rag_lines = [f"RAG injected ({len(hits)} hit(s) from store size {lt}):"]
            for hit in hits:
                tags = ", ".join(hit.get("tags") or []) or "—"
                rag_lines.append(
                    f"  • [{hit.get('score', 0):.2f}] ({tags}) {hit.get('text', '')}"
                )
            body = f"User: {body}\n\n" + "\n".join(rag_lines)
        else:
            body = f"User: {body}"
    elif ev == AgentEvent.RUN_END.value:
        body = body or "Run finished."

    safe = _esc(str(body))
    return (
        f'<div class="{kind}">'
        f'<div class="step-label">{label}</div>'
        f'<div class="step-body">{safe}</div>'
        f"{hover}"
        f"</div>"
    )


go = st.button("Run agent", type="primary", use_container_width=True)

if go and user_prompt.strip():
    harness = AgentHarness(
        backend=backend,  # type: ignore[arg-type]
        model=model,
        memory=st.session_state.memory,
        max_iterations=max_iterations,
        temperature=temperature,
    )
    register_builtin_tools(harness.tools, memory=st.session_state.memory)
    harness._ensure_system_message()

    st.session_state.steps = []
    st.session_state.think_messages = []
    timeline = st.empty()
    status = st.status("Agent loop running…", expanded=True)

    collected = []
    try:
        for step in harness.stream(user_prompt.strip(), use_rag=use_rag):
            # Store plain dicts so Streamlit session_state keeps meta intact
            payload = step.to_dict()
            collected.append(payload)
            st.session_state.steps = collected
            if step.event == AgentEvent.RUN_START:
                st.session_state.system_prompt = step.meta.get("system_prompt") or ""
                st.session_state.tool_schemas = step.meta.get("tool_schemas") or []
                st.session_state.rag_hits = step.meta.get("rag_hits") or []
                st.session_state.prompt_sent = step.meta.get("prompt_sent_to_model") or ""
                st.session_state.rag_injected = bool(step.meta.get("rag_injected"))
            if step.event == AgentEvent.THINK_START and step.meta.get("messages") is not None:
                st.session_state.think_messages.append(
                    {
                        "iteration": step.iteration,
                        "messages": step.meta.get("messages") or [],
                    }
                )
            # Live timeline
            html = "\n".join(_render_step(s) for s in collected)
            timeline.markdown(html, unsafe_allow_html=True)

            # Status line
            if step.event == AgentEvent.THINK_START:
                rag_n = len(step.meta.get("rag_hits") or []) if step.meta.get("rag_injected") else 0
                if rag_n:
                    status.write(f"Iteration {step.iteration}: thinking… (rag {rag_n})")
                else:
                    status.write(f"Iteration {step.iteration}: thinking…")
            elif step.event == AgentEvent.TOOL_CALL:
                status.write(f"Calling tool `{step.tool_name}`…")
            elif step.event == AgentEvent.OBSERVATION:
                status.write(f"Got observation from `{step.tool_name}`")
            elif step.event == AgentEvent.FINAL:
                status.write("Final answer ready")
            elif step.event == AgentEvent.ERROR:
                status.write(f"Error: {step.content}")

            # Tiny pause so the UI feels alive on fast local models
            time.sleep(0.05)

        st.session_state.answer = harness.last_answer
        # Keep authoritative copies from the harness too
        if harness.last_system_prompt:
            st.session_state.system_prompt = harness.last_system_prompt
        if harness.last_rag_hits is not None:
            st.session_state.rag_hits = harness.last_rag_hits
        try:
            trace = build_trace_from_harness(harness)
            st.session_state.last_trace_json = json.dumps(
                trace.to_dict(), indent=2, ensure_ascii=False
            )
            st.session_state.last_trace_id = trace.id
            traces_dir = ROOT / "traces"
            traces_dir.mkdir(parents=True, exist_ok=True)
            trace.save(traces_dir / f"ui_{trace.id}.json")
        except Exception:
            st.session_state.last_trace_json = ""
            st.session_state.last_trace_id = ""
        st.session_state.history.append(
            {"q": user_prompt.strip(), "a": harness.last_answer or ""}
        )
        status.update(label="Done", state="complete")
    except Exception as exc:  # noqa: BLE001
        status.update(label="Failed", state="error")
        st.error(f"{type(exc).__name__}: {exc}")


def _anatomy_notes(steps: list) -> list[str]:
    """Plain-English captions for what happened in this run."""
    notes: list[str] = []
    for step in steps:
        ev = _step_event(step)
        meta = _step_meta(step)
        content = _step_field(step, "content") or ""
        iteration = _step_field(step, "iteration") or 0
        tool_name = _step_field(step, "tool_name")
        tool_args = _step_field(step, "tool_args")

        if ev == AgentEvent.RUN_START.value:
            if meta.get("rag_injected"):
                n = len(meta.get("rag_hits") or [])
                notes.append(
                    f"**RUN_START** — User message accepted. RAG injected {n} long-term hit(s) "
                    f"(store size {meta.get('long_term_count', '?')}) "
                    "into the prompt before the first THINK."
                )
            else:
                notes.append(
                    "**RUN_START** — User message accepted. No RAG context was prepended "
                    f"(store size {meta.get('long_term_count', '?')}, "
                    "RAG off, or nothing matched)."
                )
        elif ev == AgentEvent.THINK_END.value:
            notes.append(
                f"**THINK #{iteration}** — The model saw the chat history "
                f"({meta.get('message_count', '?')} messages) and replied. "
                "Next we parse that reply for tool JSON or a final answer."
            )
        elif ev == AgentEvent.TOOL_CALL.value:
            auto = " *(auto safety-net)*" if meta.get("auto") else ""
            notes.append(
                f"**ACT · `{tool_name}`**{auto} — The harness executed a real Python "
                f"function with args `{tool_args}`."
            )
        elif ev == AgentEvent.OBSERVATION.value:
            notes.append(
                f"**OBSERVE · `{tool_name}`** — Tool output was appended as an "
                "`observation` message so the *next* THINK can read it."
            )
        elif ev == AgentEvent.FINAL.value:
            notes.append(
                "**FINAL** — Loop stops. The string below is what you’d show the end user."
            )
        elif ev == AgentEvent.ERROR.value:
            notes.append(f"**ERROR** — {content}")
    return notes


# ---------------------------------------------------------------------------
# Results layout
# ---------------------------------------------------------------------------

left, right = st.columns([1.35, 1])

with left:
    st.subheader("Loop timeline")
    if st.session_state.steps:
        html = "\n".join(_render_step(s) for s in st.session_state.steps)
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.info("Run the agent to see think → act → observe events here.")

    if st.session_state.answer:
        st.success(st.session_state.answer)

    if st.session_state.get("last_trace_json"):
        tid = st.session_state.get("last_trace_id") or "run"
        st.download_button(
            label="Download run trace (JSON)",
            data=st.session_state.last_trace_json,
            file_name=f"trace_{tid}.json",
            mime="application/json",
            use_container_width=True,
        )

with right:
    st.subheader("Memory")
    snap = st.session_state.memory.snapshot()
    m1, m2 = st.columns(2)
    m1.metric("Short-term messages", snap["short_term_count"])
    m2.metric("Long-term snippets", snap["long_term_count"])

    with st.expander("Short-term conversation", expanded=False):
        for msg in snap["short_term"]:
            role = msg["role"]
            content = msg["content"]
            preview = content if len(content) < 400 else content[:400] + "…"
            st.markdown(f"**{role}**")
            st.code(preview, language="text")

    with st.expander("Long-term (RAG) store", expanded=True):
        items = snap.get("long_term_clean") or []
        if not items:
            st.caption("Empty — use the **Pin** button above the Run agent control.")
        for item in items:
            tags = ", ".join(item.get("tags") or []) or "—"
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(f"`{item['id']}` · _{tags}_")
                st.write(item["text"])
            with c2:
                if st.button("Forget", key=f"forget_{item['id']}", use_container_width=True):
                    st.session_state.memory.forget(item["id"])
                    try:
                        st.session_state.memory.save()
                    except Exception:
                        pass
                    st.rerun()

    with st.expander("Registered tools", expanded=False):
        st.markdown(
            """
            | Tool | Purpose |
            |---|---|
            | `calculator` | Safe arithmetic AST |
            | `search` | Live DuckDuckGo / Tavily (mock offline) |
            | `python_exec` | Restricted REPL |
            | `wikipedia` | Free encyclopedia summary |
            | `datetime_now` | Real clock + timezone |
            | `http_get` | Fetch a public URL |
            | `unit_convert` | Length / mass / temp / volume |
            | `remember` | Save a fact to long-term RAG |
            """
        )

# ---------------------------------------------------------------------------
# Learner mode — opt-in internals (hidden unless sidebar toggle is on)
# ---------------------------------------------------------------------------

if st.session_state.learner_mode:
    st.markdown("---")
    st.subheader("Learner mode")
    st.caption("Opt-in internals — turn off anytime from the sidebar.")

    a1, a2 = st.columns(2)
    with a1:
        with st.expander("Anatomy of this run", expanded=True):
            notes = _anatomy_notes(st.session_state.steps or [])
            if not notes:
                st.caption("Run the agent once to get a narrated walkthrough of each beat.")
            else:
                for note in notes:
                    st.markdown(f"- {note}")
    with a2:
        hits = st.session_state.rag_hits or []
        if st.session_state.rag_injected and hits:
            with st.expander("RAG hits (this turn)", expanded=True):
                for hit in hits:
                    tags = ", ".join(hit.get("tags") or []) or "—"
                    st.markdown(
                        f"`{hit.get('id', '?')}` · score **{hit.get('score', 0):.2f}** · _{tags}_"
                    )
                    st.write(hit.get("text", ""))
                if st.session_state.prompt_sent:
                    with st.popover("Full user prompt sent to model"):
                        st.code(st.session_state.prompt_sent, language="text")

    with st.expander("Raw messages per THINK (what the model saw)", expanded=True):
        batches = st.session_state.think_messages or []
        if not batches:
            st.caption("Run the agent to capture the exact chat payload for each THINK.")
        else:
            for batch in batches:
                st.markdown(
                    f"**Iteration {batch['iteration']}** — {len(batch['messages'])} messages"
                )
                st.code(
                    json.dumps(batch["messages"], indent=2, ensure_ascii=False),
                    language="json",
                )

    with st.expander("Tool schemas", expanded=False):
        schemas = st.session_state.tool_schemas or []
        if not schemas:
            st.caption("Run the agent once to capture registered tool schemas.")
        else:
            st.code(json.dumps(schemas, indent=2), language="json")
        st.caption("System prompt is hidden from the UI — see `core/harness.py` if you need it.")
else:
    st.caption("Tip: enable **Learner mode** in the sidebar to inspect anatomy, RAG, and raw model messages.")

st.markdown("---")
st.caption(
    "Visual Agent Harness · MIT · Education & prototyping — "
    "star the repo if this made agents click for you."
)
