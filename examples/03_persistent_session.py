#!/usr/bin/env python3
"""
Example 03 — Persistent session (short-term + long-term memory).

Demonstrates:
  - Saving / loading memory JSON across process runs
  - RAG recall injected into a later question

    python examples/03_persistent_session.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.panel import Panel

from core import AgentEvent, AgentHarness, Memory
from tools import register_builtin_tools

console = Console()
SESSION = ROOT / ".sessions" / "example03.json"


def build_harness(memory: Memory) -> AgentHarness:
    h = AgentHarness(
        backend="ollama",
        model="llama3.2",
        memory=memory,
        max_iterations=6,
        temperature=0.2,
    )
    register_builtin_tools(h.tools)
    h._ensure_system_message()
    return h


def run_once(harness: AgentHarness, prompt: str) -> str:
    console.print(f"\n[bold]User:[/] {prompt}")
    for step in harness.stream(prompt, use_rag=True):
        if step.event == AgentEvent.TOOL_CALL:
            console.print(f"  [yellow]→ tool[/] {step.tool_name} {step.tool_args}")
        elif step.event == AgentEvent.FINAL:
            console.print(Panel(step.content, title="Final", border_style="green"))
    return harness.last_answer or ""


def main() -> None:
    console.print(Panel.fit("[bold cyan]03 · Persistent Session[/]", subtitle="memory save/load + RAG"))

    SESSION.parent.mkdir(parents=True, exist_ok=True)
    memory = Memory(max_short_term=30, store_path=SESSION)

    # Seed long-term facts (simulates prior sessions / user prefs)
    if not memory.long_term:
        memory.remember("The user's name is Alex.", tags=["profile"])
        memory.remember("Alex prefers answers in Celsius and metric units.", tags=["preference"])
        memory.remember("Alex is learning how LLM agents use tools.", tags=["context"])
        console.print("[dim]Seeded long-term memories.[/]")
    else:
        console.print(f"[dim]Loaded {len(memory.long_term)} long-term memories from {SESSION}[/]")

    harness = build_harness(memory)

    run_once(
        harness,
        "Using what you remember about me, recommend whether I should check "
        "Tokyo weather in C or F — then search for Tokyo weather and answer briefly.",
    )

    # Persist everything (short-term chat + long-term store)
    path = memory.save()
    console.print(f"\n[bold green]Saved session →[/] {path}")
    console.print("[dim]Re-run this script to resume with the same long-term memories.[/]")


if __name__ == "__main__":
    main()
