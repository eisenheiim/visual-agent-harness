#!/usr/bin/env python3
"""
Example 02 — Tool use (search + calculator).

The agent should:
  1. Call `search` for Paris weather (mock corpus — no API key)
  2. Optionally use `calculator` if it wants to convert units
  3. Emit a final answer

    python examples/02_tool_use_weather.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from core import AgentEvent, AgentHarness
from tools import register_builtin_tools

console = Console()


def main() -> None:
    harness = AgentHarness(
        backend="ollama",
        model="llama3.2",
        max_iterations=8,
        temperature=0.1,
    )
    register_builtin_tools(harness.tools)
    harness._ensure_system_message()

    console.print(Panel.fit("[bold cyan]02 · Tool Use — Weather[/]", subtitle="search + calculator"))

    question = (
        "Search for the weather in Paris. "
        "Then tell me the temperature in both Celsius (as reported) and Fahrenheit "
        "(convert with the calculator: F = C * 9/5 + 32)."
    )
    console.print(f"\n[bold]User:[/] {question}\n")

    table = Table(title="Agent loop", show_header=True, header_style="bold")
    table.add_column("Iter", style="dim", width=4)
    table.add_column("Event", style="cyan", width=14)
    table.add_column("Detail")

    for step in harness.stream(question, use_rag=False):
        detail = ""
        if step.event == AgentEvent.TOOL_CALL:
            detail = f"{step.tool_name} {json.dumps(step.tool_args)}"
        elif step.event == AgentEvent.OBSERVATION:
            detail = (step.content[:120] + "…") if len(step.content) > 120 else step.content
        elif step.event == AgentEvent.THINK_END:
            detail = (step.content[:100] + "…") if len(step.content) > 100 else step.content
        elif step.event == AgentEvent.FINAL:
            detail = step.content
        elif step.event == AgentEvent.ERROR:
            detail = step.content
        else:
            continue

        table.add_row(str(step.iteration or "—"), step.event.value, detail)

    console.print(table)
    console.print(Panel(harness.last_answer or "(no answer)", title="Final", border_style="green"))


if __name__ == "__main__":
    main()
