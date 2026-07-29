#!/usr/bin/env python3
"""
Example 01 — Basic reasoning (no tools required).

Shows the harness running a single think → final cycle and printing
each AgentEvent. Great smoke test after `ollama pull llama3.2`.

    python examples/01_basic_reasoning.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.panel import Panel

from core import AgentEvent, AgentHarness

console = Console()


def main() -> None:
    harness = AgentHarness(
        backend="ollama",
        model="llama3.2",
        max_iterations=4,
        temperature=0.3,
    )
    # Intentionally NO tools — pure reasoning

    console.print(Panel.fit("[bold cyan]01 · Basic Reasoning[/]", subtitle="no tools"))

    question = "In two short sentences, what is an LLM agent harness?"
    console.print(f"\n[bold]User:[/] {question}\n")

    for step in harness.stream(question, use_rag=False):
        if step.event == AgentEvent.THINK_END:
            console.print(f"[dim]think (iter {step.iteration}):[/]\n{step.content}\n")
        elif step.event == AgentEvent.FINAL:
            console.print(Panel(step.content, title="Final", border_style="green"))
        elif step.event == AgentEvent.ERROR:
            console.print(f"[red]Error:[/] {step.content}")


if __name__ == "__main__":
    main()
