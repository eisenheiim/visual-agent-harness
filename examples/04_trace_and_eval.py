#!/usr/bin/env python3
"""
Example 04 — Tracing + evals smoke demo.

Saves one run trace to traces/ and prints a summary. For the full suite::

    VAH_SEARCH_BACKEND=mock python -m evals.run

    python examples/04_trace_and_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.panel import Panel

from core import AgentHarness, build_trace_from_harness
from core.env import load_env
from tools import register_builtin_tools

console = Console()


def main() -> None:
    load_env()
    harness = AgentHarness(backend="ollama", model="llama3.2", max_iterations=6, temperature=0.1)
    register_builtin_tools(harness.tools, memory=harness.memory)
    harness._ensure_system_message()

    prompt = "Use the calculator to compute (12 + 3) * 2."
    console.print(Panel.fit("[bold cyan]04 · Trace export[/]", subtitle=prompt))
    answer = harness.run(prompt, use_rag=False)
    console.print(Panel(answer or "(empty)", title="Final", border_style="green"))

    trace = build_trace_from_harness(harness)
    out = ROOT / "traces" / f"demo_{trace.id}.json"
    trace.save(out)
    console.print(f"[bold]Trace saved →[/] {out}")
    console.print(trace.summary())


if __name__ == "__main__":
    main()
