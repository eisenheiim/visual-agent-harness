"""
Evals — small, readable regression suite for the agent loop.

Each case is a JSON object:
  id, prompt, checks...

Checks (all optional):
  must_call_tools   — every name in this list must appear as a TOOL_CALL
  forbid_tools      — none of these tool names may be called
  answer_contains   — final answer must include each substring (case-insensitive)
  answer_regex      — final answer must match this regex
  max_thinks        — think_end count must be <= this
  require_no_error  — fail if an ERROR event occurred (default true)

Run::

    python -m evals.run
    python -m evals.run --case calc_basic
    VAH_SEARCH_BACKEND=mock python -m evals.run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import AgentEvent, AgentHarness, Memory  # noqa: E402
from core.tracing import RunTrace, build_trace_from_harness  # noqa: E402

CASES_PATH = Path(__file__).parent / "cases.json"
TRACES_DIR = ROOT / "traces" / "evals"


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    checks: list[CheckResult]
    trace_path: Optional[Path] = None
    answer: str = ""
    error: str = ""


def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def score_trace(trace: RunTrace, case: dict[str, Any]) -> list[CheckResult]:
    """Apply declarative checks to a finished trace."""
    results: list[CheckResult] = []
    answer = (trace.final_answer or "").strip()
    tools = trace.tool_names()
    tools_set = set(tools)

    if case.get("require_no_error", True):
        ok = not trace.has_error()
        results.append(
            CheckResult("require_no_error", ok, "ERROR event present" if not ok else "ok")
        )

    must = case.get("must_call_tools") or []
    if must:
        missing = [t for t in must if t not in tools_set]
        results.append(
            CheckResult(
                "must_call_tools",
                not missing,
                f"called={tools}; missing={missing}" if missing else f"called={tools}",
            )
        )

    forbid = case.get("forbid_tools") or []
    if forbid:
        bad = [t for t in forbid if t in tools_set]
        results.append(
            CheckResult(
                "forbid_tools",
                not bad,
                f"forbidden called: {bad}" if bad else "ok",
            )
        )

    for needle in case.get("answer_contains") or []:
        ok = needle.lower() in answer.lower()
        results.append(
            CheckResult(
                f"answer_contains:{needle!r}",
                ok,
                f"answer={answer[:180]!r}" if not ok else "ok",
            )
        )

    pattern = case.get("answer_regex")
    if pattern:
        ok = re.search(pattern, answer, flags=re.IGNORECASE | re.DOTALL) is not None
        results.append(
            CheckResult(
                f"answer_regex:{pattern}",
                ok,
                f"answer={answer[:180]!r}" if not ok else "ok",
            )
        )

    max_thinks = case.get("max_thinks")
    if max_thinks is not None:
        n = trace.think_count()
        ok = n <= int(max_thinks)
        results.append(CheckResult("max_thinks", ok, f"thinks={n} max={max_thinks}"))

    return results


def run_case(
    case: dict[str, Any],
    *,
    backend: str = "ollama",
    model: str = "llama3.2",
    save_traces: bool = True,
) -> CaseResult:
    """Execute one eval case against a fresh harness."""
    from core.env import load_env
    from tools import register_builtin_tools

    load_env()
    memory = Memory()
    # Seed long-term memory when the case asks for it
    for item in case.get("seed_memories") or []:
        memory.remember(item.get("text", ""), tags=item.get("tags") or ["eval"])

    harness = AgentHarness(
        backend=backend,  # type: ignore[arg-type]
        model=model,
        memory=memory,
        max_iterations=int(case.get("max_iterations") or 8),
        temperature=float(case.get("temperature") or 0.1),
    )
    register_builtin_tools(harness.tools, memory=memory)
    harness._ensure_system_message()

    try:
        harness.run(case["prompt"], use_rag=bool(case.get("use_rag", True)))
        trace = build_trace_from_harness(harness)
        trace.user_message = case["prompt"]
        trace.meta["case_id"] = case["id"]
        checks = score_trace(trace, case)
        path = None
        if save_traces:
            TRACES_DIR.mkdir(parents=True, exist_ok=True)
            path = trace.save(TRACES_DIR / f"{case['id']}_{trace.id}.json")
        passed = all(c.passed for c in checks)
        return CaseResult(
            case_id=case["id"],
            passed=passed,
            checks=checks,
            trace_path=path,
            answer=trace.final_answer or "",
        )
    except Exception as exc:  # noqa: BLE001
        return CaseResult(
            case_id=case["id"],
            passed=False,
            checks=[CheckResult("runtime", False, f"{type(exc).__name__}: {exc}")],
            error=str(exc),
        )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Visual Agent Harness — eval runner")
    parser.add_argument("--case", help="Run a single case id")
    parser.add_argument("--backend", default="ollama", choices=["ollama", "openai"])
    parser.add_argument("--model", default="llama3.2")
    parser.add_argument("--no-save", action="store_true", help="Do not write trace JSON files")
    parser.add_argument("--list", action="store_true", help="List case ids and exit")
    args = parser.parse_args(argv)

    cases = load_cases()
    if args.list:
        for c in cases:
            print(f"{c['id']:20} {c.get('prompt', '')[:70]}")
        return 0

    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            print(f"Unknown case: {args.case}", file=sys.stderr)
            return 2

    print(f"Running {len(cases)} case(s) · backend={args.backend} model={args.model}")
    print("Tip: VAH_SEARCH_BACKEND=mock for offline-deterministic search\n")

    results: list[CaseResult] = []
    for case in cases:
        print(f"→ {case['id']} …", flush=True)
        result = run_case(
            case,
            backend=args.backend,
            model=args.model,
            save_traces=not args.no_save,
        )
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"  {status}  answer={result.answer[:100]!r}")
        for check in result.checks:
            mark = "✓" if check.passed else "✗"
            print(f"    {mark} {check.name}: {check.detail}")
        if result.trace_path:
            print(f"    trace → {result.trace_path}")
        print()

    passed = sum(1 for r in results if r.passed)
    print(f"Summary: {passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
