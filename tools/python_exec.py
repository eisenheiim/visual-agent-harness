"""
Python Exec tool — give the agent a tiny scratchpad interpreter.

This is the "reasoning amplifier": the model can write a short Python
snippet, run it, and read the stdout/result. Classic use cases:

  - Multi-step calculations that are awkward as one calculator expression
  - Parsing / transforming small bits of text
  - Exploring algorithms while you watch the loop in Streamlit

Security note
-------------
This is an **educational sandbox**, not a secure multitenant jail.
We remove many builtins and block obvious imports, but a determined
user can still do harmful things on a shared machine. Only run code
you trust; never expose this tool to the public internet as-is.
"""

from __future__ import annotations

import contextlib
import io
import traceback
from typing import Any

from core.tool_registry import Tool


# Builtins we allow inside the sandbox (keep this list short and boring)
_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}

_BLOCKED_SUBSTRINGS = (
    "import ",
    "from ",
    "__import__",
    "open(",
    "exec(",
    "eval(",
    "compile(",
    "os.",
    "sys.",
    "subprocess",
    "socket",
    "pathlib",
    "shutil",
    "ctypes",
    "pickle",
)


def python_exec(code: str) -> str:
    """Execute a short Python snippet and return stdout + the last expression.

    Parameters
    ----------
    code:
        Python source. Prefer small scripts (a few lines). No imports.
    """
    code = (code or "").strip()
    if not code:
        return "ERROR: empty code"

    lowered = code.lower()
    for bad in _BLOCKED_SUBSTRINGS:
        if bad in lowered:
            return (
                f"ERROR: blocked pattern {bad!r} detected. "
                "This sandbox only allows pure Python with a limited builtin set "
                "(no imports, no file/network access)."
            )

    # Separate last line if it looks like an expression so we can show its value
    # (mimics REPL behavior — hugely helpful for agent reasoning)
    lines = code.splitlines()
    last = lines[-1].strip()
    body = "\n".join(lines[:-1]) if len(lines) > 1 else ""
    show_last_value = bool(last) and not last.startswith(
        ("def ", "class ", "for ", "while ", "if ", "elif ", "else:", "try:", "except", "with ", "return", "#")
    ) and "=" not in last.split("==")[0]  # allow == comparisons, block assignments

    stdout = io.StringIO()
    globals_dict: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
    locals_dict: dict[str, Any] = {}

    try:
        with contextlib.redirect_stdout(stdout):
            if body:
                exec(body, globals_dict, locals_dict)  # noqa: S102 — intentional sandbox
            if show_last_value:
                result = eval(last, globals_dict, locals_dict)  # noqa: S307 — intentional sandbox
            else:
                exec(last, globals_dict, locals_dict)  # noqa: S102
                result = None
    except Exception:  # noqa: BLE001
        return f"ERROR during execution:\n{traceback.format_exc(limit=3)}"

    parts: list[str] = []
    printed = stdout.getvalue()
    if printed:
        parts.append(printed.rstrip())
    if result is not None:
        parts.append(f"=> {result!r}")
    if not parts:
        parts.append("(ok — no output)")
    return "\n".join(parts)


PYTHON_EXEC_TOOL = Tool(
    name="python_exec",
    description=(
        "Run a short Python snippet in a restricted sandbox and read the output. "
        "Use for multi-step logic, loops, list processing, or anything awkward "
        "for the calculator. No imports, no files, no network. "
        "The value of the last expression is returned like a REPL."
    ),
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute (small snippets only).",
            }
        },
        "required": ["code"],
    },
    func=python_exec,
)


def run(**kwargs: Any) -> str:
    return python_exec(**kwargs)
