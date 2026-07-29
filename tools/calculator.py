"""
Calculator tool — precise math without letting the LLM freestyle arithmetic.

LLMs are notoriously bad at multi-digit math. Route those questions here.
We parse a restricted expression AST (numbers + arithmetic ops only) —
no ``eval``, no attribute access, no function calls.
"""

from __future__ import annotations

import ast
import operator
from typing import Any

from core.tool_registry import Tool


# Only these binary / unary operators are allowed
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate a safe AST node."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    # Python < 3.8 used ast.Num; it was removed in 3.14 — guard with getattr
    _Num = getattr(ast, "Num", None)
    if _Num is not None and isinstance(node, _Num):  # type: ignore[arg-type]
        return float(node.n)  # type: ignore[attr-defined]

    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        # Guard absurd exponentiation that could hang the process
        if isinstance(node.op, ast.Pow) and (abs(left) > 1e6 or abs(right) > 1000):
            raise ValueError("Exponentiation too large — keep bases/exponents modest.")
        return _BIN_OPS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))

    raise ValueError(
        f"Unsupported expression element: {type(node).__name__}. "
        "Only numbers and + - * / // % ** are allowed."
    )


def calculator(expression: str) -> str:
    """Evaluate a math expression and return the result as a string.

    Parameters
    ----------
    expression:
        e.g. ``"(12.5 + 3) * 2 ** 3"``
    """
    expression = (expression or "").strip()
    if not expression:
        return "ERROR: empty expression"

    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"

    # Pretty-print integers without trailing .0
    if isinstance(result, float) and result.is_integer():
        return str(int(result))
    return str(result)


CALCULATOR_TOOL = Tool(
    name="calculator",
    description=(
        "Evaluate a mathematical expression with precise arithmetic. "
        "Use for any calculation involving numbers. "
        "Supports + - * / // % ** and parentheses. No variables or functions."
    ),
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Math expression to evaluate, e.g. '(3+5)*2'.",
            }
        },
        "required": ["expression"],
    },
    func=calculator,
)


# Allow `from tools.calculator import calculator` style demos
def run(**kwargs: Any) -> str:
    return calculator(**kwargs)
