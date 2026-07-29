"""
Built-in tools for Visual Agent Harness.

  calculator     — safe arithmetic (no eval of arbitrary code)
  search         — live DuckDuckGo / optional Tavily / mock offline
  python_exec    — restricted Python for multi-step reasoning / data work
  wikipedia      — English Wikipedia summaries (no API key)
  datetime_now   — real clock in an IANA timezone
  http_get       — fetch a public URL (truncated)
  unit_convert   — length / mass / volume / temperature conversions
  remember       — write into long-term (RAG) memory

Call :func:`register_builtin_tools` to attach all of them to a registry.
Pass ``memory=`` so ``remember`` can store facts.
"""

from __future__ import annotations

from typing import Any, Optional

from core.memory import Memory
from core.tool_registry import Tool, ToolRegistry

from .calculator import CALCULATOR_TOOL, calculator
from .datetime_now import DATETIME_TOOL, datetime_now
from .http_get import HTTP_GET_TOOL, http_get
from .python_exec import PYTHON_EXEC_TOOL, python_exec
from .remember import REMEMBER_TOOL, bind_memory, remember
from .search import SEARCH_TOOL, search
from .unit_convert import UNIT_CONVERT_TOOL, unit_convert
from .wikipedia import WIKIPEDIA_TOOL, wikipedia

BUILTIN_TOOLS = (
    CALCULATOR_TOOL,
    SEARCH_TOOL,
    PYTHON_EXEC_TOOL,
    WIKIPEDIA_TOOL,
    DATETIME_TOOL,
    HTTP_GET_TOOL,
    UNIT_CONVERT_TOOL,
    REMEMBER_TOOL,
)


def register_builtin_tools(
    registry: ToolRegistry,
    memory: Optional[Memory] = None,
) -> ToolRegistry:
    """Register every built-in tool on ``registry``.

    If ``memory`` is provided, the ``remember`` tool writes into that store.
    """
    for tool in BUILTIN_TOOLS:
        registry.register(tool)
    if memory is not None:
        bind_memory(memory.remember)
    return registry


__all__ = [
    "calculator",
    "search",
    "python_exec",
    "wikipedia",
    "datetime_now",
    "http_get",
    "unit_convert",
    "remember",
    "bind_memory",
    "CALCULATOR_TOOL",
    "SEARCH_TOOL",
    "PYTHON_EXEC_TOOL",
    "WIKIPEDIA_TOOL",
    "DATETIME_TOOL",
    "HTTP_GET_TOOL",
    "UNIT_CONVERT_TOOL",
    "REMEMBER_TOOL",
    "BUILTIN_TOOLS",
    "register_builtin_tools",
    "Tool",
]
