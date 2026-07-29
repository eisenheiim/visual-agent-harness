"""
Tool Registry — the bridge between LLM text and real function calls.

LLMs cannot "run code" by themselves. They emit structured *tool calls*
(name + arguments). This registry:

  1. Holds callable Python functions with JSON-like schemas
  2. Formats those schemas into the system prompt (so the model knows what's available)
  3. Parses the model's TOOL_CALL JSON and executes the matching function
  4. Returns observations the model can read on the next turn

This is the educational heart of "tool use" — keep it small and readable.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    """A single LLM-callable tool.

    Attributes:
        name: Unique identifier the model must use in TOOL_CALL JSON.
        description: Plain-English what/when — models lean heavily on this.
        parameters: JSON-Schema-ish dict describing arguments.
        func: The actual Python callable.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]

    def schema(self) -> dict[str, Any]:
        """Return the public schema (no Python function — safe to put in prompts)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def run(self, **kwargs: Any) -> str:
        """Execute and always return a string observation for the LLM."""
        result = self.func(**kwargs)
        return result if isinstance(result, str) else json.dumps(result, default=str)


@dataclass
class ToolCall:
    """Parsed tool invocation requested by the model."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw: str = ""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Common mistakes small local models make — map to the real registered name.
_TOOL_ALIASES: dict[str, str] = {
    "weather": "search",
    "weather_search": "search",
    "get_weather": "search",
    "web_search": "search",
    "websearch": "search",
    "google": "search",
    "bing": "search",
    "wiki": "wikipedia",
    "wiki_search": "wikipedia",
    "calc": "calculator",
    "calculate": "calculator",
    "math": "calculator",
    "python": "python_exec",
    "code": "python_exec",
    "code_exec": "python_exec",
    "fetch": "http_get",
    "browse": "http_get",
    "url_fetch": "http_get",
    "time": "datetime_now",
    "date": "datetime_now",
    "clock": "datetime_now",
    "convert": "unit_convert",
    "units": "unit_convert",
    "memory": "remember",
    "save_memory": "remember",
    "store": "remember",
}


class ToolRegistry:
    """Register tools, expose schemas to the LLM, and execute tool calls.

    Usage::

        registry = ToolRegistry()
        registry.register(Tool(name="add", description="Add two numbers", ...))
        prompt_block = registry.prompt_block()   # inject into system prompt
        result = registry.execute("add", {"a": 1, "b": 2})
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    # -- registration -------------------------------------------------------

    def register(self, tool: Tool) -> None:
        """Add or replace a tool by name."""
        self._tools[tool.name] = tool

    def register_function(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        func: Callable[..., Any],
    ) -> None:
        """Convenience wrapper around :meth:`register`."""
        self.register(Tool(name=name, description=description, parameters=parameters, func=func))

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def resolve_name(self, name: str) -> str:
        """Map alias / casing to a registered tool name when possible."""
        key = (name or "").strip()
        if key in self._tools:
            return key
        lower = key.lower().replace("-", "_").replace(" ", "_")
        if lower in self._tools:
            return lower
        alias = _TOOL_ALIASES.get(lower)
        if alias and alias in self._tools:
            return alias
        return key

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    # -- prompt formatting --------------------------------------------------

    def prompt_block(self) -> str:
        """Human-readable tool catalog for the system prompt.

        We intentionally use a simple text format (not OpenAI function-calling
        APIs) so the same loop works with Ollama *and* OpenAI chat completions,
        and so students can *see* exactly what the model receives.
        """
        if not self._tools:
            return "You have no tools available. Answer using reasoning alone."

        lines = [
            "You have access to the following tools.",
            "When you need a tool, respond with EXACTLY this JSON (and nothing else):",
            "",
            '  {"tool": "<name>", "arguments": { ... }}',
            "",
            "When you are ready to give the final answer to the user, respond with:",
            "",
            '  {"final": "<your answer here>"}',
            "",
            "Important:",
            "- Use only tool names listed below (e.g. `search`, not `weather`).",
            "- For weather, call `search` with a query like \"weather in Paris\".",
            "- Put every argument inside the `arguments` object.",
            "",
            "Available tools:",
        ]
        for tool in self._tools.values():
            params = json.dumps(tool.parameters, indent=2)
            lines.append(f"\n### {tool.name}")
            lines.append(tool.description)
            lines.append(f"Parameters schema:\n{params}")
        return "\n".join(lines)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    # -- parsing & execution ------------------------------------------------

    @staticmethod
    def parse_tool_call(text: str) -> Optional[ToolCall]:
        """Extract a TOOL_CALL or final answer from model output.

        Tolerates markdown fences and surrounding prose — models are messy.
        Returns None if neither a tool call nor a final answer is found.
        """
        cleaned = text.strip()

        # Strip ```json ... ``` fences if present
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if fence:
            cleaned = fence.group(1)

        # Find the first JSON object in the text
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None

        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        # Final answer shortcut — treated as a special "tool" by the harness
        if "final" in data:
            return ToolCall(name="__final__", arguments={"answer": data["final"]}, raw=match.group(0))

        # Canonical: {"tool": "name", "arguments": {...}}
        if "tool" in data:
            args = data.get("arguments") or data.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            # Models often flatten args: {"tool":"search","query":"..."}
            if not args:
                args = {
                    k: v
                    for k, v in data.items()
                    if k not in {"tool", "name", "arguments", "args", "final"}
                }
            return ToolCall(name=str(data["tool"]), arguments=args, raw=match.group(0))

        # Alternate: {"name": "...", "arguments": {...}}
        if "name" in data and ("arguments" in data or "args" in data or "query" in data):
            args = data.get("arguments") or data.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            if not args:
                args = {
                    k: v
                    for k, v in data.items()
                    if k not in {"tool", "name", "arguments", "args", "final"}
                }
            return ToolCall(name=str(data["name"]), arguments=args, raw=match.group(0))

        return None

    def execute(self, name: str, arguments: Optional[dict[str, Any]] = None) -> str:
        """Run a registered tool. Returns an observation string (never raises to the LLM)."""
        arguments = dict(arguments or {})
        resolved = self.resolve_name(name)
        tool = self._tools.get(resolved)
        if tool is None:
            available = ", ".join(self._tools) or "(none)"
            return f"ERROR: Unknown tool '{name}'. Available: {available}"

        # If the model invented `weather` with a city/location field, fold it into search.query
        if name != resolved and resolved == "search":
            if "query" not in arguments:
                for key in ("location", "city", "place", "q", "text"):
                    if key in arguments and arguments[key]:
                        arguments["query"] = str(arguments[key])
                        break
            if "query" in arguments and name.lower().replace("-", "_") in {
                "weather",
                "get_weather",
                "weather_search",
            }:
                q = str(arguments["query"])
                if "weather" not in q.lower():
                    arguments["query"] = f"weather in {q}"

        # Drop unknown kwargs small models invent (city, location, …)
        if resolved == "search":
            arguments = {
                k: v for k, v in arguments.items() if k in {"query", "max_results"}
            }
        elif resolved == "wikipedia":
            arguments = {k: v for k, v in arguments.items() if k in {"query"}}
        elif resolved == "calculator":
            arguments = {k: v for k, v in arguments.items() if k in {"expression"}}
        elif resolved == "python_exec":
            arguments = {k: v for k, v in arguments.items() if k in {"code"}}
        elif resolved == "http_get":
            arguments = {k: v for k, v in arguments.items() if k in {"url", "max_chars"}}
        elif resolved == "datetime_now":
            arguments = {k: v for k, v in arguments.items() if k in {"timezone_name"}}
        elif resolved == "unit_convert":
            arguments = {
                k: v for k, v in arguments.items() if k in {"value", "from_unit", "to_unit"}
            }
        elif resolved == "remember":
            arguments = {k: v for k, v in arguments.items() if k in {"text", "tags"}}

        try:
            result = tool.run(**arguments)
        except TypeError as exc:
            return f"ERROR: Bad arguments for '{resolved}': {exc}"
        except Exception as exc:  # noqa: BLE001 — surface any tool failure to the model
            return f"ERROR while running '{resolved}': {type(exc).__name__}: {exc}"

        if name != resolved:
            return f"(Mapped tool `{name}` → `{resolved}`)\n{result}"
        return result

    def execute_call(self, call: ToolCall) -> str:
        """Execute a parsed :class:`ToolCall`."""
        if call.name == "__final__":
            return str(call.arguments.get("answer", ""))
        return self.execute(call.name, call.arguments)
