"""
Visual Agent Harness — core package.

Public surface for the educational agent loop:
  - AgentHarness  → think → act → observe loop
  - Memory        → short-term chat + long-term RAG store
  - ToolRegistry  → register / call LLM tools
"""

from .harness import AgentHarness, AgentEvent, AgentStep
from .memory import Memory
from .tool_registry import ToolRegistry, Tool
from .tracing import RunTrace, build_trace_from_harness

__all__ = [
    "AgentHarness",
    "AgentEvent",
    "AgentStep",
    "Memory",
    "ToolRegistry",
    "Tool",
    "RunTrace",
    "build_trace_from_harness",
]

__version__ = "0.1.0"
