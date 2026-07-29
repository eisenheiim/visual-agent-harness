"""
Datetime tool — grounded "what time is it?" answers.

LLMs invent dates. This tool returns the real clock so agents can
timestamp answers, reason about deadlines, or greet by time of day.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from core.tool_registry import Tool


def datetime_now(timezone_name: str = "UTC") -> str:
    """Return the current date and time in a named IANA timezone.

    Parameters
    ----------
    timezone_name:
        e.g. ``"UTC"``, ``"America/New_York"``, ``"Europe/Istanbul"``.
    """
    timezone_name = (timezone_name or "UTC").strip() or "UTC"

    try:
        if timezone_name.upper() == "UTC":
            tz = timezone.utc
            label = "UTC"
        else:
            tz = ZoneInfo(timezone_name)
            label = timezone_name
    except ZoneInfoNotFoundError:
        return (
            f"ERROR: Unknown timezone {timezone_name!r}. "
            "Use an IANA name like 'UTC', 'Europe/London', or 'America/Los_Angeles'."
        )

    now = datetime.now(tz)
    return (
        f"Current datetime ({label}):\n"
        f"  ISO:      {now.isoformat()}\n"
        f"  Human:    {now.strftime('%A, %B %d, %Y · %H:%M:%S %Z')}\n"
        f"  Unix:     {int(now.timestamp())}"
    )


DATETIME_TOOL = Tool(
    name="datetime_now",
    description=(
        "Get the real current date and time in a timezone. "
        "Use whenever the answer depends on 'today', 'now', or a local clock. "
        "Do not invent the date yourself."
    ),
    parameters={
        "type": "object",
        "properties": {
            "timezone_name": {
                "type": "string",
                "description": (
                    "IANA timezone, e.g. 'UTC', 'Europe/Istanbul', "
                    "'America/New_York'. Default: UTC."
                ),
            }
        },
        "required": [],
    },
    func=datetime_now,
)


def run(**kwargs: Any) -> str:
    return datetime_now(**kwargs)
