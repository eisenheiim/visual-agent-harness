"""
Unit conversion tool — grounded conversions without calculator gymnastics.

Covers common length, mass, temperature, and volume pairs so the agent
doesn't have to memorize factors (or botch °C ↔ °F).
"""

from __future__ import annotations

from typing import Any

from core.tool_registry import Tool

# Canonical unit aliases → (family, factor_to_base_or_special)
# Temperature is handled separately (affine transform, not a pure scale).

_LENGTH = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "km": 1000.0,
    "kilometer": 1000.0,
    "kilometers": 1000.0,
    "cm": 0.01,
    "centimeter": 0.01,
    "mm": 0.001,
    "mi": 1609.344,
    "mile": 1609.344,
    "miles": 1609.344,
    "yd": 0.9144,
    "yard": 0.9144,
    "ft": 0.3048,
    "foot": 0.3048,
    "feet": 0.3048,
    "in": 0.0254,
    "inch": 0.0254,
    "inches": 0.0254,
}

_MASS = {
    "kg": 1.0,
    "kilogram": 1.0,
    "kilograms": 1.0,
    "g": 0.001,
    "gram": 0.001,
    "grams": 0.001,
    "mg": 1e-6,
    "lb": 0.45359237,
    "lbs": 0.45359237,
    "pound": 0.45359237,
    "pounds": 0.45359237,
    "oz": 0.028349523125,
    "ounce": 0.028349523125,
}

_VOLUME = {
    "l": 1.0,
    "liter": 1.0,
    "liters": 1.0,
    "litre": 1.0,
    "ml": 0.001,
    "milliliter": 0.001,
    "gal": 3.785411784,
    "gallon": 3.785411784,
    "gallons": 3.785411784,
    "cup": 0.2365882365,
    "cups": 0.2365882365,
}

_TEMP = {"c", "celsius", "f", "fahrenheit", "k", "kelvin"}

_FAMILIES: list[tuple[str, dict[str, float]]] = [
    ("length", _LENGTH),
    ("mass", _MASS),
    ("volume", _VOLUME),
]


def _norm(unit: str) -> str:
    return unit.strip().lower().replace("°", "").replace(" ", "")


def _temp_to_c(value: float, unit: str) -> float:
    if unit in {"c", "celsius"}:
        return value
    if unit in {"f", "fahrenheit"}:
        return (value - 32.0) * 5.0 / 9.0
    if unit in {"k", "kelvin"}:
        return value - 273.15
    raise ValueError(unit)


def _c_to_temp(celsius: float, unit: str) -> float:
    if unit in {"c", "celsius"}:
        return celsius
    if unit in {"f", "fahrenheit"}:
        return celsius * 9.0 / 5.0 + 32.0
    if unit in {"k", "kelvin"}:
        return celsius + 273.15
    raise ValueError(unit)


def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """Convert ``value`` from ``from_unit`` to ``to_unit``.

    Parameters
    ----------
    value:
        Numeric quantity.
    from_unit / to_unit:
        Unit names, e.g. ``km``, ``miles``, ``celsius``, ``fahrenheit``.
    """
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return f"ERROR: value must be a number, got {value!r}"

    src = _norm(str(from_unit or ""))
    dst = _norm(str(to_unit or ""))
    if not src or not dst:
        return "ERROR: from_unit and to_unit are required"

    # Temperature
    if src in _TEMP or dst in _TEMP:
        if src not in _TEMP or dst not in _TEMP:
            return "ERROR: temperature units only convert to other temperature units."
        try:
            result = _c_to_temp(_temp_to_c(amount, src), dst)
        except ValueError:
            return "ERROR: unsupported temperature unit."
        return f"{amount} {from_unit} = {result:.6g} {to_unit}"

    src_family = dst_family = None
    src_factor = dst_factor = None
    for family, table in _FAMILIES:
        if src in table:
            src_family, src_factor = family, table[src]
        if dst in table:
            dst_family, dst_factor = family, table[dst]

    if src_factor is None:
        return f"ERROR: unknown from_unit {from_unit!r}."
    if dst_factor is None:
        return f"ERROR: unknown to_unit {to_unit!r}."
    if src_family != dst_family:
        return f"ERROR: cannot convert {src_family} → {dst_family}."

    base = amount * float(src_factor)
    result = base / float(dst_factor)
    return f"{amount} {from_unit} = {result:.6g} {to_unit}"


UNIT_CONVERT_TOOL = Tool(
    name="unit_convert",
    description=(
        "Convert a number between common units: length (m, km, mi, ft, in), "
        "mass (kg, g, lb, oz), volume (l, ml, gal, cup), and temperature "
        "(celsius/C, fahrenheit/F, kelvin/K). Prefer this over inventing factors."
    ),
    parameters={
        "type": "object",
        "properties": {
            "value": {"type": "number", "description": "Numeric amount to convert."},
            "from_unit": {"type": "string", "description": "Source unit, e.g. 'km'."},
            "to_unit": {"type": "string", "description": "Target unit, e.g. 'miles'."},
        },
        "required": ["value", "from_unit", "to_unit"],
    },
    func=unit_convert,
)


def run(**kwargs: Any) -> str:
    return unit_convert(**kwargs)
