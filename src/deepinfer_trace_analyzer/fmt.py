from __future__ import annotations

from decimal import Decimal


def dec(value: Decimal) -> str:
    """Plain-notation decimal string with no exponent and no trailing zeros, for JSON."""
    if value == 0:
        return "0"
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def us(value: Decimal) -> str:
    text = f"{value:.3f}"
    return "0.000" if text == "-0.000" else text


def signed(value: Decimal) -> str:
    text = f"{value:+.3f}"
    return "+0.000" if text == "-0.000" else text


def percent(value: Decimal | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"
