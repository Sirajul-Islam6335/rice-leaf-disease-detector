from __future__ import annotations

from django import template

register = template.Library()


@register.filter
def pct(value, digits=0):
    """0.8333 -> '83%'."""
    try:
        return f"{float(value) * 100:.{int(digits)}f}%"
    except (TypeError, ValueError):
        return ""


@register.filter
def get_item(mapping, key):
    return mapping.get(key) if hasattr(mapping, "get") else None


@register.filter
def shade(share):
    """Background for a confusion-matrix cell: deeper green for a larger share of the row."""
    share = max(0.0, min(1.0, float(share)))
    return f"rgba(31, 77, 46, {0.06 + 0.86 * share:.2f})"


@register.filter
def ink_on(share):
    return "#ffffff" if float(share) > 0.5 else "var(--ink)"
