"""Color helpers for Plotly charts."""

from __future__ import annotations

from plotly.colors import hex_to_rgb, unlabel_rgb


def to_rgba(color: str, alpha: float) -> str:
    """Return an rgba(...) color string from Plotly rgb(...) or hex color input."""
    if color.startswith("rgb"):
        r, g, b = (int(v) for v in unlabel_rgb(color))
    else:
        r, g, b = hex_to_rgb(color)
    return f"rgba({r},{g},{b},{alpha})"
