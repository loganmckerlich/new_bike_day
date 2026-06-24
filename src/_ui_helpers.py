"""Shared UI helpers for the bike comparison app pages.

Provides unit-conversion helpers, formatting utilities, and small display
helpers that are used by multiple app pages.  All functions that require
Streamlit session state (like the metric/imperial toggle) read directly from
``st.session_state`` so they reflect the current user preference at call time.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

__all__ = [
    # Unit toggles
    "use_metric",
    "spd_label",
    "dist_label",
    "elev_label",
    # Converters
    "convert_speed",
    "convert_dist_m",
    "convert_elev_m",
    # Display helpers
    "gear_label",
    "fmt_duration",
    "compute_speed_kmh",
    "has_col",
]


# ── Metric / Imperial toggle ──────────────────────────────────────────────────

def use_metric() -> bool:
    """Return True when the user has selected metric units."""
    return st.session_state.get("use_metric", True)


def spd_label() -> str:
    """Return the display label for speed units."""
    return "km/h" if use_metric() else "mph"


def dist_label() -> str:
    """Return the display label for distance units."""
    return "km" if use_metric() else "mi"


def elev_label() -> str:
    """Return the display label for elevation units."""
    return "m" if use_metric() else "ft"


# ── Unit conversions ──────────────────────────────────────────────────────────

def convert_speed(kmh: float) -> float:
    """Convert km/h to the user's preferred display unit (km/h or mph)."""
    return kmh if use_metric() else kmh * 0.621371


def convert_dist_m(meters: float) -> float:
    """Convert metres to the user's preferred display unit (km or mi)."""
    return meters / 1000 if use_metric() else meters / 1609.34


def convert_elev_m(meters: float) -> float:
    """Convert metres to the user's preferred display unit (m or ft)."""
    return meters if use_metric() else meters * 3.28084


# ── General helpers ───────────────────────────────────────────────────────────

def gear_label(gear_id: str | None, bikes: dict[str, str]) -> str:
    """Map a Strava gear ID to its human-readable bike name.

    Parameters
    ----------
    gear_id:
        Raw gear ID string from the Strava API (e.g. ``"b12345678"``).
    bikes:
        Mapping of gear_id → bike name, typically from ``st.session_state["bikes"]``.
    """
    if gear_id is None:
        return "Unknown"
    return bikes.get(str(gear_id), str(gear_id))

def get_available_bikes() -> list[str]:
    if "available_bikes" in st.session_state:
        return st.session_state["available_bikes"]

    efforts = st.session_state.get("cleaned_efforts")

    watt_efforts = efforts[efforts["average_watts"].notna()].copy()
    watt_efforts = watt_efforts.dropna(subset=["gear_id"])

    available_bikes = (
        watt_efforts.groupby("bike_name")["effort_id"]
        .count()
        .sort_values(ascending=False)
        .index.tolist()
    )

    st.session_state["available_bikes"] = available_bikes
    return available_bikes


def fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as ``mm:ss``."""
    total = int(round(seconds))
    return f"{total // 60}:{total % 60:02d}"


def compute_speed_kmh(
    df: pd.DataFrame,
    distance_m: float | None = None,
) -> pd.Series:
    """Compute speed in km/h from ``moving_time`` and a segment distance.

    When *distance_m* is provided and positive, that fixed segment distance is
    used for every row (appropriate when all efforts are on the same segment).
    Otherwise each row's own ``distance`` column is used.
    """
    safe_time = df["moving_time"].replace(0, pd.NA)
    if distance_m is not None and distance_m > 0:
        return (distance_m / safe_time * 3.6).where(safe_time.notna())
    dist = df.get("distance", pd.Series(dtype=float))
    return (dist / safe_time * 3.6).where(safe_time.notna() & dist.notna())


def has_col(df: pd.DataFrame, col: str) -> bool:
    """Return True if *col* exists in *df* and has at least one non-null value."""
    return col in df.columns and df[col].notna().any()
