"""Weather enrichment helpers for causal analysis."""

from __future__ import annotations

from typing import Any

import pandas as pd


def get_weather(lat: float, lon: float, date: str) -> dict[str, float]:
    """Return weather features for a given point/date.

    This is currently a stub and returns deterministic dummy values.

    Args:
        lat: Latitude for the lookup location.
        lon: Longitude for the lookup location.
        date: Date string in ``YYYY-MM-DD`` format.

    Returns:
        Mapping of weather features used by the causal model.
    """
    # TODO: Replace with Open-Meteo historical API integration.
    _ = (lat, lon, date)
    return {
        "temp_c": 18.0,
        "wind_speed_kph": 15.0,
        "wind_direction_deg": 180.0,
        "precipitation_mm": 0.0,
    }


def _date_only(series: pd.Series) -> pd.Series:
    """Extract ``YYYY-MM-DD`` from an effort timestamp series."""
    return pd.to_datetime(series, utc=True, errors="coerce").dt.strftime("%Y-%m-%d")


def get_weather_for_efforts(efforts_df: pd.DataFrame, segments_df: pd.DataFrame) -> pd.DataFrame:
    """Attach weather columns to efforts via unique ``(segment_id, date)`` keys.

    Args:
        efforts_df: Efforts table that includes ``segment_id`` and ``start_date``.
        segments_df: Segment metadata used to resolve latitude/longitude.

    Returns:
        Copy of ``efforts_df`` with columns:
        ``temp_c``, ``wind_speed_kph``, ``wind_direction_deg``, ``precipitation_mm``.
    """
    if efforts_df.empty:
        out = efforts_df.copy()
        for col in ("temp_c", "wind_speed_kph", "wind_direction_deg", "precipitation_mm"):
            out[col] = pd.Series(dtype=float)
        return out

    efforts = efforts_df.copy()
    efforts["effort_date"] = _date_only(efforts["start_date"])

    segment_cols = [c for c in ("segment_id", "start_lat", "start_lng") if c in segments_df.columns]
    segment_lookup = segments_df[segment_cols].drop_duplicates(subset=["segment_id"]) if segment_cols else pd.DataFrame()

    weather_keys = efforts[["segment_id", "effort_date"]].dropna().drop_duplicates()
    if not segment_lookup.empty:
        weather_keys = weather_keys.merge(segment_lookup, on="segment_id", how="left")
    else:
        weather_keys["start_lat"] = 0.0
        weather_keys["start_lng"] = 0.0

    weather_rows: list[dict[str, Any]] = []
    for row in weather_keys.itertuples(index=False):
        lat = float(getattr(row, "start_lat", 0.0) or 0.0)
        lon = float(getattr(row, "start_lng", 0.0) or 0.0)
        effort_date = getattr(row, "effort_date")
        if not effort_date:
            continue
        weather = get_weather(lat, lon, str(effort_date))
        weather_rows.append(
            {
                "segment_id": row.segment_id,
                "effort_date": effort_date,
                "temp_c": float(weather["temp_c"]),
                "wind_speed_kph": float(weather["wind_speed_kph"]),
                "wind_direction_deg": float(weather["wind_direction_deg"]),
                "precipitation_mm": float(weather["precipitation_mm"]),
            }
        )

    weather_df = pd.DataFrame(weather_rows)
    if weather_df.empty:
        for col in ("temp_c", "wind_speed_kph", "wind_direction_deg", "precipitation_mm"):
            efforts[col] = pd.NA
    else:
        efforts = efforts.merge(weather_df, on=["segment_id", "effort_date"], how="left")

    return efforts.drop(columns=["effort_date"])
