"""Weather lookup helpers using Open-Meteo."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, Union

import requests

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def _normalize_date(target_date: Union[str, date, datetime]) -> str:
    """Normalize input date values into YYYY-MM-DD format."""
    if isinstance(target_date, datetime):
        return target_date.date().isoformat()
    if isinstance(target_date, date):
        return target_date.isoformat()
    return datetime.fromisoformat(target_date).date().isoformat()


def get_weather(lat: float, lon: float, target_date: Union[str, date, datetime]) -> Dict[str, Any]:
    """Fetch daily weather metrics from Open-Meteo for a location and day.

    Args:
        lat: Latitude of the location.
        lon: Longitude of the location.
        target_date: Date to query.

    Returns:
        A dictionary with selected daily weather fields.

    Raises:
        requests.HTTPError: If the API request fails.
    """
    date_str = _normalize_date(target_date)
    response = requests.get(
        OPEN_METEO_ARCHIVE_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": date_str,
            "end_date": date_str,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
            "timezone": "auto",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    daily = payload.get("daily", {})
    return {
        "date": date_str,
        "temperature_2m_max": (daily.get("temperature_2m_max") or [None])[0],
        "temperature_2m_min": (daily.get("temperature_2m_min") or [None])[0],
        "precipitation_sum": (daily.get("precipitation_sum") or [None])[0],
        "windspeed_10m_max": (daily.get("windspeed_10m_max") or [None])[0],
    }
