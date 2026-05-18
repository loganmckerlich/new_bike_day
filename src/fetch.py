"""Data fetching helpers for Strava activities and streams."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

from stravalib import Client


def _to_seconds(value: Any) -> Optional[int]:
    """Convert a timedelta-like value to seconds."""
    if value is None:
        return None
    seconds = getattr(value, "total_seconds", None)
    if callable(seconds):
        return int(seconds())
    return int(value)


def _to_float(value: Any) -> Optional[float]:
    """Convert a value to float when possible."""
    if value is None:
        return None
    return float(value)


def _extract_lat_lon(activity: Any) -> tuple[Optional[float], Optional[float]]:
    """Extract latitude and longitude from an activity."""
    latlng = getattr(activity, "start_latlng", None)
    if not latlng:
        return None, None
    if isinstance(latlng, (list, tuple)) and len(latlng) >= 2:
        return _to_float(latlng[0]), _to_float(latlng[1])
    lat = getattr(latlng, "lat", None)
    lon = getattr(latlng, "lon", None)
    if lat is not None and lon is not None:
        return _to_float(lat), _to_float(lon)
    try:
        return _to_float(latlng[0]), _to_float(latlng[1])
    except (TypeError, IndexError, KeyError):
        return None, None


def get_activities(
    client: Client,
    after: Optional[datetime] = None,
    before: Optional[datetime] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Fetch activities from Strava and normalize them into dictionaries.

    Args:
        client: Authenticated Strava client.
        after: Optional lower timestamp bound.
        before: Optional upper timestamp bound.
        limit: Optional maximum number of activities to return.

    Returns:
        A list of normalized activity dictionaries.
    """
    activities_iter: Iterable[Any] = client.get_activities(after=after, before=before, limit=limit)
    activities: List[Dict[str, Any]] = []
    for activity in activities_iter:
        lat, lon = _extract_lat_lon(activity)
        start_date_local = getattr(activity, "start_date_local", None)
        activities.append(
            {
                "id": int(activity.id),
                "name": str(getattr(activity, "name", "")),
                "distance_m": _to_float(getattr(activity, "distance", None)),
                "moving_time_s": _to_seconds(getattr(activity, "moving_time", None)),
                "elapsed_time_s": _to_seconds(getattr(activity, "elapsed_time", None)),
                "total_elevation_gain_m": _to_float(getattr(activity, "total_elevation_gain", None)),
                "average_speed_mps": _to_float(getattr(activity, "average_speed", None)),
                "max_speed_mps": _to_float(getattr(activity, "max_speed", None)),
                "average_heartrate": _to_float(getattr(activity, "average_heartrate", None)),
                "max_heartrate": _to_float(getattr(activity, "max_heartrate", None)),
                "average_watts": _to_float(getattr(activity, "average_watts", None)),
                "weighted_average_watts": _to_float(getattr(activity, "weighted_average_watts", None)),
                "kilojoules": _to_float(getattr(activity, "kilojoules", None)),
                "suffer_score": _to_float(getattr(activity, "suffer_score", None)),
                "start_date_local": start_date_local.isoformat() if start_date_local else None,
                "timezone": str(getattr(activity, "timezone", "")) or None,
                "gear_id": str(getattr(activity, "gear_id", "")) or None,
                "type": str(getattr(activity, "type", "")) or None,
                "sport_type": str(getattr(activity, "sport_type", "")) or None,
                "start_lat": lat,
                "start_lon": lon,
            }
        )
    return activities


def get_streams(
    client: Client,
    activity_id: int,
    keys: Optional[Sequence[str]] = None,
) -> Dict[str, List[Any]]:
    """Fetch time series streams for a Strava activity.

    Args:
        client: Authenticated Strava client.
        activity_id: Activity identifier.
        keys: Optional stream keys to request.

    Returns:
        A mapping from stream key to stream data list.
    """
    requested_keys = list(keys) if keys else ["time", "distance", "latlng", "velocity_smooth", "heartrate", "watts"]
    stream_set = client.get_activity_streams(activity_id=activity_id, types=requested_keys)
    normalized: Dict[str, List[Any]] = {}
    for key in requested_keys:
        stream = stream_set.get(key)
        if stream is None:
            continue
        normalized[key] = list(getattr(stream, "data", []) or [])
    return normalized
