"""Data fetching helpers for Strava activities and streams."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

from stravalib import Client

_KG_TO_LBS: float = 2.20462
_METERS_TO_MILES: float = 0.000621371

# Strava frame-type integer → human-readable label
_FRAME_TYPE_LABELS: Dict[int, str] = {
    1: "Mountain",
    2: "Cross",
    3: "Road",
    4: "Time Trial",
    5: "Triathlon",
}


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


def get_gear(client: Client, gear_id: str) -> Dict[str, Any]:
    """Fetch detailed gear information from Strava for a single gear ID.

    Args:
        client: Authenticated Strava client.
        gear_id: Strava gear identifier (e.g. ``"b12345678"``).

    Returns:
        A dictionary of gear attributes.  On API error only ``gear_id`` and
        ``gear_name`` are guaranteed to be present.
    """
    try:
        gear = client.get_gear(gear_id)
    except Exception:
        return {"gear_id": gear_id, "gear_name": gear_id}

    raw_frame_type = getattr(gear, "frame_type", None)
    try:
        frame_type_int = int(raw_frame_type) if raw_frame_type is not None else None
    except (TypeError, ValueError):
        frame_type_int = None
    frame_type_label = (
        _FRAME_TYPE_LABELS.get(frame_type_int, str(raw_frame_type))
        if frame_type_int is not None
        else None
    )

    raw_distance = _to_float(getattr(gear, "distance", None))
    strava_total_miles = round(raw_distance * _METERS_TO_MILES, 1) if raw_distance is not None else None

    weight_kg = _to_float(getattr(gear, "weight", None))
    weight_lbs = round(weight_kg * _KG_TO_LBS, 1) if weight_kg is not None else None

    def _clean(val: Any) -> Optional[str]:
        s = str(val or "").strip()
        return s if s else None

    return {
        "gear_id": gear_id,
        "gear_name": _clean(getattr(gear, "name", None)) or gear_id,
        "brand_name": _clean(getattr(gear, "brand_name", None)),
        "model_name": _clean(getattr(gear, "model_name", None)),
        "frame_type": frame_type_label,
        "description": _clean(getattr(gear, "description", None)),
        "weight_lbs": weight_lbs,
        "strava_total_miles": strava_total_miles,
        "primary": bool(getattr(gear, "primary", False)),
    }


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
