"""Data fetching helpers for Strava API."""

from __future__ import annotations

import time
from typing import Any, Optional

import pandas as pd
import requests

_STRAVA_API_BASE: str = "https://www.strava.com/api/v3"
_PREMIUM_ONLY_ERROR_MESSAGE: str = (
    "This feature requires a Strava premium membership. "
    "The segment data endpoints used by this tool are only available to premium members."
)

# Strava sport types that represent cycling activities.
_BIKE_SPORT_TYPES: frozenset[str] = frozenset(
    {
        "Ride",
        "MountainBikeRide",
        "GravelRide",
        "VirtualRide",
        "EBikeRide",
        "EMountainBikeRide",
        "Handcycle",
        "Velomobile",
    }
)


class PremiumOnlyError(Exception):
    """Raised when a 402 Payment Required response is received from Strava API.

    This occurs when segment data endpoints are accessed by non-premium Strava users,
    as these endpoints are only available to premium members. The error message
    from this exception should be displayed to the user to explain that a premium
    membership is required.
    """


# Segment classification thresholds
_SPRINT_MAX_DISTANCE: float = 500.0   # metres
_ASCENT_MIN_GRADE: float = 2.0        # percent
_DESCENT_MAX_GRADE: float = -1.0      # percent


def _auth_headers(access_token: str) -> dict[str, str]:
    """Return HTTP headers required to authenticate against the Strava API."""
    return {"Authorization": f"Bearer {access_token}"}


def _classify_segment(distance: Optional[float], average_grade: Optional[float]) -> str:
    """Return a segment type label based on distance and average grade.

    Sprint is checked first (per specification), meaning a short but steep
    segment is always classified as ``"sprint"`` regardless of grade.

    Args:
        distance: Segment length in metres.
        average_grade: Average gradient in percent.

    Returns:
        One of ``"sprint"``, ``"ascent"``, ``"descent"``, or ``"flat"``.
    """
    if distance is not None and distance < _SPRINT_MAX_DISTANCE:
        return "sprint"
    if average_grade is not None and average_grade > _ASCENT_MIN_GRADE:
        return "ascent"
    if average_grade is not None and average_grade < _DESCENT_MAX_GRADE:
        return "descent"
    return "flat"


def get_starred_segments(access_token: str) -> pd.DataFrame:
    """Fetch all starred segments for the authenticated athlete.

    Paginates through ``GET /segments/starred`` until the API returns an
    empty page.

    Args:
        access_token: Valid Strava OAuth access token.

    Returns:
        DataFrame with columns: ``segment_id``, ``name``, ``distance``,
        ``average_grade``, ``climb_category``, ``total_elevation_gain``,
        ``start_lat``, ``start_lng``, ``segment_type``.

    Raises:
        PremiumOnlyError: If the endpoint returns a 402 Payment Required error,
            indicating the user needs Strava premium membership.
    """
    url = f"{_STRAVA_API_BASE}/segments/starred"
    headers = _auth_headers(access_token)
    rows: list[dict[str, Any]] = []
    page = 1
    per_page = 200

    while True:
        try:
            resp = requests.get(
                url,
                headers=headers,
                params={"page": page, "per_page": per_page},
                timeout=30,
            )
            resp.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 402:
                raise PremiumOnlyError(_PREMIUM_ONLY_ERROR_MESSAGE) from exc
            raise

        data: list[dict[str, Any]] = resp.json()
        if not data:
            break

        for seg in data:
            # Only include cycling segments (skip if activity_type is explicitly non-cycling)
            activity_type: str = seg.get("activity_type") or ""
            if activity_type and activity_type.lower() not in {"ride", "cycling"}:
                continue

            distance: Optional[float] = seg.get("distance")
            average_grade: Optional[float] = seg.get("average_grade")
            start_latlng: list[float] = seg.get("start_latlng") or []
            rows.append(
                {
                    "segment_id": seg.get("id"),
                    "name": seg.get("name"),
                    "distance": distance,
                    "average_grade": average_grade,
                    "climb_category": seg.get("climb_category"),
                    "total_elevation_gain": seg.get("total_elevation_gain"),
                    "start_lat": start_latlng[0] if len(start_latlng) > 0 else None,
                    "start_lng": start_latlng[1] if len(start_latlng) > 1 else None,
                    "segment_type": _classify_segment(distance, average_grade),
                }
            )

        if len(data) < per_page:
            break
        page += 1

    return pd.DataFrame(rows)


def get_segment_efforts(access_token: str, segment_id: int) -> pd.DataFrame:
    """Fetch all efforts recorded on a single segment.

    Paginates through ``GET /segment_efforts`` until the API returns an
    empty page.

    Args:
        access_token: Valid Strava OAuth access token.
        segment_id: Strava segment identifier.

    Returns:
        DataFrame with columns: ``effort_id``, ``segment_id``,
        ``activity_id``, ``start_date``, ``elapsed_time``, ``moving_time``,
        ``average_watts``, ``average_heartrate``.
        ``gear_id`` is resolved later by joining against the activities dict
        (see :func:`ingest_all`).

    Raises:
        PremiumOnlyError: If the endpoint returns a 402 Payment Required error,
            indicating the user needs Strava premium membership.
    """
    url = f"{_STRAVA_API_BASE}/segment_efforts"
    headers = _auth_headers(access_token)
    rows: list[dict[str, Any]] = []
    page = 1
    per_page = 200

    while True:
        try:
            resp = requests.get(
                url,
                headers=headers,
                params={"segment_id": segment_id, "page": page, "per_page": per_page},
                timeout=30,
            )
            resp.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 402:
                raise PremiumOnlyError(_PREMIUM_ONLY_ERROR_MESSAGE) from exc
            raise

        data: list[dict[str, Any]] = resp.json()
        if not data:
            break

        for effort in data:
            activity: dict[str, Any] = effort.get("activity") or {}
            rows.append(
                {
                    "effort_id": effort.get("id"),
                    "segment_id": segment_id,
                    "activity_id": activity.get("id"),
                    "start_date": effort.get("start_date"),
                    "elapsed_time": effort.get("elapsed_time"),
                    "moving_time": effort.get("moving_time"),
                    "average_watts": effort.get("average_watts"),
                    "average_heartrate": effort.get("average_heartrate"),
                }
            )

        if len(data) < per_page:
            break
        page += 1

    return pd.DataFrame(rows)


def _decode_polyline(encoded: str) -> list[tuple[float, float]]:
    """Decode a Google encoded polyline string into a list of (lat, lng) tuples.

    Uses the standard algorithm described at
    https://developers.google.com/maps/documentation/utilities/polylinealgorithm

    Args:
        encoded: The encoded polyline string.

    Returns:
        List of ``(latitude, longitude)`` float tuples.
    """
    points: list[tuple[float, float]] = []
    idx = lat = lng = 0
    while idx < len(encoded):
        coords: list[int] = [0, 0]
        for i in range(2):
            shift = result = 0
            while True:
                b = ord(encoded[idx]) - 63
                idx += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            coords[i] = ~(result >> 1) if result & 1 else result >> 1
        lat += coords[0]
        lng += coords[1]
        points.append((lat / 1e5, lng / 1e5))
    return points


def get_segment_detail(access_token: str, segment_id: int) -> dict[str, Any]:
    """Fetch detailed information for a single segment, including its route polyline.

    Calls ``GET /segments/{segment_id}`` to retrieve the full segment data
    including the encoded route polyline and elevation bounds.

    Args:
        access_token: Valid Strava OAuth access token.
        segment_id: Strava segment identifier.

    Returns:
        A dict with keys:
        - ``"polyline_points"``: list of ``(lat, lng)`` tuples decoded from the
          segment's ``map.polyline`` (may be empty if unavailable).
        - ``"elevation_low"``: minimum elevation in metres, or ``None``.
        - ``"elevation_high"``: maximum elevation in metres, or ``None``.
        - ``"start_latlng"``: ``[lat, lng]`` or ``[]``.
        - ``"end_latlng"``: ``[lat, lng]`` or ``[]``.

        Returns an empty dict on request failure.
    """
    url = f"{_STRAVA_API_BASE}/segments/{segment_id}"
    headers = _auth_headers(access_token)
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
    except requests.RequestException:
        return {}

    data = resp.json()
    seg_map: dict[str, Any] = data.get("map") or {}
    encoded = seg_map.get("polyline") or seg_map.get("summary_polyline") or ""
    points = _decode_polyline(encoded) if encoded else []
    return {
        "polyline_points": points,
        "elevation_low": data.get("elevation_low"),
        "elevation_high": data.get("elevation_high"),
        "start_latlng": data.get("start_latlng") or [],
        "end_latlng": data.get("end_latlng") or [],
    }


def get_segment_streams(access_token: str, segment_id: int) -> dict[str, list[float]]:
    """Fetch distance and altitude streams for a segment.

    Calls ``GET /segments/{segment_id}/streams`` with keys ``distance`` and
    ``altitude``.  Returns an empty dict on any failure (including premium-only
    402 errors) so callers can degrade gracefully.

    Args:
        access_token: Valid Strava OAuth access token.
        segment_id: Strava segment identifier.

    Returns:
        A dict with keys ``"distance"`` and ``"altitude"``, each a list of
        float values.  Both lists are guaranteed to have the same length.
        Returns ``{}`` on any error.
    """
    url = f"{_STRAVA_API_BASE}/segments/{segment_id}/streams"
    headers = _auth_headers(access_token)
    try:
        resp = requests.get(
            url,
            headers=headers,
            params={"keys": "distance,altitude", "key_by_type": "true"},
            timeout=30,
        )
        resp.raise_for_status()
    except requests.RequestException:
        return {}

    data = resp.json()
    distance_stream = data.get("distance") or {}
    altitude_stream = data.get("altitude") or {}
    distances: list[float] = distance_stream.get("data") or []
    altitudes: list[float] = altitude_stream.get("data") or []
    if len(distances) != len(altitudes) or not distances:
        return {}
    return {"distance": distances, "altitude": altitudes}


def get_athlete_activities(
    access_token: str,
    max_activities: Optional[int] = None,
) -> dict[int, dict[str, Any]]:
    """Fetch bike activities that have power data for the authenticated athlete.

    Paginates ``GET /athlete/activities`` and filters client-side to only
    activities whose ``sport_type`` is one of the known cycling types **and**
    that contain power data (``average_watts > 0`` or ``device_watts`` is
    ``True``).

    Args:
        access_token: Valid Strava OAuth access token.
        max_activities: Upper bound on the total number of activities fetched
            from the API before filtering.  ``None`` means no limit (fetch
            everything).

    Returns:
        A dict mapping ``activity_id`` (``int``) to a dict with keys:
        ``gear_id`` (``str | None``), ``name`` (``str``),
        ``start_date`` (``str``), ``average_watts`` (``float | None``).
    """
    url = f"{_STRAVA_API_BASE}/athlete/activities"
    headers = _auth_headers(access_token)
    activities: dict[int, dict[str, Any]] = {}
    page = 1
    per_page = 200
    fetched = 0

    while True:
        batch_size = per_page
        if max_activities is not None:
            remaining = max_activities - fetched
            if remaining <= 0:
                break
            batch_size = min(per_page, remaining)

        resp = requests.get(
            url,
            headers=headers,
            params={"page": page, "per_page": batch_size},
            timeout=30,
        )
        resp.raise_for_status()

        data: list[dict[str, Any]] = resp.json()
        if not data:
            break

        fetched += len(data)

        for act in data:
            sport_type: str = act.get("sport_type") or act.get("type") or ""
            if sport_type not in _BIKE_SPORT_TYPES:
                continue

            average_watts: Optional[float] = act.get("average_watts")
            device_watts: bool = bool(act.get("device_watts"))
            has_power = (average_watts is not None and average_watts > 0) or device_watts
            if not has_power:
                continue

            activity_id = act.get("id")
            if activity_id is None:
                continue

            activities[int(activity_id)] = {
                "gear_id": act.get("gear_id"),
                "name": act.get("name") or "",
                "start_date": act.get("start_date") or "",
                "average_watts": average_watts,
            }

        if len(data) < batch_size:
            break
        page += 1

    return activities


def get_athlete_bikes(access_token: str) -> dict[str, str]:
    """Fetch the authenticated athlete's bikes and return a gear_id → name mapping.

    Args:
        access_token: Valid Strava OAuth access token.

    Returns:
        Dict mapping gear_id (e.g. ``"b1234567"``) to a human-readable name
        (e.g. ``"Trek Domane SL5"``).  Returns an empty dict on error.
    """
    url = f"{_STRAVA_API_BASE}/athlete"
    headers = _auth_headers(access_token)
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
    except requests.RequestException:
        return {}

    data = resp.json()
    bikes: list[dict[str, Any]] = data.get("bikes") or []
    return {
        str(bike["id"]): bike.get("name") or bike.get("model_name") or str(bike["id"])
        for bike in bikes
        if bike.get("id")
    }


def ingest_all(
    access_token: str,
    max_activities: Optional[int] = None,
) -> dict[str, pd.DataFrame | dict[str, str]]:
    """Ingest all data needed to compare segment efforts by bike.

    Follows the ordered API flow:

    1. ``GET /athlete`` — fetch the athlete's bike inventory.
    2. ``GET /athlete/activities`` — fetch cycling activities that have power
       data; build an ``activity_id → gear_id`` lookup.
    3. ``GET /segments/starred`` — fetch the athlete's starred segments.
    4. ``GET /segment_efforts`` — fetch efforts for every starred segment.
    5. Join efforts to activities on ``activity_id`` to resolve ``gear_id``.

    Args:
        access_token: Valid Strava OAuth access token.
        max_activities: Upper bound on the number of activities fetched before
            filtering (passed through to :func:`get_athlete_activities`).

    Returns:
        A dict with keys:
        - ``"bikes"``: ``dict[str, str]`` mapping gear_id to bike name.
        - ``"segments"``: :class:`pandas.DataFrame` of starred segments.
        - ``"efforts"``: :class:`pandas.DataFrame` of all efforts, with a
          ``gear_id`` column resolved from the power-filtered activities.
    """
    # Step 1: bike inventory
    bikes = get_athlete_bikes(access_token)

    # Step 2: cycling activities with power — build activity_id → gear_id map
    activities = get_athlete_activities(access_token, max_activities=max_activities)

    # Step 3: starred segments
    segments_df = get_starred_segments(access_token)

    # Step 4: segment efforts for each starred segment
    all_efforts: list[pd.DataFrame] = []
    if not segments_df.empty:
        for segment_id in segments_df["segment_id"]:
            efforts_df = get_segment_efforts(access_token, int(segment_id))
            if not efforts_df.empty:
                all_efforts.append(efforts_df)
            time.sleep(1)

    efforts = pd.concat(all_efforts, ignore_index=True) if all_efforts else pd.DataFrame()

    # Step 5: resolve gear_id from the activities lookup
    if not efforts.empty and activities:
        efforts["gear_id"] = efforts["activity_id"].map(
            lambda aid: activities.get(int(aid), {}).get("gear_id") if pd.notna(aid) else None
        )
    elif not efforts.empty:
        efforts["gear_id"] = None

    return {"bikes": bikes, "segments": segments_df, "efforts": efforts}
