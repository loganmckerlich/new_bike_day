"""Data fetching helpers for Strava API."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

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

def is_near_read_rate_limit(headers, threshold=0.9):
    limits, usage = parse_rate_limit_headers(headers)
    if limits is None or usage is None:
        return False
    limit_15min, limit_daily = limits
    usage_15min, usage_daily = usage
    if not limit_15min or not limit_daily:
        return False
    return (usage_15min / limit_15min >= threshold) or (usage_daily / limit_daily >= threshold)


def parse_rate_limit_headers(headers: Any) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    if headers is None:
        return None, None
    if hasattr(headers, "get"):
        get_header = headers.get
    else:
        return None, None
    limit_value = (
        get_header("X-RateLimit-Limit")
        or get_header("x-ratelimit-limit")
        or get_header("x-readratelimit-limit")
    )
    usage_value = (
        get_header("X-RateLimit-Usage")
        or get_header("x-ratelimit-usage")
        or get_header("x-readratelimit-usage")
    )
    if not limit_value or not usage_value:
        return None, None
    try:
        limit_15min, limit_daily = map(int, str(limit_value).split(","))
        usage_15min, usage_daily = map(int, str(usage_value).split(","))
    except (TypeError, ValueError):
        return None, None
    return (limit_15min, limit_daily), (usage_15min, usage_daily)
class PremiumOnlyError(Exception):
    """Raised when a 402 Payment Required response is received from Strava API.

    This occurs when segment data endpoints are accessed by non-premium Strava users,
    as these endpoints are only available to premium members. The error message
    from this exception should be displayed to the user to explain that a premium
    membership is required.
    """


# ---------------------------------------------------------------------------
# Dev-mode request interception
# ---------------------------------------------------------------------------

_DEV_DIR = Path(__file__).resolve().parents[1] / "data" / "dev"
_REAL_DIR = Path(__file__).resolve().parents[1] / "data" / "real"


def _response_filename(url: str, params: dict[str, Any]) -> Path:
    """Derive a ``data/real/<name>.json`` path from a Strava API URL + params."""
    path = url.replace(_STRAVA_API_BASE, "").strip("/")
    # Replace path separators with underscores: segments/123/streams → segments_123_streams
    name = path.replace("/", "_")
    # Append meaningful query-param suffixes so per-segment files don't collide.
    for key in ("segment_id",):
        if key in params:
            name = f"{name}_{params[key]}"
    if params.get("page", 1) != 1:
        name = f"{name}_p{params['page']}"
    return _REAL_DIR / f"{name}.json"


class _LoggingSession:
    """Thin wrapper around ``requests`` that persists each API response to disk.

    Responses are written to ``data/real/<endpoint>.json``, overwriting the
    previous call so only the most recent response is kept.
    """

    def get(
        self,
        url: str,
        *,
        params: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> requests.Response:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout, **kwargs)
        try:
            _REAL_DIR.mkdir(parents=True, exist_ok=True)
            dest = _response_filename(url, params or {})
            dest.write_text(json.dumps(resp.json(), indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"[_LoggingSession] WARNING: could not save {url} → {exc}")
        return resp


class _MockResponse:
    """Minimal requests.Response stand-in returned by _DevSession."""

    status_code: int = 200

    def __init__(self, data: Any) -> None:
        self._data = data

    def json(self) -> Any:
        return self._data

    def raise_for_status(self) -> None:  # noqa: D401
        pass


class _DevSession:
    """Intercepts every requests.get call and returns data from local JSON files.

    URL routing (path relative to the Strava API base):
      /athlete                 → athlete.json            (raw GET /athlete)
      /athlete/activities      → athlete_activities.json  (raw GET /athlete/activities)
      /gear/{id}               → gear_{id}.json           (raw GET /gear/{id})
      /segments/starred        → segments_starred.json    (raw GET /segments/starred)
      /segment_efforts         → segment_efforts_{id}.json  (raw GET /segment_efforts?segment_id=id)
      /segments/{id}/streams   → segment_streams_{id}.json  (raw GET /segments/{id}/streams)
      /segments/{id}           → segment_detail_{id}.json   (raw GET /segments/{id})

    All paginated endpoints return an empty list for page > 1, simulating a
    single full page of results.
    """

    def get(
        self,
        url: str,
        *,
        params: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> _MockResponse:
        params = params or {}
        page = int(params.get("page", 1))

        # Strip the base URL to get just the path component
        path = url.replace("https://www.strava.com/api/v3", "")

        if path == "/athlete":
            data = json.loads((_DEV_DIR / "athlete.json").read_text())

        elif path == "/athlete/activities":
            if page == 1:
                data = json.loads((_DEV_DIR / "athlete_activities.json").read_text())
            else:
                f = _DEV_DIR / f"athlete_activities_p{page}.json"
                data = json.loads(f.read_text()) if f.exists() else []

        elif path.startswith("/gear/"):
            gear_id = path.split("/")[-1]
            f = _DEV_DIR / f"gear_{gear_id}.json"
            data = json.loads(f.read_text()) if f.exists() else {}

        elif path == "/segments/starred":
            data = [] if page > 1 else json.loads((_DEV_DIR / "segments_starred.json").read_text())

        elif path == "/segment_efforts":
            if page > 1:
                data = []
            else:
                segment_id = params.get("segment_id", 0)
                f = _DEV_DIR / f"segment_efforts_{segment_id}.json"
                data = json.loads(f.read_text()) if f.exists() else []

        elif path.endswith("/streams"):
            segment_id = path.split("/")[-2]
            f = _DEV_DIR / f"segment_streams_{segment_id}.json"
            data = json.loads(f.read_text()) if f.exists() else {}

        else:
            # /segments/{id}
            segment_id = path.split("/")[-1]
            f = _DEV_DIR / f"segment_detail_{segment_id}.json"
            data = json.loads(f.read_text()) if f.exists() else {}

        return _MockResponse(data)


# Segment classification thresholds
# Sprint threshold updated to 400 m to match spider chart category spec.
_SPRINT_MAX_DISTANCE: float = 1000.0   # metres
_FLAT_MIN_GRADE: float = -0.5         # percent
_FLAT_MAX_GRADE: float = 0.5          # percent
_ASCENT_MIN_GRADE: float = 2.0        # percent
_DESCENT_MAX_GRADE: float = -1.0      # percent


def _auth_headers(access_token: str) -> dict[str, str]:
    """Return HTTP headers required to authenticate against the Strava API."""
    return {"Authorization": f"Bearer {access_token}"}


def _classify_segment(
    distance: Optional[float], average_grade: Optional[float]
) -> tuple[str, str]:
    """Return segment category + subcategory labels based on distance and grade.

    Sprint is checked first (per specification), meaning a short but steep
    segment is always classified as ``"sprint"`` regardless of grade.

    Args:
        distance: Segment length in metres.
        average_grade: Average gradient in percent.

    Returns:
        Tuple ``(segment_type, segment_type_detail)``.
    """
    grade = average_grade if average_grade is not None else 0.0
    dist = distance if distance is not None else float("inf")

    if dist < _SPRINT_MAX_DISTANCE:
        if grade < -0.5:
            return "sprint", "sprint_downhill"
        if grade > 0.5:
            return "sprint", "sprint_uphill"
        return "sprint", "sprint_flat"

    if _FLAT_MIN_GRADE <= grade <= _FLAT_MAX_GRADE:
        if dist < 3000.0:
            return "flat", "flat_short"
        return "flat", "flat_long"

    if grade > _ASCENT_MIN_GRADE:
        if grade <= 3.0:
            return "ascent", "ascent_shallow"
        if grade <= 6.0:
            return "ascent", "ascent_moderate"
        return "ascent", "ascent_steep"

    if grade < _DESCENT_MAX_GRADE:
        if grade >= -4.0:
            return "descent", "descent_gentle"
        return "descent", "descent_steep"

    if dist < 3000.0:
        return "flat", "flat_short"
    return "flat", "flat_long"


def get_starred_segments(access_token: str, *, _http: Any = requests) -> pd.DataFrame:
    """Fetch all starred segments for the authenticated athlete.

    Paginates through ``GET /segments/starred`` until the API returns an
    empty page.

    Args:
        access_token: Valid Strava OAuth access token.

    Returns:
        DataFrame with columns: ``segment_id``, ``name``, ``distance``,
        ``average_grade``, ``climb_category``, ``total_elevation_gain``,
        ``start_lat``, ``start_lng``, ``segment_type``,
        ``segment_type_detail``.

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
            resp = _http.get(
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
        near_limit = is_near_read_rate_limit(getattr(resp, "headers", {}))
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
            elev_gain = seg.get("total_elevation_gain")
            if elev_gain is None:
                elev_high = seg.get("elevation_high")
                elev_low = seg.get("elevation_low")
                if elev_high is not None and elev_low is not None:
                    elev_gain = max(0.0, elev_high - elev_low)
            segment_type, segment_type_detail = _classify_segment(distance, average_grade)
            rows.append(
                {
                    "segment_id": seg.get("id"),
                    "name": seg.get("name"),
                    "distance": distance,
                    "average_grade": average_grade,
                    "maximum_grade": seg.get("maximum_grade"),
                    "climb_category": seg.get("climb_category"),
                    "hazardous": seg.get("hazardous"),
                    "total_elevation_gain": elev_gain,
                    "start_lat": start_latlng[0] if len(start_latlng) > 0 else None,
                    "start_lng": start_latlng[1] if len(start_latlng) > 1 else None,
                    "segment_type": segment_type,
                    "segment_type_detail": segment_type_detail,
                }
            )

        if near_limit:
            print("[get_starred_segments] WARNING: near Strava API read rate limit.")
            break
        if len(data) < per_page:
            break
        page += 1

    return pd.DataFrame(rows)


def get_segment_efforts(access_token: str, segment_id: int, *, _http: Any = requests) -> pd.DataFrame:
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
            resp = _http.get(
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
        if is_near_read_rate_limit(resp.headers):
            print("[get_segment_efforts] WARNING: near Strava API read rate limit.")
            break
        page += 1

    return pd.DataFrame(rows)


def get_activity_map_for_window(
    access_token: str,
    window_start: datetime,
    window_end: datetime,
    *,
    _http: Any = requests,
) -> tuple[dict[int, dict[str, Any]], pd.DataFrame, bool, bool]:
    """Fetch cycling activities in a time window and build activity_id lookup.

    Returns:
        Tuple of ``(activities, rides, threshold_reached, incomplete_window)``.
    """
    url = f"{_STRAVA_API_BASE}/athlete/activities"
    headers = _auth_headers(access_token)
    rows: list[dict[str, Any]] = []
    activities: dict[int, dict[str, Any]] = {}
    page = 1
    per_page = 200
    threshold_reached = False
    incomplete_window = False
    after_epoch = int(window_start.replace(tzinfo=timezone.utc).timestamp())
    before_epoch = int(window_end.replace(tzinfo=timezone.utc).timestamp())

    while True:
        try:
            resp = _http.get(
                url,
                headers=headers,
                params={
                    "page": page,
                    "per_page": per_page,
                    "after": after_epoch,
                    "before": before_epoch,
                },
                timeout=30,
            )
            resp.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 429:
                incomplete_window = True
                threshold_reached = True
                break
            raise
        threshold_reached = threshold_reached or is_near_read_rate_limit(getattr(resp, "headers", {}))

        data: list[dict[str, Any]] = resp.json()
        if not data:
            break

        for act in data:
            sport_type: str = act.get("sport_type") or act.get("type") or ""
            if sport_type not in _BIKE_SPORT_TYPES:
                continue
            activity_id = act.get("id")
            if activity_id is None:
                continue
            activity_key = int(activity_id)
            payload = {
                "gear_id": act.get("gear_id"),
                "name": act.get("name") or "",
                "start_date": act.get("start_date") or "",
                "average_watts": act.get("average_watts"),
                "moving_time": act.get("moving_time"),
                "elapsed_time": act.get("elapsed_time"),
                "distance": act.get("distance"),
                "total_elevation_gain": act.get("total_elevation_gain"),
                "average_heartrate": act.get("average_heartrate"),
                "average_speed": act.get("average_speed"),
                "sport_type": sport_type,
            }
            activities[activity_key] = payload
            rows.append({"activity_id": str(activity_key), **payload})

        if len(data) < per_page:
            break
        if threshold_reached:
            incomplete_window = True
            break
        page += 1

    rides = pd.DataFrame(rows) if rows else pd.DataFrame()
    return activities, rides, threshold_reached, incomplete_window


def get_segment_efforts_for_window(
    access_token: str,
    segment_id: int,
    window_start: datetime,
    window_end: datetime,
    *,
    _http: Any = requests,
) -> tuple[pd.DataFrame, bool, bool]:
    """Fetch all efforts for one segment in a time window.

    Returns:
        Tuple of ``(efforts_df, threshold_reached, incomplete_segment)``.
    """
    url = f"{_STRAVA_API_BASE}/segment_efforts"
    headers = _auth_headers(access_token)
    rows: list[dict[str, Any]] = []
    page = 1
    per_page = 200
    threshold_reached = False
    incomplete_segment = False
    start_iso = window_start.replace(tzinfo=timezone.utc).isoformat()
    end_iso = window_end.replace(tzinfo=timezone.utc).isoformat()

    while True:
        try:
            resp = _http.get(
                url,
                headers=headers,
                params={
                    "segment_id": segment_id,
                    "start_date_local": start_iso,
                    "end_date_local": end_iso,
                    "page": page,
                    "per_page": per_page,
                },
                timeout=30,
            )
            resp.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 402:
                raise PremiumOnlyError(_PREMIUM_ONLY_ERROR_MESSAGE) from exc
            if exc.response is not None and exc.response.status_code == 429:
                incomplete_segment = True
                threshold_reached = True
                break
            raise

        threshold_reached = threshold_reached or is_near_read_rate_limit(getattr(resp, "headers", {}))
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
        if threshold_reached:
            incomplete_segment = True
            break
        page += 1

    return pd.DataFrame(rows), threshold_reached, incomplete_segment


def ingest_window(
    access_token: str,
    segments: pd.DataFrame,
    window_start: datetime,
    window_end: datetime,
    *,
    progress_callback: Optional[Callable[[str, int], None]] = None,
    _http: Any = requests,
) -> dict[str, Any]:
    """Ingest one 30-day window and return data + completion status."""
    activities, rides, threshold_reached, incomplete_window = get_activity_map_for_window(
        access_token,
        window_start,
        window_end,
        _http=_http,
    )

    if incomplete_window:
        return {
            "complete": False,
            "threshold_reached": threshold_reached,
            "mid_window_rate_limit": True,
            "efforts": pd.DataFrame(),
            "rides": pd.DataFrame(),
            "activities": {},
            "processed_segments": 0,
            "total_segments": int(len(segments.index)),
        }

    activity_gear_map = {activity_id: data.get("gear_id") for activity_id, data in activities.items()}
    all_efforts: list[pd.DataFrame] = []
    n_segments = int(len(segments.index))
    segment_ids = [int(segment_id) for segment_id in segments.get("segment_id", pd.Series(dtype=int)).tolist()]
    processed_segments = 0
    mid_window_rate_limit = False

    for index, segment_id in enumerate(segment_ids):
        if progress_callback is not None:
            progress_callback(
                f"Fetching segment efforts ({index + 1}/{n_segments}) for {window_start.date()} → {window_end.date()}",
                index + 1,
            )
        segment_efforts, reached_now, incomplete_segment = get_segment_efforts_for_window(
            access_token,
            segment_id,
            window_start,
            window_end,
            _http=_http,
        )
        threshold_reached = threshold_reached or reached_now
        if incomplete_segment or (reached_now and index < n_segments - 1):
            mid_window_rate_limit = True
            break
        if not segment_efforts.empty:
            all_efforts.append(segment_efforts)
        processed_segments += 1

    if mid_window_rate_limit:
        return {
            "complete": False,
            "threshold_reached": True,
            "mid_window_rate_limit": True,
            "efforts": pd.DataFrame(),
            "rides": pd.DataFrame(),
            "activities": {},
            "processed_segments": processed_segments,
            "total_segments": n_segments,
        }

    efforts = pd.concat(all_efforts, ignore_index=True) if all_efforts else pd.DataFrame()
    if not efforts.empty:
        efforts_activity_ids = pd.to_numeric(efforts["activity_id"], errors="coerce")
        efforts["gear_id"] = efforts_activity_ids.map(activity_gear_map)
        # ponytail: keep only efforts tied to cycling activities that exposed gear_id in this window.
        efforts = efforts[efforts["gear_id"].notna()].copy()

    return {
        "complete": True,
        "threshold_reached": threshold_reached,
        "mid_window_rate_limit": False,
        "efforts": efforts,
        "rides": rides,
        "activities": activities,
        "processed_segments": processed_segments,
        "total_segments": n_segments,
    }


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


def get_segment_detail(access_token: str, segment_id: int, *, _http: Any = requests) -> dict[str, Any]:
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
        resp = _http.get(url, headers=headers, timeout=30)
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


def get_segment_streams(access_token: str, segment_id: int, *, _http: Any = requests) -> dict[str, list[float]]:
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
        resp = _http.get(
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
    *,
    _http: Any = requests,
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

        resp = _http.get(
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
                "moving_time": act.get("moving_time"),
                "elapsed_time": act.get("elapsed_time"),
                "distance": act.get("distance"),
                "total_elevation_gain": act.get("total_elevation_gain"),
                "average_heartrate": act.get("average_heartrate"),
                "average_speed": act.get("average_speed"),
                "sport_type": act.get("sport_type") or act.get("type") or "",
            }

        if len(data) < batch_size:
            break

        if is_near_read_rate_limit(resp.headers):
            print("[get_athlete_activities] WARNING: near Strava API read rate limit.")
            break
        page += 1

    return activities


def get_athlete_bikes(
    access_token: str,
    *,
    gear_ids: Optional[set[str]] = None,
    _http: Any = None,
) -> tuple[dict[str, str], dict[str, float], Optional[int]]:
    """Resolve gear_id → bike name for both active and retired bikes.

    **Step 1** — ``GET /athlete``: returns active bikes and FTP when the token
    has ``profile:read_all`` scope.  One API call, fast.

    **Step 2** — ``GET /gear/{id}`` per unknown gear_id: fetches any gear_id
    seen in activities that was not in the active bike list (i.e. retired bikes).

    Args:
        access_token: Valid Strava OAuth access token.
        gear_ids: Set of gear_ids seen in activities. Any id not returned by
            ``GET /athlete`` (retired bikes) will be fetched via ``GET /gear/{id}``.
        _http: HTTP session override (defaults to :class:`_LoggingSession`).

    Returns:
        Tuple of (bikes dict, distances dict, ftp). bikes maps gear_id to bike
        name. ftp is the athlete's FTP in watts, or None if unavailable.
    """
    if _http is None:
        _http = _LoggingSession()
    headers = _auth_headers(access_token)
    bikes: dict[str, str] = {}
    distances: dict[str, float] = {}
    ftp: Optional[int] = None

    # Step 1: GET /athlete — active bikes only
    try:
        resp = _http.get(f"{_STRAVA_API_BASE}/athlete", headers=headers, timeout=30)
        resp.raise_for_status()
        athlete_data = resp.json()
        bikes_list: list[dict[str, Any]] = athlete_data.get("bikes") or []
        bikes = {
            str(b["id"]): b.get("name") or b.get("model_name") or str(b["id"])
            for b in bikes_list
            if b.get("id")
        }
        distances = {
            str(b["id"]): float(b["converted_distance"])
            for b in bikes_list
            if b.get("id") and b.get("converted_distance") is not None
        }
        raw_ftp = athlete_data.get("ftp")
        ftp = int(raw_ftp) if raw_ftp is not None else None
    except requests.RequestException:
        pass

    # Step 2: GET /gear/{id} — retired bikes not returned by /athlete
    for gear_id in (gear_ids or set()):
        if gear_id in bikes:
            continue
        try:
            resp = _http.get(f"{_STRAVA_API_BASE}/gear/{gear_id}", headers=headers, timeout=30)
            resp.raise_for_status()
            gear = resp.json()
            name = gear.get("name") or gear.get("model_name") or str(gear_id)
            bikes[str(gear_id)] = name
            if gear.get("converted_distance") is not None:
                distances[str(gear_id)] = float(gear["converted_distance"])
        except requests.RequestException:
            pass


    return bikes, distances, ftp


def ingest_all(
    access_token: str,
    max_activities: Optional[int] = None,
    *,
    dev: bool = False,
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> dict[str, pd.DataFrame | dict[str, str]]:
    """Ingest all data needed to compare segment efforts by bike.

    Follows the ordered API flow:

    1. ``GET /athlete/activities`` — fetch cycling activities that have power
       data; collect unique gear_ids.
    2. ``GET /athlete`` — fetch active bikes and FTP.
       ``GET /gear/{id}`` — fetch any retired bike not in the active list.
    3. ``GET /segments/starred`` — fetch the athlete's starred segments.
    4. ``GET /segment_efforts`` — fetch efforts for every starred segment.
    5. Join efforts to activities on ``activity_id`` to resolve ``gear_id``.

    Args:
        access_token: Valid Strava OAuth access token.
        max_activities: Upper bound on the number of activities fetched before
            filtering (passed through to :func:`get_athlete_activities`).
        progress_callback: Optional callable ``(message, percent)`` invoked at
            key steps so callers can display a progress bar or log progress.
            ``percent`` is an integer in ``[0, 100]``.

    Returns:
        A dict with keys:
        - ``"bikes"``: ``dict[str, str]`` mapping gear_id to bike name.
        - ``"segments"``: :class:`pandas.DataFrame` of starred segments.
        - ``"efforts"``: :class:`pandas.DataFrame` of all efforts, with a
          ``gear_id`` column resolved from the power-filtered activities.
    """

    _http: Any = _DevSession() if dev else _LoggingSession()

    def _progress(msg: str, pct: int) -> None:
        
        if progress_callback is not None:
            progress_callback(msg, pct)

    
    # Step 1: cycling activities with power — build activity_id → gear_id map
    _progress("📋 GET /athlete/activities — fetching cycling activities with power data…", 10)
    activities = get_athlete_activities(access_token, max_activities=max_activities, _http=_http)

    # Step 2: resolve bike names — GET /athlete for active bikes, GET /gear/{id} for retired
    _progress("🚴 Resolving bike names…", 22)
    gear_ids: set[str] = {
        act_data["gear_id"]
        for act_data in activities.values()
        if act_data.get("gear_id")
    }
    bikes, bike_distances, ftp = get_athlete_bikes(access_token, gear_ids=gear_ids, _http=_http)

    # Step 3: starred segments
    _progress("⭐ GET /segments/starred — fetching your starred segments…", 35)
    segments_df = get_starred_segments(access_token, _http=_http)

    # Step 4: segment efforts for each starred segment
    all_efforts: list[pd.DataFrame] = []
    if not segments_df.empty:
        n_segments = len(segments_df)
        segment_names: dict = segments_df.set_index("segment_id")["name"].to_dict()
        for i, segment_id in enumerate(segments_df["segment_id"]):
            seg_name = segment_names.get(segment_id, str(segment_id))
            pct = 42 + (int(50 * i / n_segments) if n_segments > 0 else 0)
            _progress(
                f"💪 GET /segment_efforts — '{seg_name}' ({i + 1} of {n_segments})…",
                pct,
            )
            efforts_df = get_segment_efforts(access_token, int(segment_id), _http=_http)
            if not efforts_df.empty:
                all_efforts.append(efforts_df)
            if not dev:
                time.sleep(1)

    _progress("🔗 Joining effort data with activity info…", 93)
    efforts = pd.concat(all_efforts, ignore_index=True) if all_efforts else pd.DataFrame()

    # Step 5: resolve gear_id from the activities lookup
    if not efforts.empty and activities:
        efforts["gear_id"] = efforts["activity_id"].map(
            lambda aid: activities.get(int(aid), {}).get("gear_id") if pd.notna(aid) else None
        )
    elif not efforts.empty:
        efforts["gear_id"] = None

    # Build a rides DataFrame (one row per activity) from the activities dict
    rides = pd.DataFrame(
        [{"activity_id": str(aid), **data} for aid, data in activities.items()]
    ) if activities else pd.DataFrame()

    return {"bikes": bikes, "bike_distances": bike_distances, "ftp": ftp, "segments": segments_df, "efforts": efforts, "rides": rides, "activities": activities}
