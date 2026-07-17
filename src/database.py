"""Supabase-backed persistence for starred segments, segment efforts, and athlete tokens.

All API responses are stored here as a shared cache in Supabase. The application
serves data from this cache and only calls the Strava API when a refresh is
requested or when the cache is empty on first use.
"""

from __future__ import annotations
        
from streamlit.runtime.scriptrunner import get_script_run_ctx
import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import streamlit as st
from supabase import create_client

_CLEANUP_TRIGGER_MB = 450.0
_CLEANUP_TARGET_MB = 425.0

_SUPABASE_URL = st.secrets.get("SUPABASE_URL")
_SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")

_PAGE_SIZE = 1000
_UPSERT_BATCH_SIZE = 500


def _upsert_batched(table: str, records: list[dict], on_conflict: str) -> None:
    """Upsert records in chunks to avoid Supabase PostgREST payload size limits.

    PostgREST rejects request bodies above ~1 MB; for large users a single
    ingest window can contain thousands of rows, so we split into batches.
    ponytail: 500 rows × ~10 fields × ~30 bytes ≈ 150 KB per batch, well
    under the 1 MB limit. Upgrade path: increase _UPSERT_BATCH_SIZE if
    Supabase raises its limit or if fields grow.
    """
    client = _get_supabase()
    total_batches = (len(records) + _UPSERT_BATCH_SIZE - 1) // _UPSERT_BATCH_SIZE
    for batch_index, i in enumerate(range(0, len(records), _UPSERT_BATCH_SIZE)):
        try:
            client.table(table).upsert(records[i : i + _UPSERT_BATCH_SIZE], on_conflict=on_conflict).execute()
        except Exception as exc:
            raise RuntimeError(
                f"Upsert to '{table}' failed on batch {batch_index + 1}/{total_batches} "
                f"(rows {i}–{min(i + _UPSERT_BATCH_SIZE, len(records)) - 1})"
            ) from exc


def _paginated_select(table: str, eq_col: str, eq_val: str) -> list[dict]:
    """Fetch all rows from a table using range-based pagination."""
    # ponytail: Supabase PostgREST caps responses at 1000 rows by default;
    # this walks pages until a short page signals the end.
    client = _get_supabase()
    rows: list[dict] = []
    start = 0
    while True:
        end = start + _PAGE_SIZE - 1
        page = client.table(table).select("*").eq(eq_col, eq_val).range(start, end).execute()
        batch = page.data or []
        rows.extend(batch)
        if len(batch) < _PAGE_SIZE:
            break
        start += _PAGE_SIZE
    return rows

supabase = create_client(_SUPABASE_URL, _SUPABASE_KEY) if _SUPABASE_URL and _SUPABASE_KEY else None

_CREATE_STARRED_SEGMENTS: str = """
CREATE TABLE IF NOT EXISTS starred_segments (
    athlete_id           TEXT NOT NULL,
    segment_id           TEXT NOT NULL,
    name                 TEXT,
    distance             REAL,
    average_grade        REAL,
    maximum_grade        REAL,
    climb_category       INTEGER,
    hazardous            BOOLEAN,
    total_elevation_gain REAL,
    start_lat            REAL,
    start_lng            REAL,
    segment_type         TEXT,
    segment_type_detail  TEXT,
    PRIMARY KEY (athlete_id, segment_id)
)
"""

_CREATE_SEGMENT_EFFORTS: str = """
CREATE TABLE IF NOT EXISTS segment_efforts (
    athlete_id        TEXT NOT NULL,
    effort_id         TEXT NOT NULL,
    segment_id        TEXT,
    activity_id       TEXT,
    gear_id           TEXT,
    start_date        TIMESTAMP,
    elapsed_time           INTEGER,
    moving_time            INTEGER,
    average_watts          REAL,
    average_heartrate      REAL,
    PRIMARY KEY (athlete_id, effort_id)
)
"""

_CREATE_RIDES: str = """
CREATE TABLE IF NOT EXISTS rides (
    athlete_id           TEXT NOT NULL,
    activity_id          TEXT NOT NULL,
    gear_id              TEXT,
    name                 TEXT,
    sport_type           TEXT,
    start_date           TIMESTAMP,
    moving_time          INTEGER,
    elapsed_time         INTEGER,
    distance             REAL,
    total_elevation_gain REAL,
    average_watts        REAL,
    average_heartrate    REAL,
    average_speed        REAL,
    PRIMARY KEY (athlete_id, activity_id)
)
"""

_CREATE_BIKES: str = """
CREATE TABLE IF NOT EXISTS bikes (
    athlete_id         TEXT NOT NULL,
    gear_id            TEXT NOT NULL,
    name               TEXT,
    converted_distance REAL,
    PRIMARY KEY (athlete_id, gear_id)
)
"""

_CREATE_FTP: str = """
CREATE TABLE IF NOT EXISTS athlete_ftp (
    athlete_id TEXT PRIMARY KEY,
    ftp        INTEGER
)
"""

_CREATE_ATHLETE_TOKENS: str = """
CREATE TABLE IF NOT EXISTS athlete_tokens (
    athlete_id    TEXT PRIMARY KEY,
    access_token  TEXT NOT NULL,
    refresh_token TEXT NOT NULL,
    expires_at    INTEGER NOT NULL
)
"""

_CREATE_USERS: str = """
CREATE TABLE IF NOT EXISTS users (
    athlete_id TEXT PRIMARY KEY,
    last_accessed TIMESTAMP,
    created_at TIMESTAMP,
    last_ingested_date TIMESTAMP,
    oldest_ingested_date TIMESTAMP
)
"""

_CREATE_SEGMENT_GEO: str = """
CREATE TABLE IF NOT EXISTS segment_geo_cache (
    segment_id     TEXT PRIMARY KEY,
    polyline_json  TEXT,
    elevation_low  REAL,
    elevation_high REAL,
    start_lat      REAL,
    start_lng      REAL,
    end_lat        REAL,
    end_lng        REAL,
    streams_json   TEXT
)
"""

_SEGMENTS_COLS: list[str] = [
    "athlete_id",
    "segment_id",
    "name",
    "distance",
    "average_grade",
    "maximum_grade",
    "climb_category",
    "hazardous",
    "total_elevation_gain",
    "start_lat",
    "start_lng",
    "segment_type",
    "segment_type_detail",
]

_EFFORTS_COLS: list[str] = [
    "athlete_id",
    "effort_id",
    "segment_id",
    "activity_id",
    "gear_id",
    "start_date",
    "elapsed_time",
    "moving_time",
    "average_watts",
    "average_heartrate",
]

_RIDES_COLS: list[str] = [
    "athlete_id",
    "activity_id",
    "gear_id",
    "name",
    "sport_type",
    "start_date",
    "moving_time",
    "elapsed_time",
    "distance",
    "total_elevation_gain",
    "average_watts",
    "average_heartrate",
    "average_speed",
]


def _get_supabase() -> Any:
    if supabase is None:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be configured")
    return supabase


def _normalize_athlete_id(athlete_id: int | str | None) -> str | None:
    if athlete_id is None:
        return None
    return str(athlete_id)


def _clean_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (float, int)):
        return None if pd.isna(value) else value
    if isinstance(value, str):
        return value
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime().isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _rows_to_dataframe(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)
    cleaned_rows = []
    for row in rows:
        cleaned_rows.append({column: _clean_value(row.get(column)) for column in columns})
    return pd.DataFrame(cleaned_rows, columns=columns)


def init_db() -> None:
    """Ensure the Supabase client is configured for the app.

    The schema itself is expected to be created in the Supabase SQL editor so
    the app can focus on data access and user-scoping.
    """
    _get_supabase()


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def touch_user(athlete_id: int | str) -> None:
    """Create or update the user row and refresh last_accessed."""
    athlete_key = _normalize_athlete_id(athlete_id)
    if not athlete_key:
        return
    now = datetime.now(timezone.utc).isoformat()
    client = _get_supabase()
    existing = (
        client.table("users")
        .select("created_at,last_ingested_date,oldest_ingested_date")
        .eq("athlete_id", athlete_key)
        .limit(1)
        .execute()
    )
    existing_row = existing.data[0] if existing.data else {}
    created_at = existing_row.get("created_at") or now
    payload = {
        "athlete_id": athlete_key,
        "last_accessed": now,
        "created_at": created_at,
        "last_ingested_date": existing_row.get("last_ingested_date"),
        "oldest_ingested_date": existing_row.get("oldest_ingested_date"),
    }
    client.table("users").upsert(payload, on_conflict="athlete_id").execute()


def load_user_ingest_dates(athlete_id: int | str) -> tuple[str | None, str | None]:
    """Load (last_ingested_date, oldest_ingested_date) for a user."""
    athlete_key = _normalize_athlete_id(athlete_id)
    if not athlete_key:
        return None, None
    try:
        response = (
            _get_supabase()
            .table("users")
            .select("last_ingested_date,oldest_ingested_date")
            .eq("athlete_id", athlete_key)
            .limit(1)
            .execute()
        )
        if not response.data:
            return None, None
        row = response.data[0]
        return row.get("last_ingested_date"), row.get("oldest_ingested_date")
    except Exception:
        return None, None


def save_user_ingest_dates(
    athlete_id: int | str,
    *,
    last_ingested_date: Any | None = None,
    oldest_ingested_date: Any | None = None,
) -> None:
    """Persist ingestion watermark dates on the users table."""
    athlete_key = _normalize_athlete_id(athlete_id)
    if not athlete_key:
        return
    now = datetime.now(timezone.utc).isoformat()
    client = _get_supabase()
    try:
        existing = (
            client.table("users")
            .select("created_at,last_ingested_date,oldest_ingested_date")
            .eq("athlete_id", athlete_key)
            .limit(1)
            .execute()
        )
    except Exception:
        return
    row = existing.data[0] if existing.data else {}
    payload = {
        "athlete_id": athlete_key,
        "last_accessed": now,
        "created_at": row.get("created_at") or now,
        "last_ingested_date": _clean_value(
            last_ingested_date if last_ingested_date is not None else row.get("last_ingested_date")
        ),
        "oldest_ingested_date": _clean_value(
            oldest_ingested_date if oldest_ingested_date is not None else row.get("oldest_ingested_date")
        ),
    }
    try:
        client.table("users").upsert(payload, on_conflict="athlete_id").execute()
    except Exception:
        return


# ---------------------------------------------------------------------------
# Page view tracking
# ---------------------------------------------------------------------------


def log_page_view(page: str) -> None:
    """Insert one row into page_views. Silent no-op if Supabase is not configured."""
    # ponytail: session_id is stable per browser tab; use it for distinct-session queries.
    # Not truly unique users — incognito or a new tab = new session.
    try:
        ctx = get_script_run_ctx()
        session_id = ctx.session_id if ctx else None
        client = _get_supabase()
        client.table("page_views").insert({"page": page, "session_id": session_id}).execute()
    except Exception:
        pass  # never break the app over analytics


# ---------------------------------------------------------------------------
# Starred segments
# ---------------------------------------------------------------------------

def save_segments(df: pd.DataFrame, athlete_id: int | str) -> None:
    """Upsert rows into the starred_segments table."""
    if df.empty:
        return
    athlete_key = _normalize_athlete_id(athlete_id)
    if not athlete_key:
        return
    columns = [c for c in _SEGMENTS_COLS if c in df.columns and c != "athlete_id"]
    records = []
    for row in df[columns].itertuples(index=False, name=None):
        record = {column: _clean_value(value) for column, value in zip(columns, row)}
        record["athlete_id"] = athlete_key
        records.append(record)
    if records:
        _upsert_batched("starred_segments", records, "athlete_id,segment_id")


def load_segments(athlete_id: int | str) -> pd.DataFrame:
    """Load all starred segments for a given athlete."""
    athlete_key = _normalize_athlete_id(athlete_id)
    if not athlete_key:
        return pd.DataFrame(columns=_SEGMENTS_COLS)
    try:
        rows = _paginated_select("starred_segments", "athlete_id", athlete_key)
        return _rows_to_dataframe(rows, _SEGMENTS_COLS)
    except Exception:
        return pd.DataFrame(columns=_SEGMENTS_COLS)


def clear_segments(athlete_id: int | str) -> None:
    """Delete all starred segment rows for a given athlete."""
    athlete_key = _normalize_athlete_id(athlete_id)
    if not athlete_key:
        return
    try:
        _get_supabase().table("starred_segments").delete().eq("athlete_id", athlete_key).execute()
    except Exception:
        return


# ---------------------------------------------------------------------------
# Segment efforts
# ---------------------------------------------------------------------------

def save_efforts(df: pd.DataFrame, athlete_id: int | str) -> None:
    """Upsert rows into the segment_efforts table."""
    if df.empty:
        return
    athlete_key = _normalize_athlete_id(athlete_id)
    if not athlete_key:
        return
    columns = [c for c in _EFFORTS_COLS if c in df.columns and c != "athlete_id"]
    records = []
    for row in df[columns].itertuples(index=False, name=None):
        record = {column: _clean_value(value) for column, value in zip(columns, row)}
        record["athlete_id"] = athlete_key
        records.append(record)
    if records:
        _upsert_batched("segment_efforts", records, "athlete_id,effort_id")


def load_efforts(athlete_id: int | str) -> pd.DataFrame:
    """Load all segment efforts for a given athlete."""
    athlete_key = _normalize_athlete_id(athlete_id)
    if not athlete_key:
        return pd.DataFrame(columns=_EFFORTS_COLS)
    try:
        rows = _paginated_select("segment_efforts", "athlete_id", athlete_key)
        return _rows_to_dataframe(rows, _EFFORTS_COLS)
    except Exception:
        return pd.DataFrame(columns=_EFFORTS_COLS)


def clear_efforts(athlete_id: int | str) -> None:
    """Delete all segment effort rows for a given athlete."""
    athlete_key = _normalize_athlete_id(athlete_id)
    if not athlete_key:
        return
    try:
        _get_supabase().table("segment_efforts").delete().eq("athlete_id", athlete_key).execute()
    except Exception:
        return


# ---------------------------------------------------------------------------
# Rides (activity-level)
# ---------------------------------------------------------------------------

def save_rides(df: pd.DataFrame, athlete_id: int | str) -> None:
    """Upsert rows into the rides table."""
    if df.empty:
        return
    athlete_key = _normalize_athlete_id(athlete_id)
    if not athlete_key:
        return
    columns = [c for c in _RIDES_COLS if c in df.columns and c != "athlete_id"]
    records = []
    for row in df[columns].itertuples(index=False, name=None):
        record = {col: _clean_value(val) for col, val in zip(columns, row)}
        record["athlete_id"] = athlete_key
        records.append(record)
    if records:
        _upsert_batched("rides", records, "athlete_id,activity_id")


def load_rides(athlete_id: int | str) -> pd.DataFrame:
    """Load all rides for a given athlete."""
    athlete_key = _normalize_athlete_id(athlete_id)
    if not athlete_key:
        return pd.DataFrame(columns=_RIDES_COLS)
    try:
        rows = _paginated_select("rides", "athlete_id", athlete_key)
        return _rows_to_dataframe(rows, _RIDES_COLS)
    except Exception:
        return pd.DataFrame(columns=_RIDES_COLS)


def clear_rides(athlete_id: int | str) -> None:
    """Delete all ride rows for a given athlete."""
    athlete_key = _normalize_athlete_id(athlete_id)
    if not athlete_key:
        return
    try:
        _get_supabase().table("rides").delete().eq("athlete_id", athlete_key).execute()
    except Exception:
        return


# =======
# FTP
# =======

def save_ftp(ftp: int | None, athlete_id: int | str) -> None:
    """Persist the athlete's FTP value in the database."""
    athlete_key = _normalize_athlete_id(athlete_id)
    if not athlete_key:
        return
    payload = {"athlete_id": athlete_key, "ftp": ftp}
    _get_supabase().table("athlete_ftp").upsert(payload, on_conflict="athlete_id").execute()


def load_ftp(athlete_id: int | str) -> int | None:
    """Load the athlete's FTP value from the database."""
    athlete_key = _normalize_athlete_id(athlete_id)
    if not athlete_key:
        return None
    try:
        response = _get_supabase().table("athlete_ftp").select("ftp").eq("athlete_id", athlete_key).limit(1).execute()
        if not response.data:
            return None
        return response.data[0].get("ftp")
    except Exception:
        return None


def clear_ftp(athlete_id: int | str) -> None:
    """Delete the athlete's FTP value from the database."""
    athlete_key = _normalize_athlete_id(athlete_id)
    if not athlete_key:
        return
    try:
        _get_supabase().table("athlete_ftp").delete().eq("athlete_id", athlete_key).execute()
    except Exception:
        return


# ---------------------------------------------------------------------------
# Bikes (gear_id → name mapping)
# ---------------------------------------------------------------------------

def save_bikes(
    bikes: dict[str, str],
    athlete_id: int | str,
    distances: dict[str, float] | None = None,
) -> None:
    """Upsert the athlete's bike inventory into the bikes table."""
    if not bikes:
        return
    athlete_key = _normalize_athlete_id(athlete_id)
    if not athlete_key:
        return
    distances = distances or {}
    records = [
        {
            "athlete_id": athlete_key,
            "gear_id": gear_id,
            "name": name,
            "converted_distance": _clean_value(distances.get(gear_id)),
        }
        for gear_id, name in bikes.items()
    ]
    _get_supabase().table("bikes").upsert(records, on_conflict="athlete_id,gear_id").execute()


def load_bikes(athlete_id: int | str) -> tuple[dict[str, str], dict[str, float]]:
    """Load the bike inventory for a given athlete."""
    athlete_key = _normalize_athlete_id(athlete_id)
    if not athlete_key:
        return {}, {}
    try:
        response = _get_supabase().table("bikes").select("gear_id, name, converted_distance").eq("athlete_id", athlete_key).execute()
        rows = response.data or []
        names = {row.get("gear_id"): row.get("name") for row in rows if row.get("gear_id") is not None}
        distances = {
            row.get("gear_id"): row.get("converted_distance")
            for row in rows
            if row.get("gear_id") is not None and row.get("converted_distance") is not None
        }
        return names, distances
    except Exception:
        return {}, {}


def clear_bikes(athlete_id: int | str) -> None:
    """Delete all bike rows for a given athlete."""
    athlete_key = _normalize_athlete_id(athlete_id)
    if not athlete_key:
        return
    try:
        _get_supabase().table("bikes").delete().eq("athlete_id", athlete_key).execute()
    except Exception:
        return


# ---------------------------------------------------------------------------
# Athlete tokens (for webhook-triggered re-ingestion)
# ---------------------------------------------------------------------------

def save_athlete_token(
    athlete_id: int | str,
    access_token: str,
    refresh_token: str,
    expires_at: int,
) -> None:
    """Persist or update the OAuth tokens for an athlete."""
    athlete_key = _normalize_athlete_id(athlete_id)
    if not athlete_key:
        return
    payload = {
        "athlete_id": athlete_key,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": expires_at,
    }
    _get_supabase().table("athlete_tokens").upsert(payload, on_conflict="athlete_id").execute()


def load_athlete_token(athlete_id: int | str | None = None) -> dict[str, Any] | None:
    """Load token information for an athlete.

    In multi-user mode, an athlete_id should be provided so reads stay scoped to a
    single user and never cross-contaminate another athlete's data. Passing
    ``None`` returns ``None`` to keep the data access boundary explicit.
    """
    if athlete_id is None:
        return None
    athlete_key = _normalize_athlete_id(athlete_id)
    if not athlete_key:
        return None
    try:
        response = _get_supabase().table("athlete_tokens").select("*").eq("athlete_id", athlete_key).limit(1).execute()
        if not response.data:
            return None
        row = response.data[0]
        return {
            "athlete_id": row.get("athlete_id"),
            "access_token": row.get("access_token"),
            "refresh_token": row.get("refresh_token"),
            "expires_at": row.get("expires_at"),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Segment geo cache (polyline, elevation, streams)
# ---------------------------------------------------------------------------

def save_segment_geo(segment_id: int | str, detail: dict[str, Any], streams: dict[str, list[float]]) -> None:
    """Persist geo and stream data for a segment."""
    polyline_points = detail.get("polyline_points") or []
    start_latlng = detail.get("start_latlng") or []
    end_latlng = detail.get("end_latlng") or []
    payload = {
        "segment_id": str(segment_id),
        "polyline_json": json.dumps(polyline_points),
        "elevation_low": detail.get("elevation_low"),
        "elevation_high": detail.get("elevation_high"),
        "start_lat": start_latlng[0] if len(start_latlng) > 0 else None,
        "start_lng": start_latlng[1] if len(start_latlng) > 1 else None,
        "end_lat": end_latlng[0] if len(end_latlng) > 0 else None,
        "end_lng": end_latlng[1] if len(end_latlng) > 1 else None,
        "streams_json": json.dumps(streams) if streams else None,
    }
    _get_supabase().table("segment_geo_cache").upsert(payload, on_conflict="segment_id").execute()


def load_segment_geo(segment_id: int | str) -> dict[str, Any] | None:
    """Load cached geo and stream data for a segment."""
    try:
        response = _get_supabase().table("segment_geo_cache").select("*").eq("segment_id", str(segment_id)).limit(1).execute()
        if not response.data:
            return None
        row = response.data[0]
        polyline_points = json.loads(row.get("polyline_json") or "[]") if row.get("polyline_json") else []
        streams = json.loads(row.get("streams_json") or "{}") if row.get("streams_json") else {}
        start_lat, start_lng = row.get("start_lat"), row.get("start_lng")
        end_lat, end_lng = row.get("end_lat"), row.get("end_lng")
        return {
            "polyline_points": [tuple(point) for point in polyline_points if isinstance(point, (list, tuple))],
            "elevation_low": row.get("elevation_low"),
            "elevation_high": row.get("elevation_high"),
            "start_latlng": [start_lat, start_lng] if start_lat is not None else [],
            "end_latlng": [end_lat, end_lng] if end_lat is not None else [],
            "streams": streams,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Storage management
# ---------------------------------------------------------------------------

def get_db_size_mb() -> float:
    """Return the current database size in megabytes via the Supabase RPC."""
    try:
        response = _get_supabase().rpc("get_db_size_mb").execute()
        data = response.data or []
        if not data:
            return 0.0
        if isinstance(data, list):
            first = data[0]
            if isinstance(first, dict):
                for key in ("get_db_size_mb", "db_size_mb", "value"):
                    if key in first:
                        return float(first[key])
                return float(first.get("value", 0.0))
            return float(first)
        return float(data)
    except Exception:
        return 0.0


def _delete_user_data(athlete_id: int | str) -> None:
    athlete_key = _normalize_athlete_id(athlete_id)
    if not athlete_key:
        return
    client = _get_supabase()
    try:
        client.table("segment_efforts").delete().eq("athlete_id", athlete_key).execute()
        client.table("starred_segments").delete().eq("athlete_id", athlete_key).execute()
        client.table("bikes").delete().eq("athlete_id", athlete_key).execute()
        client.table("athlete_ftp").delete().eq("athlete_id", athlete_key).execute()
        client.table("athlete_tokens").delete().eq("athlete_id", athlete_key).execute()
        client.table("users").delete().eq("athlete_id", athlete_key).execute()
    except Exception:
        return


def cleanup_if_needed(athlete_id: int | str) -> None:
    """Evict least recently accessed users when database usage is high."""
    athlete_key = _normalize_athlete_id(athlete_id)
    if not athlete_key:
        return
    current_size_mb = get_db_size_mb()
    # ponytail: 450MB is the trigger point and 425MB is the safety target to stay under the 500MB free-tier limit (85% threshold).
    if current_size_mb <= _CLEANUP_TRIGGER_MB:
        return
    while current_size_mb > _CLEANUP_TARGET_MB:
        try:
            response = _get_supabase().table("users").select("athlete_id,last_accessed").order("last_accessed", desc=False).limit(1).execute()
            candidates = [
                row for row in (response.data or [])
                if str(row.get("athlete_id")) != athlete_key and row.get("athlete_id") is not None
            ]
            if not candidates:
                break
            target = candidates[0]
            _delete_user_data(target["athlete_id"])
        except Exception:
            break
        current_size_mb = get_db_size_mb()
