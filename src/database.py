"""SQLite-backed persistence for starred segments, segment efforts, and athlete tokens.

All API responses are stored here as a static cache.  The application serves data
from this cache and only calls the Strava API when a webhook event notifies that
data has changed, or when the cache is empty on first use.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

_DEFAULT_DB_PATH: Path = Path(__file__).resolve().parents[1] / "data" / "new_bike_day.db"

# Allow override via environment variable for testing or alternate deployments.
_DB_PATH: Path = Path(os.environ["NEW_BIKE_DAY_DB_PATH"]) if "NEW_BIKE_DAY_DB_PATH" in os.environ else _DEFAULT_DB_PATH

_CREATE_STARRED_SEGMENTS: str = """
CREATE TABLE IF NOT EXISTS starred_segments (
    segment_id           INTEGER PRIMARY KEY,
    name                 TEXT,
    distance             REAL,
    average_grade        REAL,
    climb_category       INTEGER,
    total_elevation_gain REAL,
    start_lat            REAL,
    start_lng            REAL,
    segment_type         TEXT
)
"""

_CREATE_SEGMENT_EFFORTS: str = """
CREATE TABLE IF NOT EXISTS segment_efforts (
    effort_id         INTEGER PRIMARY KEY,
    segment_id        INTEGER,
    activity_id       INTEGER,
    gear_id           TEXT,
    start_date        TEXT,
    elapsed_time      INTEGER,
    moving_time       INTEGER,
    average_watts     REAL,
    average_heartrate REAL
)
"""

_CREATE_BIKES: str = """
CREATE TABLE IF NOT EXISTS bikes (
    gear_id TEXT PRIMARY KEY,
    name    TEXT
)
"""

_CREATE_ATHLETE_TOKENS: str = """
CREATE TABLE IF NOT EXISTS athlete_tokens (
    athlete_id    INTEGER PRIMARY KEY,
    access_token  TEXT NOT NULL,
    refresh_token TEXT NOT NULL,
    expires_at    INTEGER NOT NULL
)
"""

_CREATE_SEGMENT_GEO: str = """
CREATE TABLE IF NOT EXISTS segment_geo_cache (
    segment_id     INTEGER PRIMARY KEY,
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
    "segment_id",
    "name",
    "distance",
    "average_grade",
    "climb_category",
    "total_elevation_gain",
    "start_lat",
    "start_lng",
    "segment_type",
]

_EFFORTS_COLS: list[str] = [
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


def _connect() -> sqlite3.Connection:
    """Open a connection to the SQLite database, creating the file if needed."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(_DB_PATH)


def init_db() -> None:
    """Create all tables and apply schema migrations if needed."""
    with _connect() as conn:
        conn.execute(_CREATE_STARRED_SEGMENTS)
        conn.execute(_CREATE_SEGMENT_EFFORTS)
        conn.execute(_CREATE_BIKES)
        conn.execute(_CREATE_ATHLETE_TOKENS)
        conn.execute(_CREATE_SEGMENT_GEO)

        # Migration: add activity_id column to segment_efforts if it was created
        # before this column was introduced.
        try:
            conn.execute("ALTER TABLE segment_efforts ADD COLUMN activity_id INTEGER")
        except sqlite3.OperationalError:
            pass  # Column already exists — nothing to do.


# ---------------------------------------------------------------------------
# Starred segments
# ---------------------------------------------------------------------------

def save_segments(df: pd.DataFrame) -> None:
    """Upsert rows into the starred_segments table.

    Existing rows are replaced so that segment metadata stays current after a
    re-ingest triggered by a webhook.

    Args:
        df: DataFrame containing starred segment data.  Only columns that
            appear in the database schema are written.
    """
    if df.empty:
        return
    available = [c for c in _SEGMENTS_COLS if c in df.columns]
    col_list = ", ".join(available)
    placeholders = ", ".join("?" * len(available))
    sql = f"INSERT OR REPLACE INTO starred_segments ({col_list}) VALUES ({placeholders})"
    with _connect() as conn:
        conn.executemany(sql, df[available].itertuples(index=False, name=None))


def load_segments() -> pd.DataFrame:
    """Load all rows from the starred_segments table.

    Returns:
        DataFrame with columns matching the starred_segments schema, or an
        empty DataFrame if the table is empty or does not exist yet.
    """
    with _connect() as conn:
        try:
            return pd.read_sql_query("SELECT * FROM starred_segments", conn)
        except sqlite3.OperationalError:
            return pd.DataFrame(columns=_SEGMENTS_COLS)


def clear_segments() -> None:
    """Delete all rows from the starred_segments table."""
    with _connect() as conn:
        conn.execute("DELETE FROM starred_segments")


# ---------------------------------------------------------------------------
# Segment efforts
# ---------------------------------------------------------------------------

def save_efforts(df: pd.DataFrame) -> None:
    """Upsert rows into the segment_efforts table.

    Existing rows are replaced so that gear_id and other fields stay current
    after a re-ingest triggered by a webhook.

    Args:
        df: DataFrame containing segment effort data.  Only columns that
            appear in the database schema are written.
    """
    if df.empty:
        return
    available = [c for c in _EFFORTS_COLS if c in df.columns]
    col_list = ", ".join(available)
    placeholders = ", ".join("?" * len(available))
    sql = f"INSERT OR REPLACE INTO segment_efforts ({col_list}) VALUES ({placeholders})"
    with _connect() as conn:
        conn.executemany(sql, df[available].itertuples(index=False, name=None))


def load_efforts() -> pd.DataFrame:
    """Load all rows from the segment_efforts table.

    Returns:
        DataFrame with columns matching the segment_efforts schema, or an
        empty DataFrame if the table is empty or does not exist yet.
    """
    with _connect() as conn:
        try:
            return pd.read_sql_query("SELECT * FROM segment_efforts", conn)
        except sqlite3.OperationalError:
            return pd.DataFrame(columns=_EFFORTS_COLS)


def clear_efforts() -> None:
    """Delete all rows from the segment_efforts table."""
    with _connect() as conn:
        conn.execute("DELETE FROM segment_efforts")


# ---------------------------------------------------------------------------
# Bikes (gear_id → name mapping)
# ---------------------------------------------------------------------------

def save_bikes(bikes: dict[str, str]) -> None:
    """Upsert the athlete's bike inventory into the bikes table.

    Args:
        bikes: Dict mapping gear_id to a human-readable bike name.
    """
    if not bikes:
        return
    sql = "INSERT OR REPLACE INTO bikes (gear_id, name) VALUES (?, ?)"
    with _connect() as conn:
        conn.executemany(sql, bikes.items())


def load_bikes() -> dict[str, str]:
    """Load the bike inventory from the bikes table.

    Returns:
        Dict mapping gear_id to bike name, or an empty dict if the table is
        empty or does not exist yet.
    """
    with _connect() as conn:
        try:
            rows = conn.execute("SELECT gear_id, name FROM bikes").fetchall()
            return {gear_id: name for gear_id, name in rows}
        except sqlite3.OperationalError:
            return {}


def clear_bikes() -> None:
    """Delete all rows from the bikes table."""
    with _connect() as conn:
        conn.execute("DELETE FROM bikes")


# ---------------------------------------------------------------------------
# Athlete tokens (for webhook-triggered re-ingestion)
# ---------------------------------------------------------------------------

def save_athlete_token(
    athlete_id: int,
    access_token: str,
    refresh_token: str,
    expires_at: int,
) -> None:
    """Persist or update the OAuth tokens for an athlete.

    Tokens are needed by the webhook server to call the Strava API when an
    event arrives without user interaction.

    Args:
        athlete_id: The athlete's Strava user ID.
        access_token: Current short-lived access token.
        refresh_token: Long-lived refresh token used to obtain a new access token.
        expires_at: Unix timestamp at which the access_token expires.
    """
    sql = """
        INSERT OR REPLACE INTO athlete_tokens
            (athlete_id, access_token, refresh_token, expires_at)
        VALUES (?, ?, ?, ?)
    """
    with _connect() as conn:
        conn.execute(sql, (athlete_id, access_token, refresh_token, expires_at))


def load_athlete_token(athlete_id: int | None = None) -> dict[str, Any] | None:
    """Load token information for an athlete.

    Args:
        athlete_id: The athlete's Strava user ID.  If ``None``, returns the
            first row found (useful for single-user deployments).

    Returns:
        Dict with keys ``athlete_id``, ``access_token``, ``refresh_token``,
        ``expires_at``, or ``None`` if no token is stored.
    """
    with _connect() as conn:
        try:
            if athlete_id is not None:
                row = conn.execute(
                    "SELECT athlete_id, access_token, refresh_token, expires_at "
                    "FROM athlete_tokens WHERE athlete_id = ?",
                    (athlete_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT athlete_id, access_token, refresh_token, expires_at "
                    "FROM athlete_tokens LIMIT 1"
                ).fetchone()
        except sqlite3.OperationalError:
            return None

    if row is None:
        return None
    return {
        "athlete_id": row[0],
        "access_token": row[1],
        "refresh_token": row[2],
        "expires_at": row[3],
    }


# ---------------------------------------------------------------------------
# Segment geo cache (polyline, elevation, streams)
# ---------------------------------------------------------------------------

def save_segment_geo(segment_id: int, detail: dict[str, Any], streams: dict[str, list[float]]) -> None:
    """Persist geo and stream data for a segment.

    Args:
        segment_id: Strava segment identifier.
        detail: Dict returned by :func:`src.fetch.get_segment_detail`.
        streams: Dict returned by :func:`src.fetch.get_segment_streams`.
    """
    polyline_points = detail.get("polyline_points") or []
    start_latlng = detail.get("start_latlng") or []
    end_latlng = detail.get("end_latlng") or []

    sql = """
        INSERT OR REPLACE INTO segment_geo_cache
            (segment_id, polyline_json, elevation_low, elevation_high,
             start_lat, start_lng, end_lat, end_lng, streams_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with _connect() as conn:
        conn.execute(sql, (
            segment_id,
            json.dumps(polyline_points),
            detail.get("elevation_low"),
            detail.get("elevation_high"),
            start_latlng[0] if len(start_latlng) > 0 else None,
            start_latlng[1] if len(start_latlng) > 1 else None,
            end_latlng[0] if len(end_latlng) > 0 else None,
            end_latlng[1] if len(end_latlng) > 1 else None,
            json.dumps(streams) if streams else None,
        ))


def load_segment_geo(segment_id: int) -> dict[str, Any] | None:
    """Load cached geo and stream data for a segment.

    Returns:
        Dict compatible with what :func:`app.pages.1_Segment_Comparison._get_segment_geo`
        expects, or ``None`` if the segment has not been cached yet.
    """
    with _connect() as conn:
        try:
            row = conn.execute(
                "SELECT polyline_json, elevation_low, elevation_high, "
                "start_lat, start_lng, end_lat, end_lng, streams_json "
                "FROM segment_geo_cache WHERE segment_id = ?",
                (segment_id,),
            ).fetchone()
        except sqlite3.OperationalError:
            return None

    if row is None:
        return None

    polyline_points = json.loads(row[0]) if row[0] else []
    streams = json.loads(row[7]) if row[7] else {}

    start_lat, start_lng = row[3], row[4]
    end_lat, end_lng = row[5], row[6]

    return {
        "polyline_points": [tuple(p) for p in polyline_points],
        "elevation_low": row[1],
        "elevation_high": row[2],
        "start_latlng": [start_lat, start_lng] if start_lat is not None else [],
        "end_latlng": [end_lat, end_lng] if end_lat is not None else [],
        "streams": streams,
    }
