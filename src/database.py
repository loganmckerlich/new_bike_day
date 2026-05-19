"""SQLite-backed persistence for starred segments and segment efforts."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

_DB_PATH: Path = Path(__file__).resolve().parents[1] / "data" / "new_bike_day.db"

_CREATE_STARRED_SEGMENTS: str = """
CREATE TABLE IF NOT EXISTS starred_segments (
    segment_id         INTEGER PRIMARY KEY,
    name               TEXT,
    distance           REAL,
    average_grade      REAL,
    climb_category     INTEGER,
    total_elevation_gain REAL,
    start_lat          REAL,
    start_lng          REAL,
    segment_type       TEXT
)
"""

_CREATE_SEGMENT_EFFORTS: str = """
CREATE TABLE IF NOT EXISTS segment_efforts (
    effort_id          INTEGER PRIMARY KEY,
    segment_id         INTEGER,
    gear_id            TEXT,
    start_date         TEXT,
    elapsed_time       INTEGER,
    moving_time        INTEGER,
    average_watts      REAL,
    average_heartrate  REAL
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
    """Create the starred_segments and segment_efforts tables if they do not exist."""
    with _connect() as conn:
        conn.execute(_CREATE_STARRED_SEGMENTS)
        conn.execute(_CREATE_SEGMENT_EFFORTS)


def save_segments(df: pd.DataFrame) -> None:
    """Upsert rows into the starred_segments table, skipping existing rows.

    Rows whose ``segment_id`` already exists in the database are silently
    ignored (``INSERT OR IGNORE`` semantics).

    Args:
        df: DataFrame containing starred segment data.  Only columns that
            appear in the database schema are written.
    """
    if df.empty:
        return
    available = [c for c in _SEGMENTS_COLS if c in df.columns]
    col_list = ", ".join(available)
    placeholders = ", ".join("?" * len(available))
    sql = f"INSERT OR IGNORE INTO starred_segments ({col_list}) VALUES ({placeholders})"
    with _connect() as conn:
        conn.executemany(sql, df[available].itertuples(index=False, name=None))


def save_efforts(df: pd.DataFrame) -> None:
    """Upsert rows into the segment_efforts table, skipping existing rows.

    Rows whose ``effort_id`` already exists in the database are silently
    ignored (``INSERT OR IGNORE`` semantics).

    Args:
        df: DataFrame containing segment effort data.  Only columns that
            appear in the database schema are written.
    """
    if df.empty:
        return
    available = [c for c in _EFFORTS_COLS if c in df.columns]
    col_list = ", ".join(available)
    placeholders = ", ".join("?" * len(available))
    sql = f"INSERT OR IGNORE INTO segment_efforts ({col_list}) VALUES ({placeholders})"
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
        except Exception:
            return pd.DataFrame(columns=_SEGMENTS_COLS)


def load_efforts() -> pd.DataFrame:
    """Load all rows from the segment_efforts table.

    Returns:
        DataFrame with columns matching the segment_efforts schema, or an
        empty DataFrame if the table is empty or does not exist yet.
    """
    with _connect() as conn:
        try:
            return pd.read_sql_query("SELECT * FROM segment_efforts", conn)
        except Exception:
            return pd.DataFrame(columns=_EFFORTS_COLS)
