"""SQLite helpers for activity, stream, and weather persistence."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


def get_connection(db_path: str) -> sqlite3.Connection:
    """Create and return a SQLite connection for a database path."""
    target_path = Path(db_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(target_path)


def init_db(conn: sqlite3.Connection) -> None:
    """Initialize SQLite tables required by the ingestion pipeline."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS activities (
            activity_id INTEGER PRIMARY KEY,
            name TEXT,
            distance_m REAL,
            moving_time_s INTEGER,
            elapsed_time_s INTEGER,
            total_elevation_gain_m REAL,
            average_speed_mps REAL,
            max_speed_mps REAL,
            average_heartrate REAL,
            max_heartrate REAL,
            average_watts REAL,
            weighted_average_watts REAL,
            kilojoules REAL,
            suffer_score REAL,
            start_date_local TEXT,
            timezone TEXT,
            gear_id TEXT,
            type TEXT,
            sport_type TEXT,
            start_lat REAL,
            start_lon REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS weather (
            activity_id INTEGER PRIMARY KEY,
            date TEXT,
            temperature_2m_max REAL,
            temperature_2m_min REAL,
            precipitation_sum REAL,
            windspeed_10m_max REAL,
            FOREIGN KEY(activity_id) REFERENCES activities(activity_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS streams (
            activity_id INTEGER PRIMARY KEY,
            streams_json TEXT NOT NULL,
            FOREIGN KEY(activity_id) REFERENCES activities(activity_id)
        )
        """
    )
    conn.commit()


def activity_exists(conn: sqlite3.Connection, activity_id: int) -> bool:
    """Return True when an activity already exists in the database."""
    result = conn.execute(
        "SELECT 1 FROM activities WHERE activity_id = ? LIMIT 1",
        (activity_id,),
    ).fetchone()
    return result is not None


def save_activities(conn: sqlite3.Connection, activities: Iterable[Dict[str, Any]]) -> None:
    """Persist activities to SQLite while skipping rows that already exist."""
    conn.executemany(
        """
        INSERT OR IGNORE INTO activities (
            activity_id, name, distance_m, moving_time_s, elapsed_time_s,
            total_elevation_gain_m, average_speed_mps, max_speed_mps,
            average_heartrate, max_heartrate, average_watts, weighted_average_watts,
            kilojoules, suffer_score, start_date_local, timezone, gear_id, type,
            sport_type, start_lat, start_lon
        ) VALUES (
            :id, :name, :distance_m, :moving_time_s, :elapsed_time_s,
            :total_elevation_gain_m, :average_speed_mps, :max_speed_mps,
            :average_heartrate, :max_heartrate, :average_watts, :weighted_average_watts,
            :kilojoules, :suffer_score, :start_date_local, :timezone, :gear_id, :type,
            :sport_type, :start_lat, :start_lon
        )
        """,
        list(activities),
    )
    conn.commit()


def save_streams(conn: sqlite3.Connection, activity_id: int, streams: Dict[str, Any]) -> None:
    """Persist activity stream payload as JSON for a single activity."""
    conn.execute(
        """
        INSERT OR REPLACE INTO streams (activity_id, streams_json)
        VALUES (?, ?)
        """,
        (activity_id, json.dumps(streams)),
    )
    conn.commit()


def save_weather(conn: sqlite3.Connection, activity_id: int, weather: Dict[str, Any]) -> None:
    """Persist weather values for a single activity."""
    conn.execute(
        """
        INSERT OR REPLACE INTO weather (
            activity_id, date, temperature_2m_max, temperature_2m_min,
            precipitation_sum, windspeed_10m_max
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            activity_id,
            weather.get("date"),
            weather.get("temperature_2m_max"),
            weather.get("temperature_2m_min"),
            weather.get("precipitation_sum"),
            weather.get("windspeed_10m_max"),
        ),
    )
    conn.commit()


def load_activities(conn: sqlite3.Connection, gear_id: Optional[str] = None) -> pd.DataFrame:
    """Load all stored activities or only those for a given gear ID."""
    query = "SELECT * FROM activities"
    params: tuple[Any, ...] = ()
    if gear_id:
        query += " WHERE gear_id = ?"
        params = (gear_id,)
    return pd.read_sql_query(query, conn, params=params)
