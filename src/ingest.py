"""Ingestion entrypoint for pulling Strava and weather data into SQLite."""

from __future__ import annotations

import logging
import os
import argparse
from typing import Optional

import requests
from stravalib import Client

try:
    from .auth import get_access_token
    from .database import activity_exists, get_connection, init_db, save_activities, save_streams, save_weather
    from .fetch import get_activities, get_streams
    from .weather import get_weather
except ImportError:  # pragma: no cover
    from auth import get_access_token
    from database import activity_exists, get_connection, init_db, save_activities, save_streams, save_weather
    from fetch import get_activities, get_streams
    from weather import get_weather


def ingest(db_path: str = "data/strava.db", max_activities: Optional[int] = None) -> int:
    """Run idempotent ingestion from Strava and Open-Meteo into SQLite.

    Args:
        db_path: Path to the SQLite database.
        max_activities: Optional cap on activities fetched from Strava.

    Returns:
        Number of newly inserted activities.
    """
    access_token = get_access_token()
    client = Client(access_token=access_token)
    activities = get_activities(client=client, limit=max_activities)

    inserted = 0
    with get_connection(db_path) as conn:
        init_db(conn)
        for activity in activities:
            activity_id = int(activity["id"])
            if activity_exists(conn, activity_id):
                continue

            save_activities(conn, [activity])

            try:
                streams = get_streams(client=client, activity_id=activity_id)
                save_streams(conn, activity_id=activity_id, streams=streams)
            except requests.RequestException as exc:
                logging.warning("Unable to fetch streams for activity %s: %s", activity_id, exc)

            lat = activity.get("start_lat")
            lon = activity.get("start_lon")
            start_date = activity.get("start_date_local")
            if lat is not None and lon is not None and start_date:
                try:
                    weather = get_weather(lat=float(lat), lon=float(lon), target_date=str(start_date))
                    save_weather(conn, activity_id=activity_id, weather=weather)
                except requests.RequestException as exc:
                    logging.warning("Unable to fetch weather for activity %s: %s", activity_id, exc)
                except ValueError as exc:
                    logging.warning("Unable to parse weather date for activity %s: %s", activity_id, exc)

            inserted += 1

    return inserted


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for ingestion execution."""
    parser = argparse.ArgumentParser(description="Ingest Strava activities and weather into SQLite.")
    parser.add_argument(
        "--db-path",
        default=os.getenv("STRAVA_DB_PATH", "data/strava.db"),
        help="Path to SQLite database file.",
    )
    parser.add_argument(
        "--max-activities",
        type=int,
        default=None,
        help="Optional maximum number of activities to ingest.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute ingestion and print insert count."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    inserted = ingest(db_path=args.db_path, max_activities=args.max_activities)
    logging.info("Ingestion complete. New activities inserted: %s", inserted)


if __name__ == "__main__":
    main()
