"""Ingestion entrypoint for pulling Strava and weather data into flat files."""

from __future__ import annotations

import logging
import os
import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from stravalib import Client

try:
    from .auth import get_access_token
    from .fetch import get_activities, get_streams
    from .weather import get_weather
except ImportError:  # pragma: no cover
    from auth import get_access_token
    from fetch import get_activities, get_streams
    from weather import get_weather


def ingest(data_path: str = "data/activities.csv", max_activities: Optional[int] = None) -> int:
    """Run idempotent ingestion from Strava and Open-Meteo into a CSV file.

    Args:
        data_path: Path to the activities CSV file.
        max_activities: Optional cap on activities fetched from Strava.

    Returns:
        Number of newly inserted activities.

    Notes:
        Existing activities are skipped based on the activity `id`, so repeated runs
        do not duplicate already ingested rows.
    """
    access_token = get_access_token()
    client = Client(access_token=access_token)
    activities = get_activities(client=client, limit=max_activities)

    target_path = Path(data_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        existing = pd.read_csv(target_path)
    else:
        existing = pd.DataFrame()

    existing_ids: set[int] = set()
    if "id" in existing.columns:
        existing_ids = set(existing["id"].dropna().astype(int).tolist())

    new_rows = []
    for activity in activities:
        activity_id = int(activity["id"])
        if activity_id in existing_ids:
            continue

        row = dict(activity)
        row["streams_json"] = None
        row["weather_date"] = None
        row["temperature_2m_max"] = None
        row["temperature_2m_min"] = None
        row["precipitation_sum"] = None
        row["windspeed_10m_max"] = None

        try:
            streams = get_streams(client=client, activity_id=activity_id)
            row["streams_json"] = json.dumps(streams)
        except requests.RequestException as exc:
            logging.warning("Unable to fetch streams for activity %s: %s", activity_id, exc)

        lat = activity.get("start_lat")
        lon = activity.get("start_lon")
        start_date = activity.get("start_date_local")
        if lat is not None and lon is not None and start_date:
            try:
                weather = get_weather(lat=float(lat), lon=float(lon), target_date=str(start_date))
                row["weather_date"] = weather.get("date")
                row["temperature_2m_max"] = weather.get("temperature_2m_max")
                row["temperature_2m_min"] = weather.get("temperature_2m_min")
                row["precipitation_sum"] = weather.get("precipitation_sum")
                row["windspeed_10m_max"] = weather.get("windspeed_10m_max")
            except requests.RequestException as exc:
                logging.warning("Unable to fetch weather for activity %s: %s", activity_id, exc)
            except ValueError as exc:
                logging.warning("Unable to parse weather date for activity %s: %s", activity_id, exc)

        new_rows.append(row)

    if not new_rows:
        return 0

    updated = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    updated.to_csv(target_path, index=False)
    return len(new_rows)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for ingestion execution."""
    parser = argparse.ArgumentParser(description="Ingest Strava activities and weather into a CSV file.")
    parser.add_argument(
        "--data-path",
        default=os.getenv("STRAVA_DATA_PATH", "data/activities.csv"),
        help="Path to activities CSV file.",
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
    inserted = ingest(data_path=args.data_path, max_activities=args.max_activities)
    logging.info("Ingestion complete. New activities inserted: %s", inserted)


if __name__ == "__main__":
    main()
