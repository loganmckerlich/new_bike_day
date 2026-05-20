"""Helpers for lightweight home-page personality customizations."""

from __future__ import annotations

import json
from pathlib import Path


_ATHLETE_JSON = Path(__file__).resolve().parents[1] / "data" / "dev" / "athlete.json"


def load_dev_athlete_profile() -> dict[str, str]:
    """Load a small athlete profile from dev fixtures for home-page personalization."""
    try:
        payload = json.loads(_ATHLETE_JSON.read_text())
    except (OSError, json.JSONDecodeError):
        return {}

    first_name = str(payload.get("firstname") or "").strip()
    last_name = str(payload.get("lastname") or "").strip()
    full_name = " ".join(part for part in [first_name, last_name] if part).strip()

    bikes = payload.get("bikes") if isinstance(payload.get("bikes"), list) else []
    primary_bike = ""
    for bike in bikes:
        if isinstance(bike, dict) and bike.get("primary"):
            primary_bike = str(bike.get("name") or "").strip()
            break
    if not primary_bike and bikes and isinstance(bikes[0], dict):
        primary_bike = str(bikes[0].get("name") or "").strip()

    return {
        "first_name": first_name,
        "full_name": full_name,
        "city": str(payload.get("city") or "").strip(),
        "country": str(payload.get("country") or "").strip(),
        "primary_bike": primary_bike,
    }


def build_cheeky_conclusion(
    *,
    athlete_name: str | None,
    legs_status: str,
    vibe: str,
    takeaway: str,
) -> str:
    """Build a playful ride conclusion sentence for the home page form."""
    intro = athlete_name.strip() if athlete_name and athlete_name.strip() else "The rider"
    headline = takeaway.strip() or "I respected the watts and feared the climbs."
    return f"{intro} says: {headline} Legs status: {legs_status}. Ride vibe: {vibe}."
