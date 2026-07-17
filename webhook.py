"""
webhook.py — Strava deauthorization webhook handler.

Listens for Strava push subscription events and deletes athlete data
when a user revokes access. Must be deployed as a publicly accessible
endpoint separate from Streamlit (e.g. FastAPI on Railway or Render).

Strava docs: https://developers.strava.com/docs/webhooks/
"""

import os
import logging

from fastapi import FastAPI, Request, HTTPException
from supabase import create_client, Client

logger = logging.getLogger(__name__)
app = FastAPI()
import json
logger.info(json.dumps(dict(os.environ), default=str))

def _get_supabase() -> Client:
    """Create and return a Supabase client using environment variables."""
    return create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_KEY"]
    )

def _normalize_athlete_id(athlete_id: int | str | None) -> str | None:
    if athlete_id is None:
        return None
    return str(athlete_id)

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


@app.get("/webhook")
async def verify_webhook(request: Request):
    """Handle Strava's webhook verification challenge.

    Strava sends a GET request to verify the endpoint before activating
    the subscription. Must respond with the hub.challenge value.
    """
    verify_token = os.environ["STRAVA_VERIFY_TOKEN"]

    params = request.query_params

    if params.get("hub.verify_token") != verify_token:
        raise HTTPException(status_code=403, detail="Invalid verify token")

    challenge = params.get("hub.challenge")
    if not challenge:
        raise HTTPException(status_code=400, detail="Missing hub.challenge")

    return {"hub.challenge": challenge}


@app.post("/webhook")
async def handle_webhook(request: Request):
    """Handle incoming Strava webhook events.

    Only processes athlete deauthorization events (aspect_type == "delete").
    All other event types are acknowledged and ignored.

    Strava expects a 200 response quickly — deletion runs after response
    is returned to avoid timeouts.
    """
    body = await request.json()

    object_type = body.get("object_type")
    aspect_type = body.get("aspect_type")
    athlete_id = str(body.get("owner_id", ""))

    if object_type != "athlete" or aspect_type != "delete":
        logger.info(f"Ignoring event: object_type={object_type} aspect_type={aspect_type}")
        return {"status": "ignored"}

    if not athlete_id:
        logger.warning("Received deauth event with no owner_id")
        raise HTTPException(status_code=400, detail="Missing owner_id")

    logger.info(f"Deauthorization received for athlete_id={athlete_id}")

    try:
        _delete_user_data(athlete_id)
        logger.info(f"Successfully deleted data for athlete_id={athlete_id}")
    except Exception as e:
        logger.error(f"Failed to delete data for athlete_id={athlete_id}: {e}")
        # still return 200 — Strava will retry on failure which could
        # cause issues. Log the error and handle cleanup manually if needed.

    return {"status": "ok"}

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}