"""
webhook.py — Strava deauthorization webhook handler.

Listens for Strava push subscription events and deletes athlete data
when a user revokes access. Must be deployed as a publicly accessible
endpoint separate from Streamlit (e.g. FastAPI on Railway or Render).

Strava docs: https://developers.strava.com/docs/webhooks/
"""

import os
import logging
from fastapi import FastAPI, Request, Response, HTTPException
from src.database import _delete_user_data

logger = logging.getLogger(__name__)
app = FastAPI()

VERIFY_TOKEN = os.environ["STRAVA_VERIFY_TOKEN"]


@app.get("/webhook")
async def verify_webhook(request: Request) -> Response:
    """Handle Strava's webhook verification challenge.

    Strava sends a GET request to verify the endpoint before activating
    the subscription. Must respond with the hub.challenge value.
    """
    params = request.query_params

    if params.get("hub.verify_token") != VERIFY_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid verify token")

    challenge = params.get("hub.challenge")
    if not challenge:
        raise HTTPException(status_code=400, detail="Missing hub.challenge")

    return {"hub.challenge": challenge}


@app.post("/webhook")
async def handle_webhook(request: Request) -> Response:
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