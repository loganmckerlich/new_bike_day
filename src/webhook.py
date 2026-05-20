"""Strava webhook receiver and subscription management.

Strava pushes events to your callback URL when activities are created, updated,
or deleted, or when an athlete revokes app access.  This module provides:

- A Flask HTTP server that handles the Strava webhook verification handshake and
  incoming event POSTs.  On any activity event, the server re-ingests all data
  for that athlete and writes the result to the SQLite static cache.  Strava
  requires a 200 OK response within 2 seconds, so the actual re-ingest runs in
  a background thread.

- CLI helpers for managing the webhook subscription with Strava
  (subscribe, view, unsubscribe).

Running the server
------------------
::

    STRAVA_CLIENT_ID=... STRAVA_CLIENT_SECRET=... \\
    STRAVA_WEBHOOK_VERIFY_TOKEN=... \\
    python -m src.webhook serve [--host 0.0.0.0] [--port 8502]

Managing subscriptions
----------------------
::

    python -m src.webhook subscribe --callback-url https://yourhost.example.com/webhook
    python -m src.webhook view
    python -m src.webhook unsubscribe --subscription-id 12345

Environment variables
---------------------
``STRAVA_CLIENT_ID``
    Strava application client ID.

``STRAVA_CLIENT_SECRET``
    Strava application client secret.

``STRAVA_WEBHOOK_VERIFY_TOKEN``
    A secret string you choose.  Strava sends this back during subscription
    verification so you can confirm the request is genuine.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any

import requests

# Ensure the project root is on sys.path so ``src.*`` imports work when the
# module is run directly (``python -m src.webhook``).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.auth import get_valid_access_token
from src.database import (
    clear_bikes,
    clear_efforts,
    clear_segments,
    init_db,
    load_athlete_token,
    save_athlete_token,
    save_bikes,
    save_efforts,
    save_segments,
)
from src.fetch import ingest_all

_STRAVA_API_BASE = "https://www.strava.com/api/v3"
_PUSH_SUBSCRIPTIONS_URL = f"{_STRAVA_API_BASE}/push_subscriptions"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Re-ingest helper (runs in background thread on webhook events)
# ---------------------------------------------------------------------------

def _reingest_for_athlete(athlete_id: int, client_id: str, client_secret: str) -> None:
    """Re-fetch all Strava data for *athlete_id* and update the SQLite cache.

    This function is intended to be called from a background thread so that
    the webhook endpoint can return 200 OK to Strava immediately.

    Args:
        athlete_id: The Strava athlete ID from the webhook event's ``owner_id``.
        client_id: Strava application client ID (needed for token refresh).
        client_secret: Strava application client secret (needed for token refresh).
    """
    stored = load_athlete_token(athlete_id)
    if stored is None:
        logger.warning(
            "Webhook event for athlete %s but no stored token — skipping re-ingest.",
            athlete_id,
        )
        return

    try:
        access_token, updated_token = get_valid_access_token(client_id, client_secret, stored)
    except Exception:
        logger.exception("Failed to obtain a valid access token for athlete %s.", athlete_id)
        return

    # Persist any refreshed token data.
    if updated_token is not stored:
        save_athlete_token(
            athlete_id=athlete_id,
            access_token=updated_token["access_token"],
            refresh_token=updated_token["refresh_token"],
            expires_at=updated_token["expires_at"],
        )

    logger.info("Re-ingesting data for athlete %s…", athlete_id)
    try:
        # Clear stale data so deleted activities / segments are removed.
        clear_efforts()
        clear_segments()
        clear_bikes()

        result = ingest_all(access_token)

        init_db()
        save_segments(result["segments"])
        save_efforts(result["efforts"])
        save_bikes(result.get("bikes", {}))
        logger.info("Re-ingest complete for athlete %s.", athlete_id)
    except Exception:
        logger.exception("Re-ingest failed for athlete %s.", athlete_id)


# ---------------------------------------------------------------------------
# Flask webhook server
# ---------------------------------------------------------------------------

def create_app(
    verify_token: str,
    client_id: str,
    client_secret: str,
) -> "Flask":  # noqa: F821 – avoid hard import at module level
    """Create and return the Flask application.

    Args:
        verify_token: The verify token you registered with Strava.  Incoming
            verification requests must supply the same value.
        client_id: Strava application client ID.
        client_secret: Strava application client secret.

    Returns:
        A configured Flask application instance.
    """
    try:
        from flask import Flask, jsonify, request as flask_request
    except ImportError as exc:
        raise ImportError(
            "Flask is required to run the webhook server. "
            "Install it with: pip install flask"
        ) from exc

    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health() -> Any:
        return "OK", 200

    @app.route("/webhook", methods=["GET"])
    def verify_webhook() -> Any:
        """Handle Strava's subscription verification challenge (GET)."""
        mode = flask_request.args.get("hub.mode")
        token = flask_request.args.get("hub.verify_token")
        challenge = flask_request.args.get("hub.challenge")

        if mode == "subscribe" and token == verify_token:
            logger.info("Webhook verification challenge accepted.")
            return jsonify({"hub.challenge": challenge})

        logger.warning("Webhook verification failed: unexpected mode=%r token=%r", mode, token)
        return "Forbidden", 403

    @app.route("/webhook", methods=["POST"])
    def handle_webhook() -> Any:
        """Receive a Strava event and trigger an async re-ingest (POST)."""
        event: dict[str, Any] = flask_request.get_json(silent=True) or {}
        object_type: str = event.get("object_type", "")
        aspect_type: str = event.get("aspect_type", "")
        owner_id: int | None = event.get("owner_id")

        logger.info(
            "Webhook event received: object_type=%r aspect_type=%r owner_id=%r",
            object_type,
            aspect_type,
            owner_id,
        )

        if object_type == "athlete" and aspect_type == "update":
            updates: dict[str, Any] = event.get("updates") or {}
            # authorized may be the string "false" or the boolean False depending on Strava's version
            if updates.get("authorized") in ("false", False) and owner_id is not None:
                # Athlete revoked access — clear their cached data.
                logger.info("Athlete %s deauthorized; clearing cache.", owner_id)
                clear_efforts()
                clear_segments()
                clear_bikes()
            return "OK", 200

        if object_type == "activity" and owner_id is not None:
            # Kick off a background re-ingest so we can return 200 immediately.
            thread = threading.Thread(
                target=_reingest_for_athlete,
                args=(owner_id, client_id, client_secret),
                daemon=True,
            )
            thread.start()

        return "OK", 200

    return app


# ---------------------------------------------------------------------------
# Subscription management helpers
# ---------------------------------------------------------------------------

def subscribe_webhook(
    client_id: str,
    client_secret: str,
    callback_url: str,
    verify_token: str,
) -> dict[str, Any]:
    """Register a new Strava webhook subscription.

    Args:
        client_id: Strava application client ID.
        client_secret: Strava application client secret.
        callback_url: Publicly accessible HTTPS URL for the ``/webhook`` endpoint.
        verify_token: The secret string you chose for verification.

    Returns:
        The JSON response from Strava (contains ``id``, ``callback_url``, etc.).

    Raises:
        requests.HTTPError: If Strava returns an error response.
    """
    resp = requests.post(
        _PUSH_SUBSCRIPTIONS_URL,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "callback_url": callback_url,
            "verify_token": verify_token,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def view_subscriptions(client_id: str, client_secret: str) -> list[dict[str, Any]]:
    """List all active Strava webhook subscriptions for this application.

    Args:
        client_id: Strava application client ID.
        client_secret: Strava application client secret.

    Returns:
        List of subscription dicts as returned by Strava.
    """
    resp = requests.get(
        _PUSH_SUBSCRIPTIONS_URL,
        params={"client_id": client_id, "client_secret": client_secret},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def unsubscribe_webhook(
    client_id: str,
    client_secret: str,
    subscription_id: int,
) -> None:
    """Delete a Strava webhook subscription.

    Args:
        client_id: Strava application client ID.
        client_secret: Strava application client secret.
        subscription_id: The numeric ID of the subscription to delete.

    Raises:
        requests.HTTPError: If Strava returns an error response.
    """
    resp = requests.delete(
        f"{_PUSH_SUBSCRIPTIONS_URL}/{subscription_id}",
        params={"client_id": client_id, "client_secret": client_secret},
        timeout=30,
    )
    resp.raise_for_status()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Manage Strava webhook subscriptions and run the webhook server.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── serve ──────────────────────────────────────────────────────────────
    serve_parser = subparsers.add_parser("serve", help="Start the webhook HTTP server.")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=8502, help="Port to listen on (default: 8502)")

    # ── subscribe ──────────────────────────────────────────────────────────
    sub_parser = subparsers.add_parser("subscribe", help="Register a new webhook subscription.")
    sub_parser.add_argument("--callback-url", required=True, help="Public HTTPS callback URL.")
    sub_parser.add_argument(
        "--verify-token",
        default=None,
        help="Verify token (default: $STRAVA_WEBHOOK_VERIFY_TOKEN).",
    )

    # ── view ───────────────────────────────────────────────────────────────
    subparsers.add_parser("view", help="List active webhook subscriptions.")

    # ── unsubscribe ────────────────────────────────────────────────────────
    unsub_parser = subparsers.add_parser("unsubscribe", help="Delete a webhook subscription.")
    unsub_parser.add_argument("--subscription-id", type=int, required=True)

    args = parser.parse_args()

    client_id = os.environ.get("STRAVA_CLIENT_ID", "")
    client_secret = os.environ.get("STRAVA_CLIENT_SECRET", "")
    verify_token = os.environ.get("STRAVA_WEBHOOK_VERIFY_TOKEN", "")

    if not client_id or not client_secret:
        print(
            "Error: STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET must be set as environment variables.",
            file=sys.stderr,
        )
        sys.exit(1)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.command == "serve":
        effective_verify_token = verify_token
        if not effective_verify_token:
            print(
                "Error: STRAVA_WEBHOOK_VERIFY_TOKEN must be set to run the webhook server.",
                file=sys.stderr,
            )
            sys.exit(1)
        init_db()
        app = create_app(
            verify_token=effective_verify_token,
            client_id=client_id,
            client_secret=client_secret,
        )
        print(f"Starting webhook server on {args.host}:{args.port} …")
        app.run(host=args.host, port=args.port)

    elif args.command == "subscribe":
        effective_verify_token = args.verify_token or verify_token
        if not effective_verify_token:
            print(
                "Error: Provide --verify-token or set STRAVA_WEBHOOK_VERIFY_TOKEN.",
                file=sys.stderr,
            )
            sys.exit(1)
        result = subscribe_webhook(client_id, client_secret, args.callback_url, effective_verify_token)
        print("Subscription created:", result)

    elif args.command == "view":
        subs = view_subscriptions(client_id, client_secret)
        if not subs:
            print("No active subscriptions.")
        else:
            for sub in subs:
                print(sub)

    elif args.command == "unsubscribe":
        unsubscribe_webhook(client_id, client_secret, args.subscription_id)
        print(f"Subscription {args.subscription_id} deleted.")


if __name__ == "__main__":
    _cli()
