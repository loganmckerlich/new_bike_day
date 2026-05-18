"""Authentication helpers for Strava API access."""

from __future__ import annotations

import os
from typing import Final, Optional

import requests
from dotenv import load_dotenv

TOKEN_URL: Final[str] = "https://www.strava.com/oauth/token"


def get_access_token(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    refresh_token: Optional[str] = None,
) -> str:
    """Refresh a Strava access token and return it.

    Args:
        client_id: Optional Strava client ID. Falls back to STRAVA_CLIENT_ID.
        client_secret: Optional Strava client secret. Falls back to STRAVA_CLIENT_SECRET.
        refresh_token: Optional Strava refresh token. Falls back to STRAVA_REFRESH_TOKEN.

    Returns:
        A valid Strava access token.

    Raises:
        ValueError: If required credentials are missing.
        requests.HTTPError: If Strava token refresh fails.
    """
    load_dotenv()
    resolved_client_id: str = client_id or os.getenv("STRAVA_CLIENT_ID", "")
    resolved_client_secret: str = client_secret or os.getenv("STRAVA_CLIENT_SECRET", "")
    resolved_refresh_token: str = refresh_token or os.getenv("STRAVA_REFRESH_TOKEN", "")

    if not resolved_client_id or not resolved_client_secret or not resolved_refresh_token:
        raise ValueError(
            "Missing Strava credentials. Set STRAVA_CLIENT_ID, STRAVA_CLIENT_SECRET, and STRAVA_REFRESH_TOKEN."
        )

    response = requests.post(
        TOKEN_URL,
        data={
            "client_id": resolved_client_id,
            "client_secret": resolved_client_secret,
            "refresh_token": resolved_refresh_token,
            "grant_type": "refresh_token",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    access_token = payload.get("access_token")
    if not access_token:
        raise ValueError("Strava token response did not include an access token.")
    return str(access_token)
