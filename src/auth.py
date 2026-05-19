"""Authentication helpers for Strava API access."""

from __future__ import annotations

from urllib.parse import urlencode
from typing import Final

import requests

TOKEN_URL: Final[str] = "https://www.strava.com/oauth/token"
AUTHORIZE_URL: Final[str] = "https://www.strava.com/oauth/authorize"


def get_authorization_url(
    client_id: str,
    redirect_uri: str,
    scope: str = "read,activity:read_all",
) -> str:
    """Build the Strava OAuth authorization URL for SSO."""
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "approval_prompt": "auto",
        "scope": scope,
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


def exchange_code_for_token(
    client_id: str,
    client_secret: str,
    code: str,
    redirect_uri: str,
) -> str:
    """Exchange a Strava OAuth authorization code for an access token.

    Args:
        client_id: Strava client ID.
        client_secret: Strava client secret.
        code: OAuth authorization code.
        redirect_uri: Redirect URI configured in Strava app settings.

    Returns:
        A valid Strava access token.

    Raises:
        requests.HTTPError: If Strava token refresh fails.
        ValueError: If Strava token response does not include access token.
    """
    response = requests.post(
        TOKEN_URL,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    access_token = payload.get("access_token")
    if not access_token:
        raise ValueError("Strava token response did not include an access token.")
    return str(access_token)
