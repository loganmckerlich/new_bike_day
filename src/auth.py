"""Authentication helpers for Strava API access."""

from __future__ import annotations

import time
from typing import Any
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


def exchange_code(
    client_id: str,
    client_secret: str,
    code: str,
    redirect_uri: str,
) -> dict[str, Any]:
    """Exchange a Strava OAuth authorization code for a full token payload.

    Args:
        client_id: Strava client ID.
        client_secret: Strava client secret.
        code: OAuth authorization code.
        redirect_uri: Redirect URI configured in Strava app settings.

    Returns:
        Dict with keys ``access_token`` (str), ``refresh_token`` (str),
        ``expires_at`` (int, Unix timestamp), and ``athlete_id`` (int).

    Raises:
        requests.HTTPError: If Strava token exchange fails.
        ValueError: If the response is missing expected fields.
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
    return _parse_token_response(response.json())


def exchange_code_for_token(
    client_id: str,
    client_secret: str,
    code: str,
    redirect_uri: str,
) -> str:
    """Exchange a Strava OAuth authorization code for an access token.

    This is a convenience wrapper around :func:`exchange_code` that returns
    only the access token string.

    Args:
        client_id: Strava client ID.
        client_secret: Strava client secret.
        code: OAuth authorization code.
        redirect_uri: Redirect URI configured in Strava app settings.

    Returns:
        A valid Strava access token string.

    Raises:
        requests.HTTPError: If Strava token refresh fails.
        ValueError: If Strava token response does not include access token.
    """
    return exchange_code(client_id, client_secret, code, redirect_uri)["access_token"]


def refresh_access_token(
    client_id: str,
    client_secret: str,
    refresh_token: str,
) -> dict[str, Any]:
    """Obtain a new access token using the stored refresh token.

    Strava access tokens expire after 6 hours.  The webhook server calls this
    before making any API call to ensure it always has a valid token.

    Args:
        client_id: Strava client ID.
        client_secret: Strava client secret.
        refresh_token: Long-lived refresh token from a previous token exchange.

    Returns:
        Dict with keys ``access_token`` (str), ``refresh_token`` (str),
        ``expires_at`` (int, Unix timestamp), and ``athlete_id`` (int).

    Raises:
        requests.HTTPError: If Strava token refresh fails.
        ValueError: If the response is missing expected fields.
    """
    response = requests.post(
        TOKEN_URL,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
        timeout=30,
    )
    response.raise_for_status()
    return _parse_token_response(response.json())


def get_valid_access_token(
    client_id: str,
    client_secret: str,
    stored_token: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Return a valid access token, refreshing it if it has expired.

    Args:
        client_id: Strava client ID.
        client_secret: Strava client secret.
        stored_token: Token dict as returned by :func:`exchange_code` or
            :func:`refresh_access_token` (or loaded from the database).

    Returns:
        A tuple of ``(access_token, updated_token_dict)``.  If the token was
        still valid, ``updated_token_dict`` is the same object that was passed
        in.  If a refresh occurred, it contains the new token data and the
        caller should persist it to the database.
    """
    # Give a 60-second buffer so a token is not used right before expiry.
    if stored_token.get("expires_at", 0) > time.time() + 60:
        return stored_token["access_token"], stored_token

    new_token = refresh_access_token(client_id, client_secret, stored_token["refresh_token"])
    return new_token["access_token"], new_token


def _parse_token_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Parse and validate a Strava token response payload.

    Args:
        payload: JSON response body from the Strava token endpoint.

    Returns:
        Normalised dict with ``access_token``, ``refresh_token``,
        ``expires_at``, and ``athlete_id``.

    Raises:
        ValueError: If the payload is missing required fields.
    """
    access_token = payload.get("access_token")
    refresh_token = payload.get("refresh_token")
    expires_at = payload.get("expires_at")

    if not access_token:
        raise ValueError("Strava token response did not include an access token.")
    if not refresh_token:
        raise ValueError("Strava token response did not include a refresh token.")
    if expires_at is None:
        raise ValueError("Strava token response did not include expires_at.")

    athlete: dict[str, Any] = payload.get("athlete") or {}
    athlete_id: int | None = athlete.get("id")

    return {
        "access_token": str(access_token),
        "refresh_token": str(refresh_token),
        "expires_at": int(expires_at),
        "athlete_id": int(athlete_id) if athlete_id is not None else None,
    }
