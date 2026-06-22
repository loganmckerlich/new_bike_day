"""Authentication helpers for Strava API access."""

from __future__ import annotations

from urllib.parse import urlencode
from typing import Final
import streamlit as st
import requests

TOKEN_URL: Final[str] = "https://www.strava.com/oauth/token"
AUTHORIZE_URL: Final[str] = "https://www.strava.com/oauth/authorize"


def custom_auth_button() -> None:
    auth_url = get_authorization_url(
        client_id=st.secrets["STRAVA_CLIENT_ID"],
        redirect_uri=st.secrets["STRAVA_REDIRECT_URI"],
        scope="read,activity:read_all,profile:read_all",
    )

    st.markdown("### Connect your Strava account")
    st.markdown(
            f"""
            <a href="{auth_url}"
            target="_top"
            style="
                display: inline-block;
                padding: 0.5rem 1.2rem;
                background-color: #FC4C02;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 6px;
                text-decoration: none;
            ">
            🚴 Connect Strava
            </a>
            """,
            unsafe_allow_html=True,
        )


def get_authorization_url(
    client_id: str,
    redirect_uri: str,
    scope: str = "read,activity:read_all,profile:read_all",
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


def handle_redirect() -> None:
    query_params = st.query_params

    if "code" not in query_params:
        return

    code = query_params["code"]
    st.info("Authorizing with Strava...")

    token_response = requests.post(
        TOKEN_URL,
        data={
            "client_id": st.secrets["STRAVA_CLIENT_ID"],
            "client_secret": st.secrets["STRAVA_CLIENT_SECRET"],
            "code": code,
            "grant_type": "authorization_code",
        },
    )

    token_data = token_response.json()

    if "access_token" in token_data:
        st.session_state["strava_token"] = token_data["access_token"]
        st.session_state["strava_athlete"] = token_data.get("athlete", {})
        st.success("✅ Connected to Strava!")
        st.query_params.clear()
        st.rerun()
    else:
        st.error("❌ Failed to authenticate")
        st.json(token_data)