"""Streamlit app entrypoint for cloud-ready Strava analysis."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from stravalib import Client

# Ensure `src` imports work when launching Streamlit from different working directories.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.auth import exchange_code_for_token, get_authorization_url
from src.fetch import get_activities

DEFAULT_MAX_ACTIVITIES = 200


def _build_analysis_frame(raw_activities: list[dict[str, object]]) -> pd.DataFrame:
    """Transform Strava activity payload into an analysis-friendly dataframe."""
    if not raw_activities:
        return pd.DataFrame()

    frame = pd.DataFrame(raw_activities)
    frame["distance_km"] = frame["distance_m"] / 1000
    frame["moving_time_h"] = frame["moving_time_s"] / 3600
    frame["avg_speed_kph"] = frame["average_speed_mps"] * 3.6
    frame["date"] = pd.to_datetime(frame["start_date_local"], errors="coerce")
    return frame


def _process_data(
    access_token: str,
    max_activities: int,
) -> pd.DataFrame:
    """Fetch Strava activities and run in-memory data processing."""
    progress = st.progress(0, text="Starting…")

    progress.progress(35, text="Loading data: pulling activities from Strava API…")
    client = Client(access_token=access_token)
    activities = get_activities(client=client, limit=max_activities)

    progress.progress(65, text="Processing data…")
    frame = _build_analysis_frame(activities)

    progress.progress(90, text="Running algorithms…")
    if not frame.empty:
        watts = pd.to_numeric(frame["average_watts"], errors="coerce")
        # Zero/negative watts are excluded to avoid invalid division in this derived metric.
        valid_watts = watts.mask(watts <= 0)
        frame["kph_per_watt"] = frame["avg_speed_kph"] / valid_watts

    progress.progress(100, text="Complete.")
    return frame


def _query_param_value(value: object) -> str | None:
    """Normalize Streamlit query param values into a single string."""
    if isinstance(value, list):
        return str(value[0]) if value else None
    if value is None:
        return None
    return str(value)


def _exchange_access_token(client_id: str, client_secret: str, redirect_uri: str, code: str) -> str:
    """Exchange a Strava auth code for an access token."""
    return exchange_code_for_token(
        client_id=client_id,
        client_secret=client_secret,
        code=code,
        redirect_uri=redirect_uri,
    )


def _normalized_redirect_uri(raw_value: str) -> str:
    """Normalize redirect URI to reduce callback mismatches for localhost defaults."""
    value = raw_value.strip() if raw_value else ""
    if not value:
        return "http://localhost:8501/"

    parsed = urlsplit(value)
    if parsed.scheme and parsed.netloc and not parsed.path:
        return urlunsplit((parsed.scheme, parsed.netloc, "/", parsed.query, parsed.fragment))
    return value


def main() -> None:
    """Render the cloud-ready Streamlit workflow."""
    load_dotenv()
    st.set_page_config(page_title="New Bike Day", layout="wide")
    st.title("🚴 New Bike Day")
    st.caption("One-click Strava SSO to validate access and load all activities in-memory.")

    env_client_id = os.getenv("STRAVA_CLIENT_ID", "")
    env_client_secret = os.getenv("STRAVA_CLIENT_SECRET", "")
    default_redirect_uri = _normalized_redirect_uri(os.getenv("STRAVA_REDIRECT_URI", "http://localhost:8501"))
    env_access_token = os.getenv("STRAVA_ACCESS_TOKEN", "")

    st.subheader("1) Connect Strava")
    if not env_client_id or not env_client_secret:
        st.error("Set STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET in .env to enable SSO.")
        return

    auth_url = get_authorization_url(client_id=env_client_id, redirect_uri=default_redirect_uri)
    st.link_button("Sign in with Strava SSO", auth_url, use_container_width=True)

    code_from_params = st.query_params.get("code")
    error_from_params = st.query_params.get("error")
    max_activities = st.number_input("Max Activities", min_value=1, max_value=500, value=DEFAULT_MAX_ACTIVITIES)

    if error_from_params:
        st.error(f"Strava authorization failed: {error_from_params}")
        return

    code = _query_param_value(code_from_params)

    selected_max_activities = int(max_activities)

    if code:
        last_processed_code = st.session_state.get("last_processed_code")
        last_loaded_max_activities = st.session_state.get("last_loaded_max_activities")
        should_process = code != last_processed_code or last_loaded_max_activities != selected_max_activities
        if should_process:
            with st.spinner("Working…"):
                try:
                    access_token = st.session_state.get("access_token")
                    if code != last_processed_code or not access_token:
                        access_token = _exchange_access_token(env_client_id, env_client_secret, default_redirect_uri, code)
                    data = _process_data(
                        access_token=access_token,
                        max_activities=selected_max_activities,
                    )
                except (requests.RequestException, ValueError) as exc:
                    st.error(f"Unable to process data: {exc}")
                    return
            st.session_state["activities"] = data
            st.session_state["last_processed_code"] = code
            st.session_state["last_loaded_max_activities"] = selected_max_activities
            st.session_state["access_token"] = access_token
            st.success("Strava validated. Activities loaded.")
        else:
            st.info("Using already-loaded activities for this authorization and activity limit.")
    else:
        st.info("Click Sign in with Strava SSO, authorize access, and return here to auto-load data.")

    if st.button("Reload Activities", type="secondary"):
        access_token = st.session_state.get("access_token") or env_access_token
        if not access_token and code:
            try:
                access_token = _exchange_access_token(env_client_id, env_client_secret, default_redirect_uri, code)
            except (requests.RequestException, ValueError) as exc:
                st.error(f"Unable to process data: {exc}")
                return
        if not access_token:
            st.warning(
                "Please authorize with Strava using the Sign in button above, "
                "or set STRAVA_ACCESS_TOKEN in your .env file to enable reloading."
            )
            return
        with st.spinner("Working…"):
            try:
                data = _process_data(
                    access_token=access_token,
                    max_activities=selected_max_activities,
                )
            except (requests.RequestException, ValueError) as exc:
                st.error(f"Unable to process data: {exc}")
                return
        st.session_state["activities"] = data
        st.session_state["access_token"] = access_token
        st.session_state["last_loaded_max_activities"] = selected_max_activities
        if code:
            st.session_state["last_processed_code"] = code

    data = st.session_state.get("activities")
    if data is None or data.empty:
        st.info("No in-memory data yet. Complete Strava SSO to load activities.")
        return

    st.subheader("Activity Preview")
    st.dataframe(data.head(200), use_container_width=True)
    st.subheader("Bike Comparison")
    metrics = {
        "rides": ("id", "count"),
        "total_distance_km": ("distance_km", "sum"),
        "avg_speed_kph": ("avg_speed_kph", "mean"),
    }
    bike_stats = data.groupby("gear_id", dropna=False).agg(**metrics).reset_index()
    bike_stats = bike_stats.sort_values("total_distance_km", ascending=False)
    st.dataframe(bike_stats, use_container_width=True)


if __name__ == "__main__":
    main()
