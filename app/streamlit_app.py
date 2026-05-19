"""Streamlit app entrypoint for cloud-ready Strava analysis."""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from stravalib import Client

from src.auth import exchange_code_for_token, get_authorization_url
from src.fetch import get_activities


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
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    code: str,
    max_activities: int,
) -> pd.DataFrame:
    """Authenticate, fetch Strava activities, and run in-memory data processing."""
    progress = st.progress(0, text="Starting…")

    progress.progress(15, text="Loading data: authenticating with Strava…")
    access_token = exchange_code_for_token(
        client_id=client_id,
        client_secret=client_secret,
        code=code,
        redirect_uri=redirect_uri,
    )

    progress.progress(45, text="Loading data: pulling activities from Strava API…")
    client = Client(access_token=access_token)
    activities = get_activities(client=client, limit=max_activities)

    progress.progress(70, text="Processing data…")
    frame = _build_analysis_frame(activities)

    progress.progress(90, text="Running algorithms…")
    if not frame.empty:
        _ = (
            frame.groupby("gear_id", dropna=False)
            .agg(total_distance_km=("distance_km", "sum"), avg_speed_kph=("avg_speed_kph", "mean"))
            .sort_values("total_distance_km", ascending=False)
        )

    progress.progress(100, text="Complete.")
    return frame


def main() -> None:
    """Render the cloud-ready Streamlit workflow."""
    st.set_page_config(page_title="New Bike Day", layout="wide")
    st.title("🚴 New Bike Day")
    st.caption("Sign in with Strava SSO, process data, and run analysis in-memory.")

    env_client_id = os.getenv("STRAVA_CLIENT_ID", "")
    env_client_secret = os.getenv("STRAVA_CLIENT_SECRET", "")
    default_redirect_uri = os.getenv("STRAVA_REDIRECT_URI", "http://localhost:8501")

    st.subheader("1) Connect Strava")
    client_id = st.text_input("Strava Client ID", value=env_client_id)
    client_secret = st.text_input("Strava Client Secret", value=env_client_secret, type="password")
    redirect_uri = st.text_input("Redirect URI", value=default_redirect_uri)

    if client_id and redirect_uri:
        auth_url = get_authorization_url(client_id=client_id, redirect_uri=redirect_uri)
        st.link_button("Sign in with Strava SSO", auth_url)

    query_code = st.query_params.get("code")
    code = st.text_input("Authorization Code", value=query_code or "")
    max_activities = st.number_input("Max Activities", min_value=1, max_value=500, value=200)

    if st.button("Process Data", type="primary"):
        if not client_id or not client_secret or not code:
            st.error("Client ID, client secret, and authorization code are required.")
            return
        with st.spinner("Working…"):
            try:
                data = _process_data(
                    client_id=client_id,
                    client_secret=client_secret,
                    redirect_uri=redirect_uri,
                    code=code,
                    max_activities=int(max_activities),
                )
            except Exception as exc:  # pragma: no cover
                st.error(f"Unable to process data: {exc}")
                return
        st.session_state["activities"] = data

    data = st.session_state.get("activities")
    if data is None or data.empty:
        st.info("No in-memory data yet. Complete Strava SSO and click Process Data.")
        return

    st.subheader("Activity Preview")
    st.dataframe(data.head(200), use_container_width=True)
    st.subheader("Bike Comparison")
    bike_stats = (
        data.groupby("gear_id", dropna=False)
        .agg(rides=("id", "count"), total_distance_km=("distance_km", "sum"), avg_speed_kph=("avg_speed_kph", "mean"))
        .reset_index()
        .sort_values("total_distance_km", ascending=False)
    )
    st.dataframe(bike_stats, use_container_width=True)


if __name__ == "__main__":
    main()
