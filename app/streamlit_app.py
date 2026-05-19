"""Streamlit home page: sign in with Strava and view bike summaries."""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import requests
import streamlit as st

# Ensure `src` imports work when launching Streamlit from different working directories.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.auth import exchange_code_for_token, get_authorization_url
from src.fetch import ingest_all, PremiumOnlyError

DEFAULT_MAX_ACTIVITIES = 2000


def _process_data(
    access_token: str,
    max_activities: int,  # noqa: ARG001 — kept for signature compatibility
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch starred segments and all efforts from Strava, returning both as DataFrames."""
    progress = st.progress(0, text="Starting…")

    progress.progress(20, text="Fetching starred segments from Strava API…")
    result = ingest_all(access_token)

    progress.progress(100, text="Complete.")
    return result["efforts"], result["segments"]


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


def _render_bike_summaries(efforts: pd.DataFrame, segments: pd.DataFrame) -> None:
    """Render the bike summaries section based on segment efforts grouped by gear."""
    st.subheader("🚲 Bike Summaries")

    agg_metrics = {
        "total_efforts": ("effort_id", "count"),
        "total_moving_hours": ("moving_time", "sum"),
        "avg_watts": ("average_watts", "mean"),
        "avg_heartrate": ("average_heartrate", "mean"),
    }
    bike_stats = efforts.groupby("gear_id", dropna=False).agg(**agg_metrics).reset_index()
    bike_stats["total_moving_hours"] = (bike_stats["total_moving_hours"] / 3600).round(1)
    bike_stats["avg_watts"] = bike_stats["avg_watts"].round(0)
    bike_stats["avg_heartrate"] = bike_stats["avg_heartrate"].round(0)
    bike_stats = bike_stats.sort_values("total_efforts", ascending=False)

    st.dataframe(bike_stats, use_container_width=True)

    if not segments.empty:
        st.subheader("⭐ Starred Segments")
        preferred_seg = [
            "name", "segment_type", "distance", "average_grade",
            "climb_category", "total_elevation_gain", "start_lat", "start_lng",
        ]
        display_cols = [c for c in preferred_seg if c in segments.columns]
        st.dataframe(segments[display_cols], use_container_width=True)


def _save_session(data: pd.DataFrame, segments: pd.DataFrame, code: str | None, max_activities: int, access_token: str) -> None:
    st.session_state["efforts"] = data
    st.session_state["segments"] = segments
    st.session_state["last_loaded_max_activities"] = max_activities
    st.session_state["access_token"] = access_token
    if code:
        st.session_state["last_processed_code"] = code


def main() -> None:
    """Render the home page: Strava sign-in and bike summaries."""
    st.set_page_config(page_title="New Bike Day", layout="wide")
    st.title("🚴 New Bike Day")
    st.caption("Sign in with Strava to load your activities and see your bike summaries.")

    env_client_id = st.secrets.get("STRAVA_CLIENT_ID", "")
    env_client_secret = st.secrets.get("STRAVA_CLIENT_SECRET", "")
    default_redirect_uri = _normalized_redirect_uri(st.secrets.get("STRAVA_REDIRECT_URI", "http://localhost:8501"))
    env_access_token = st.secrets.get("STRAVA_ACCESS_TOKEN", "")

    st.subheader("1) Connect Strava")
    if not env_client_id or not env_client_secret:
        st.error("Set STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET in .streamlit/secrets.toml to enable SSO.")
        return

    auth_url = get_authorization_url(client_id=env_client_id, redirect_uri=default_redirect_uri)
    st.link_button("Sign in with Strava SSO", auth_url, use_container_width=True)

    code_from_params = st.query_params.get("code")
    error_from_params = st.query_params.get("error")
    max_activities = st.number_input(
        "Max Activities",
        min_value=1,
        max_value=10000,
        value=DEFAULT_MAX_ACTIVITIES,
    )

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
                    data, gear_frame = _process_data(
                        access_token=access_token,
                        max_activities=selected_max_activities,
                    )
                except PremiumOnlyError as exc:
                    st.error(f"Premium Membership Required: {exc}")
                    return
                except (requests.RequestException, ValueError) as exc:
                    st.error(f"Unable to process data: {exc}")
                    return
            _save_session(data, gear_frame, code, selected_max_activities, access_token)
            st.success("Strava validated. Segment data loaded.")
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
                "or set STRAVA_ACCESS_TOKEN in your .streamlit/secrets.toml file to enable reloading."
            )
            return
        with st.spinner("Working…"):
            try:
                data, gear_frame = _process_data(
                    access_token=access_token,
                    max_activities=selected_max_activities,
                )
            except PremiumOnlyError as exc:
                st.error(f"Premium Membership Required: {exc}")
                return
            except (requests.RequestException, ValueError) as exc:
                st.error(f"Unable to process data: {exc}")
                return
        _save_session(data, gear_frame, code, selected_max_activities, access_token)

    data = st.session_state.get("efforts")
    if data is None or data.empty:
        st.info("No data yet. Complete Strava SSO above to load segment data.")
        return

    segments = st.session_state.get("segments", pd.DataFrame())
    _render_bike_summaries(data, segments)


if __name__ == "__main__":
    main()
