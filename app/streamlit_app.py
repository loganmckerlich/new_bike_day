"""Streamlit home page: sign in with Strava and view bike summaries."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import requests
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from stravalib import Client

# Ensure `src` imports work when launching Streamlit from different working directories.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.auth import exchange_code_for_token, get_authorization_url
from src.fetch import get_activities, get_gear, get_segment_efforts, get_starred_segments

DEFAULT_MAX_ACTIVITIES = 2000

# Unit-conversion constants
_METERS_TO_MILES: float = 0.000621371
_MPS_TO_MPH: float = 2.23694
_METERS_TO_FEET: float = 3.28084


def _build_analysis_frame(raw_activities: list[dict[str, object]]) -> pd.DataFrame:
    """Transform Strava activity payload into an analysis-friendly dataframe."""
    if not raw_activities:
        return pd.DataFrame()

    frame = pd.DataFrame(raw_activities)
    frame["distance_miles"] = frame["distance_m"] * _METERS_TO_MILES
    frame["moving_time_h"] = frame["moving_time_s"] / 3600
    frame["avg_speed_mph"] = frame["average_speed_mps"] * _MPS_TO_MPH
    frame["elevation_gain_ft"] = frame["total_elevation_gain_m"] * _METERS_TO_FEET
    frame["date"] = pd.to_datetime(frame["start_date_local"], errors="coerce")
    return frame


def _process_data(
    access_token: str,
    max_activities: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Fetch Strava activities, gear, and starred segments, returning all three."""
    progress = st.progress(0, text="Starting…")

    progress.progress(20, text="Pulling activities from Strava API…")
    client = Client(access_token=access_token)
    activities = get_activities(client=client, limit=max_activities)

    progress.progress(50, text="Processing activity data…")
    frame = _build_analysis_frame(activities)

    if not frame.empty:
        watts = pd.to_numeric(frame["average_watts"], errors="coerce")
        # Zero/negative watts are excluded to avoid invalid division in this derived metric.
        valid_watts = watts.mask(watts <= 0)
        frame["mph_per_watt"] = frame["avg_speed_mph"] / valid_watts

    progress.progress(70, text="Fetching equipment details…")
    gear_ids: list[str] = (
        [g for g in frame["gear_id"].dropna().unique() if g] if not frame.empty else []
    )
    gear_rows = [get_gear(client=client, gear_id=gid) for gid in gear_ids]
    gear_frame = pd.DataFrame(gear_rows) if gear_rows else pd.DataFrame(columns=["gear_id"])

    progress.progress(85, text="Fetching starred segments…")
    starred = get_starred_segments(client=client)

    progress.progress(100, text="Complete.")
    return frame, gear_frame, starred


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


def _render_bike_summaries(activities: pd.DataFrame, gear_frame: pd.DataFrame) -> None:
    """Render the bike summaries section."""
    st.subheader("🚲 Bike Summaries")

    agg_metrics = {
        "rides": ("id", "count"),
        "total_miles": ("distance_miles", "sum"),
        "avg_speed_mph": ("avg_speed_mph", "mean"),
        "total_elevation_ft": ("elevation_gain_ft", "sum"),
        "total_moving_hours": ("moving_time_h", "sum"),
        "avg_watts": ("average_watts", "mean"),
    }
    bike_stats = activities.groupby("gear_id", dropna=False).agg(**agg_metrics).reset_index()
    bike_stats["total_miles"] = bike_stats["total_miles"].round(1)
    bike_stats["avg_speed_mph"] = bike_stats["avg_speed_mph"].round(1)
    bike_stats["total_elevation_ft"] = bike_stats["total_elevation_ft"].round(0)
    bike_stats["total_moving_hours"] = bike_stats["total_moving_hours"].round(1)
    bike_stats["avg_watts"] = bike_stats["avg_watts"].round(0)

    if not gear_frame.empty and "gear_id" in gear_frame.columns:
        bike_stats = bike_stats.merge(gear_frame, on="gear_id", how="left")

    bike_stats = bike_stats.sort_values("total_miles", ascending=False)

    # Preferred column order – only show columns that actually exist
    preferred = [
        "gear_name", "brand_name", "model_name", "frame_type",
        "rides", "total_miles", "avg_speed_mph",
        "total_elevation_ft", "total_moving_hours", "avg_watts",
        "weight_lbs", "strava_total_miles", "primary", "description",
    ]
    display_cols = [c for c in preferred if c in bike_stats.columns]
    st.dataframe(bike_stats[display_cols], use_container_width=True)


def _fmt_pace(minutes: float) -> str:
    """Format a decimal-minutes pace value as 'mm:ss'."""
    if pd.isna(minutes):
        return "—"
    m = int(minutes)
    s = int(round((minutes - m) * 60))
    return f"{m}:{s:02d}"


def _render_segment_analysis(
    activities: pd.DataFrame,
    gear_frame: pd.DataFrame,
    starred_segments: list[dict],
    access_token: str,
) -> None:
    """Render the starred segment comparison section."""
    st.subheader("🏁 Starred Segment Analysis")

    if not starred_segments:
        st.info("No starred segments found for your account.")
        return

    # Build display labels and lookup map
    seg_labels = [
        f"{s['name']}  ({(s['distance_m'] or 0) * _METERS_TO_MILES:.2f} mi)"
        for s in starred_segments
    ]
    seg_by_label = dict(zip(seg_labels, starred_segments))

    selected_label = st.selectbox("Select a segment", seg_labels, key="segment_select")
    selected_seg = seg_by_label[selected_label]
    segment_id = selected_seg["segment_id"]
    distance_m = selected_seg.get("distance_m") or 0.0

    # Fetch efforts lazily, cached per segment
    cache_key = f"segment_efforts_{segment_id}"
    if cache_key not in st.session_state:
        with st.spinner("Fetching segment efforts…"):
            try:
                client = Client(access_token=access_token)
                efforts = get_segment_efforts(client=client, segment_id=segment_id)
            except Exception as exc:
                st.error(f"Unable to fetch efforts for this segment: {exc}")
                return
        st.session_state[cache_key] = efforts

    efforts = st.session_state[cache_key]
    if not efforts:
        st.info("No efforts recorded for this segment yet.")
        return

    efforts_df = pd.DataFrame(efforts)
    efforts_df["date"] = pd.to_datetime(efforts_df["start_date_local"], errors="coerce")

    # Cross-reference with activities to get gear_id
    if not activities.empty and "id" in activities.columns:
        act_lookup = activities[["id", "gear_id"]].rename(columns={"id": "activity_id"})
        efforts_df = efforts_df.merge(act_lookup, on="activity_id", how="left")
    else:
        efforts_df["gear_id"] = None

    # Attach gear names
    gear_name_map: dict[str, str] = {}
    if not gear_frame.empty and "gear_id" in gear_frame.columns and "gear_name" in gear_frame.columns:
        gear_name_map = dict(zip(gear_frame["gear_id"], gear_frame["gear_name"]))
    efforts_df["gear_label"] = efforts_df["gear_id"].map(gear_name_map).fillna("Unknown gear")

    # Compute pace (min/mile) from elapsed time and segment distance
    if distance_m > 0:
        distance_miles = distance_m * _METERS_TO_MILES
        efforts_df["pace_min_per_mile"] = (efforts_df["elapsed_time_s"] / 60) / distance_miles

    # ── Summary stats table ───────────────────────────────────────────────────
    st.markdown("#### Summary by Gear")
    agg_spec: dict = {"efforts": ("effort_id", "count")}
    if "pace_min_per_mile" in efforts_df.columns:
        agg_spec["avg_pace"] = ("pace_min_per_mile", "mean")
        agg_spec["best_pace"] = ("pace_min_per_mile", "min")
    if "average_watts" in efforts_df.columns:
        agg_spec["avg_watts"] = ("average_watts", "mean")
        agg_spec["max_watts"] = ("average_watts", "max")
    if "average_heartrate" in efforts_df.columns:
        agg_spec["avg_hr"] = ("average_heartrate", "mean")

    summary = efforts_df.groupby("gear_label", dropna=False).agg(**agg_spec).reset_index()
    summary = summary.sort_values("efforts", ascending=False)

    display = summary.copy()
    if "avg_pace" in display.columns:
        display["avg_pace"] = display["avg_pace"].apply(_fmt_pace)
        display["best_pace"] = display["best_pace"].apply(_fmt_pace)
    for col in ("avg_watts", "max_watts", "avg_hr"):
        if col in display.columns:
            display[col] = display[col].round(1)

    display = display.rename(columns={
        "gear_label": "Gear",
        "efforts": "Efforts",
        "avg_pace": "Avg Pace (min/mi)",
        "best_pace": "Best Pace (min/mi)",
        "avg_watts": "Avg Watts",
        "max_watts": "Max Watts",
        "avg_hr": "Avg HR",
    })
    st.dataframe(display, use_container_width=True, hide_index=True)

    # ── Plots ─────────────────────────────────────────────────────────────────
    has_pace = "pace_min_per_mile" in efforts_df.columns
    has_watts = (
        "average_watts" in efforts_df.columns
        and efforts_df["average_watts"].notna().any()
    )

    if has_pace or has_watts:
        st.markdown("#### Distributions by Gear")
        cols = st.columns(2 if (has_pace and has_watts) else 1)

        if has_pace:
            pace_df = efforts_df.dropna(subset=["pace_min_per_mile"])
            fig_pace = px.box(
                pace_df,
                x="gear_label",
                y="pace_min_per_mile",
                color="gear_label",
                labels={"gear_label": "Gear", "pace_min_per_mile": "Pace (min/mi)"},
                title="Pace by Gear",
            )
            fig_pace.update_layout(showlegend=False)
            cols[0].plotly_chart(fig_pace, use_container_width=True)

        if has_watts:
            watts_df = efforts_df.dropna(subset=["average_watts"])
            fig_watts = px.box(
                watts_df,
                x="gear_label",
                y="average_watts",
                color="gear_label",
                labels={"gear_label": "Gear", "average_watts": "Avg Watts"},
                title="Power by Gear",
            )
            fig_watts.update_layout(showlegend=False)
            cols[-1].plotly_chart(fig_watts, use_container_width=True)

    # ── Efforts over time ─────────────────────────────────────────────────────
    if has_pace and efforts_df["date"].notna().any():
        st.markdown("#### Efforts Over Time")
        scatter_df = efforts_df.dropna(subset=["pace_min_per_mile", "date"])
        fig_time = px.scatter(
            scatter_df,
            x="date",
            y="pace_min_per_mile",
            color="gear_label",
            hover_data={
                "date": "|%Y-%m-%d",
                "pace_min_per_mile": ":.2f",
                "average_watts": True,
                "gear_label": False,
            },
            labels={
                "date": "Date",
                "pace_min_per_mile": "Pace (min/mi)",
                "gear_label": "Gear",
                "average_watts": "Avg Watts",
            },
            title="Segment Pace Over Time",
        )
        st.plotly_chart(fig_time, use_container_width=True)


def _save_session(
    data: pd.DataFrame, gear_frame: pd.DataFrame, starred: list[dict], code: str | None, max_activities: int, access_token: str) -> None:
    st.session_state["activities"] = data
    st.session_state["gear"] = gear_frame
    st.session_state["starred_segments"] = starred
    st.session_state["last_loaded_max_activities"] = max_activities
    st.session_state["access_token"] = access_token
    if code:
        st.session_state["last_processed_code"] = code


def main() -> None:
    """Render the home page: Strava sign-in and bike summaries."""
    load_dotenv()
    st.set_page_config(page_title="New Bike Day", layout="wide")
    st.title("🚴 New Bike Day")
    st.caption("Sign in with Strava to load your activities and see your bike summaries.")

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
                    data, gear_frame, starred = _process_data(
                        access_token=access_token,
                        max_activities=selected_max_activities,
                    )
                except (requests.RequestException, ValueError) as exc:
                    st.error(f"Unable to process data: {exc}")
                    return
            _save_session(data, gear_frame, starred, code, selected_max_activities, access_token)
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
                data, gear_frame, starred = _process_data(
                    access_token=access_token,
                    max_activities=selected_max_activities,
                )
            except (requests.RequestException, ValueError) as exc:
                st.error(f"Unable to process data: {exc}")
                return
        _save_session(data, gear_frame, starred, code, selected_max_activities, access_token)

    data = st.session_state.get("activities")
    if data is None or data.empty:
        st.info("No data yet. Complete Strava SSO above to load activities.")
        return

    gear_frame = st.session_state.get("gear", pd.DataFrame(columns=["gear_id"]))
    _render_bike_summaries(data, gear_frame)

    starred_segments = st.session_state.get("starred_segments", [])
    access_token = st.session_state.get("access_token") or env_access_token
    if access_token:
        _render_segment_analysis(data, gear_frame, starred_segments, access_token)


if __name__ == "__main__":
    main()
