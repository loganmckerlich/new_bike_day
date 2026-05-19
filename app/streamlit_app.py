"""Streamlit home page: sign in with Strava and view bike summaries."""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import plotly.express as px
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
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Fetch starred segments, all efforts, and bike names from Strava."""
    progress = st.progress(0, text="Starting…")
    progress.progress(20, text="Fetching starred segments and efforts from Strava…")
    result = ingest_all(access_token)
    progress.progress(100, text="Complete.")
    return result["efforts"], result["segments"], result.get("bikes", {})


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


def _fmt_time(seconds: float) -> str:
    """Format a duration in seconds as m:ss."""
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def _gear_label(gear_id: str | None, bikes: dict[str, str]) -> str:
    """Return a human-readable bike name, falling back to the gear_id."""
    if not gear_id:
        return "Unknown"
    return bikes.get(str(gear_id), str(gear_id))


def _render_bike_summaries(efforts: pd.DataFrame, segments: pd.DataFrame, bikes: dict[str, str]) -> None:
    """Render the bike summaries section."""
    st.markdown("---")
    st.subheader("🚲 Your Bikes at a Glance")

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
    bike_stats["bike_name"] = bike_stats["gear_id"].map(lambda g: _gear_label(g, bikes))
    bike_stats = bike_stats.sort_values("total_efforts", ascending=False)

    # Metric cards for top bikes
    top = bike_stats.head(4)
    cols = st.columns(len(top)) if len(top) > 0 else []
    for col, (_, row) in zip(cols, top.iterrows()):
        with col:
            watts_str = f"{int(row['avg_watts'])} W" if pd.notna(row["avg_watts"]) else "—"
            hr_str = f"{int(row['avg_heartrate'])} bpm" if pd.notna(row["avg_heartrate"]) else "—"
            st.metric(label=row["bike_name"], value=f"{row['total_efforts']} efforts")
            st.caption(f"⏱ {row['total_moving_hours']} hrs · ⚡ {watts_str} · ❤️ {hr_str}")

    if len(bike_stats) > 0:
        with st.expander("Full bike stats table", expanded=False):
            display = bike_stats[["bike_name", "total_efforts", "total_moving_hours", "avg_watts", "avg_heartrate"]].copy()
            display.columns = ["Bike", "Efforts", "Moving Hours", "Avg Watts", "Avg HR"]
            st.dataframe(display, use_container_width=True, hide_index=True)

        # Watts bar chart
        chart_data = bike_stats.dropna(subset=["avg_watts"])
        if not chart_data.empty:
            fig = px.bar(
                chart_data,
                x="bike_name",
                y="avg_watts",
                labels={"bike_name": "Bike", "avg_watts": "Avg Watts"},
                title="Average Power by Bike",
                color="bike_name",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    # Starred segments table
    if not segments.empty:
        st.markdown("---")
        st.subheader("⭐ Starred Segments")
        preferred_seg = [
            "name", "segment_type", "distance", "average_grade",
            "climb_category", "total_elevation_gain",
        ]
        display_cols = [c for c in preferred_seg if c in segments.columns]
        seg_display = segments[display_cols].copy()
        if "distance" in seg_display.columns:
            seg_display["distance"] = (seg_display["distance"] / 1000).round(2).astype(str) + " km"
        if "average_grade" in seg_display.columns:
            seg_display["average_grade"] = seg_display["average_grade"].round(1).astype(str) + "%"
        seg_display.columns = [c.replace("_", " ").title() for c in seg_display.columns]
        st.dataframe(seg_display, use_container_width=True, hide_index=True)


def _save_session(
    data: pd.DataFrame,
    segments: pd.DataFrame,
    bikes: dict[str, str],
    code: str | None,
    max_activities: int,
    access_token: str,
) -> None:
    st.session_state["efforts"] = data
    st.session_state["segments"] = segments
    st.session_state["bikes"] = bikes
    st.session_state["last_loaded_max_activities"] = max_activities
    st.session_state["access_token"] = access_token
    if code:
        st.session_state["last_processed_code"] = code


def main() -> None:
    """Render the home page: Strava sign-in and bike summaries."""
    st.set_page_config(page_title="New Bike Day", page_icon="🚴", layout="wide")

    # Hero header
    col_title, col_logo = st.columns([4, 1])
    with col_title:
        st.title("🚴 New Bike Day")
        st.markdown(
            "Compare your rides across different bikes on the same Strava segments. "
            "Sign in with Strava to get started."
        )

    env_client_id = st.secrets.get("STRAVA_CLIENT_ID", "")
    env_client_secret = st.secrets.get("STRAVA_CLIENT_SECRET", "")
    default_redirect_uri = _normalized_redirect_uri(st.secrets.get("STRAVA_REDIRECT_URI", "http://localhost:8501"))
    env_access_token = st.secrets.get("STRAVA_ACCESS_TOKEN", "")

    st.markdown("---")

    if not env_client_id or not env_client_secret:
        st.error("⚠️ Set STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET in `.streamlit/secrets.toml` to enable sign-in.")
        return

    # Auth section
    auth_col, status_col = st.columns([2, 3])
    with auth_col:
        auth_url = get_authorization_url(client_id=env_client_id, redirect_uri=default_redirect_uri)
        st.link_button("🔗 Sign in with Strava", auth_url, use_container_width=True)

    code_from_params = st.query_params.get("code")
    error_from_params = st.query_params.get("error")
    max_activities = st.number_input(
        "Max Activities to Load",
        min_value=1,
        max_value=10000,
        value=DEFAULT_MAX_ACTIVITIES,
        help="Limit how many recent activities are fetched from Strava.",
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
            with st.spinner("Fetching your Strava data…"):
                try:
                    access_token = st.session_state.get("access_token")
                    if code != last_processed_code or not access_token:
                        access_token = _exchange_access_token(env_client_id, env_client_secret, default_redirect_uri, code)
                    data, gear_frame, bikes = _process_data(
                        access_token=access_token,
                        max_activities=selected_max_activities,
                    )
                except PremiumOnlyError as exc:
                    st.error(str(exc))
                    return
                except (requests.RequestException, ValueError) as exc:
                    st.error(f"Unable to process data: {exc}")
                    return
            _save_session(data, gear_frame, bikes, code, selected_max_activities, access_token)
            with status_col:
                st.success("✅ Connected to Strava — data loaded!")
        else:
            with status_col:
                st.info("✅ Using cached Strava data.")
    else:
        with status_col:
            st.info("👆 Click **Sign in with Strava** to authorize and load your segment data.")

    reload_col, _ = st.columns([2, 4])
    with reload_col:
        if st.button("🔄 Reload Activities", type="secondary", use_container_width=True):
            access_token = st.session_state.get("access_token") or env_access_token
            if not access_token and code:
                try:
                    access_token = _exchange_access_token(env_client_id, env_client_secret, default_redirect_uri, code)
                except (requests.RequestException, ValueError) as exc:
                    st.error(f"Unable to exchange token: {exc}")
                    return
            if not access_token:
                st.warning(
                    "Please authorize with Strava using the Sign in button above, "
                    "or set STRAVA_ACCESS_TOKEN in your `.streamlit/secrets.toml`."
                )
                return
            with st.spinner("Reloading…"):
                try:
                    data, gear_frame, bikes = _process_data(
                        access_token=access_token,
                        max_activities=selected_max_activities,
                    )
                except PremiumOnlyError as exc:
                    st.error(str(exc))
                    return
                except (requests.RequestException, ValueError) as exc:
                    st.error(f"Unable to process data: {exc}")
                    return
            _save_session(data, gear_frame, bikes, code, selected_max_activities, access_token)
            st.success("Activities reloaded.")

    data = st.session_state.get("efforts")
    if data is None or data.empty:
        st.info("No data loaded yet. Sign in with Strava above to fetch your segment efforts.")
        return

    segments = st.session_state.get("segments", pd.DataFrame())
    bikes = st.session_state.get("bikes", {})
    _render_bike_summaries(data, segments, bikes)


if __name__ == "__main__":
    main()

