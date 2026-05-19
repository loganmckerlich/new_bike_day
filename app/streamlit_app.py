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

from src.auth import exchange_code, get_authorization_url
from src.database import (
    clear_bikes,
    clear_efforts,
    clear_segments,
    init_db,
    load_bikes,
    load_efforts,
    load_segments,
    save_athlete_token,
    save_bikes,
    save_efforts,
    save_segments,
)
from src.dev_data import load_dev_data
from src.fetch import ingest_all, PremiumOnlyError


# ---------------------------------------------------------------------------
# Static cache helpers (SQLite-backed)
# ---------------------------------------------------------------------------

def _load_from_db() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]] | None:
    """Return (efforts, segments, bikes) from the SQLite cache, or None if empty."""
    init_db()
    segments = load_segments()
    efforts = load_efforts()
    bikes = load_bikes()
    if segments.empty and efforts.empty:
        return None
    return efforts, segments, bikes


def _save_to_db(
    efforts: pd.DataFrame,
    segments: pd.DataFrame,
    bikes: dict[str, str],
) -> None:
    """Persist ingested data to the SQLite static cache."""
    init_db()
    save_segments(segments)
    save_efforts(efforts)
    save_bikes(bikes)


def _process_data(
    access_token: str,
    *,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Return efforts, segments, bikes — from the SQLite cache or from Strava.

    The Strava API is only called when:
    - ``force_refresh`` is ``True`` (the user clicked the "Reload" button), or
    - the SQLite cache is empty (first run after sign-in).

    In all other cases data is served directly from the on-disk SQLite cache so
    that no network requests are made.

    On a forced refresh the cache is cleared and rebuilt from the API, then
    written back to SQLite before returning.
    """
    if force_refresh:
        # Clear stale cache so deleted segments / efforts are removed.
        clear_efforts()
        clear_segments()
        clear_bikes()

        progress = st.progress(0, text="Starting…")

        def on_progress(msg: str, pct: int) -> None:
            progress.progress(pct, text=msg)

        result = ingest_all(access_token, progress_callback=on_progress)
        progress.progress(100, text="✅ Complete!")

        efforts, segments, bikes = result["efforts"], result["segments"], result.get("bikes", {})
        _save_to_db(efforts, segments, bikes)
        return efforts, segments, bikes

    # Normal path: serve from SQLite cache.
    cached = _load_from_db()
    if cached is not None:
        return cached

    # Cache miss (first sign-in): fetch from API and populate the cache.
    with st.spinner("⏳ Fetching your Strava data for the first time…"):
        result = ingest_all(access_token)

    efforts, segments, bikes = result["efforts"], result["segments"], result.get("bikes", {})
    _save_to_db(efforts, segments, bikes)
    return efforts, segments, bikes


def _query_param_value(value: object) -> str | None:
    """Normalize Streamlit query param values into a single string."""
    if isinstance(value, list):
        return str(value[0]) if value else None
    if value is None:
        return None
    return str(value)


def _exchange_access_token(client_id: str, client_secret: str, redirect_uri: str, code: str) -> str:
    """Exchange a Strava auth code for an access token, persisting the full token to the DB."""
    token = exchange_code(
        client_id=client_id,
        client_secret=client_secret,
        code=code,
        redirect_uri=redirect_uri,
    )
    # Persist refresh token so the webhook server can re-ingest without user interaction.
    if token.get("athlete_id") and token.get("refresh_token"):
        init_db()
        save_athlete_token(
            athlete_id=token["athlete_id"],
            access_token=token["access_token"],
            refresh_token=token["refresh_token"],
            expires_at=token["expires_at"],
        )
    return token["access_token"]


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

        # Build attempts data from efforts
        seg_display = segments.copy()
        if not efforts.empty and "segment_id" in efforts.columns:
            # Total attempts per segment
            total_attempts = (
                efforts.groupby("segment_id")["effort_id"].count().rename("Total Attempts")
            )
            seg_display = seg_display.merge(total_attempts, on="segment_id", how="left")

            # Per-bike attempts (using bike names)
            if "gear_id" in efforts.columns:
                bike_attempts = (
                    efforts.groupby(["segment_id", "gear_id"])["effort_id"]
                    .count()
                    .unstack(fill_value=0)
                )
                bike_attempts.columns = [
                    _gear_label(c, bikes) for c in bike_attempts.columns
                ]
                seg_display = seg_display.merge(bike_attempts, on="segment_id", how="left")

        preferred_seg = [
            "name", "segment_type", "distance", "average_grade",
            "climb_category", "total_elevation_gain",
        ]
        _internal_cols = {"segment_id", "start_lat", "start_lng"}
        display_cols = [c for c in preferred_seg if c in seg_display.columns]
        # Append attempts columns (everything not in the preferred list or internal-only columns)
        attempts_cols = [c for c in seg_display.columns if c not in preferred_seg and c not in _internal_cols]
        display_cols = display_cols + attempts_cols

        seg_display = seg_display[display_cols].copy()
        if "distance" in seg_display.columns:
            seg_display["distance"] = (seg_display["distance"] / 1000).round(2).astype(str) + " km"
        if "average_grade" in seg_display.columns:
            seg_display["average_grade"] = seg_display["average_grade"].round(1).astype(str) + "%"
        # Apply title-casing to all columns (bike name columns from _gear_label are already readable)
        seg_display.columns = [c.replace("_", " ").title() for c in seg_display.columns]
        st.dataframe(seg_display, use_container_width=True, hide_index=True)


def _save_session(
    data: pd.DataFrame,
    segments: pd.DataFrame,
    bikes: dict[str, str],
    code: str | None,
    access_token: str,
) -> None:
    st.session_state["efforts"] = data
    st.session_state["segments"] = segments
    st.session_state["bikes"] = bikes
    st.session_state["access_token"] = access_token
    if code:
        st.session_state["last_processed_code"] = code


def main() -> None:
    """Render the home page: Strava sign-in and bike summaries."""
    st.set_page_config(page_title="New Bike Day", page_icon="🚴", layout="wide")

    # ---------------------------------------------------------------------------
    # Dev mode toggle (sidebar so it's always accessible)
    # ---------------------------------------------------------------------------
    with st.sidebar:
        dev_mode = st.toggle(
            "🛠️ Dev Mode",
            value=False,
            help=(
                "Load static sample data stored in the repository instead of "
                "hitting the Strava API. No network calls are made."
            ),
        )

    # Hero header
    col_title, col_logo = st.columns([4, 1])
    with col_title:
        st.title("🚴 New Bike Day")
        st.markdown(
            "Compare your rides across different bikes on the same Strava segments. "
            "Sign in with Strava to get started."
        )
        if dev_mode:
            st.info(
                "🛠️ **Dev Mode is ON** — showing static sample data. "
                "No Strava API calls are made.",
                icon="🛠️",
            )

    # ---------------------------------------------------------------------------
    # Dev mode: load static JSON and skip all OAuth / API logic
    # ---------------------------------------------------------------------------
    if dev_mode:
        result = load_dev_data()
        _save_session(
            result["efforts"],
            result["segments"],
            result["bikes"],
            code=None,
            access_token="",
        )
        data = st.session_state.get("efforts")
        segments = st.session_state.get("segments", pd.DataFrame())
        bikes = st.session_state.get("bikes", {})
        _render_bike_summaries(data, segments, bikes)
        return

    # ---------------------------------------------------------------------------
    # Live mode: OAuth → Strava API
    # ---------------------------------------------------------------------------
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

    if error_from_params:
        st.error(f"Strava authorization failed: {error_from_params}")
        return

    code = _query_param_value(code_from_params)

    if code:
        last_processed_code = st.session_state.get("last_processed_code")
        should_process = code != last_processed_code
        if should_process:
            with st.spinner("Connecting to Strava…"):
                try:
                    access_token = st.session_state.get("access_token")
                    if code != last_processed_code or not access_token:
                        access_token = _exchange_access_token(env_client_id, env_client_secret, default_redirect_uri, code)
                except (requests.RequestException, ValueError) as exc:
                    st.error(f"Unable to exchange token: {exc}")
                    return
            try:
                data, gear_frame, bikes = _process_data(access_token=access_token)
            except PremiumOnlyError as exc:
                st.error(str(exc))
                return
            except (requests.RequestException, ValueError) as exc:
                st.error(f"Unable to process data: {exc}")
                return
            _save_session(data, gear_frame, bikes, code, access_token)
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
            try:
                data, gear_frame, bikes = _process_data(
                    access_token=access_token,
                    force_refresh=True,
                )
            except PremiumOnlyError as exc:
                st.error(str(exc))
                return
            except (requests.RequestException, ValueError) as exc:
                st.error(f"Unable to process data: {exc}")
                return
            _save_session(data, gear_frame, bikes, code, access_token)
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

