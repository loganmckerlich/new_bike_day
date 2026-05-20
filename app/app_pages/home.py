"""Home page: sign in with Strava and view bike summaries."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import requests
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[2]
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
from src.fetch import ingest_all, PremiumOnlyError, get_athlete_bikes
from src.utils import link_button_no_tab


# ---------------------------------------------------------------------------
# Static cache helpers (SQLite-backed)
# ---------------------------------------------------------------------------

def _load_from_db() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], dict[str, float]] | None:
    init_db()
    segments = load_segments()
    efforts = load_efforts()
    bikes, bike_distances = load_bikes()
    if segments.empty and efforts.empty:
        return None
    return efforts, segments, bikes, bike_distances


def _save_to_db(
    efforts: pd.DataFrame,
    segments: pd.DataFrame,
    bikes: dict[str, str],
    bike_distances: dict[str, float] | None = None,
) -> None:
    init_db()
    save_segments(segments)
    save_efforts(efforts)
    save_bikes(bikes, bike_distances or {})


def _process_data(
    access_token: str,
    *,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], dict[str, float]]:
    if force_refresh:
        clear_efforts()
        clear_segments()
        clear_bikes()

        progress = st.progress(0, text="Starting…")

        def on_progress(msg: str, pct: int) -> None:
            progress.progress(pct, text=msg)

        result = ingest_all(access_token, progress_callback=on_progress)
        progress.progress(100, text="✅ Complete!")

        efforts, segments, bikes = result["efforts"], result["segments"], result.get("bikes", {})
        bike_distances: dict[str, float] = result.get("bike_distances", {})
        _save_to_db(efforts, segments, bikes, bike_distances)
        return efforts, segments, bikes, bike_distances

    cached = _load_from_db()
    if cached is not None:
        efforts, segments, bikes, bike_distances = cached
        # If bikes table is empty, re-resolve names: try GET /athlete first,
        # then fall back to activity details using the cached efforts' activity_ids.
        if not bikes and access_token:
            try:
                gear_to_activity: dict[str, int] = (
                    efforts.dropna(subset=["gear_id", "activity_id"])
                    .drop_duplicates("gear_id")
                    .assign(activity_id=lambda d: d["activity_id"].astype(int))
                    .set_index("gear_id")["activity_id"]
                    .to_dict()
                )
                bikes, bike_distances = get_athlete_bikes(access_token, gear_to_activity=gear_to_activity)
                if bikes:
                    save_bikes(bikes, bike_distances)
            except Exception:
                pass
        return efforts, segments, bikes, bike_distances

    with st.spinner("⏳ Fetching your Strava data for the first time…"):
        result = ingest_all(access_token)

    efforts, segments, bikes = result["efforts"], result["segments"], result.get("bikes", {})
    bike_distances = result.get("bike_distances", {})
    _save_to_db(efforts, segments, bikes, bike_distances)
    return efforts, segments, bikes, bike_distances


def _query_param_value(value: object) -> str | None:
    if isinstance(value, list):
        return str(value[0]) if value else None
    if value is None:
        return None
    return str(value)


def _exchange_access_token(client_id: str, client_secret: str, redirect_uri: str, code: str) -> str:
    token = exchange_code(
        client_id=client_id,
        client_secret=client_secret,
        code=code,
        redirect_uri=redirect_uri,
    )
    if token.get("athlete_id") and token.get("refresh_token") and token.get("access_token") and token.get("expires_at") is not None:
        init_db()
        save_athlete_token(
            athlete_id=token["athlete_id"],
            access_token=token["access_token"],
            refresh_token=token["refresh_token"],
            expires_at=token["expires_at"],
        )
    return token["access_token"]


def _normalized_redirect_uri(raw_value: str) -> str:
    value = raw_value.strip() if raw_value else ""
    if not value:
        return "http://localhost:8501/"
    parsed = urlsplit(value)
    if parsed.scheme and parsed.netloc and not parsed.path:
        return urlunsplit((parsed.scheme, parsed.netloc, "/", parsed.query, parsed.fragment))
    return value


def _fmt_time(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def _gear_label(gear_id: str | None, bikes: dict[str, str]) -> str:
    # pd.isna() handles None, float NaN, and pd.NA consistently.
    try:
        if pd.isna(gear_id):
            return "Unknown"
    except (TypeError, ValueError):
        pass
    if not gear_id:
        return "Unknown"
    return bikes.get(str(gear_id), str(gear_id))


# ---------------------------------------------------------------------------
# Bike card CSS
# ---------------------------------------------------------------------------

_BIKE_CARD_CSS = """
<style>
.bike-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(252, 76, 2, 0.25);
    border-radius: 14px;
    padding: 1.2rem 1.1rem 1rem;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
    margin-bottom: 0.25rem;
}
.bike-card .bc-brand {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #FC4C02;
    margin-bottom: 0.15rem;
}
.bike-card .bc-name {
    font-size: 1.05rem;
    font-weight: 700;
    line-height: 1.25;
    margin-bottom: 0.85rem;
}
.bike-card .bc-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 0.3rem 0;
    border-top: 1px solid rgba(128, 128, 128, 0.12);
    font-size: 0.875rem;
}
.bike-card .bc-label { color: rgba(160, 160, 160, 0.85); }
.bike-card .bc-val   { font-weight: 600; }
</style>
"""


def _render_bike_card(row: pd.Series) -> None:
    use_metric = st.session_state.get("use_metric", True)
    name = str(row["bike_name"])
    parts = name.split(" ", 1)
    brand = parts[0].upper()
    model = parts[1] if len(parts) > 1 else parts[0]

    watts_str = f"{int(row['avg_watts'])} W" if pd.notna(row["avg_watts"]) else "—"
    hr_str = f"{int(row['avg_heartrate'])} bpm" if pd.notna(row["avg_heartrate"]) else "—"
    hours = f"{row['total_moving_hours']:.1f} hrs"
    dist = row.get("converted_distance")  # stored as miles from Strava odometer
    if pd.notna(dist):
        if use_metric:
            dist_str = f"{dist * 1.60934:,.0f} km"
        else:
            dist_str = f"{dist:,.0f} mi"
    else:
        dist_str = "—"

    st.markdown(
        f"""
        <div class="bike-card">
          <div class="bc-brand">{brand}</div>
          <div class="bc-name">{model}</div>
          <div class="bc-row">
            <span class="bc-label">Total distance</span>
            <span class="bc-val">{dist_str}</span>
          </div>
          <div class="bc-row">
            <span class="bc-label">Efforts</span>
            <span class="bc-val">{int(row['total_efforts'])}</span>
          </div>
          <div class="bc-row">
            <span class="bc-label">Moving time</span>
            <span class="bc-val">{hours}</span>
          </div>
          <div class="bc-row">
            <span class="bc-label">Avg power</span>
            <span class="bc-val">{watts_str}</span>
          </div>
          <div class="bc-row">
            <span class="bc-label">Avg heart rate</span>
            <span class="bc-val">{hr_str}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_bike_summaries(
    efforts: pd.DataFrame,
    segments: pd.DataFrame,
    bikes: dict[str, str],
    bike_distances: dict[str, float] | None = None,
) -> None:
    st.subheader("Your bikes at a glance")

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
    # Drop entries with no gear (efforts logged without a bike).
    bike_stats = bike_stats[bike_stats["bike_name"] != "Unknown"]
    # Merge in total mileage from Strava odometer (converted_distance).
    if bike_distances:
        bike_stats["converted_distance"] = bike_stats["gear_id"].map(bike_distances)
    else:
        bike_stats["converted_distance"] = float("nan")
    bike_stats = bike_stats.sort_values("total_efforts", ascending=False)

    st.markdown(_BIKE_CARD_CSS, unsafe_allow_html=True)

    cols = st.columns(max(len(bike_stats), 1))
    for col, (_, row) in zip(cols, bike_stats.iterrows()):
        with col:
            _render_bike_card(row)

    # ── Starred segments ─────────────────────────────────────────────────────
    if not segments.empty:
        st.divider()
        st.subheader("⭐ Starred segments")

        seg_display = segments.copy()
        if not efforts.empty and "segment_id" in efforts.columns:
            total_attempts = (
                efforts.groupby("segment_id")["effort_id"].count().rename("Total Attempts")
            )
            seg_display = seg_display.merge(total_attempts, on="segment_id", how="left")

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
        attempts_cols = [c for c in seg_display.columns if c not in preferred_seg and c not in _internal_cols]
        display_cols = display_cols + attempts_cols

        seg_display = seg_display[display_cols].copy()
        if "distance" in seg_display.columns:
            use_metric = st.session_state.get("use_metric", True)
            if use_metric:
                seg_display["distance"] = (seg_display["distance"] / 1000).round(2).astype(str) + " km"
            else:
                seg_display["distance"] = (seg_display["distance"] / 1609.34).round(2).astype(str) + " mi"
        if "average_grade" in seg_display.columns:
            seg_display["average_grade"] = seg_display["average_grade"].round(1).astype(str) + "%"
        seg_display.columns = [c.replace("_", " ").title() for c in seg_display.columns]
        st.dataframe(seg_display, width="stretch", hide_index=True)


def _save_session(
    data: pd.DataFrame,
    segments: pd.DataFrame,
    bikes: dict[str, str],
    code: str | None,
    access_token: str,
    bike_distances: dict[str, float] | None = None,
) -> None:
    st.session_state["efforts"] = data
    st.session_state["segments"] = segments
    st.session_state["bikes"] = bikes
    st.session_state["bike_distances"] = bike_distances or {}
    st.session_state["access_token"] = access_token
    if code:
        st.session_state["last_processed_code"] = code


def main() -> None:
    # ── Dev mode toggle ───────────────────────────────────────────────────────
    with st.sidebar:
        dev_mode = st.toggle(
            "🛠️ Dev mode",
            value=False,
            help="Load static sample data instead of hitting the Strava API.",
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
            st.caption("🛠️ Dev mode is on — showing static sample data.")

    # ── Dev mode: load static JSON and skip all OAuth / API logic ─────────────
    if dev_mode:
        result = ingest_all(access_token="", dev=True)
        _save_session(
            result["efforts"],
            result["segments"],
            result["bikes"],
            code=None,
            access_token="",
            bike_distances=result.get("bike_distances", {}),
        )
        data = st.session_state.get("efforts")
        segments = st.session_state.get("segments", pd.DataFrame())
        bikes = st.session_state.get("bikes", {})
        bike_distances = st.session_state.get("bike_distances", {})
        _render_bike_summaries(data, segments, bikes, bike_distances)
        return

    # ── Live mode: OAuth → Strava API ─────────────────────────────────────────
    env_client_id = st.secrets.get("STRAVA_CLIENT_ID", "")
    env_client_secret = st.secrets.get("STRAVA_CLIENT_SECRET", "")
    default_redirect_uri = _normalized_redirect_uri(st.secrets.get("STRAVA_REDIRECT_URI", "http://localhost:8501"))
    env_access_token = st.secrets.get("STRAVA_ACCESS_TOKEN", "")

    st.divider()

    if not env_client_id or not env_client_secret:
        st.error("⚠️ Set STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET in `.streamlit/secrets.toml` to enable sign-in.")
        return

    auth_col, status_col = st.columns([2, 3])
    with auth_col:
        auth_url = get_authorization_url(client_id=env_client_id, redirect_uri=default_redirect_uri)
        # st.link_button("🔗 Sign in with Strava", auth_url, width="stretch")
        link_button_no_tab("🔗 Sign in with Strava", auth_url)

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
                data, gear_frame, bikes, bike_distances = _process_data(access_token=access_token)
            except PremiumOnlyError as exc:
                st.error(str(exc))
                return
            except (requests.RequestException, ValueError) as exc:
                traceback.print_exc(file=sys.stderr)
                st.error(f"Unable to process data: {exc}")
                return
            _save_session(data, gear_frame, bikes, code, access_token, bike_distances)
            with status_col:
                st.success("✅ Connected to Strava — data loaded!")
        else:
            with status_col:
                st.caption("✅ Using cached Strava data.")
    else:
        with status_col:
            st.caption("👆 Click **Sign in with Strava** to authorize and load your segment data.")

    reload_col, _ = st.columns([2, 4])
    with reload_col:
        if st.button("🔄 Reload activities", type="secondary", width="stretch"):
            access_token = st.session_state.get("access_token") or env_access_token
            if not access_token and code:
                try:
                    access_token = _exchange_access_token(env_client_id, env_client_secret, default_redirect_uri, code)
                except (requests.RequestException, ValueError) as exc:
                    st.error(f"Unable to exchange token: {exc}")
                    return
            if not access_token:
                st.warning("Please authorize with Strava first.")
                return
            try:
                data, gear_frame, bikes, bike_distances = _process_data(
                    access_token=access_token,
                    force_refresh=True,
                )
            except PremiumOnlyError as exc:
                st.error(str(exc))
                return
            except (requests.RequestException, ValueError) as exc:
                traceback.print_exc(file=sys.stderr)
                st.error(f"Unable to process data: {exc}")
                return
            _save_session(data, gear_frame, bikes, code, access_token, bike_distances)
            st.success("Activities reloaded.")

    data = st.session_state.get("efforts")
    if data is None or data.empty:
        st.caption("No data loaded yet. Sign in with Strava above to fetch your segment efforts.")
        return

    segments = st.session_state.get("segments", pd.DataFrame())
    bikes = st.session_state.get("bikes", {})
    bike_distances = st.session_state.get("bike_distances", {})
    _render_bike_summaries(data, segments, bikes, bike_distances)


main()
