"""Data Collection page: sign in with Strava and view bike/segment summaries."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.database import (
    clear_bikes,
    clear_efforts,
    clear_segments,
    init_db,
    load_bikes,
    load_efforts,
    load_segments,
    save_bikes,
    save_efforts,
    save_segments,
)
from src.fetch import ingest_all, PremiumOnlyError, get_athlete_bikes
from src.home_personality import load_dev_athlete_profile
from src.auth import custom_auth_button, handle_redirect

def get_and_save_data(access_token: str) -> None:
    try:
        data, gear_frame, bikes, bike_distances, ftp = _process_data(
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
    _save_session(data, gear_frame, bikes, access_token, bike_distances, ftp)
    st.success("Activities reloaded.")

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
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], dict[str, float], int | None]:
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
        ftp: int | None = result.get("ftp")
        _save_to_db(efforts, segments, bikes, bike_distances)
        return efforts, segments, bikes, bike_distances, ftp

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
                bikes, bike_distances, _ = get_athlete_bikes(access_token, gear_to_activity=gear_to_activity)
                if bikes:
                    save_bikes(bikes, bike_distances)
            except Exception:
                pass
        return efforts, segments, bikes, bike_distances, None

    with st.spinner("⏳ Fetching your Strava data for the first time…"):
        result = ingest_all(access_token)

    efforts, segments, bikes = result["efforts"], result["segments"], result.get("bikes", {})
    bike_distances = result.get("bike_distances", {})
    ftp = result.get("ftp")
    _save_to_db(efforts, segments, bikes, bike_distances)
    return efforts, segments, bikes, bike_distances, ftp

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
    for i, (_, row) in enumerate(bike_stats.iterrows()):
        with cols[i]:
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
            "name", "segment_type", "segment_type_detail", "distance", "average_grade",
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
    access_token: str,
    bike_distances: dict[str, float] | None = None,
    ftp: int | None = None,
) -> None:
    st.session_state["efforts"] = data
    st.session_state["segments"] = segments
    st.session_state["bikes"] = bikes
    st.session_state["bike_distances"] = bike_distances or {}
    st.session_state["access_token"] = access_token
    if ftp is not None:
        st.session_state["ftp"] = ftp

def _load_sample_data() -> None:
    """Load dev sample data into session state and render results."""
    result = ingest_all(access_token="", dev=True)
    _save_session(
        result["efforts"],
        result["segments"],
        result["bikes"],
        access_token="",
        bike_distances=result.get("bike_distances", {}),
        ftp=result.get("ftp"),
    )
    data = st.session_state.get("efforts")
    segments = st.session_state.get("segments", pd.DataFrame())
    bikes = st.session_state.get("bikes", {})
    bike_distances = st.session_state.get("bike_distances", {})
    _render_bike_summaries(data, segments, bikes, bike_distances)


def _fallback_to_sample_data(error_message: str) -> None:
    """Show an error and fall back to sample data."""
    st.error(error_message)
    st.session_state["use_sample_data"] = True
    _load_sample_data()


def main() -> None:
    handle_redirect()
    use_sample_data = st.session_state.get("use_sample_data", False)

    # Hero header
    col_title, col_logo = st.columns([4, 1])
    with col_title:
        st.title("📡 Step 1 — Data Collection")
        if use_sample_data:
            athlete_profile = load_dev_athlete_profile()
            athlete_name = athlete_profile.get("first_name") or None
            st.info("Using Sample Data")
        else:
            athlete_name = st.session_state.get("athlete_name")
        if athlete_name:
            st.header(f"Hello, {athlete_name} 👋")
        st.markdown(
            "Sign in with Strava to load your segment efforts and bike data. "
            "Once loaded, proceed to **Step 2 — Data Cleaning**."
        )

    # ── Live mode: OAuth → Strava API ─────────────────────────────────────────
    st.divider()

    if st.session_state.get("strava_token"):
        st.success("✅ Connected to Strava!")
    else:
        custom_auth_button()
    if not st.session_state.get("strava_token") and st.button("📊 Use sample data", width="stretch"):
        st.session_state["use_sample_data"] = True
        st.rerun()
    if st.button("🔄 Reload activities", type="secondary", width="stretch"):
        if not access_token:
            st.warning("Please authorize with Strava first.")
            return
        get_and_save_data(access_token)
    # ── Sample data mode ───────────────────────────────────────────────────────
    if st.session_state.get("use_sample_data"):
        _load_sample_data()
        return

    error_from_params = st.query_params.get("error") or st.session_state.pop("oauth_error", None)

    if error_from_params:
        _fallback_to_sample_data(f"Strava sign-in failed: {error_from_params}. Showing sample data instead.")
        return

    access_token = st.session_state.get("strava_token")
    if not access_token:
        # dont load rest of page, wait for sign in
        return
    
    if st.session_state.get("efforts") is None:
        # initial load after auth, fetch data and save to session
        get_and_save_data(access_token)

    ## some basic viz

    data = st.session_state.get("efforts")
    if data is None or data.empty:
        st.caption("No data loaded yet. Sign in with Strava above to fetch your segment efforts.")
        return

    segments = st.session_state.get("segments", pd.DataFrame())
    bikes = st.session_state.get("bikes", {})
    bike_distances = st.session_state.get("bike_distances", {})
    _render_bike_summaries(data, segments, bikes, bike_distances)

main()
