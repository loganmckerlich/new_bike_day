"""Data Collection page: sign in with Strava and view bike/segment summaries."""

from __future__ import annotations

import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils import navigator

from src.database import (
    clear_bikes,
    clear_efforts,
    clear_rides,
    clear_segments,
    clear_ftp,
    cleanup_if_needed,
    init_db,
    load_bikes,
    load_efforts,
    load_rides,
    load_segments,
    load_ftp,
    load_user_ingest_dates,
    save_bikes,
    save_efforts,
    save_rides,
    save_segments,
    save_ftp,
    save_user_ingest_dates,
)
from src.fetch import ingest_all, ingest_window, get_athlete_bikes, get_starred_segments, PremiumOnlyError
from src.home_personality import load_dev_athlete_profile
from src.auth import custom_auth_button, handle_redirect, get_demo_access_token

def get_and_save_data(access_token: str, athlete_id: int, force_refresh: bool = False) -> None:
    try:
        db_cached = _load_from_db(athlete_id)
        if force_refresh:
            clear_efforts(athlete_id)
            clear_segments(athlete_id)
            clear_bikes(athlete_id)
            clear_ftp(athlete_id)
            clear_rides(athlete_id)
            message = _run_chunked_ingest(access_token, athlete_id, direction="older", initial=True)
            st.info(message)
            return
        if db_cached is None:
            message = _run_chunked_ingest(access_token, athlete_id, direction="older", initial=True)
            st.info(message)
            return
        data, gear_frame, bikes, bike_distances, ftp, rides = db_cached
    except PremiumOnlyError as exc:
        st.error(str(exc))
        return
    except (requests.RequestException, ValueError) as exc:
        traceback.print_exc(file=sys.stderr)
        st.error(f"Unable to process data: {exc}")
        return
    _save_session(data, gear_frame, bikes, access_token, bike_distances, ftp, rides)

# ---------------------------------------------------------------------------
# Static cache helpers (Supabase-backed (soon))
# ---------------------------------------------------------------------------

def _load_from_db(athlete_id: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], dict[str, float]] | None:
    init_db()
    segments = load_segments(athlete_id)
    efforts = load_efforts(athlete_id)
    bikes, bike_distances = load_bikes(athlete_id)
    ftp = load_ftp(athlete_id)
    rides = load_rides(athlete_id)
    if segments.empty and efforts.empty:
        return None
    return efforts, segments, bikes, bike_distances, ftp, rides


def _save_to_db(
    efforts: pd.DataFrame,
    segments: pd.DataFrame,
    bikes: dict[str, str],
    athlete_id: int,
    bike_distances: dict[str, float] | None = None,
    ftp: int | None = None,
    rides: pd.DataFrame | None = None,
) -> None:
    print("Saving data to local cache for athlete_id", athlete_id)
    init_db()
    save_segments(segments, athlete_id)
    save_efforts(efforts, athlete_id)
    save_bikes(bikes, athlete_id, bike_distances or {})
    save_ftp(ftp, athlete_id)
    if rides is not None and not rides.empty:
        save_rides(rides, athlete_id)


def _to_utc_timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def _date_floor(dt: datetime) -> datetime:
    return datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)


def _window_message(start: datetime, end: datetime, segments: int, efforts: int) -> str:
    return (
        f"Fetched {start.date()} → {end.date()} across {segments} segments "
        f"({efforts} efforts)."
    )


def _run_chunked_ingest(
    access_token: str,
    athlete_id: int,
    *,
    direction: str,
    initial: bool = False,
) -> str:
    existing_efforts = load_efforts(athlete_id)
    existing_segments = load_segments(athlete_id)
    existing_bikes, existing_distances = load_bikes(athlete_id)
    existing_ftp = load_ftp(athlete_id)
    existing_rides = load_rides(athlete_id)
    saved_last, saved_oldest = load_user_ingest_dates(athlete_id)

    now = _date_floor(datetime.now(timezone.utc))
    three_years_ago = now - timedelta(days=365 * 3)
    last_ingested = _to_utc_timestamp(saved_last)
    oldest_ingested = _to_utc_timestamp(saved_oldest)
    start_dates = existing_efforts.get("start_date") if not existing_efforts.empty else None
    newest_effort = _to_utc_timestamp(start_dates.max() if start_dates is not None else None)
    oldest_effort = _to_utc_timestamp(start_dates.min() if start_dates is not None else None)

    last_ingested = last_ingested or newest_effort
    oldest_ingested = oldest_ingested or oldest_effort

    progress = st.progress(0, text="Starting chunked ingest…")

    def update_progress(text: str, current: int, total: int) -> None:
        pct = int((current / max(total, 1)) * 100)
        progress.progress(pct, text=text)

    segments = existing_segments
    if segments.empty:
        segments = get_starred_segments(access_token)
        save_segments(segments, athlete_id)

    window_bounds: list[tuple[datetime, datetime]] = []
    if initial:
        end_cursor = now
        while end_cursor > three_years_ago:
            start_cursor = max(three_years_ago, end_cursor - timedelta(days=30))
            window_bounds.append((start_cursor, end_cursor))
            end_cursor = start_cursor
    elif direction == "newer":
        start_cursor = _date_floor(last_ingested or now - timedelta(days=30))
        while start_cursor < now:
            window_end = min(start_cursor + timedelta(days=30), now)
            window_bounds.append((start_cursor, window_end))
            start_cursor = window_end
    else:
        end_cursor = _date_floor(oldest_ingested or now)
        lower_bound = max(three_years_ago, end_cursor - timedelta(days=30))
        if end_cursor > lower_bound:
            window_bounds.append((lower_bound, end_cursor))

    if not window_bounds:
        progress.progress(100, text="No new windows to ingest.")
        return "No new windows to ingest."

    completed_ranges: list[tuple[datetime, datetime]] = []
    threshold_hit = False
    for index, (window_start, window_end) in enumerate(window_bounds, start=1):
        update_progress(
            f"Loading {window_start.date()} → {window_end.date()} ({index}/{len(window_bounds)})…",
            index - 1,
            len(window_bounds),
        )
        window_result = ingest_window(
            access_token,
            segments,
            window_start,
            window_end,
        )
        if window_result["mid_window_rate_limit"]:
            threshold_hit = True
            break

        window_efforts: pd.DataFrame = window_result["efforts"]
        window_rides: pd.DataFrame = window_result["rides"]
        if not window_efforts.empty:
            save_efforts(window_efforts, athlete_id)
            existing_efforts = pd.concat([existing_efforts, window_efforts], ignore_index=True)
            if "effort_id" in existing_efforts.columns:
                existing_efforts = existing_efforts.drop_duplicates(subset=["effort_id"], keep="last")
        if not window_rides.empty:
            save_rides(window_rides, athlete_id)
            existing_rides = pd.concat([existing_rides, window_rides], ignore_index=True)
            if "activity_id" in existing_rides.columns:
                existing_rides = existing_rides.drop_duplicates(subset=["activity_id"], keep="last")

        gear_ids = {entry.get("gear_id") for entry in window_result["activities"].values() if entry.get("gear_id")}
        bikes, distances, ftp = get_athlete_bikes(access_token, gear_ids=gear_ids)
        if bikes:
            existing_bikes.update(bikes)
            existing_distances.update(distances)
            save_bikes(bikes, athlete_id, distances)
        if ftp is not None:
            existing_ftp = ftp
            save_ftp(ftp, athlete_id)

        if direction == "newer":
            last_ingested = window_end
            save_user_ingest_dates(athlete_id, last_ingested_date=window_end)
        else:
            oldest_ingested = window_start
            save_user_ingest_dates(athlete_id, oldest_ingested_date=window_start)
        completed_ranges.append((window_start, window_end))

        threshold_hit = window_result["threshold_reached"]
        if threshold_hit:
            break
        update_progress(
            _window_message(window_start, window_end, window_result["total_segments"], len(window_efforts.index)),
            index,
            len(window_bounds),
        )

    current_start_dates = existing_efforts.get("start_date") if not existing_efforts.empty else None
    actual_oldest = _to_utc_timestamp(current_start_dates.min() if current_start_dates is not None else None)
    actual_newest = _to_utc_timestamp(current_start_dates.max() if current_start_dates is not None else None)
    if initial and current_start_dates is not None:
        save_user_ingest_dates(
            athlete_id,
            oldest_ingested_date=actual_oldest,
            last_ingested_date=actual_newest,
        )
    elif current_start_dates is not None and not completed_ranges:
        save_user_ingest_dates(
            athlete_id,
            oldest_ingested_date=actual_oldest,
            last_ingested_date=actual_newest,
        )

    _save_session(
        existing_efforts,
        segments,
        existing_bikes,
        access_token,
        existing_distances,
        existing_ftp,
        existing_rides,
    )

    if threshold_hit:
        if completed_ranges:
            _, last_end = completed_ranges[-1]
            progress.progress(100, text="Rate limit reached. Saved completed windows.")
            return (
                f"Rate limit reached mid-window. Progress up to {last_end.date()} has been saved. "
                "Come back later to continue."
            )
        progress.progress(100, text="Rate limit reached before any full window completed.")
        return "Rate limit reached before a full window completed. Come back later to continue."

    progress.progress(100, text="✅ Chunked ingest complete.")
    total_segments = int(len(segments.index))
    if completed_ranges:
        first_start, _ = completed_ranges[0]
        _, last_end = completed_ranges[-1]
        effort_count = len(existing_efforts.index)
        return _window_message(first_start, last_end, total_segments, effort_count)
    return "No data fetched."


@st.cache_data(ttl=3600, show_spinner=False)
def _process_data(
    access_token: str,
    athlete_id: int,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], dict[str, float], int | None]:
    db_cached = _load_from_db(athlete_id)

    if force_refresh or db_cached is None:
        if force_refresh:
            print("Force refresh enabled - clearing cache for athlete_id", athlete_id)
            clear_efforts(athlete_id)
            clear_segments(athlete_id)
            clear_bikes(athlete_id)
            clear_ftp(athlete_id)
            clear_rides(athlete_id)
        else:
            print("No db cache found for athlete_id", athlete_id, "- fetching from Strava API")

        cleanup_if_needed(athlete_id)
        progress = st.progress(0, text="Starting…")
        def on_progress(msg: str, pct: int) -> None:
            progress.progress(pct, text=msg)
        result = ingest_all(access_token, progress_callback=on_progress)
        progress.progress(100, text="✅ Complete!")

        efforts, segments, bikes = result["efforts"], result["segments"], result.get("bikes", {})
        bike_distances: dict[str, float] = result.get("bike_distances", {})
        ftp = result.get("ftp")
        rides: pd.DataFrame = result.get("rides", pd.DataFrame())
        _save_to_db(efforts, segments, bikes, athlete_id, bike_distances, ftp, rides)
        return efforts, segments, bikes, bike_distances, ftp, rides

    elif db_cached is not None:
        print("Loaded data from db cache for athlete_id", athlete_id)
        efforts, segments, bikes, bike_distances, ftp, rides = db_cached
        return efforts, segments, bikes, bike_distances, ftp, rides
    else:
        st.error("OOP, This shouldnt happen")
        st.stop()


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

    watts_str = f"{int(row['avg_watts'])} W *" if pd.notna(row["avg_watts"]) else "—"
    hr_str = f"{int(row['avg_heartrate'])} bpm *" if pd.notna(row["avg_heartrate"]) else "—"
    moving_hours = row.get("total_moving_hours")
    hours = f"{moving_hours:.1f} hrs" if pd.notna(moving_hours) else "—"
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
            <span class="bc-label">Rides</span>
            <span class="bc-val">{int(row['total_rides'])}</span>
          </div>
          <div class="bc-row">
            <span class="bc-label">Segment Efforts</span>
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
        <p style="font-size:0.65rem; color: rgba(255,255,255,0.4); margin-top:0.2rem;">* mean of per-effort averages</p>
        """,
        unsafe_allow_html=True,
    )


def _render_bike_summaries(
    efforts: pd.DataFrame,
    segments: pd.DataFrame,
    bikes: dict[str, str],
    bike_distances: dict[str, float] | None = None,
    rides: pd.DataFrame | None = None,
) -> None:
    st.subheader("Your bikes at a glance")

    agg_metrics = {
        "total_efforts": ("effort_id", "count"),
        "avg_watts": ("average_watts", "mean"),
        "avg_heartrate": ("average_heartrate", "mean"),
    }
    bike_stats = efforts.groupby("gear_id", dropna=False).agg(**agg_metrics).reset_index()

    # Rides + moving time — sourced from the rides table (one row per activity)
    if rides is not None and not rides.empty and "gear_id" in rides.columns:
        ride_agg = (
            rides.dropna(subset=["gear_id"])
            .groupby("gear_id")
            .agg(total_rides=("activity_id", "count"), total_moving_seconds=("moving_time", "sum"))
            .reset_index()
        )
        ride_agg["total_moving_hours"] = (ride_agg["total_moving_seconds"] / 3600).round(1)
        bike_stats = bike_stats.merge(ride_agg[["gear_id", "total_rides", "total_moving_hours"]], on="gear_id", how="left")
    else:
        # Fallback: count distinct activity_ids from efforts (moving time unavailable)
        rides_per_bike = (
            efforts.dropna(subset=["activity_id"])
            .groupby("gear_id")["activity_id"]
            .nunique()
            .rename("total_rides")
        )
        bike_stats = bike_stats.merge(rides_per_bike, on="gear_id", how="left")
        bike_stats["total_moving_hours"] = float("nan")

    bike_stats["total_rides"] = bike_stats.get("total_rides", pd.Series(0, index=bike_stats.index)).fillna(0).astype(int)
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
    st.caption("Fields with * are calculated as mean of mean unweighted so are not exact")
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
    rides: pd.DataFrame | None = None,
) -> None:
    st.session_state["efforts"] = data
    st.session_state["segments"] = segments
    st.session_state["bikes"] = bikes
    st.session_state["bike_distances"] = bike_distances or {}
    st.session_state["access_token"] = access_token
    st.session_state["rides"] = rides if rides is not None else pd.DataFrame()
    if ftp is not None:
        st.session_state["ftp"] = ftp

def _load_demo_data() -> None:
    """Load demo data: live from my Strava account, falling back to static dev JSON."""
    token_result = get_demo_access_token()
    if token_result is not None:
        access_token, athlete_id = token_result
        if athlete_id is not None:
            athlete_id = int(athlete_id)
            st.session_state["strava_athlete"] = {"id": athlete_id}
            get_and_save_data(access_token, athlete_id, force_refresh=False)
            return
    # ponytail: fall back to static snapshots if secret is missing or token refresh fails
    result = ingest_all(access_token="", dev=True)
    _save_session(
        result["efforts"],
        result["segments"],
        result["bikes"],
        access_token="",
        bike_distances=result.get("bike_distances", {}),
        ftp=result.get("ftp"),
        rides=result.get("rides", pd.DataFrame()),
    )
    data = st.session_state.get("efforts")
    segments = st.session_state.get("segments", pd.DataFrame())
    bikes = st.session_state.get("bikes", {})
    bike_distances = st.session_state.get("bike_distances", {})
    rides = st.session_state.get("rides", pd.DataFrame())
    _render_bike_summaries(data, segments, bikes, bike_distances, rides)


def _fallback_to_sample_data(error_message: str) -> None:
    """Show an error and fall back to demo data."""
    st.error(error_message)
    st.session_state["use_sample_data"] = True
    _load_demo_data()


def main() -> None:
    handle_redirect()
    athlete_id = st.session_state.get("strava_athlete", {}).get("id")
    athlete_id = int(athlete_id) if athlete_id is not None else None
    use_sample_data = st.session_state.get("use_sample_data", False)

    # Hero header
    col_title, col_logo = st.columns([4, 1])
    with col_title:
        st.title("📡 Step 1 — Data Collection")
        if use_sample_data:
            athlete_profile = load_dev_athlete_profile()
            athlete_name = athlete_profile.get("first_name") or None
            st.info("Viewing Logan's Strava data")
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
        status, bio = st.columns(2)
        with status:
            st.success("✅ Connected to Strava!")
        with bio:
            # st.image(st.session_state.get("strava_athlete", {}).get("profile", ""), width=80)
            st.caption(st.session_state.get("strava_athlete", {}).get("bio", ""))

    else:
        st.caption("""
            User created Strava applications are limited on how many users can sign in.
            If you dont record a large number of rides with heart rate and power meter
            data, and/or you dont have rides recorded and labeled under multiple bikes,
            please use my sample data to explore the app!
            """)
        custom_auth_button()
        if not st.session_state.get("use_sample_data") and st.button("📊 View Logans Data", width="stretch"):
            st.session_state["use_sample_data"] = True
            _load_demo_data()
            st.rerun()

    if st.session_state.get("strava_token"):
        newer_col, older_col = st.columns(2)
        with newer_col:
            if st.button("Get newer data", type="secondary", width="stretch"):
                message = _run_chunked_ingest(
                    st.session_state.get("strava_token"),
                    athlete_id,
                    direction="newer",
                )
                st.success(message)
        with older_col:
            if st.button("Get older data", type="secondary", width="stretch"):
                message = _run_chunked_ingest(
                    st.session_state.get("strava_token"),
                    athlete_id,
                    direction="older",
                )
                st.success(message)

    error_from_params = st.query_params.get("error") or st.session_state.pop("oauth_error", None)

    if error_from_params:
        _fallback_to_sample_data(f"Strava sign-in failed: {error_from_params}. Showing sample data instead.")
        return

    if not st.session_state.get("strava_token") and not st.session_state.get("use_sample_data"):
        # dont load rest of page, wait for sign in
        return
    elif st.session_state.get("efforts") is None and st.session_state.get("strava_token"):
        # signed in - havent loaded yet - load from db if its there
        get_and_save_data(st.session_state.get("strava_token"), athlete_id, force_refresh=False)


    oldest_raw = st.session_state.get("efforts", pd.DataFrame()).get("start_date").min()
    newest_raw = st.session_state.get("efforts", pd.DataFrame()).get("start_date").max()
    oldest = pd.to_datetime(oldest_raw).strftime("%Y-%m-%d") if pd.notna(oldest_raw) else "N/A"
    newest = pd.to_datetime(newest_raw).strftime("%Y-%m-%d") if pd.notna(newest_raw) else "N/A"

    st.markdown(f"Using Data from {oldest} to {newest}")

    ## some basic viz
    data = st.session_state.get("efforts")
    if data is None or data.empty:
        st.caption("No data loaded yet. Sign in with Strava above to fetch your segment efforts.")
        return

    segments = st.session_state.get("segments", pd.DataFrame())
    bikes = st.session_state.get("bikes", {})
    bike_distances = st.session_state.get("bike_distances", {})
    rides = st.session_state.get("rides", pd.DataFrame())
    _render_bike_summaries(data, segments, bikes, bike_distances, rides)

navigator("data_collection1")
main()
navigator("data_collection2")