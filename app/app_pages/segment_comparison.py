"""Segment comparison page – valid segment selection and bike performance analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.fetch import get_segment_detail, get_segment_streams
from src.database import init_db, load_segment_geo, save_segment_geo

# ── Session state ─────────────────────────────────────────────────────────────
efforts: pd.DataFrame | None = st.session_state.get("efforts")
segments: pd.DataFrame | None = st.session_state.get("segments")
bikes: dict[str, str] = st.session_state.get("bikes", {})
access_token: str | None = st.session_state.get("access_token")

if efforts is None or (hasattr(efforts, "empty") and efforts.empty):
    st.info("👈 Head to the **Home** page to sign in with Strava and load your data first.")
    st.stop()

if segments is None or segments.empty:
    st.warning("No starred segments found. Star some segments on Strava and reload.")
    st.stop()

# ── Constants ─────────────────────────────────────────────────────────────────
SEGMENT_TYPES: list[str] = ["sprint", "flat", "ascent", "descent"]
TYPE_ICONS: dict[str, str] = {
    "sprint": "⚡",
    "flat": "➡️",
    "ascent": "⬆️",
    "descent": "⬇️",
}
_COLOR_SEQ: list[str] = px.colors.qualitative.Set2

# ── Unit helpers ─────────────────────────────────────────────────────────────

def _use_metric() -> bool:
    return st.session_state.get("use_metric", True)


def _spd_label() -> str:
    return "km/h" if _use_metric() else "mph"


def _dist_label() -> str:
    return "km" if _use_metric() else "mi"


def _elev_label() -> str:
    return "m" if _use_metric() else "ft"


def _convert_speed(kmh: float) -> float:
    """Convert km/h to display unit."""
    return kmh if _use_metric() else kmh * 0.621371


def _convert_dist_m(meters: float) -> float:
    """Convert metres to display unit (km or mi)."""
    return meters / 1000 if _use_metric() else meters / 1609.34


def _convert_elev_m(meters: float) -> float:
    """Convert metres to display unit (m or ft)."""
    return meters if _use_metric() else meters * 3.28084


# ── Helpers ───────────────────────────────────────────────────────────────────

def _gear_label(gear_id: str | None) -> str:
    if gear_id is None:
        return "Unknown"
    return bikes.get(str(gear_id), str(gear_id))


def _fmt_duration(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 60}:{total % 60:02d}"


def _compute_speed_kmh(df: pd.DataFrame, distance_m: float | None = None) -> pd.Series:
    safe_time = df["moving_time"].replace(0, pd.NA)
    if distance_m is not None and distance_m > 0:
        return (distance_m / safe_time * 3.6).where(safe_time.notna())
    dist = df.get("distance", pd.Series(dtype=float))
    return (dist / safe_time * 3.6).where(safe_time.notna() & dist.notna())


def _has_col(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and df[col].notna().any()


def _get_segment_geo(segment_id: int) -> dict:
    cache_key = f"segment_geo_{segment_id}"
    cached = st.session_state.get(cache_key)
    if cached is not None:
        return cached

    init_db()
    db_cached = load_segment_geo(segment_id)
    if db_cached is not None:
        st.session_state[cache_key] = db_cached
        return db_cached

    if not access_token:
        st.session_state[cache_key] = {}
        return {}

    detail = get_segment_detail(access_token, segment_id)
    streams = get_segment_streams(access_token, segment_id)
    save_segment_geo(segment_id, detail, streams)
    result = {**detail, "streams": streams}
    st.session_state[cache_key] = result
    return result


def _render_segment_map(geo: dict, seg_name: str) -> None:
    points = geo.get("polyline_points") or []
    start_ll = geo.get("start_latlng") or []
    end_ll = geo.get("end_latlng") or []

    if not points and not start_ll:
        st.caption("No route data available for this segment.")
        return

    traces = []

    if points:
        lats = [p[0] for p in points]
        lngs = [p[1] for p in points]
        traces.append(
            go.Scattermap(
                lat=lats,
                lon=lngs,
                mode="lines",
                line={"width": 4, "color": "#FC4C02"},
                name="Route",
                hoverinfo="skip",
            )
        )
        center_lat = sum(lats) / len(lats)
        center_lng = sum(lngs) / len(lngs)
    elif start_ll:
        center_lat, center_lng = float(start_ll[0]), float(start_ll[1])
    else:
        return

    if start_ll and len(start_ll) == 2:
        traces.append(
            go.Scattermap(
                lat=[float(start_ll[0])],
                lon=[float(start_ll[1])],
                mode="markers+text",
                marker={"size": 14, "color": "#22c55e"},
                text=["Start"],
                textposition="top right",
                name="Start",
            )
        )

    if end_ll and len(end_ll) == 2:
        traces.append(
            go.Scattermap(
                lat=[float(end_ll[0])],
                lon=[float(end_ll[1])],
                mode="markers+text",
                marker={"size": 14, "color": "#ef4444"},
                text=["Finish"],
                textposition="top right",
                name="Finish",
            )
        )

    fig = go.Figure(traces)
    fig.update_layout(
        map={
            "style": "open-street-map",
            "center": {"lat": center_lat, "lon": center_lng},
            "zoom": 13,
        },
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=280,
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch", config={"scrollZoom": True})


def _render_elevation_profile(geo: dict, seg_distance_m: float) -> None:
    streams = geo.get("streams") or {}
    elev_low = geo.get("elevation_low")
    elev_high = geo.get("elevation_high")
    d_label = _dist_label()
    e_label = _elev_label()

    if streams and "distance" in streams and "altitude" in streams:
        dist_vals = [_convert_dist_m(d) for d in streams["distance"]]
        alt = [_convert_elev_m(v) for v in streams["altitude"]]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dist_vals,
                y=alt,
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(252, 76, 2, 0.15)",
                line={"color": "#FC4C02", "width": 2},
                hovertemplate=f"Distance: %{{x:.2f}} {d_label}<br>Elevation: %{{y:.0f}} {e_label}<extra></extra>",
            )
        )
        fig.update_layout(
            xaxis_title=f"Distance ({d_label})",
            yaxis_title=f"Elevation ({e_label})",
            margin={"l": 40, "r": 10, "t": 10, "b": 40},
            height=280,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")

    elif elev_low is not None and elev_high is not None and seg_distance_m > 0:
        dist_total = _convert_dist_m(seg_distance_m)
        mid = dist_total / 2
        elev_lo = _convert_elev_m(float(elev_low))
        elev_hi = _convert_elev_m(float(elev_high))
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[0, mid, dist_total],
                y=[elev_lo, elev_hi, elev_lo],
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(252, 76, 2, 0.15)",
                line={"color": "#FC4C02", "width": 2, "dash": "dot"},
                hovertemplate=f"~Elevation: %{{y:.0f}} {e_label}<extra></extra>",
            )
        )
        fig.update_layout(
            xaxis_title=f"Distance ({d_label})",
            yaxis_title=f"Elevation ({e_label})",
            margin={"l": 40, "r": 10, "t": 10, "b": 40},
            height=280,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.caption("📐 Approximate elevation profile")
        st.plotly_chart(fig, width="stretch")
    else:
        st.caption("No elevation data available.")


# ── Data preparation ──────────────────────────────────────────────────────────

# Keep only efforts with power data
watt_efforts = efforts[efforts["average_watts"].notna()].copy()

if watt_efforts.empty:
    st.warning("No efforts with power data found. Ensure your rides are recorded with a power meter.")
    st.stop()

watt_efforts["bike_name"] = watt_efforts["gear_id"].map(_gear_label)

seg_meta = segments[
    ["segment_id", "name", "distance", "average_grade", "total_elevation_gain", "segment_type"]
]
watt_efforts = watt_efforts.merge(seg_meta, on="segment_id", how="inner")
watt_efforts["speed_kmh"] = _compute_speed_kmh(watt_efforts)

available_bikes = sorted(watt_efforts["bike_name"].dropna().unique().tolist())

# ── Sidebar: analysis settings ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Analysis settings")

    if len(available_bikes) < 2:
        st.warning(
            "Need at least 2 bikes with power data to compare. "
            "Ride more segments on different bikes."
        )
        st.stop()

    min_sample_size = st.number_input(
        "Minimum rides per bike per segment",
        min_value=1,
        max_value=20,
        value=2,
        step=1,
        help="Both bikes must have at least this many power-measured rides on a segment.",
    )

    bikes_to_compare = st.multiselect(
        "Bikes to compare",
        options=available_bikes,
        default=available_bikes[:2],
        max_selections=5,
        help="Select 2–5 bikes to compare.",
    )

if len(bikes_to_compare) < 2:
    st.warning("Please select at least **2 bikes** in the sidebar to compare.")
    st.stop()

# ── Compute valid segments ────────────────────────────────────────────────────
selected_efforts = watt_efforts[watt_efforts["bike_name"].isin(bikes_to_compare)].copy()

seg_counts = (
    selected_efforts.groupby(["segment_id", "bike_name"])["effort_id"]
    .count()
    .unstack(fill_value=0)
)
for b in bikes_to_compare:
    if b not in seg_counts.columns:
        seg_counts[b] = 0

# All selected bikes must meet the minimum sample size on a segment
valid_mask = (seg_counts[bikes_to_compare] >= min_sample_size).all(axis=1)
valid_segment_ids = seg_counts[valid_mask].index.tolist()

# Per-bike rides columns for the segment table
rides_cols: dict[str, str] = {b: f"Rides ({b})" for b in bikes_to_compare}
valid_segs = segments[segments["segment_id"].isin(valid_segment_ids)].copy()
if valid_segment_ids:
    valid_segs = valid_segs.merge(
        seg_counts[bikes_to_compare].rename(columns=rides_cols),
        left_on="segment_id",
        right_index=True,
        how="left",
    )

bikes_label = " vs ".join(f"**{b}**" for b in bikes_to_compare)

# ── Performance profile (spider chart) ──────────────────────────────────────
st.subheader("Performance profile")
st.caption(
    f"Average speed across all valid segments per type — {bikes_label}. "
    "When a category has multiple segments the scores are averaged. "
    "A larger covered area indicates overall stronger performance."
)

spider_efforts = selected_efforts[selected_efforts["segment_id"].isin(valid_segment_ids)].copy()
speed_profile: dict[str, list[float]] = {b: [] for b in bikes_to_compare}

for seg_type in SEGMENT_TYPES:
    _type_eff = spider_efforts[spider_efforts["segment_type"] == seg_type].copy()
    for b in bikes_to_compare:
        _b_eff = _type_eff[_type_eff["bike_name"] == b].copy()
        if _b_eff.empty or _b_eff["speed_kmh"].isna().all():
            speed_profile[b].append(0.0)
        else:
            per_seg_avg = _b_eff.groupby("segment_id")["speed_kmh"].mean()
            speed_profile[b].append(_convert_speed(float(per_seg_avg.mean())))

categories = [f"{TYPE_ICONS.get(t, '')} {t.capitalize()}" for t in SEGMENT_TYPES]
categories_closed = categories + [categories[0]]
_spd = _spd_label()

fig_spider = go.Figure()
for idx, b in enumerate(bikes_to_compare):
    vals = speed_profile[b]
    vals_closed = vals + [vals[0]]
    fig_spider.add_trace(
        go.Scatterpolar(
            r=vals_closed,
            theta=categories_closed,
            fill="toself",
            name=b,
            opacity=0.45,
            line={"color": _COLOR_SEQ[idx % len(_COLOR_SEQ)], "width": 2},
            fillcolor=_COLOR_SEQ[idx % len(_COLOR_SEQ)],
            hovertemplate="%{theta}: %{r:.1f} " + _spd + "<extra>" + b + "</extra>",
        )
    )

fig_spider.update_layout(
    polar={"radialaxis": {"visible": True, "title": {"text": f"Avg speed ({_spd})"}}},
    showlegend=True,
    legend={"orientation": "h", "yanchor": "bottom", "y": -0.15},
    title="Speed profile by segment type — " + " vs ".join(bikes_to_compare),
    height=500,
)
spider_col, _ = st.columns([2, 1])
with spider_col:
    st.plotly_chart(fig_spider, width="stretch")

# ── Valid segment selector ────────────────────────────────────────────────────
st.divider()
st.subheader("Valid segments")

if not valid_segment_ids:
    st.info(
        f"No segments where all selected bikes have ≥ {int(min_sample_size)} "
        "power-measured rides. Try reducing the minimum sample size or selecting fewer bikes."
    )
    st.stop()

st.caption(
    f"Segments where all selected bikes have ≥ {int(min_sample_size)} "
    "power-measured rides. Check any segment to open the comparison."
)

# Two rows of two segment types each (2×2 grid)
selected_segment_ids: list[int] = []
all_rides_cols = list(rides_cols.values())

for row_types in [SEGMENT_TYPES[:2], SEGMENT_TYPES[2:]]:
    row_cols = st.columns(2)
    for col, seg_type in zip(row_cols, row_types):
        with col:
            icon = TYPE_ICONS.get(seg_type, "")
            st.markdown(f"**{icon} {seg_type.capitalize()}**")

            type_segs = valid_segs[valid_segs["segment_type"] == seg_type].copy()

            if type_segs.empty:
                st.caption("No valid segments.")
                continue

            display = type_segs[
                ["segment_id", "name", *all_rides_cols, "distance", "average_grade"]
            ].copy()
            _dist_col = f"Dist ({_dist_label()})"
            display["distance"] = (
                (display["distance"] / 1000).round(2)
                if _use_metric()
                else (display["distance"] / 1609.34).round(2)
            )
            display["average_grade"] = display["average_grade"].round(1)
            display.insert(0, "Select", False)
            display = display.rename(
                columns={
                    "name": "Segment",
                    "distance": _dist_col,
                    "average_grade": "Grade (%)",
                }
            )

            edited = st.data_editor(
                display,
                column_config={
                    "Select": st.column_config.CheckboxColumn("✓", default=False),
                    "segment_id": None,
                },
                hide_index=True,
                width="stretch",
                key=f"table_{seg_type}",
                disabled=["Segment", *all_rides_cols, _dist_col, "Grade (%)"],
            )


            selected_rows = edited[edited["Select"]]
            if not selected_rows.empty:
                selected_segment_ids.extend(selected_rows["segment_id"].tolist())

# ── Comparison detail ─────────────────────────────────────────────────────────
if not selected_segment_ids:
    st.caption("☝️ Check any segment above to open the bike-to-bike comparison.")
else:
    st.divider()
    st.subheader("Bike comparison — " + " vs ".join(bikes_to_compare))

    tab_labels = []
    for sid in selected_segment_ids:
        row = segments[segments["segment_id"] == sid]
        tab_labels.append(row["name"].iloc[0] if not row.empty else str(sid))

    seg_tabs = st.tabs(tab_labels)

    for seg_tab, seg_id in zip(seg_tabs, selected_segment_ids):
        with seg_tab:
            seg_row = segments[segments["segment_id"] == seg_id].iloc[0]
            seg_distance_m: float = float(seg_row.get("distance", 0) or 0)

            seg_efforts = selected_efforts[selected_efforts["segment_id"] == seg_id].copy()
            if seg_efforts.empty:
                st.info("No power-measured efforts found for this segment.")
                continue

            seg_efforts["speed_kmh"] = _compute_speed_kmh(seg_efforts, distance_m=seg_distance_m)
            seg_efforts["start_date"] = pd.to_datetime(seg_efforts["start_date"], errors="coerce")
            seg_efforts["date_str"] = seg_efforts["start_date"].dt.strftime("%Y-%m-%d")

            # Segment info metrics
            info_cols = st.columns(4)
            with info_cols[0]:
                if seg_distance_m:
                    dist_disp = _convert_dist_m(seg_distance_m)
                    dist_metric_str = f"{dist_disp:.2f} {_dist_label()}"
                else:
                    dist_metric_str = "—"
                st.metric("Distance", dist_metric_str)
            with info_cols[1]:
                grade = seg_row.get("average_grade")
                st.metric("Avg grade", f"{grade:.1f}%" if pd.notna(grade) else "—")
            with info_cols[2]:
                elev = seg_row.get("total_elevation_gain")
                if pd.notna(elev):
                    elev_disp = _convert_elev_m(float(elev))
                    elev_metric_str = f"{elev_disp:.0f} {_elev_label()}"
                else:
                    elev_metric_str = "—"
                st.metric("Elevation gain", elev_metric_str)
            with info_cols[3]:
                stype = seg_row.get("segment_type", "—")
                st.metric("Type", str(stype).capitalize() if stype else "—")

            # Map + elevation
            with st.spinner("Loading segment map…"):
                geo = _get_segment_geo(int(seg_id))

            map_col, elev_col = st.columns(2)
            with map_col:
                st.markdown("**Route**")
                _render_segment_map(geo, str(seg_row.get("name", "")))
            with elev_col:
                st.markdown("**Elevation profile**")
                _render_elevation_profile(geo, seg_distance_m)

            # Summary table
            agg: dict[str, tuple] = {
                "Rides": ("effort_id", "count"),
                "Best time": ("moving_time", "min"),
                "Avg time": ("moving_time", "mean"),
                "Avg power (W)": ("average_watts", "mean"),
                "Max power (W)": ("average_watts", "max"),
            }
            if _has_col(seg_efforts, "average_heartrate"):
                agg["Avg HR (bpm)"] = ("average_heartrate", "mean")
            _spd_col_avg = f"Avg speed ({_spd_label()})"
            _spd_col_max = f"Max speed ({_spd_label()})"
            if _has_col(seg_efforts, "speed_kmh"):
                agg[_spd_col_avg] = ("speed_kmh", "mean")
                agg[_spd_col_max] = ("speed_kmh", "max")

            summary = seg_efforts.groupby("bike_name").agg(**agg).reset_index()
            summary.rename(columns={"bike_name": "Bike"}, inplace=True)

            # Convert speed columns to display unit
            for col_name in [_spd_col_avg, _spd_col_max]:
                if col_name in summary.columns:
                    summary[col_name] = summary[col_name].apply(
                        lambda v: _convert_speed(v) if pd.notna(v) else v
                    )

            for col_name in ["Best time", "Avg time"]:
                if col_name in summary.columns:
                    summary[col_name] = summary[col_name].apply(
                        lambda s: _fmt_duration(s) if pd.notna(s) else "—"
                    )
            for col_name in ["Avg power (W)", "Max power (W)", "Avg HR (bpm)",
                             _spd_col_avg, _spd_col_max]:
                if col_name in summary.columns:
                    summary[col_name] = summary[col_name].apply(
                        lambda v: f"{v:.1f}" if pd.notna(v) else "—"
                    )

            st.dataframe(summary, width="stretch", hide_index=True)

            # Metric tabs: Speed | Power | Heart Rate | Timeline
            chart_tab_labels: list[str] = []
            has_speed = _has_col(seg_efforts, "speed_kmh")
            has_watts = _has_col(seg_efforts, "average_watts")
            has_hr = _has_col(seg_efforts, "average_heartrate")

            if has_speed:
                chart_tab_labels.append("🚀 Speed")
            if has_watts:
                chart_tab_labels.append("⚡ Power")
            if has_hr:
                chart_tab_labels.append("❤️ Heart rate")
            chart_tab_labels.append("📅 Timeline")

            chart_tabs = st.tabs(chart_tab_labels)
            ct_idx = 0

            if has_speed:
                with chart_tabs[ct_idx]:
                    _spd_display = seg_efforts.dropna(subset=["speed_kmh"]).copy()
                    _spd_display["speed_kmh"] = _spd_display["speed_kmh"].apply(_convert_speed)
                    _spd_axis = f"Speed ({_spd_label()})"
                    st.caption(f"Speed distribution by bike ({_spd_label()})")
                    fig = px.box(
                        _spd_display,
                        x="bike_name",
                        y="speed_kmh",
                        color="bike_name",
                        color_discrete_sequence=_COLOR_SEQ,
                        labels={"bike_name": "Bike", "speed_kmh": _spd_axis},
                        points="all",
                    )
                    fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, width="stretch")
                ct_idx += 1

            if has_watts:
                with chart_tabs[ct_idx]:
                    st.caption("Power distribution by bike (W)")
                    fig = px.box(
                        seg_efforts.dropna(subset=["average_watts"]),
                        x="bike_name",
                        y="average_watts",
                        color="bike_name",
                        color_discrete_sequence=_COLOR_SEQ,
                        labels={"bike_name": "Bike", "average_watts": "Power (W)"},
                        points="all",
                    )
                    fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, width="stretch")
                ct_idx += 1

            if has_hr:
                with chart_tabs[ct_idx]:
                    st.caption("Heart rate distribution by bike (bpm)")
                    fig = px.box(
                        seg_efforts.dropna(subset=["average_heartrate"]),
                        x="bike_name",
                        y="average_heartrate",
                        color="bike_name",
                        color_discrete_sequence=_COLOR_SEQ,
                        labels={"bike_name": "Bike", "average_heartrate": "Heart rate (bpm)"},
                        points="all",
                    )
                    fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, width="stretch")
                ct_idx += 1

            # Timeline tab
            with chart_tabs[ct_idx]:
                st.caption("Performance over time — each dot is one effort")
                timeline_data = seg_efforts.dropna(subset=["start_date"])
                if not timeline_data.empty:
                    y_col = "speed_kmh" if has_speed else "moving_time"
                    y_label = f"Speed ({_spd_label()})" if has_speed else "Time (s)"
                    if has_speed:
                        timeline_data = timeline_data.copy()
                        timeline_data["speed_kmh"] = timeline_data["speed_kmh"].apply(_convert_speed)
                    fig = px.scatter(
                        timeline_data.sort_values("start_date"),
                        x="start_date",
                        y=y_col,
                        color="bike_name",
                        color_discrete_sequence=_COLOR_SEQ,
                        labels={"start_date": "Date", y_col: y_label, "bike_name": "Bike"},
                        hover_data={"date_str": True, "average_watts": True, "average_heartrate": True},
                    )
                    fig.update_traces(marker_size=8)
                    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.caption("No date information available.")

            # Individual efforts expander
            with st.expander("All efforts for this segment", expanded=False):
                disp_cols = ["date_str", "bike_name", "moving_time", "speed_kmh",
                             "average_watts", "average_heartrate"]
                available = [c for c in disp_cols if c in seg_efforts.columns]
                sort_col = ["start_date"] if "start_date" in seg_efforts.columns else []
                detail = (
                    seg_efforts[available + sort_col].copy()
                    .sort_values(sort_col, ascending=False)
                    .drop(columns=sort_col)
                ) if sort_col else seg_efforts[available].copy()

                if "speed_kmh" in detail.columns:
                    detail["speed_kmh"] = detail["speed_kmh"].apply(
                        lambda v: _convert_speed(v) if pd.notna(v) else v
                    )
                detail.rename(columns={
                    "date_str": "Date",
                    "bike_name": "Bike",
                    "moving_time": "Time (s)",
                    "speed_kmh": f"Speed ({_spd_label()})",
                    "average_watts": "Avg power (W)",
                    "average_heartrate": "Avg HR (bpm)",
                }, inplace=True)

                if "Time (s)" in detail.columns:
                    detail["Time (s)"] = detail["Time (s)"].apply(
                        lambda s: _fmt_duration(s) if pd.notna(s) else "—"
                    )
                    detail.rename(columns={"Time (s)": "Time"}, inplace=True)

                st.dataframe(detail, width="stretch", hide_index=True)


