"""Streamlit segment comparison page – compare gear performance on the same segment."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure `src` imports work when launching Streamlit from different working directories.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.fetch import get_segment_detail, get_segment_streams

st.set_page_config(page_title="Segment Comparison – New Bike Day", page_icon="📊", layout="wide")
st.title("📊 Segment Comparison")
st.caption("Select a starred segment and compare your efforts across different bikes.")

# ── Check session state ──────────────────────────────────────────────────────
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

# ── Helpers ──────────────────────────────────────────────────────────────────

def _gear_label(gear_id: str | None) -> str:
    """Return a human-readable bike name, falling back to the gear_id."""
    if gear_id is None:
        return "Unknown"
    return bikes.get(str(gear_id), str(gear_id))


def _fmt_pace(seconds_per_km: float) -> str:
    """Format seconds-per-km as m:ss /km."""
    total = int(round(seconds_per_km))
    return f"{total // 60}:{total % 60:02d} /km"


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as m:ss."""
    total = int(round(seconds))
    return f"{total // 60}:{total % 60:02d}"


def _get_segment_geo(segment_id: int) -> dict:
    """Return cached segment geo data, fetching from Strava if needed."""
    cache_key = f"segment_geo_{segment_id}"
    cached = st.session_state.get(cache_key)
    if cached is not None:
        return cached

    if not access_token:
        st.session_state[cache_key] = {}
        return {}

    detail = get_segment_detail(access_token, segment_id)
    streams = get_segment_streams(access_token, segment_id)
    result = {**detail, "streams": streams}
    st.session_state[cache_key] = result
    return result


def _render_segment_map(geo: dict, seg_name: str) -> None:
    """Render a Plotly minimap of the segment route."""
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
            go.Scattermapbox(
                lat=lats,
                lon=lngs,
                mode="lines",
                line={"width": 4, "color": "#FC4C02"},  # Strava orange
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

    # Start marker
    if start_ll and len(start_ll) == 2:
        traces.append(
            go.Scattermapbox(
                lat=[float(start_ll[0])],
                lon=[float(start_ll[1])],
                mode="markers+text",
                marker={"size": 14, "color": "#22c55e", "symbol": "circle"},
                text=["Start"],
                textposition="top right",
                name="Start",
                hoverinfo="text",
            )
        )

    # End marker
    if end_ll and len(end_ll) == 2:
        traces.append(
            go.Scattermapbox(
                lat=[float(end_ll[0])],
                lon=[float(end_ll[1])],
                mode="markers+text",
                marker={"size": 14, "color": "#ef4444", "symbol": "circle"},
                text=["Finish"],
                textposition="top right",
                name="Finish",
                hoverinfo="text",
            )
        )

    fig = go.Figure(traces)
    fig.update_layout(
        mapbox={
            "style": "open-street-map",
            "center": {"lat": center_lat, "lon": center_lng},
            "zoom": 13,
        },
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=300,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})


def _render_elevation_profile(geo: dict, seg_distance_m: float) -> None:
    """Render the elevation profile for the segment."""
    streams = geo.get("streams") or {}
    elev_low = geo.get("elevation_low")
    elev_high = geo.get("elevation_high")

    if streams and "distance" in streams and "altitude" in streams:
        # Full altitude stream available
        dist_km = [d / 1000 for d in streams["distance"]]
        alt = streams["altitude"]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dist_km,
                y=alt,
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(252, 76, 2, 0.15)",
                line={"color": "#FC4C02", "width": 2},
                hovertemplate="Distance: %{x:.2f} km<br>Elevation: %{y:.0f} m<extra></extra>",
                name="Elevation",
            )
        )
        fig.update_layout(
            xaxis_title="Distance (km)",
            yaxis_title="Elevation (m)",
            margin={"l": 40, "r": 10, "t": 10, "b": 40},
            height=300,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    elif elev_low is not None and elev_high is not None and seg_distance_m > 0:
        # Approximate schematic: flat line at elev_low → peak → elev_low
        dist_km_total = seg_distance_m / 1000
        mid = dist_km_total / 2
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[0, mid, dist_km_total],
                y=[float(elev_low), float(elev_high), float(elev_low)],
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(252, 76, 2, 0.15)",
                line={"color": "#FC4C02", "width": 2, "dash": "dot"},
                hovertemplate="Distance: %{x:.2f} km<br>~Elevation: %{y:.0f} m<extra></extra>",
                name="Approx. Elevation",
            )
        )
        fig.add_annotation(
            x=mid,
            y=float(elev_high),
            text=f"{elev_high:.0f} m",
            showarrow=True,
            arrowhead=2,
            font={"size": 11},
            ay=-30,
        )
        fig.update_layout(
            xaxis_title="Distance (km)",
            yaxis_title="Elevation (m)",
            margin={"l": 40, "r": 10, "t": 10, "b": 40},
            height=300,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.caption("📐 Approximate elevation profile (detailed stream unavailable)")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.caption("No elevation data available for this segment.")


# ── Segment selector ─────────────────────────────────────────────────────────
seg_options = dict(zip(segments["name"], segments["segment_id"]))
selected_name = st.selectbox(
    "Choose a segment",
    options=list(seg_options.keys()),
    help="Only segments with recorded efforts are meaningful here.",
)

if selected_name is None:
    st.stop()

selected_segment_id = seg_options[selected_name]
seg_row = segments[segments["segment_id"] == selected_segment_id].iloc[0]
seg_distance_m: float = float(seg_row.get("distance", 0) or 0)

# Segment info cards
info_cols = st.columns(4)
with info_cols[0]:
    dist_km = seg_distance_m / 1000 if seg_distance_m else None
    st.metric("Distance", f"{dist_km:.2f} km" if dist_km else "—")
with info_cols[1]:
    grade = seg_row.get("average_grade")
    st.metric("Avg Grade", f"{grade:.1f}%" if pd.notna(grade) else "—")
with info_cols[2]:
    elev = seg_row.get("total_elevation_gain")
    st.metric("Elevation Gain", f"{elev:.0f} m" if pd.notna(elev) else "—")
with info_cols[3]:
    stype = seg_row.get("segment_type", "—")
    st.metric("Type", str(stype).capitalize() if stype else "—")

# ── Minimap + Elevation profile ───────────────────────────────────────────────
st.markdown("---")

with st.spinner("Loading segment map…"):
    geo = _get_segment_geo(int(selected_segment_id))

map_col, elev_col = st.columns(2)
with map_col:
    st.markdown("#### 🗺️ Route")
    _render_segment_map(geo, selected_name)

with elev_col:
    st.markdown("#### 📈 Elevation Profile")
    _render_elevation_profile(geo, seg_distance_m)

st.markdown("---")

# ── Filter efforts for this segment ──────────────────────────────────────────
seg_efforts = efforts[efforts["segment_id"] == selected_segment_id].copy()

if seg_efforts.empty:
    st.info(f"No recorded efforts found for **{selected_name}**.")
    st.stop()

# Derive pace (seconds/km) from moving_time and segment distance
if seg_distance_m > 0:
    safe_time = seg_efforts["moving_time"].replace(0, pd.NA)
    seg_efforts["pace_sec_per_km"] = safe_time / (seg_distance_m / 1000)
    seg_efforts["speed_kmh"] = (seg_distance_m / safe_time * 3.6)
else:
    seg_efforts["pace_sec_per_km"] = None
    seg_efforts["speed_kmh"] = None

seg_efforts["bike_name"] = seg_efforts["gear_id"].map(_gear_label)
seg_efforts["start_date"] = pd.to_datetime(seg_efforts["start_date"], errors="coerce")
seg_efforts["date_str"] = seg_efforts["start_date"].dt.strftime("%Y-%m-%d")

# ── Summary stats per gear ────────────────────────────────────────────────────
st.subheader("📋 Summary by Bike")

agg: dict[str, tuple] = {
    "Efforts": ("effort_id", "count"),
    "Best Time": ("moving_time", "min"),
    "Avg Time": ("moving_time", "mean"),
}
if seg_efforts["average_watts"].notna().any():
    agg["Avg Watts"] = ("average_watts", "mean")
    agg["Max Watts"] = ("average_watts", "max")
if seg_efforts["average_heartrate"].notna().any():
    agg["Avg HR"] = ("average_heartrate", "mean")
if seg_efforts["pace_sec_per_km"].notna().any():
    agg["Best Pace"] = ("pace_sec_per_km", "min")
    agg["Avg Pace"] = ("pace_sec_per_km", "mean")

summary = seg_efforts.groupby("bike_name", dropna=False).agg(**agg).reset_index()
summary.rename(columns={"bike_name": "Bike"}, inplace=True)

# Format time columns
for col in ["Best Time", "Avg Time"]:
    if col in summary.columns:
        summary[col] = summary[col].apply(lambda s: _fmt_duration(s) if pd.notna(s) else "—")

for col in ["Best Pace", "Avg Pace"]:
    if col in summary.columns:
        summary[col] = summary[col].apply(lambda s: _fmt_pace(s) if pd.notna(s) else "—")

for col in ["Avg Watts", "Max Watts"]:
    if col in summary.columns:
        summary[col] = summary[col].apply(lambda v: f"{v:.0f} W" if pd.notna(v) else "—")

if "Avg HR" in summary.columns:
    summary["Avg HR"] = summary["Avg HR"].apply(lambda v: f"{v:.0f} bpm" if pd.notna(v) else "—")

st.dataframe(summary, use_container_width=True, hide_index=True)

# ── Charts ────────────────────────────────────────────────────────────────────
has_pace = seg_efforts["pace_sec_per_km"].notna().any()
has_watts = seg_efforts["average_watts"].notna().any()
has_hr = seg_efforts["average_heartrate"].notna().any()

chart_tab_labels = []
if has_pace:
    chart_tab_labels.append("⏱ Pace")
if has_watts:
    chart_tab_labels.append("⚡ Power")
if has_hr:
    chart_tab_labels.append("❤️ Heart Rate")
chart_tab_labels.append("📅 Timeline")

if chart_tab_labels:
    tabs = st.tabs(chart_tab_labels)
    tab_idx = 0

    color_seq = px.colors.qualitative.Set2

    if has_pace:
        with tabs[tab_idx]:
            st.markdown("**Pace distribution by bike** (lower is faster)")
            fig = px.box(
                seg_efforts.dropna(subset=["pace_sec_per_km"]),
                x="bike_name",
                y="pace_sec_per_km",
                color="bike_name",
                color_discrete_sequence=color_seq,
                labels={"bike_name": "Bike", "pace_sec_per_km": "Pace (sec/km)"},
                points="all",
            )
            fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
            max_pace_tick = int(seg_efforts["pace_sec_per_km"].max() or 300) + 30
            fig.update_yaxes(
                tickvals=list(range(0, max_pace_tick, 30)),
                ticktext=[_fmt_pace(v) for v in range(0, max_pace_tick, 30)],
            )
            st.plotly_chart(fig, use_container_width=True)
        tab_idx += 1

    if has_watts:
        with tabs[tab_idx]:
            st.markdown("**Power distribution by bike**")
            fig = px.box(
                seg_efforts.dropna(subset=["average_watts"]),
                x="bike_name",
                y="average_watts",
                color="bike_name",
                color_discrete_sequence=color_seq,
                labels={"bike_name": "Bike", "average_watts": "Watts"},
                points="all",
            )
            fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        tab_idx += 1

    if has_hr:
        with tabs[tab_idx]:
            st.markdown("**Heart rate distribution by bike**")
            fig = px.box(
                seg_efforts.dropna(subset=["average_heartrate"]),
                x="bike_name",
                y="average_heartrate",
                color="bike_name",
                color_discrete_sequence=color_seq,
                labels={"bike_name": "Bike", "average_heartrate": "Heart Rate (bpm)"},
                points="all",
            )
            fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        tab_idx += 1

    # Timeline tab
    with tabs[tab_idx]:
        st.markdown("**Performance over time** – each dot is one effort")
        timeline_data = seg_efforts.dropna(subset=["start_date"])
        if not timeline_data.empty:
            y_col = "moving_time"
            y_label = "Time (seconds)"
            if has_pace:
                y_col = "pace_sec_per_km"
                y_label = "Pace (sec/km)"

            fig = px.scatter(
                timeline_data.sort_values("start_date"),
                x="start_date",
                y=y_col,
                color="bike_name",
                color_discrete_sequence=color_seq,
                labels={"start_date": "Date", y_col: y_label, "bike_name": "Bike"},
                hover_data={"date_str": True, "average_watts": True, "average_heartrate": True},
            )
            fig.update_traces(marker_size=8)
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
            if has_pace:
                max_pace = int(timeline_data[y_col].max() or 300)
                fig.update_yaxes(
                    tickvals=list(range(0, max_pace + 30, 30)),
                    ticktext=[_fmt_pace(v) for v in range(0, max_pace + 30, 30)],
                )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No date information available to plot timeline.")

# ── Individual efforts table ──────────────────────────────────────────────────
st.markdown("---")
with st.expander("🗂 All efforts for this segment", expanded=False):
    display_cols = ["date_str", "bike_name", "moving_time", "average_watts", "average_heartrate"]
    if has_pace:
        display_cols.insert(3, "pace_sec_per_km")
    available = [c for c in display_cols if c in seg_efforts.columns]
    detail = seg_efforts[available].copy().sort_values("start_date", ascending=False, na_position="last")

    col_rename = {
        "date_str": "Date",
        "bike_name": "Bike",
        "moving_time": "Time (s)",
        "pace_sec_per_km": "Pace (sec/km)",
        "average_watts": "Avg Watts",
        "average_heartrate": "Avg HR",
    }
    detail.rename(columns=col_rename, inplace=True)

    if "Time (s)" in detail.columns:
        detail["Time (s)"] = detail["Time (s)"].apply(lambda s: _fmt_duration(s) if pd.notna(s) else "—")
        detail.rename(columns={"Time (s)": "Time"}, inplace=True)
    if "Pace (sec/km)" in detail.columns:
        detail["Pace (sec/km)"] = detail["Pace (sec/km)"].apply(lambda s: _fmt_pace(s) if pd.notna(s) else "—")
        detail.rename(columns={"Pace (sec/km)": "Pace"}, inplace=True)

    st.dataframe(detail, use_container_width=True, hide_index=True)


