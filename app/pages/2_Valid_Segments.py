"""Valid Segments Analysis – find segments where both bikes have sufficient data to compare."""

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

st.set_page_config(page_title="Valid Segments – New Bike Day", page_icon="🔍", layout="wide")
st.title("🔍 Valid Segment Analysis")
st.caption(
    "Find segments where you can meaningfully compare two bikes — "
    "both must have enough power-measured rides on the same segment."
)

# ── Check session state ──────────────────────────────────────────────────────
efforts: pd.DataFrame | None = st.session_state.get("efforts")
segments: pd.DataFrame | None = st.session_state.get("segments")
bikes: dict[str, str] = st.session_state.get("bikes", {})

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


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as m:ss."""
    total = int(round(seconds))
    return f"{total // 60}:{total % 60:02d}"


def _has_speed(df: pd.DataFrame) -> bool:
    """Return True when the DataFrame has a populated ``speed_kmh`` column."""
    return "speed_kmh" in df.columns and df["speed_kmh"].notna().any()


def _compute_speed_kmh(df: pd.DataFrame, distance_m: float | None = None) -> pd.Series:
    """Compute speed (km/h) from moving_time and an optional fixed distance.

    If ``distance_m`` is provided (e.g. the known segment distance) it is used
    for every row.  Otherwise the per-row ``distance`` column is used.
    """
    if distance_m is not None and distance_m > 0:
        safe_time = df["moving_time"].replace(0, pd.NA)
        return (distance_m / safe_time * 3.6).where(safe_time.notna())
    dist = df.get("distance", pd.Series(dtype=float))
    safe_time = df["moving_time"].replace(0, pd.NA)
    return (dist / safe_time * 3.6).where(safe_time.notna() & dist.notna())


# ── Filter to watt-measured efforts ─────────────────────────────────────────
watt_efforts = efforts[efforts["average_watts"].notna()].copy()

if watt_efforts.empty:
    st.warning(
        "No efforts with power data found. Ensure your rides are recorded with a power meter."
    )
    st.stop()

# Add bike label
watt_efforts["bike_name"] = watt_efforts["gear_id"].map(_gear_label)

# Join with segment metadata (segment_type, distance, grade, elevation)
seg_meta = segments[
    ["segment_id", "name", "distance", "average_grade", "total_elevation_gain", "segment_type"]
]
watt_efforts = watt_efforts.merge(seg_meta, on="segment_id", how="inner")

# Compute speed in km/h from segment distance and moving time
watt_efforts["speed_kmh"] = _compute_speed_kmh(watt_efforts)

# ── Sidebar: analysis settings ────────────────────────────────────────────────
available_bikes = sorted(watt_efforts["bike_name"].dropna().unique().tolist())

with st.sidebar:
    st.markdown("### ⚙️ Analysis Settings")

    if len(available_bikes) < 2:
        st.warning(
            "Need at least 2 bikes with power data. "
            "Ride more segments on different bikes to enable comparison."
        )
        st.stop()

    min_sample_size = st.number_input(
        "Minimum rides per bike per segment",
        min_value=1,
        max_value=20,
        value=2,
        step=1,
        help=(
            "Only include segments where both selected bikes have at least "
            "this many rides with power data."
        ),
    )

    bikes_to_compare = st.multiselect(
        "Bikes to compare",
        options=available_bikes,
        default=available_bikes[:2],
        max_selections=2,
        help="Select exactly 2 bikes to compare.",
    )

if len(bikes_to_compare) != 2:
    st.warning("Please select exactly **2 bikes** to compare in the sidebar.")
    st.stop()

bike_a, bike_b = bikes_to_compare[0], bikes_to_compare[1]

# ── Compute valid segments ────────────────────────────────────────────────────
# Filter efforts for the two selected bikes only
selected_efforts = watt_efforts[watt_efforts["bike_name"].isin([bike_a, bike_b])].copy()

# Count power-measured rides per (segment, bike)
seg_counts = (
    selected_efforts.groupby(["segment_id", "bike_name"])["effort_id"]
    .count()
    .unstack(fill_value=0)
)
for b in [bike_a, bike_b]:
    if b not in seg_counts.columns:
        seg_counts[b] = 0

# A segment is "valid" when both bikes meet the minimum sample size
valid_mask = (seg_counts[bike_a] >= min_sample_size) & (seg_counts[bike_b] >= min_sample_size)
valid_segment_ids = seg_counts[valid_mask].index.tolist()

if not valid_segment_ids:
    st.info(
        f"No segments found where both **{bike_a}** and **{bike_b}** have at least "
        f"**{int(min_sample_size)}** rides with power data. "
        "Try reducing the minimum sample size."
    )
    st.stop()

# Build the display table for valid segments (one row per segment)
rides_col_a = f"Rides ({bike_a})"
rides_col_b = f"Rides ({bike_b})"
valid_segs = segments[segments["segment_id"].isin(valid_segment_ids)].copy()
valid_segs = valid_segs.merge(
    seg_counts[[bike_a, bike_b]].rename(
        columns={bike_a: rides_col_a, bike_b: rides_col_b}
    ),
    left_on="segment_id",
    right_index=True,
    how="left",
)

# ── Segment type constants (ordered for both tables and spider plot) ──────────
SEGMENT_TYPES: list[str] = ["sprint", "flat", "ascent", "descent"]
TYPE_ICONS: dict[str, str] = {
    "sprint": "⚡",
    "flat": "➡️",
    "ascent": "⬆️",
    "descent": "⬇️",
}

# ── Display 4 segment-type tables ────────────────────────────────────────────

st.markdown("---")
st.subheader("📋 Valid Segments by Type")
st.caption(
    f"Segments where both **{bike_a}** and **{bike_b}** have ≥ {int(min_sample_size)} "
    "power-measured rides. Select segments below to compare them."
)

# Collect segment_ids selected via checkbox across all type tables
selected_segment_ids: list[int] = []

for seg_type in SEGMENT_TYPES:
    type_segs = valid_segs[valid_segs["segment_type"] == seg_type].copy()
    if type_segs.empty:
        continue

    icon = TYPE_ICONS.get(seg_type, "")
    st.markdown(f"#### {icon} {seg_type.capitalize()} Segments")

    display = type_segs[
        ["segment_id", "name", rides_col_a, rides_col_b, "distance", "total_elevation_gain", "average_grade"]
    ].copy()
    display["distance"] = (display["distance"] / 1000).round(2)
    display["total_elevation_gain"] = display["total_elevation_gain"].round(0)
    display["average_grade"] = display["average_grade"].round(1)
    display.insert(0, "Select", False)
    display = display.rename(
        columns={
            "name": "Segment",
            "distance": "Distance (km)",
            "total_elevation_gain": "Elevation Gain (m)",
            "average_grade": "Grade (%)",
            "segment_id": "ID",
        }
    )

    edited = st.data_editor(
        display,
        column_config={
            "Select": st.column_config.CheckboxColumn("Select", default=False),
            "ID": st.column_config.NumberColumn("ID", disabled=True),
        },
        hide_index=True,
        use_container_width=True,
        key=f"table_{seg_type}",
        disabled=[
            "Segment",
            rides_col_a,
            rides_col_b,
            "Distance (km)",
            "Elevation Gain (m)",
            "Grade (%)",
        ],
    )

    selected_rows = edited[edited["Select"]]
    if not selected_rows.empty:
        selected_segment_ids.extend(selected_rows["ID"].tolist())

# ── Comparison tabs for selected segments ────────────────────────────────────
if not selected_segment_ids:
    st.info("☝️ Check the **Select** box on any segment above to compare the two bikes.")
    st.stop()

st.markdown("---")
st.subheader(f"📊 Bike-to-Bike Comparison: {bike_a} vs {bike_b}")

tab_labels = []
for sid in selected_segment_ids:
    row = segments[segments["segment_id"] == sid]
    tab_labels.append(row["name"].iloc[0] if not row.empty else str(sid))

tabs = st.tabs(tab_labels)

for tab, seg_id in zip(tabs, selected_segment_ids):
    with tab:
        seg_row = segments[segments["segment_id"] == seg_id].iloc[0]
        seg_distance_m: float = float(seg_row.get("distance", 0) or 0)

        # Filter to this segment, both selected bikes, with watt data
        seg_efforts = selected_efforts[selected_efforts["segment_id"] == seg_id].copy()
        if seg_efforts.empty:
            st.info("No power-measured efforts found for this segment.")
            continue

        # Compute speed if segment distance is known
        if seg_distance_m > 0:
            safe_time = seg_efforts["moving_time"].replace(0, pd.NA)
            seg_efforts["speed_kmh"] = seg_distance_m / safe_time * 3.6
        else:
            seg_efforts["speed_kmh"] = None

        # ── Segment info metrics
        info_cols = st.columns(4)
        with info_cols[0]:
            st.metric("Distance", f"{seg_distance_m / 1000:.2f} km" if seg_distance_m else "—")
        with info_cols[1]:
            grade = seg_row.get("average_grade")
            st.metric("Avg Grade", f"{grade:.1f}%" if pd.notna(grade) else "—")
        with info_cols[2]:
            elev = seg_row.get("total_elevation_gain")
            st.metric("Elevation Gain", f"{elev:.0f} m" if pd.notna(elev) else "—")
        with info_cols[3]:
            stype = seg_row.get("segment_type", "—")
            st.metric("Type", str(stype).capitalize() if stype else "—")

        # ── Summary table
        agg: dict[str, tuple] = {
            "Rides": ("effort_id", "count"),
            "Best Time": ("moving_time", "min"),
            "Avg Time": ("moving_time", "mean"),
            "Avg Power (W)": ("average_watts", "mean"),
            "Max Power (W)": ("average_watts", "max"),
        }
        if seg_efforts["average_heartrate"].notna().any():
            agg["Avg HR (bpm)"] = ("average_heartrate", "mean")
        if _has_speed(seg_efforts):
            agg["Max Speed (km/h)"] = ("speed_kmh", "max")
            agg["Avg Speed (km/h)"] = ("speed_kmh", "mean")

        summary = seg_efforts.groupby("bike_name").agg(**agg).reset_index()
        summary.rename(columns={"bike_name": "Bike"}, inplace=True)

        for col in ["Best Time", "Avg Time"]:
            if col in summary.columns:
                summary[col] = summary[col].apply(
                    lambda s: _fmt_duration(s) if pd.notna(s) else "—"
                )
        for col in ["Avg Power (W)", "Max Power (W)", "Avg HR (bpm)", "Max Speed (km/h)", "Avg Speed (km/h)"]:
            if col in summary.columns:
                summary[col] = summary[col].apply(
                    lambda v: f"{v:.1f}" if pd.notna(v) else "—"
                )

        st.dataframe(summary, use_container_width=True, hide_index=True)

        # ── Charts
        color_seq = px.colors.qualitative.Set2
        chart_cols = st.columns(2)

        with chart_cols[0]:
            if seg_efforts["average_watts"].notna().any():
                fig = px.box(
                    seg_efforts.dropna(subset=["average_watts"]),
                    x="bike_name",
                    y="average_watts",
                    color="bike_name",
                    color_discrete_sequence=color_seq,
                    labels={"bike_name": "Bike", "average_watts": "Power (W)"},
                    title="Power Distribution",
                    points="all",
                )
                fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

        with chart_cols[1]:
            if _has_speed(seg_efforts):
                fig = px.box(
                    seg_efforts.dropna(subset=["speed_kmh"]),
                    x="bike_name",
                    y="speed_kmh",
                    color="bike_name",
                    color_discrete_sequence=color_seq,
                    labels={"bike_name": "Bike", "speed_kmh": "Speed (km/h)"},
                    title="Speed Distribution",
                    points="all",
                )
                fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

# ── Spider / Star Plot ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🕷️ Speed Profile by Segment Type")
st.caption(
    "Maximum speed achieved by each bike across all segment types. "
    "Both bikes are shown with low opacity so they can be compared directly."
)

speed_data: dict[str, list[float]] = {bike_a: [], bike_b: []}
for seg_type in SEGMENT_TYPES:
    type_efforts = watt_efforts[watt_efforts["segment_type"] == seg_type]
    for b in [bike_a, bike_b]:
        b_efforts = type_efforts[type_efforts["bike_name"] == b]
        has_spd = not b_efforts.empty and b_efforts["speed_kmh"].notna().any()
        max_spd = b_efforts["speed_kmh"].max() if has_spd else 0.0
        speed_data[b].append(float(max_spd) if pd.notna(max_spd) else 0.0)

categories = [t.capitalize() for t in SEGMENT_TYPES]
# Close the polygon by repeating the first value
categories_closed = categories + [categories[0]]

color_seq = px.colors.qualitative.Set2
fig_spider = go.Figure()

for idx, b in enumerate([bike_a, bike_b]):
    values_closed = speed_data[b] + [speed_data[b][0]]
    fig_spider.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill="toself",
            name=b,
            opacity=0.45,
            line={"color": color_seq[idx % len(color_seq)], "width": 2},
            fillcolor=color_seq[idx % len(color_seq)],
        )
    )

fig_spider.update_layout(
    polar={
        "radialaxis": {
            "visible": True,
            "title": {"text": "Max Speed (km/h)"},
        }
    },
    showlegend=True,
    title=f"Max Speed by Segment Type — {bike_a} vs {bike_b}",
    height=500,
)

col_spider, _ = st.columns([2, 1])
with col_spider:
    st.plotly_chart(fig_spider, use_container_width=True)
