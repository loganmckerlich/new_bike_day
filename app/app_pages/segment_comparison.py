"""Segment comparison page – valid segment selection and bike performance analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.fetch import get_segment_detail, get_segment_streams
from src.database import init_db, load_segment_geo, save_segment_geo
from src.plot_colors import to_rgba
from src.analytics import (
    compute_speed_per_watt,
    filter_outliers_by_power_speed,
    mean_profile_by_segment_type,
    power_normalized_profile,
    outlier_detection_frames,
)

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
SEGMENT_TYPE_DETAILS: list[str] = [
    "sprint_flat",
    "sprint_uphill",
    "sprint_downhill",
    "flat_short",
    "flat_long",
    "ascent_shallow",
    "ascent_moderate",
    "ascent_steep",
    "descent_gentle",
    "descent_steep",
]
TYPE_ICONS: dict[str, str] = {
    "sprint": "⚡",
    "flat": "➡️",
    "ascent": "⬆️",
    "descent": "⬇️",
}
TYPE_DETAIL_LABELS: dict[str, str] = {
    "sprint_flat": "Sprint • Flat",
    "sprint_uphill": "Sprint • Uphill",
    "sprint_downhill": "Sprint • Downhill",
    "flat_short": "Flat • Short",
    "flat_long": "Flat • Long",
    "ascent_shallow": "Ascent • Shallow",
    "ascent_moderate": "Ascent • Moderate",
    "ascent_steep": "Ascent • Steep",
    "descent_gentle": "Descent • Gentle",
    "descent_steep": "Descent • Steep",
}
_COLOR_SEQ: list[str] = px.colors.qualitative.Set2
_SPIDER_POLYGON_LINE_WIDTH: int = 3
_SPIDER_POLYGON_FILL_ALPHA: float = 0.20

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

seg_meta_cols = [
    "segment_id",
    "name",
    "distance",
    "average_grade",
    "total_elevation_gain",
    "segment_type",
]
if "segment_type_detail" in segments.columns:
    seg_meta_cols.append("segment_type_detail")
seg_meta = segments[seg_meta_cols].copy()
if "segment_type_detail" not in seg_meta.columns:
    seg_meta["segment_type_detail"] = seg_meta["segment_type"]
watt_efforts = watt_efforts.merge(seg_meta, on="segment_id", how="inner")
watt_efforts["speed_kmh"] = _compute_speed_kmh(watt_efforts)

available_bikes = sorted(watt_efforts["bike_name"].dropna().unique().tolist())

# ── Sidebar: analysis settings ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Segment settings")

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

    spider_use_subcategories = st.toggle(
        "Use subcategories in spider charts",
        value=False,
        help="Show spider charts by segment subcategory instead of parent category.",
    )

    bikes_to_compare = st.multiselect(
        "Bikes to compare",
        options=available_bikes,
        default=available_bikes[:2],
        max_selections=5,
        help="Select 2–5 bikes to compare.",
    )

# ── Read shared analysis params from session state ────────────────────────────
z_threshold: float = float(st.session_state.get("outlier_z_threshold", 2.0))
exclude_descents: bool = bool(st.session_state.get("exclude_descents", False))
min_watts: int = int(st.session_state.get("min_watts", 0))
descents_exempt_watts: bool = bool(st.session_state.get("descents_exempt_watts", False))

if len(bikes_to_compare) < 2:
    st.warning("Please select at least **2 bikes** in the sidebar to compare.")
    st.stop()

# ── Apply shared analysis filters ─────────────────────────────────────────────
# Minimum watts filter (with optional descent exemption)
if min_watts > 0:
    _is_descent = watt_efforts.get("segment_type", pd.Series("flat", index=watt_efforts.index)) == "descent"
    if descents_exempt_watts:
        watt_efforts = watt_efforts[
            (watt_efforts["average_watts"] >= min_watts) | _is_descent
        ].copy()
    else:
        watt_efforts = watt_efforts[watt_efforts["average_watts"] >= min_watts].copy()

# Exclude descent segments entirely
if exclude_descents:
    _seg_type = watt_efforts.get("segment_type", pd.Series("flat", index=watt_efforts.index))
    watt_efforts = watt_efforts[_seg_type != "descent"].copy()

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
if "segment_type_detail" not in valid_segs.columns:
    valid_segs["segment_type_detail"] = valid_segs["segment_type"]
if valid_segment_ids:
    valid_segs = valid_segs.merge(
        seg_counts[bikes_to_compare].rename(columns=rides_cols),
        left_on="segment_id",
        right_index=True,
        how="left",
    )

bikes_label = " vs ".join(f"**{b}**" for b in bikes_to_compare)

# ── Performance profile (spider charts) ──────────────────────────────────────
st.subheader("Performance profile")
st.caption(
    f"{bikes_label} — axes are normalised (0–100) so the full chart area is used; "
    "gaps between bikes are proportional to real differences. "
    "Left shows speed, right shows speed / W\u00b9\u2044\u00b3 (power-normalised efficiency using cube-root scaling). "
    "Hover to see actual values."
)

spider_efforts = selected_efforts[selected_efforts["segment_id"].isin(valid_segment_ids)].copy()
spider_efforts = compute_speed_per_watt(spider_efforts)
spider_filtered, _ = filter_outliers_by_power_speed(spider_efforts, z_threshold=z_threshold)

spider_dimension_col = "segment_type_detail" if spider_use_subcategories else "segment_type"
spider_dimensions = SEGMENT_TYPE_DETAILS if spider_use_subcategories else SEGMENT_TYPES
if spider_use_subcategories:
    categories = [TYPE_DETAIL_LABELS.get(t, t.replace("_", " ").title()) for t in spider_dimensions]
else:
    categories = [f"{TYPE_ICONS.get(t, '')} {t.capitalize()}" for t in spider_dimensions]
categories_closed = categories + [categories[0]]

# ── Chart 1: raw speed ────────────────────────────────────────────────────────
speed_profile = mean_profile_by_segment_type(
    spider_efforts,
    bikes_to_compare,
    spider_dimensions,
    valid_segment_ids,
    value_col="speed_kmh",
    segment_type_col=spider_dimension_col,
)

_spd = _spd_label()


def _normalize_profile(profile: dict[str, list[float]]) -> dict[str, list[float]]:
    """Global min-max normalize so all values map to [0, 100].

    Using a single global scale across all bikes and dimensions preserves the
    proportional gaps between bikes: if bike A is 10 km/h faster on sprints and
    5 km/h faster on hills, the radial distance on hills will be exactly half
    that of sprints.
    """
    all_vals = [v for vals in profile.values() for v in vals if v is not None and not (v != v)]
    if not all_vals:
        return profile
    lo, hi = min(all_vals), max(all_vals)
    if hi == lo:
        return {b: [50.0] * len(vals) for b, vals in profile.items()}
    return {
        b: [10.0 + (v - lo) / (hi - lo) * 90 for v in vals]
        for b, vals in profile.items()
    }


# Convert speed values for display, then normalize for radial position
speed_display = {b: [_convert_speed(v) for v in speed_profile[b]] for b in bikes_to_compare}
speed_norm = _normalize_profile(speed_display)

fig_spider = go.Figure()
for idx, b in enumerate(bikes_to_compare):
    norm_vals = speed_norm[b]
    raw_vals = speed_display[b]
    norm_closed = norm_vals + [norm_vals[0]]
    raw_closed = raw_vals + [raw_vals[0]]
    color = _COLOR_SEQ[idx % len(_COLOR_SEQ)]
    fig_spider.add_trace(
        go.Scatterpolar(
            r=norm_closed,
            theta=categories_closed,
            fill="toself",
            name=b,
            line={"color": color, "width": _SPIDER_POLYGON_LINE_WIDTH},
            fillcolor=to_rgba(color, _SPIDER_POLYGON_FILL_ALPHA),
            customdata=[f"{v:.1f} {_spd}" for v in raw_closed],
            hovertemplate="%{theta}: %{customdata}<extra>" + b + "</extra>",
        )
    )
fig_spider.update_layout(
    polar={
        "radialaxis": {"visible": True, "tickvals": [0, 25, 50, 75, 100], "ticktext": ["", "", "", "", ""], "range": [0, 100]},
        "angularaxis": {"categoryorder": "array", "categoryarray": categories},
    },
    showlegend=True,
    legend={"orientation": "h", "yanchor": "bottom", "y": -0.15},
    title=f"Speed profile by segment {'subcategory' if spider_use_subcategories else 'type'}",
    height=500,
)

# ── Chart 2: power-normalised efficiency ──────────────────────────────────────
eff_profile = power_normalized_profile(
    spider_filtered,
    bikes_to_compare,
    spider_dimensions,
    valid_segment_ids,
    segment_type_col=spider_dimension_col,
)

eff_display = {b: [_convert_speed(v) for v in eff_profile[b]] for b in bikes_to_compare}
eff_norm = _normalize_profile(eff_display)

fig_efficiency = go.Figure()
for idx, b in enumerate(bikes_to_compare):
    norm_vals = eff_norm[b]
    raw_vals = eff_display[b]
    norm_closed = norm_vals + [norm_vals[0]]
    raw_closed = raw_vals + [raw_vals[0]]
    color = _COLOR_SEQ[idx % len(_COLOR_SEQ)]
    fig_efficiency.add_trace(
        go.Scatterpolar(
            r=norm_closed,
            theta=categories_closed,
            fill="toself",
            name=b,
            line={"color": color, "width": _SPIDER_POLYGON_LINE_WIDTH},
            fillcolor=to_rgba(color, _SPIDER_POLYGON_FILL_ALPHA),
            customdata=[f"{v:.4f} {_spd}/W\u00b9\u141f\u00b3" for v in raw_closed],
            hovertemplate="%{theta}: %{customdata}<extra>" + b + "</extra>",
            showlegend=False,
        )
    )
fig_efficiency.update_layout(
    polar={
        "radialaxis": {"visible": True, "tickvals": [0, 25, 50, 75, 100], "ticktext": ["", "", "", "", ""], "range": [0, 100]},
        "angularaxis": {"categoryorder": "array", "categoryarray": categories},
    },
    showlegend=False,
    title=f"Efficiency profile by segment {'subcategory' if spider_use_subcategories else 'type'} (power-normalised)",
    height=500,
)

spider_col1, spider_col2 = st.columns(2)
with spider_col1:
    st.plotly_chart(fig_spider, width="stretch")
with spider_col2:
    st.plotly_chart(fig_efficiency, width="stretch")

# ── Methodology explainer ─────────────────────────────────────────────────────
with st.expander("🔬 How is this calculated?"):
    # pick an example segment from valid segments (prefer one with most efforts)
    _seg_effort_counts = (
        spider_efforts.groupby("segment_id")["effort_id"].count().reindex(valid_segment_ids, fill_value=0)
    )
    _default_seg_id = int(_seg_effort_counts.idxmax()) if not _seg_effort_counts.empty else None

    _seg_name_map = {
        int(row["segment_id"]): row["name"]
        for _, row in segments[segments["segment_id"].isin(valid_segment_ids)].iterrows()
    }
    _seg_options = [
        (sid, _seg_name_map.get(sid, str(sid)))
        for sid in _seg_effort_counts.sort_values(ascending=False).index
    ]

    if not _seg_options:
        st.info("No valid segments available to illustrate.")
    else:
        _example_seg_id = st.selectbox(
            "Example segment",
            options=[s[0] for s in _seg_options],
            format_func=lambda sid: _seg_name_map.get(sid, str(sid)),
            index=0,
            key="explainer_seg",
        )

        _step = st.radio(
            "Step",
            options=[
                "1 — Raw efforts",
                "2 — Outlier detection",
                "3 — After filtering",
                "4 — Efficiency metric",
            ],
            horizontal=True,
            key="explainer_step",
        )

        _raw, _annotated, _filtered_seg = outlier_detection_frames(
            spider_efforts, int(_example_seg_id), z_threshold=z_threshold
        )
        _seg_dist_m = float(
            segments.loc[segments["segment_id"] == _example_seg_id, "distance"].iloc[0]
            if not segments[segments["segment_id"] == _example_seg_id].empty
            else 0
        )
        for _df in [_raw, _annotated, _filtered_seg]:
            if "speed_kmh" not in _df.columns:
                _df["speed_kmh"] = _compute_speed_kmh(_df, distance_m=_seg_dist_m)

        _n_total = len(_raw)
        _n_outliers = int(_annotated["is_outlier"].sum()) if "is_outlier" in _annotated.columns else 0
        _n_kept = _n_total - _n_outliers

        if _step == "1 — Raw efforts":
            st.caption(
                f"**{_n_total} efforts** recorded on this segment across all selected bikes. "
                "Each point is one attempt. More power should mean more speed — but real data "
                "is noisy (drafting, wind, fatigue)."
            )
            _fig_raw = px.scatter(
                _raw.dropna(subset=["speed_kmh", "average_watts"]),
                x="average_watts",
                y="speed_kmh",
                color="bike_name",
                color_discrete_sequence=_COLOR_SEQ,
                labels={"average_watts": "Avg power (W)", "speed_kmh": f"Speed ({_spd_label()})", "bike_name": "Bike"},
                hover_data={"bike_name": True},
            )
            _fig_raw.update_traces(marker_size=9)
            _fig_raw.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=380)
            st.plotly_chart(_fig_raw, width="stretch")

        elif _step == "2 — Outlier detection":
            st.caption(
                f"We compute **speed / power\u00b9\u2044\u00b3** (speed ÷ power^\u00b9\u2044\u00b3) for every effort — "
                "because in aerodynamics speed scales as the cube root of power (v ∝ P^\u00b9\u2044\u00b3), "
                "this ratio is approximately constant for a given bike and conditions. "
                f"Efforts that deviate more than **{z_threshold:.1f} standard deviations** from the "
                "segment mean are flagged as likely outliers (drafting, strong headwind, etc.) — "
                "situations where the bike isn't the main variable."
            )
            if "is_outlier" not in _annotated.columns:
                st.info("Not enough efforts to detect outliers on this segment.")
            else:
                _plot_ann = _annotated.dropna(subset=["speed_kmh", "average_watts"]).copy()
                _plot_ann["label"] = _plot_ann["is_outlier"].map(
                    {True: "Outlier", False: "Normal"}
                )
                _plot_ann["z_label"] = _plot_ann["z_score"].apply(
                    lambda z: f"z = {z:.2f}" if pd.notna(z) else ""
                )

                _ann_bikes = [
                    b for b in bikes_to_compare
                    if b in _plot_ann["bike_name"].values
                ]
                for _bi, _bname in enumerate(_ann_bikes):
                    _bike_color = _COLOR_SEQ[_bi % len(_COLOR_SEQ)]
                    _bdata = _plot_ann[_plot_ann["bike_name"] == _bname].copy()
                    _n_b_out = int(_bdata["is_outlier"].sum())
                    _n_b_kept = len(_bdata) - _n_b_out

                    st.markdown(f"##### {_bname}")
                    _sc_col, _hs_col = st.columns(2)

                    with _sc_col:
                        st.markdown("Speed vs power")
                        _fig_b_sc = go.Figure()
                        for _is_out, _dot_color, _dot_name in [
                            (False, _bike_color, "Normal"),
                            (True, "#ef5350", "Outlier"),
                        ]:
                            _pts = _bdata[_bdata["is_outlier"] == _is_out]
                            if _pts.empty:
                                continue
                            _fig_b_sc.add_trace(go.Scatter(
                                x=_pts["average_watts"],
                                y=_pts["speed_kmh"],
                                mode="markers",
                                name=_dot_name,
                                marker={"color": _dot_color, "size": 10,
                                        "line": {"width": 1, "color": "white"}},
                                text=_pts["z_label"],
                                hovertemplate=(
                                    "Power: %{x:.0f} W<br>"
                                    f"Speed: %{{y:.1f}} {_spd_label()}<br>"
                                    "Z-score: %{text}<extra>" + _dot_name + "</extra>"
                                ),
                            ))
                        # Draw speed/W^(1/3) guidelines as curves:
                        # speed = k × watts^(1/3)  →  k = speed_per_cbrt_watt
                        _sc_spw = _bdata.dropna(subset=["speed_per_cbrt_watt", "average_watts"])
                        if len(_sc_spw) >= 2:
                            _sc_mean = _sc_spw["speed_per_cbrt_watt"].mean()
                            _sc_std = _sc_spw["speed_per_cbrt_watt"].std(ddof=1)
                            _sc_lo = _sc_mean - z_threshold * _sc_std
                            _sc_hi = _sc_mean + z_threshold * _sc_std
                            _w_min = float(_sc_spw["average_watts"].min())
                            _w_max = float(_sc_spw["average_watts"].max())
                            _w_pad = (_w_max - _w_min) * 0.05
                            _wx = list(np.linspace(_w_min - _w_pad, _w_max + _w_pad, 60))
                            _wx_rev = list(reversed(_wx))
                            # ±σ band fill (behind everything) as a curve
                            _fig_b_sc.add_trace(go.Scatter(
                                x=_wx + _wx_rev,
                                y=[_sc_lo * np.cbrt(w) for w in _wx] + [_sc_hi * np.cbrt(w) for w in _wx_rev],
                                fill="toself",
                                fillcolor="rgba(239,83,80,0.10)",
                                line={"width": 0},
                                hoverinfo="skip",
                                showlegend=False,
                            ))
                            # mean curve
                            _fig_b_sc.add_trace(go.Scatter(
                                x=_wx,
                                y=[_sc_mean * np.cbrt(w) for w in _wx],
                                mode="lines",
                                line={"color": "rgba(128,128,128,0.6)", "dash": "dot", "width": 1.5},
                                name="\u03bc (speed/W\u00b9\u2044\u00b3)",
                                hovertemplate=f"\u03bc = {_sc_mean:.4f} {_spd_label()}/W\u00b9\u2044\u00b3<extra>mean</extra>",
                            ))
                            # ±threshold curves
                            for _slope, _slabel in [(_sc_lo, f"\u2212{z_threshold:.2g}\u03c3"), (_sc_hi, f"+{z_threshold:.2g}\u03c3")]:
                                _fig_b_sc.add_trace(go.Scatter(
                                    x=_wx,
                                    y=[_slope * np.cbrt(w) for w in _wx],
                                    mode="lines",
                                    line={"color": "#ef5350", "dash": "dash", "width": 1.5},
                                    name=_slabel,
                                    hovertemplate=f"{_slabel} = {_slope:.4f} {_spd_label()}/W\u00b9\u2044\u00b3<extra>{_slabel}</extra>",
                                ))
                        _fig_b_sc.update_layout(
                            xaxis_title="Avg power (W)",
                            yaxis_title=f"Speed ({_spd_label()})",
                            plot_bgcolor="rgba(0,0,0,0)",
                            legend={"orientation": "h", "y": -0.25},
                            height=300,
                            margin={"t": 10, "b": 10},
                        )
                        st.plotly_chart(_fig_b_sc, width="stretch")

                    with _hs_col:
                        st.markdown(f"Speed/W\u00b9\u2044\u00b3 distribution  \u00b1{z_threshold:.1f}\u03c3 cutoff")
                        _spw_b = _bdata.dropna(subset=["speed_per_cbrt_watt"])
                        if len(_spw_b) >= 2:
                            _b_mean = _spw_b["speed_per_cbrt_watt"].mean()
                            _b_std = _spw_b["speed_per_cbrt_watt"].std(ddof=1)
                            _b_lo = _b_mean - z_threshold * _b_std
                            _b_hi = _b_mean + z_threshold * _b_std
                            _nbins = max(6, len(_spw_b) // 2)

                            _fig_b_h = go.Figure()
                            for _is_out, _bar_color, _bar_name in [
                                (False, _bike_color, "Normal"),
                                (True, "#ef5350", "Outlier"),
                            ]:
                                _pts_h = _spw_b[_spw_b["is_outlier"] == _is_out]
                                if _pts_h.empty:
                                    continue
                                _fig_b_h.add_trace(go.Histogram(
                                    x=_pts_h["speed_per_cbrt_watt"],
                                    name=_bar_name,
                                    marker_color=_bar_color,
                                    opacity=0.8,
                                    nbinsx=_nbins,
                                    hovertemplate="Speed/W\u00b9\u2044\u00b3: %{x:.4f}<br>Count: %{y}<extra>" + _bar_name + "</extra>",
                                ))

                            _xlo = float(_spw_b["speed_per_cbrt_watt"].min())
                            _xhi = float(_spw_b["speed_per_cbrt_watt"].max())
                            _xpad = max((_xhi - _xlo) * 0.05, 1e-6)
                            for _sx0, _sx1 in [(_xlo - _xpad, _b_lo), (_b_hi, _xhi + _xpad)]:
                                _fig_b_h.add_vrect(
                                    x0=_sx0, x1=_sx1,
                                    fillcolor="rgba(239,83,80,0.12)",
                                    line_width=0, layer="below",
                                )
                            for _vx, _vlabel in [(_b_lo, f"−{z_threshold:.1f}σ"), (_b_hi, f"+{z_threshold:.1f}σ")]:
                                _fig_b_h.add_vline(
                                    x=_vx, line_dash="dash", line_color="#ef5350",
                                    annotation_text=_vlabel, annotation_position="top",
                                    annotation_font_color="#ef5350",
                                )
                            _fig_b_h.add_vline(
                                x=_b_mean, line_dash="dot",
                                line_color="rgba(128,128,128,0.7)",
                                annotation_text="μ", annotation_position="top",
                            )
                            _fig_b_h.update_layout(
                                barmode="overlay",
                                xaxis_title=f"Speed/W\u00b9\u141f\u00b3 ({_spd_label()}/W\u00b9\u141f\u00b3)",
                                yaxis_title="Efforts",
                                plot_bgcolor="rgba(0,0,0,0)",
                                legend={"orientation": "h", "y": -0.25},
                                height=300,
                                margin={"t": 10, "b": 10},
                            )
                            st.plotly_chart(_fig_b_h, width="stretch")
                        else:
                            st.caption("Not enough efforts to show distribution.")

                    st.caption(
                        f"🔴 **{_n_b_out} outlier(s)** · 🔵 **{_n_b_kept} kept** "
                        f"(\u03bc = {_bdata['speed_per_cbrt_watt'].mean():.4f}, "
                        f"\u03c3 = {_bdata['speed_per_cbrt_watt'].std(ddof=1):.4f})"
                    )
                    if _bi < len(_ann_bikes) - 1:
                        st.divider()

        elif _step == "3 — After filtering":
            st.caption(
                f"After removing the {_n_outliers} outlier(s), **{_n_kept} clean efforts** remain. "
                "The cube-root curve (speed \u221d power^\u00b9\u2044\u00b3) should now fit the data more tightly — "
                "these are the efforts we use to compare bikes fairly."
            )
            _fig_flt = px.scatter(
                _filtered_seg.dropna(subset=["speed_kmh", "average_watts"]),
                x="average_watts",
                y="speed_kmh",
                color="bike_name",
                color_discrete_sequence=_COLOR_SEQ,
                trendline="lowess",
                labels={"average_watts": "Avg power (W)", "speed_kmh": f"Speed ({_spd_label()})", "bike_name": "Bike"},
            )
            _fig_flt.update_traces(marker_size=9, selector={"mode": "markers"})
            _fig_flt.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=380)
            st.plotly_chart(_fig_flt, width="stretch")

        else:  # Step 4
            st.caption(
                "We divide each effort's speed by **power^\u00b9\u2044\u00b3** to get **speed / W^\u00b9\u2044\u00b3** — "
                "in aerodynamics, speed scales as the cube root of power (v \u221d P^\u00b9\u2044\u00b3), "
                "so this ratio is approximately constant for a given bike and conditions. "
                "A higher value means the bike converts power into speed more efficiently."
            )
            _spw_data = _filtered_seg.dropna(subset=["speed_per_cbrt_watt", "bike_name"]).copy()
            if _spw_data.empty:
                st.info("Not enough data after filtering to compute speed efficiency on this segment.")
            else:
                _fig_spw = px.box(
                    _spw_data,
                    x="bike_name",
                    y="speed_per_cbrt_watt",
                    color="bike_name",
                    color_discrete_sequence=_COLOR_SEQ,
                    points="all",
                    labels={"bike_name": "Bike", "speed_per_cbrt_watt": f"Speed / W\u00b9\u2044\u00b3 ({_spd_label()}/W\u00b9\u2044\u00b3)"},
                )
                _fig_spw.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", height=380)
                st.plotly_chart(_fig_spw, width="stretch")
                _spw_summary = (
                    _spw_data.groupby("bike_name")["speed_per_cbrt_watt"]
                    .agg(["mean", "median", "count"])
                    .rename(columns={"mean": "Mean", "median": "Median", "count": "Efforts"})
                    .reset_index()
                    .rename(columns={"bike_name": "Bike"})
                )
                for col in ["Mean", "Median"]:
                    _spw_summary[col] = _spw_summary[col].apply(lambda v: f"{v:.4f}")
                st.dataframe(_spw_summary, hide_index=True, width="stretch")

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
                [
                    "segment_id",
                    "name",
                    "segment_type_detail",
                    *all_rides_cols,
                    "distance",
                    "average_grade",
                ]
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
                    "segment_type_detail": "Subtype",
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
                disabled=["Segment", "Subtype", *all_rides_cols, _dist_col, "Grade (%)"],
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
            # Apply outlier filtering to this segment's efforts for the summary
            seg_efforts_spw = compute_speed_per_watt(seg_efforts)
            seg_efforts_clean, seg_efforts_ann = filter_outliers_by_power_speed(
                seg_efforts_spw, z_threshold=z_threshold
            )
            _n_seg_outliers = int(seg_efforts_ann["is_outlier"].sum()) if "is_outlier" in seg_efforts_ann.columns else 0

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
            # Power-normalised efficiency (computed on clean efforts)
            _spw_col = f"Speed/W\u00b9\u141f\u00b3 ({_spd_label()}/W\u00b9\u141f\u00b3)"
            if _has_col(seg_efforts_clean, "speed_per_cbrt_watt"):
                agg[_spw_col] = ("speed_per_cbrt_watt", "mean")

            summary = seg_efforts_clean.groupby("bike_name").agg(**agg).reset_index()
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
            if _spw_col in summary.columns:
                summary[_spw_col] = summary[_spw_col].apply(
                    lambda v: f"{v:.4f}" if pd.notna(v) else "—"
                )

            if _n_seg_outliers:
                st.caption(
                    f"Stats computed on clean efforts — **{_n_seg_outliers} outlier effort(s)** "
                    f"excluded at z-threshold {z_threshold:.1f}. Adjust the threshold in the sidebar."
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
