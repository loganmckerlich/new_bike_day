"""Segment comparison page – valid segment selection and bike performance analysis."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.utils import navigator, page_guard

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.fetch import get_segment_detail, get_segment_streams
from src.database import init_db, load_segment_geo, save_segment_geo
from src.plot_colors import to_rgba
from src.analytics import (
    compute_speed_per_watt,
    mean_profile_by_segment_type,
    power_normalized_profile,
)
from src.bike_delta import power_overlap_ok
from src._ui_helpers import (
    use_metric as _use_metric,
    spd_label as _spd_label,
    dist_label as _dist_label,
    elev_label as _elev_label,
    convert_speed as _convert_speed,
    convert_dist_m as _convert_dist_m,
    convert_elev_m as _convert_elev_m,
    fmt_duration as _fmt_duration,
    compute_speed_kmh as _compute_speed_kmh,
    has_col as _has_col,
    gear_label,
    get_available_bikes
)

# ── Module-level placeholders (refreshed inside show()) ──────────────────────
efforts: pd.DataFrame | None = None
segments: pd.DataFrame | None = None
bikes: dict[str, str] = {}
access_token: str | None = None

# ── Constants ─────────────────────────────────────────────────────────────────
SEGMENT_TYPES: list[str] = ["sprint", "flat", "ascent", "descent"]
SEGMENT_TYPE_DETAILS: list[str] = [
    "sprint_flat",
    "sprint_uphill",
    "sprint_downhill",
    "flat_short",
    "flat_long",
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
    "ascent_moderate": "Ascent • Moderate",
    "ascent_steep": "Ascent • Steep",
    "descent_gentle": "Descent • Gentle",
    "descent_steep": "Descent • Steep",
}
_COLOR_SEQ: list[str] = px.colors.qualitative.Set2
_SPIDER_POLYGON_LINE_WIDTH: int = 3
_SPIDER_POLYGON_FILL_ALPHA: float = 0.20
_SPIDER_AXIS_PADDING: float = 1.2
_EFFICIENCY_LABEL: str = "Efficiency (km/h / W^(1/3))"


def _highlight_best_value(series: pd.Series, ascending: bool = False) -> list[str]:
    """Highlight the best numeric value in a column in green.

    By default, the largest numeric value is highlighted; pass ``ascending=True``
    to highlight the smallest value instead.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return [""] * len(series)

    values = series.dropna()
    if values.empty:
        return [""] * len(series)

    best_value = values.min() if ascending else values.max()
    return [
        "background-color: #dcfce7; color: #166534; font-weight: 600"
        if pd.notna(v) and ((v <= best_value) if ascending else (v >= best_value))
        else ""
        for v in series
    ]


def _straightness_index(geo: dict) -> float | None:
    """Return a 0-1 straightness index for a route, if route points are available."""
    points = geo.get("polyline_points") or []
    if len(points) < 2:
        return None

    coords = [(float(p[0]), float(p[1])) for p in points if len(p) >= 2]
    if len(coords) < 2:
        return None

    def _haversine(start: tuple[float, float], end: tuple[float, float]) -> float:
        lat1, lon1 = map(math.radians, start)
        lat2, lon2 = map(math.radians, end)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return 2 * 6371000 * math.asin(math.sqrt(a))

    total_path = sum(_haversine(coords[i], coords[i + 1]) for i in range(len(coords) - 1))
    if total_path <= 0:
        return None
    return _haversine(coords[0], coords[-1]) / total_path


# ── Page title ────────────────────────────────────────────────────────────

def comp_inputs():

    available_bikes = get_available_bikes()

    bikes_to_compare = st.multiselect(
        "Bikes to compare",
        options=available_bikes,
        default=st.session_state.get("seg_bikes_plain", available_bikes[:2]),
        max_selections=5,
        help="Select up to 5 bikes to compare.",
        key="seg_bikes_select",
    )
    st.session_state["seg_bikes_plain"] = bikes_to_compare
    st.caption("Only bikes with ≥ 20 segments after outlier filtering are shown.")

    min_efforts = st.number_input(
        "Min efforts per bike per segment",
        min_value=1,
        max_value=20,
        value=st.session_state.get("seg_min_efforts_plain", 3),
        step=1,
        help="Both bikes must have at least this many power-measured efforts on a segment.",
        key="seg_min_efforts",
    )
    st.session_state["seg_min_efforts_plain"] = min_efforts
    return bikes_to_compare, min_efforts

@st.cache_data(ttl=3600)
def _get_segment_geo(segment_id: int) -> dict:
    init_db()
    db_cached = load_segment_geo(segment_id)
    if db_cached is not None:
        return db_cached

    if not access_token:
        return {}

    detail = get_segment_detail(access_token, segment_id)
    streams = get_segment_streams(access_token, segment_id)
    save_segment_geo(segment_id, detail, streams)
    result = {**detail, "streams": streams}
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


# ── Public entry point ───────────────────────────────────────────────────────
def show(bikes_to_compare, min_efforts: int = 3) -> None:
    """Render the segmented bike comparison analysis."""
    global efforts, segments, bikes, access_token
    efforts = st.session_state.get("cleaned_efforts")
    segments = st.session_state.get("segments")
    bikes = st.session_state.get("bikes", {})
    access_token = st.session_state.get("access_token")

    # ── Tab title ────────────────────────────────────────────────────────────

    # ── Data preparation ──────────────────────────────────────────────────────────

    # cleaned_efforts from Data Cleaning page already have power filter + descent
    # filter applied; just keep entries that still have power data (safety check).
    watt_efforts = efforts[efforts["average_watts"].notna()].copy()

    if watt_efforts.empty:
        st.warning("No efforts with power data found. Ensure your rides are recorded with a power meter.")
        st.stop()

    watt_efforts["bike_name"] = watt_efforts["gear_id"].map(lambda g: gear_label(g, bikes))

    # segment_type and related columns are already merged in by data_cleaning.py;
    # merge again only for columns that may be missing (e.g. total_elevation_gain).
    _extra_seg_cols = ["segment_id"]
    for _col in ["name", "distance", "average_grade", "total_elevation_gain",
                 "segment_type", "segment_type_detail"]:
        if _col not in watt_efforts.columns and _col in segments.columns:
            _extra_seg_cols.append(_col)

    if len(_extra_seg_cols) > 1:
        _seg_extra = segments[_extra_seg_cols].copy()
        watt_efforts = watt_efforts.merge(_seg_extra, on="segment_id", how="left")

    if "segment_type_detail" not in watt_efforts.columns:
        if "segment_type" in watt_efforts.columns:
            watt_efforts["segment_type_detail"] = watt_efforts["segment_type"]

    if "speed_kmh" not in watt_efforts.columns:
        watt_efforts["speed_kmh"] = _compute_speed_kmh(watt_efforts)


    # ── Sidebar: segment settings ──────────────────────────────────────────────────

    spider_use_subcategories = st.toggle(
        "Use subcategories in spider charts",
        value=st.session_state.get("seg_spider_plain", False),
        key="seg_spider_subcategories",
        help="Show spider charts by segment subcategory instead of parent category.",
    )
    st.session_state["seg_spider_plain"] = spider_use_subcategories

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
    valid_mask = (seg_counts[bikes_to_compare] >= min_efforts).all(axis=1)
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
        f"{bikes_label} — each spoke is centred on the field average **for that terrain type**, "
        "so zero = bikes are equal on that terrain. "
        "Gap size is comparable across spokes; raw speed values are not. "
        "Left shows speed, right shows speed / W\u00b9\u2044\u00b3 (power-normalised efficiency). "
    )

    spider_filtered = selected_efforts[selected_efforts["segment_id"].isin(valid_segment_ids)].copy()

    spider_dimension_col = "segment_type_detail" if spider_use_subcategories else "segment_type"
    spider_dimensions = SEGMENT_TYPE_DETAILS if spider_use_subcategories else SEGMENT_TYPES
    if spider_use_subcategories:
        categories = [TYPE_DETAIL_LABELS.get(t, t.replace("_", " ").title()) for t in spider_dimensions]
    else:
        categories = [f"{TYPE_ICONS.get(t, '')} {t.capitalize()}" for t in spider_dimensions]
    categories_closed = categories + [categories[0]]

    # ── Chart 1: raw speed ────────────────────────────────────────────────────────
    speed_profile = mean_profile_by_segment_type(
        spider_filtered,
        bikes_to_compare,
        spider_dimensions,
        valid_segment_ids,
        value_col="speed_kmh",
        segment_type_col=spider_dimension_col,
    )

    _spd = _spd_label()

    def _center_per_axis(profile: dict[str, list[float]]) -> tuple[dict[str, list[float]], float, float]:
        """For each spoke, subtract the mean across bikes for that spoke.

        Zero = field average on that terrain type. Bounds are symmetric
        (±max absolute deviation) so gap sizes are comparable across spokes
        but raw speed levels don't dominate the scale.
        """
        bikes = list(profile.keys())
        if not bikes:
            return profile, -1.0, 1.0
        n_dims = len(profile[bikes[0]])
        scaled = {b: list(vals) for b, vals in profile.items()}
        for i in range(n_dims):
            axis_vals = [profile[b][i] for b in bikes if profile[b][i] == profile[b][i]]
            if not axis_vals:
                continue
            axis_mean = sum(axis_vals) / len(axis_vals)
            for b in scaled:
                scaled[b][i] -= axis_mean
        all_scaled = [v for vals in scaled.values() for v in vals if v == v]
        abs_max = max((abs(v) for v in all_scaled), default=1.0)
        return scaled, -abs_max, abs_max

    # Convert speed values for display, then mean-centre for radial position
    speed_display = {b: [_convert_speed(v) for v in speed_profile[b]] for b in bikes_to_compare}
    speed_scaled, spd_lo, spd_hi = _center_per_axis(speed_display)

    fig_spider = go.Figure()
    for idx, b in enumerate(bikes_to_compare):
        scaled_vals = speed_scaled[b]
        raw_vals = speed_display[b]
        scaled_closed = scaled_vals + [scaled_vals[0]]
        raw_closed = raw_vals + [raw_vals[0]]
        color = _COLOR_SEQ[idx % len(_COLOR_SEQ)]
        fig_spider.add_trace(
            go.Scatterpolar(
                r=scaled_closed,
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
            "radialaxis": {
                "visible": True,
                "title": f"Speed advantage ({_spd})",
                "range": [spd_lo * _SPIDER_AXIS_PADDING, spd_hi * _SPIDER_AXIS_PADDING],
                "showline": True,
            },
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
    eff_scaled, eff_lo, eff_hi = _center_per_axis(eff_display)

    fig_efficiency = go.Figure()
    for idx, b in enumerate(bikes_to_compare):
        scaled_vals = eff_scaled[b]
        raw_vals = eff_display[b]
        scaled_closed = scaled_vals + [scaled_vals[0]]
        raw_closed = raw_vals + [raw_vals[0]]
        color = _COLOR_SEQ[idx % len(_COLOR_SEQ)]
        fig_efficiency.add_trace(
            go.Scatterpolar(
                r=scaled_closed,
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
            "radialaxis": {
                "visible": True,
                "title": "Efficiency advantage",
                "range": [eff_lo * _SPIDER_AXIS_PADDING, eff_hi * _SPIDER_AXIS_PADDING],
                "showline": True,
            },
            "angularaxis": {"categoryorder": "array", "categoryarray": categories},
        },
        showlegend=False,
        title=f"Efficiency profile by segment {'subcategory' if spider_use_subcategories else 'type'} (power-normalised)",
        height=500,
    )

    spider_col1, spider_col2 = st.columns(2)
    with spider_col1:
        st.plotly_chart(fig_spider, width="stretch", config={"staticPlot": True})
    with spider_col2:
        st.plotly_chart(fig_efficiency, width="stretch", config={"staticPlot": True})
    st.caption(
        "Outward = advantage on that terrain type. Inward = disadvantage. "
        "The gap between polygons is what matters — not how far out they sit overall."
    )
    with st.expander("Methodology"):
        st.markdown(
            """
Only segments ridden on **both** bikes are included — this keeps the comparison fair and holds
terrain constant across bikes.

For each of those segment efforts, speed is divided by the cube root of average power
(`speed / W¹ᐟ³`). This controls for effort: going faster because you pushed harder doesn't count as a win for the bike.
The cube root comes from aerodynamics — at steady state, speed scales as power^(1/3), so this
ratio is roughly constant for a given bike regardless of effort level.

Those efficiency values are then averaged by terrain type (sprint, flat, climb, descent) to
produce each spoke of the chart.

The speed chart shows the same data but without the power-normalisation step, so it reflects raw speed.
"""
        )


    # ── Valid segment selector ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Valid segments")

    if not valid_segment_ids:
        st.info(
            f"No segments where all selected bikes have ≥ {int(min_efforts)} "
            "power-measured rides. Try reducing the minimum sample size or selecting fewer bikes."
        )
        return

    st.caption(
        f"Segments where all selected bikes have ≥ {int(min_efforts)} "
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

                # ── KS power-overlap badge ─────────────────────────────────────
                # Requires ≥ 2 bikes selected; only checks the first pair for display.
                if len(bikes_to_compare) >= 2:
                    _bike_a, _bike_b = bikes_to_compare[0], bikes_to_compare[1]
                    _ks_labels: list[str] = []
                    for _sid in display["segment_id"]:
                        _seg_eff = selected_efforts[selected_efforts["segment_id"] == _sid]
                        _ok = power_overlap_ok(_seg_eff, _bike_a, _bike_b)
                        _ks_labels.append("✅" if _ok else "⚠️")
                    display["Power overlap"] = _ks_labels

                display.insert(0, "Select", False)
                display = display.rename(
                    columns={
                        "name": "Segment",
                        "segment_type_detail": "Subtype",
                        "distance": _dist_col,
                        "average_grade": "Grade (%)",
                    }
                )

                _disabled_cols = ["Segment", "Subtype", *all_rides_cols, _dist_col, "Grade (%)"]
                if "Power overlap" in display.columns:
                    _disabled_cols.append("Power overlap")

                edited = st.data_editor(
                    display,
                    column_config={
                        "Select": st.column_config.CheckboxColumn("✓", default=False),
                        "segment_id": None,
                    },
                    hide_index=True,
                    width="stretch",
                    key=f"table_{seg_type}",
                    disabled=_disabled_cols,
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

                seg_efforts_clean = compute_speed_per_watt(seg_efforts.copy())
                seg_efforts = seg_efforts_clean.copy()

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

                with st.spinner("Loading segment map…"):
                    geo = _get_segment_geo(int(seg_id))

                segment_detail = seg_row.get("segment_type_detail") or seg_row.get("segment_type")
                segment_label = TYPE_DETAIL_LABELS.get(
                    segment_detail,
                    str(segment_detail or "—").replace("_", " ").title(),
                )
                straightness = _straightness_index(geo)
                metadata = pd.DataFrame(
                    [
                        {
                            "Segment": segment_label,
                            "Distance": dist_metric_str,
                            "Average grade": f"{grade:.1f}%" if pd.notna(grade) else "—",
                            "Straightness index": f"{straightness:.2f}" if straightness is not None else "—",
                        }
                    ]
                )
                st.dataframe(
                    metadata.style.hide(axis="index").set_properties(
                        **{"background-color": "#f8fafc", "font-weight": "600"}
                    ),
                    width="stretch",
                    hide_index=True,
                )

                bike_names = seg_efforts_clean["bike_name"].dropna().unique().tolist()
                summary = pd.DataFrame({"Bike name": bike_names})
                grouped = seg_efforts_clean.groupby("bike_name")
                summary_metrics = grouped.agg(
                    mean_speed_kmh=("speed_kmh", "mean"),
                    mean_watts=("average_watts", "mean"),
                    num_efforts=("effort_id", "count"),
                    best_effort_time=("moving_time", "min"),
                )
                summary_metrics["mean_hr"] = (
                    grouped["average_heartrate"].mean()
                    if _has_col(seg_efforts_clean, "average_heartrate")
                    else np.nan
                )
                summary_metrics["mean_efficiency"] = grouped["speed_per_cbrt_watt"].mean()
                summary_metrics = summary_metrics.reindex(bike_names)
                summary["Mean speed (km/h)"] = summary_metrics["mean_speed_kmh"].to_numpy()
                summary["Mean watts"] = summary_metrics["mean_watts"].to_numpy()
                summary["Mean HR"] = summary_metrics["mean_hr"].to_numpy()
                summary["Mean efficiency"] = summary_metrics["mean_efficiency"].to_numpy()
                summary["Number of efforts"] = summary_metrics["num_efforts"].to_numpy()
                summary["Best effort time"] = summary_metrics["best_effort_time"].to_numpy()

                styler = summary.style.hide(axis="index")
                styler = styler.format(
                    {
                        "Mean speed (km/h)": lambda v: f"{_convert_speed(v):.1f}" if pd.notna(v) else "—",
                        "Mean watts": lambda v: f"{v:.1f}" if pd.notna(v) else "—",
                        "Mean HR": lambda v: f"{v:.1f}" if pd.notna(v) else "—",
                        "Mean efficiency": lambda v: f"{v:.4f}" if pd.notna(v) else "—",
                        "Number of efforts": lambda v: f"{int(v)}" if pd.notna(v) else "—",
                        "Best effort time": lambda v: _fmt_duration(v) if pd.notna(v) else "—",
                    }
                )
                for col, ascending in [
                    ("Mean speed (km/h)", False),
                    ("Mean watts", False),
                    ("Mean HR", False),
                    ("Mean efficiency", False),
                    ("Number of efforts", False),
                    ("Best effort time", True),
                ]:
                    styler = styler.apply(
                        lambda s: _highlight_best_value(s, ascending=ascending),
                        axis=0,
                        subset=[col],
                    )
                styler = styler.set_properties(**{"white-space": "nowrap"})
                st.dataframe(styler, width="stretch", hide_index=True)

                # Map + elevation
                map_col, elev_col = st.columns(2)
                with map_col:
                    st.markdown("**Route**")
                    _render_segment_map(geo, str(seg_row.get("name", "")))
                with elev_col:
                    st.markdown("**Elevation profile**")
                    _render_elevation_profile(geo, seg_distance_m)

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
                chart_tab_labels.extend(["📅 Timeline", "📈 Efficiency scatter", "🎻 Efficiency distribution"])

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
                    ct_idx += 1

                with chart_tabs[ct_idx]:
                    efficiency_df = seg_efforts_clean.dropna(subset=["average_watts", "speed_kmh"]).copy()
                    st.caption("Efficiency scatter by bike with cube-root curve")
                    if not efficiency_df.empty:
                        fig = go.Figure()
                        fit_x = np.cbrt(efficiency_df["average_watts"].astype(float).to_numpy())
                        fit_y = efficiency_df["speed_kmh"].astype(float).to_numpy()
                        mask = np.isfinite(fit_x) & np.isfinite(fit_y)
                        if mask.sum() >= 2:
                            denom = np.dot(fit_x[mask], fit_x[mask])
                            if denom > 0:
                                slope = np.dot(fit_x[mask], fit_y[mask]) / denom
                                curve_watts = np.linspace(float(efficiency_df["average_watts"].min()), float(efficiency_df["average_watts"].max()), 100)
                                fig.add_trace(
                                    go.Scatter(
                                        x=curve_watts,
                                        y=slope * np.cbrt(curve_watts),
                                        mode="lines",
                                        line={"color": "#64748b", "width": 2, "dash": "dash"},
                                        name="Theoretical curve",
                                        showlegend=False,
                                        hoverinfo="skip",
                                    )
                                )
                        bike_names = efficiency_df["bike_name"].dropna().unique().tolist()
                        for idx, bike_name in enumerate(bike_names):
                            bike_data = efficiency_df[efficiency_df["bike_name"] == bike_name]
                            fig.add_trace(
                                go.Scatter(
                                    x=bike_data["average_watts"],
                                    y=bike_data["speed_kmh"],
                                    mode="markers",
                                    name=bike_name,
                                    marker={"color": _COLOR_SEQ[idx % len(_COLOR_SEQ)], "size": 8},
                                    hovertemplate=(
                                        f"{bike_name}<br>Power: %{{x:.0f}} W<br>Speed: %{{y:.1f}} {_spd_label()}<extra></extra>"
                                    ),
                                )
                            )
                        fig.update_layout(
                            xaxis_title="Average power (W)",
                            yaxis_title=f"Speed ({_spd_label()})",
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig, width="stretch")
                    else:
                        st.caption("No power/speed pairs available for efficiency scatter.")
                    ct_idx += 1

                with chart_tabs[ct_idx]:
                    efficiency_distribution = seg_efforts_clean.dropna(subset=["speed_per_cbrt_watt"]).copy()
                    st.caption("Efficiency distribution by bike")
                    if not efficiency_distribution.empty:
                        fig = go.Figure()
                        bike_names = efficiency_distribution["bike_name"].dropna().unique().tolist()
                        for idx, bike_name in enumerate(bike_names):
                            bike_data = efficiency_distribution[efficiency_distribution["bike_name"] == bike_name]
                            color = _COLOR_SEQ[idx % len(_COLOR_SEQ)]
                            fig.add_trace(
                                go.Violin(
                                    y=bike_data["speed_per_cbrt_watt"],
                                    x=[bike_name] * len(bike_data),
                                    name=bike_name,
                                    line={"color": color},
                                    fillcolor=to_rgba(color, 0.25),
                                    box_visible=True,
                                    meanline_visible=True,
                                    points=False,
                                    hovertemplate=(
                                        f"{bike_name}<br>Efficiency: %{{y:.4f}}<extra></extra>"
                                    ),
                                )
                            )
                        fig.update_layout(
                            xaxis_title="Bike",
                            yaxis_title=_EFFICIENCY_LABEL,
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig, width="stretch")
                    else:
                        st.caption("No efficiency values available for this segment.")

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



def main() -> None:
    st.title("📊 Step 3 — Segment Level Comparison")
    st.markdown(
        "Filters and cleaning are already applied (configured in **Step 2 — Data Cleaning**). "
        "Select bikes and segments below to compare performance."
    )

    page_guard("bike_comparison_segmented")

    bikes_to_compare, min_efforts = comp_inputs()

    show(bikes_to_compare, min_efforts)

if __name__ == "__main__":
    navigator("bike_comparison_segmented1")
    main()
    navigator("bike_comparison_segmented2")