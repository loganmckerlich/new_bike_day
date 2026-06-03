"""Data Cleaning page: configure outlier removal and filtering, then preview the effect."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.analytics import (
    apply_min_watts_filter,
    compute_speed_per_watt,
    filter_outliers_by_power_speed,
    outlier_detection_frames,
)
from app.app_pages._ui_helpers import (
    use_metric as _use_metric,
    spd_label as _spd_label,
    convert_speed as _convert_speed,
    gear_label as _gear_label_fn,
    compute_speed_kmh as _compute_speed_kmh,
)


def _gear_label(gear_id: str | None, bikes: dict[str, str]) -> str:
    return _gear_label_fn(gear_id, bikes)


# ── Page ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("🧹 Step 2 — Data Cleaning")
    st.markdown(
        "Configure how raw Strava efforts are filtered before analysis. "
        "Noisy efforts — coasting, equipment glitches, drafting — can skew comparisons. "
        "The settings you choose here are applied on every subsequent page."
    )

    # ── Guard: data must be loaded ─────────────────────────────────────────────
    raw_efforts: pd.DataFrame | None = st.session_state.get("efforts")
    segments: pd.DataFrame | None = st.session_state.get("segments")
    bikes: dict[str, str] = st.session_state.get("bikes", {})

    if raw_efforts is None or (hasattr(raw_efforts, "empty") and raw_efforts.empty):
        st.info("👈 Head to **Step 1 — Data Collection** to sign in with Strava and load your data first.")
        st.stop()

    if segments is None or segments.empty:
        st.warning("No starred segments found. Star some segments on Strava and reload from Step 1.")
        st.stop()

    # Keep only power efforts
    efforts_with_power = raw_efforts[raw_efforts["average_watts"].notna()].copy()
    if efforts_with_power.empty:
        st.warning("No efforts with power data found. Ensure your rides are recorded with a power meter.")
        st.stop()

    # Merge segment metadata so we have segment_type available
    seg_meta_cols = ["segment_id", "name", "distance", "average_grade", "maximum_grade", "segment_type"]
    if "segment_type_detail" in segments.columns:
        seg_meta_cols.append("segment_type_detail")
    seg_meta = segments[seg_meta_cols].copy()
    efforts_with_power = efforts_with_power.merge(seg_meta, on="segment_id", how="inner")
    efforts_with_power["bike_name"] = efforts_with_power["gear_id"].map(
        lambda g: _gear_label(g, bikes)
    )
    efforts_with_power["speed_kmh"] = _compute_speed_kmh(efforts_with_power)

    # ── Two-column layout: controls left, visualisations right ────────────────
    left_col, right_col = st.columns([1, 2], vertical_alignment="top")

    # ── LEFT: Cleaning controls ────────────────────────────────────────────────
    with left_col:
        st.subheader("⚙️ Cleaning parameters")

        ftp: int | None = st.session_state.get("ftp")
        _default_min_watts = int(round(ftp * 0.75)) if ftp is not None and ftp > 0 else 0
        st.session_state.setdefault("min_watts", _default_min_watts)
        min_watts: int = st.number_input(
            "Minimum watts",
            min_value=0,
            max_value=999,
            step=5,
            key="min_watts",
            help=(
                "Efforts with average power below this threshold are excluded from analysis. "
                "Default is 75 % of your FTP (if available). Set to 0 to disable."
            ),
        )

        st.session_state.setdefault("outlier_z_threshold", 2.0)
        z_threshold: float = st.slider(
            "Outlier z-score threshold",
            min_value=0.25,
            max_value=3.5,
            step=0.25,
            key="outlier_z_threshold",
            help=(
                "Z-score = how many standard deviations an effort's speed/W^(1/3) sits "
                "from the segment mean. Efforts beyond this threshold are removed as outliers. "
                "Lower = more aggressive filtering."
            ),
        )

        st.session_state.setdefault("exclude_descents", False)
        exclude_descents: bool = st.toggle(
            "Exclude descent segments",
            key="exclude_descents",
            help=(
                "Remove descent segments entirely from all analysis pages. "
                "Descents are often wind-dominated and can skew results."
            ),
        )

    # ── Apply filters and store cleaned efforts in session state ───────────────
    cleaned = apply_min_watts_filter(
        efforts_with_power,
        int(min_watts),
        descents_exempt=True,
    )
    if exclude_descents and "segment_type" in cleaned.columns:
        cleaned = cleaned[cleaned["segment_type"] != "descent"].copy()

    st.session_state["cleaned_efforts"] = cleaned

    n_raw = len(efforts_with_power)
    n_clean = len(cleaned)
    _after_mw = apply_min_watts_filter(efforts_with_power, int(min_watts), descents_exempt=True)

    # ── Build segment selector options (needs cleaned data) ────────────────────
    _seg_effort_counts_ok = not cleaned.empty
    if _seg_effort_counts_ok:
        _seg_effort_counts = cleaned.groupby("segment_id")["effort_id"].count()
        _seg_effort_counts = _seg_effort_counts[_seg_effort_counts >= 3]
        _seg_effort_counts_ok = not _seg_effort_counts.empty

    if _seg_effort_counts_ok:
        _seg_name_map = {
            int(row["segment_id"]): row["name"]
            for _, row in segments[segments["segment_id"].isin(_seg_effort_counts.index)].iterrows()
        }
        _seg_options_sorted = _seg_effort_counts.sort_values(ascending=False).index.tolist()

    # ── LEFT (continued): outlier explainer controls ───────────────────────────
    with left_col:
        st.divider()
        st.subheader("🔬 Outlier detection")
        st.caption("Pick a segment and step to explore how filtering works.")

        if not _seg_effort_counts_ok:
            st.info("Not enough efforts per segment to illustrate outlier detection (need ≥ 3).")
            example_seg_id = None
            step = None
        else:
            example_seg_id: int = st.selectbox(
                "Example segment",
                options=_seg_options_sorted,
                format_func=lambda sid: _seg_name_map.get(sid, str(sid)),
                index=0,
                key="cleaning_explainer_seg",
            )

            step = st.radio(
                "Step",
                options=[
                    "1 — Raw efforts",
                    "2 — Outlier detection",
                    "3 — After filtering",
                    "4 — Efficiency metric",
                ],
                key="cleaning_explainer_step",
            )

    # ── RIGHT: filter summary + visualisations ─────────────────────────────────
    with right_col:
        st.subheader("📊 Filter summary")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Raw efforts (with power)", n_raw)
        with m2:
            st.metric(
                "Removed by min-watts filter",
                n_raw - len(_after_mw),
                help=f"Efforts with average power < {int(min_watts)} W (descents are always included)",
            )
        with m3:
            st.metric(
                "After min-watts" + (" + descent removal" if exclude_descents else "") + " filter",
                n_clean,
            )
        st.caption(
            f"**{n_clean}** efforts remain after filtering. "
            f"The z-score threshold ({z_threshold:.2g}σ) is applied per-segment on non-descent segments."
        )

        st.divider()

        if cleaned.empty:
            st.info("No efforts remain after filtering. Adjust the parameters on the left.")
        elif not _seg_effort_counts_ok:
            st.info("Not enough efforts per segment to show outlier detection (need ≥ 3 per segment).")
        else:
            _spd = _spd_label()
            _COLOR_SEQ = px.colors.qualitative.Set2
            available_bikes = sorted(cleaned["bike_name"].dropna().unique().tolist())

            _raw, _annotated, _filtered_seg = outlier_detection_frames(
                cleaned, int(example_seg_id), z_threshold=z_threshold
            )
            _seg_dist_m = float(
                segments.loc[segments["segment_id"] == example_seg_id, "distance"].iloc[0]
                if not segments[segments["segment_id"] == example_seg_id].empty
                else 0
            )
            for _df in [_raw, _annotated, _filtered_seg]:
                if "speed_kmh" not in _df.columns:
                    _df["speed_kmh"] = _compute_speed_kmh(_df, distance_m=_seg_dist_m)

            _n_total = len(_raw)
            _n_outliers_seg = int(_annotated["is_outlier"].sum()) if "is_outlier" in _annotated.columns else 0
            _n_kept = _n_total - _n_outliers_seg

            if step == "1 — Raw efforts":
                st.caption(
                    f"**{_n_total} efforts** recorded on this segment across all bikes. "
                    "Each point is one attempt. More power should mean more speed — but real data "
                    "is noisy (drafting, wind, fatigue)."
                )
                _fig_raw = px.scatter(
                    _raw.dropna(subset=["speed_kmh", "average_watts"]),
                    x="average_watts",
                    y="speed_kmh",
                    color="bike_name",
                    color_discrete_sequence=_COLOR_SEQ,
                    labels={"average_watts": "Avg power (W)", "speed_kmh": f"Speed ({_spd})", "bike_name": "Bike"},
                    hover_data={"bike_name": True},
                )
                _fig_raw.update_traces(marker_size=9)
                _fig_raw.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=380)
                st.plotly_chart(_fig_raw, width="stretch")

            elif step == "2 — Outlier detection":
                st.caption(
                    "We compute **speed / power¹⁄³** for every effort — "
                    "because in aerodynamics speed scales as the cube root of power (v ∝ P¹⁄³), "
                    "this ratio is approximately constant for a given bike and conditions. "
                    f"Efforts that deviate more than **{z_threshold:.1f} standard deviations** from the "
                    "segment mean are flagged as likely outliers (drafting, strong headwind, etc.)."
                )
                if "is_outlier" not in _annotated.columns:
                    st.info("Not enough efforts to detect outliers on this segment.")
                else:
                    _plot_ann = _annotated.dropna(subset=["speed_kmh", "average_watts"]).copy()
                    _plot_ann["label"] = _plot_ann["is_outlier"].map({True: "Outlier", False: "Normal"})
                    _plot_ann["z_label"] = _plot_ann["z_score"].apply(
                        lambda z: f"z = {z:.2f}" if pd.notna(z) else ""
                    )

                    _ann_bikes = [b for b in available_bikes if b in _plot_ann.get("bike_name", pd.Series()).values]
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
                                        f"Speed: %{{y:.1f}} {_spd}<br>"
                                        "Z-score: %{text}<extra>" + _dot_name + "</extra>"
                                    ),
                                ))
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
                                _fig_b_sc.add_trace(go.Scatter(
                                    x=_wx + _wx_rev,
                                    y=[_sc_lo * np.cbrt(w) for w in _wx] + [_sc_hi * np.cbrt(w) for w in _wx_rev],
                                    fill="toself",
                                    fillcolor="rgba(239,83,80,0.10)",
                                    line={"width": 0},
                                    hoverinfo="skip",
                                    showlegend=False,
                                ))
                                _fig_b_sc.add_trace(go.Scatter(
                                    x=_wx,
                                    y=[_sc_mean * np.cbrt(w) for w in _wx],
                                    mode="lines",
                                    line={"color": "rgba(128,128,128,0.6)", "dash": "dot", "width": 1.5},
                                    name="μ (speed/W¹⁄³)",
                                    hovertemplate=f"μ = {_sc_mean:.4f} {_spd}/W¹⁄³<extra>mean</extra>",
                                ))
                                for _slope, _slabel in [(_sc_lo, f"−{z_threshold:.2g}σ"), (_sc_hi, f"+{z_threshold:.2g}σ")]:
                                    _fig_b_sc.add_trace(go.Scatter(
                                        x=_wx,
                                        y=[_slope * np.cbrt(w) for w in _wx],
                                        mode="lines",
                                        line={"color": "#ef5350", "dash": "dash", "width": 1.5},
                                        name=_slabel,
                                        hovertemplate=f"{_slabel} = {_slope:.4f} {_spd}/W¹⁄³<extra>{_slabel}</extra>",
                                    ))
                            _fig_b_sc.update_layout(
                                xaxis_title="Avg power (W)",
                                yaxis_title=f"Speed ({_spd})",
                                plot_bgcolor="rgba(0,0,0,0)",
                                legend={"orientation": "h", "y": -0.25},
                                height=300,
                                margin={"t": 10, "b": 10},
                            )
                            st.plotly_chart(_fig_b_sc, width="stretch")

                        with _hs_col:
                            st.markdown(f"Speed/W¹⁄³ distribution ±{z_threshold:.1f}σ cutoff")
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
                                        hovertemplate="Speed/W¹⁄³: %{x:.4f}<br>Count: %{y}<extra>" + _bar_name + "</extra>",
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
                                    xaxis_title=f"Speed/W¹⁄³ ({_spd}/W¹⁄³)",
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
                            f"(μ = {_bdata['speed_per_cbrt_watt'].mean():.4f}, "
                            f"σ = {_bdata['speed_per_cbrt_watt'].std(ddof=1):.4f})"
                        )
                        if _bi < len(_ann_bikes) - 1:
                            st.divider()

            elif step == "3 — After filtering":
                st.caption(
                    f"After removing the {_n_outliers_seg} outlier(s), **{_n_kept} clean efforts** remain. "
                    "The cube-root curve (speed ∝ power¹⁄³) should now fit the data more tightly — "
                    "these are the efforts used to compare bikes fairly."
                )
                _fig_flt = px.scatter(
                    _filtered_seg.dropna(subset=["speed_kmh", "average_watts"]),
                    x="average_watts",
                    y="speed_kmh",
                    color="bike_name",
                    color_discrete_sequence=_COLOR_SEQ,
                    trendline="lowess",
                    labels={"average_watts": "Avg power (W)", "speed_kmh": f"Speed ({_spd})", "bike_name": "Bike"},
                )
                _fig_flt.update_traces(marker_size=9, selector={"mode": "markers"})
                _fig_flt.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=380)
                st.plotly_chart(_fig_flt, width="stretch")

            else:  # Step 4
                st.caption(
                    "We divide each effort's speed by **power¹⁄³** to get **speed / W¹⁄³** — "
                    "in aerodynamics, speed scales as the cube root of power (v ∝ P¹⁄³), "
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
                        labels={"bike_name": "Bike", "speed_per_cbrt_watt": f"Speed / W¹⁄³ ({_spd}/W¹⁄³)"},
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

    st.divider()
    st.success(
        f"✅ Cleaning settings saved. **{n_clean} efforts** ready for analysis. "
        "Proceed to **Step 3 — Segment Comparison** in the navigation."
    )


main()
