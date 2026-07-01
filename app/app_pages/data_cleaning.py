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

from src.plots import(
    make_fig_b_h,
    make_fig_b_sc 
)

from src._ui_helpers import (
    use_metric as _use_metric,
    spd_label as _spd_label,
    convert_speed as _convert_speed,
    gear_label,
    compute_speed_kmh as _compute_speed_kmh,
    get_available_bikes
)

from src.utils import navigator, page_guard

# ── Page ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("🧹 Step 2 — Data Cleaning")
    st.markdown(
        "Configure how raw Strava efforts are filtered before analysis. "
        "Noisy efforts — coasting, equipment glitches, drafting — can skew comparisons. "
        "The settings you choose here are applied on every subsequent page."
    )

    page_guard("data_cleaning")
    segments: pd.DataFrame | None = st.session_state.get("segments")
    bikes: dict[str, str] = st.session_state.get("bikes", {})
    efforts_with_power: pd.DataFrame | None = st.session_state.get("efforts")

    # Merge segment metadata so we have segment_type available
    seg_meta_cols = ["segment_id", "name", "distance", "average_grade", "maximum_grade", "segment_type", "hazardous"]
    if "segment_type_detail" in segments.columns:
        seg_meta_cols.append("segment_type_detail")
    seg_meta = segments[seg_meta_cols].copy()
    efforts_with_power = efforts_with_power.merge(seg_meta, on="segment_id", how="inner")
    efforts_with_power["bike_name"] = efforts_with_power["gear_id"].map(
        lambda g: gear_label(g, bikes)
    )
    efforts_with_power["speed_kmh"] = _compute_speed_kmh(efforts_with_power)

    # ── Two-column layout: controls left, visualisations right ────────────────
    left_col, right_col = st.columns([1, 2], vertical_alignment="top")

    # ── LEFT: Cleaning controls ────────────────────────────────────────────────
    with left_col:
        st.subheader("⚙️ Cleaning parameters")

        ftp = st.session_state.get("ftp")
        _default_min_watts = int(round(ftp * 0.5)) if ftp is not None and ftp > 0 else 0
        st.session_state.setdefault("min_watts", _default_min_watts)
        min_watts: int = st.number_input(
            "Minimum watts",
            min_value=0,
            max_value=999,
            step=5,
            key="min_watts",
            help=(
                "Efforts with average power below this threshold are excluded from analysis. "
                "Default is 50 % of your FTP (if available). Set to 0 to disable."
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
                "Remove descent segments entirely from analysis pages. Separate from outlier filtering. "
                "Because power doesn't create speed as much on descents our outlier detection method doesnt work for them."
            ),
        )


    # ── Apply filters and store cleaned efforts in session state ───────────────
    efforts_with_power = compute_speed_per_watt(efforts_with_power)
    _after_z, _ = filter_outliers_by_power_speed(efforts_with_power, z_threshold=z_threshold)
    cleaned_pre_d = apply_min_watts_filter(
        _after_z,
        int(min_watts),
        descents_exempt=True,
    )
    if exclude_descents and "segment_type" in cleaned_pre_d.columns:
        cleaned = cleaned_pre_d[cleaned_pre_d["segment_type"] != "descent"].copy()
    else:
        cleaned = cleaned_pre_d

    #save for later
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

    with left_col:
        # down here because needs to happen after filtering from selections
        st.caption("Pick a segment and step to explore how filtering works.")

        if not _seg_effort_counts_ok:
            st.info("Not enough efforts per segment to illustrate outlier detection (need ≥ 3).")
            example_seg_id = None
        else:
            example_seg_id: int = st.selectbox(
                "Example segment",
                options=_seg_options_sorted,
                format_func=lambda sid: _seg_name_map.get(int(sid), str(sid)),
                index=0,
                key="cleaning_explainer_seg",
            )

    # ── RIGHT: filter summary + visualisations ─────────────────────────────────
    with right_col:
        st.subheader("📊 Filter summary")
        if exclude_descents and "segment_type" in cleaned.columns:
            m1, m2, m3, m4, mf = st.columns(5)
        else:
            m1, m2, m3, mf = st.columns(4)
            m4 = None
        with m1:
            st.metric("Raw efforts", n_raw)
        with m2:
            st.metric(
                "Z-Score Outliers",
                n_raw - len(_after_z),
                help=f"Efforts where we got meaningfully more or less speed than anticipated given the watts recorded. (Windy/Drafting/Heavy Braking)",
            )
        with m3:
            st.metric(
                "Min-Watts Filter",
                n_raw - len(_after_mw),
                help=f"Efforts with average power < {int(min_watts)} W (descents are always included)",
            )
        if m4 is not None:
            with m4:
                st.metric(
                    "Descent Filter",
                    len(cleaned_pre_d[cleaned_pre_d["segment_type"] == "descent"]),
                )
        with mf:
            st.metric(
                "Remaining",
                len(cleaned),
                help=f"These will be used in analysis on following pages",
            )

        st.divider()

        if cleaned.empty:
            st.info("No efforts remain after filtering. Adjust the parameters on the left.")
        elif not _seg_effort_counts_ok:
            st.info("Not enough efforts per segment to show outlier detection (need ≥ 3 per segment).")
        else:
            _spd = _spd_label()
            _COLOR_SEQ = px.colors.qualitative.Set2
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

                _ann_bikes = sorted(_plot_ann["bike_name"].dropna().unique().tolist())
                if _ann_bikes:
                    bike_tabs = st.tabs(_ann_bikes)
                    for _bi, _bname in enumerate(_ann_bikes):
                        with bike_tabs[_bi]:
                            _bike_color = _COLOR_SEQ[_bi % len(_COLOR_SEQ)]
                            _bdata = _plot_ann[_plot_ann["bike_name"] == _bname].copy()
                            _n_b_out = int(_bdata["is_outlier"].sum())
                            _n_b_kept = len(_bdata) - _n_b_out

                            _spw_b = _bdata.dropna(subset=["speed_per_cbrt_watt"])
                            if len(_spw_b) >= 2:
                                _b_mean = _spw_b["speed_per_cbrt_watt"].mean()
                                _b_std = _spw_b["speed_per_cbrt_watt"].std(ddof=1)
                                _b_lo = _b_mean - z_threshold * _b_std
                                _b_hi = _b_mean + z_threshold * _b_std
                                _nbins = max(6, len(_spw_b) // 2)
                            else:
                                st.caption("Not enough efforts to show distribution.")
                            st.markdown(f"##### {_bname}")
                            _sc_col, _hs_col = st.columns(2)

                            with _sc_col:
                                st.markdown("Speed vs power")
                                _fig_b_sc = make_fig_b_sc(_bike_color, _bdata, _spd, z_threshold, _b_lo, _b_hi, _b_mean) 
                                st.plotly_chart(_fig_b_sc, width="stretch", config={"staticPlot": True})

                            with _hs_col:
                                st.markdown(f"Speed/W¹⁄³ distribution ±{z_threshold:.1f}σ cutoff")
                                _fig_b_h = make_fig_b_h(_bike_color, _spw_b, _b_lo, _b_hi, _b_mean, _nbins, _spd, z_threshold) 
                                st.plotly_chart(_fig_b_h, width="stretch", config={"staticPlot": True})

                            st.caption(
                                f"🔴 **{_n_b_out} outlier(s)** · 🔵 **{_n_b_kept} kept** "
                                f"(μ = {_bdata['speed_per_cbrt_watt'].mean():.4f}, "
                                f"σ = {_bdata['speed_per_cbrt_watt'].std(ddof=1):.4f})"
                            )

    st.divider()
    st.success(
        f"✅ Cleaning settings saved. **{n_clean} efforts** ready for analysis. "
        "Proceed to **Step 3 — Segment Comparison** in the navigation."
    )

navigator("data_cleaning1")
main()
navigator("data_cleaning2")
