"""Overall bike comparison page — aggregate speed-delta estimation.

Implements the Bike Speed Delta Estimation pipeline (PDF guide):

    Phase 1: Data quality overview
    Phase 2: Spline baseline model (fit on reference bike, project onto all)
    Phase 3: KS power-overlap filter per segment
    Phase 4: Per-segment OLS delta estimation
    Phase 5: Inverse-variance weighted summary, forest plot, interpretability

Key design: the baseline is fit on the reference bike only to avoid absorbing
the bike effect into the fitness/seasonal trend.  Residuals after baseline
removal are what we attribute to the bike.
"""

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

from src.bike_delta import (
    prepare_delta_dataset,
    get_paired_segments,
    fit_baseline_model,
    compute_residuals,
    segment_power_overlap_summary,
    per_segment_delta,
    weighted_delta_summary,
    compute_i2,
    delta_to_sec_per_km,
)
from src.analytics import filter_outliers_by_power_speed
from src.plot_colors import to_rgba
from app.app_pages._ui_helpers import (
    use_metric as _use_metric,
    spd_label as _spd_label,
    convert_speed as _convert_speed,
    gear_label,
)

_COLOR_SEQ: list[str] = px.colors.qualitative.Set2


# ── Dataset builder (cached in session state) ─────────────────────────────────

def _build_delta_df(
    efforts: pd.DataFrame,
    segments: pd.DataFrame,
    bikes: dict[str, str],
    z_threshold: float,
) -> pd.DataFrame:
    """Prepare and outlier-filter the delta dataset, cached in session state."""
    shape_key = f"{len(efforts)}_{len(segments)}_{len(bikes)}_{z_threshold}"
    cached = st.session_state.get("_overall_delta_df")
    if cached is not None and st.session_state.get("_overall_shape_key") == shape_key:
        return cached

    df = prepare_delta_dataset(efforts, segments, bikes)
    if "speed_per_cbrt_watt" in df.columns:
        df, _ = filter_outliers_by_power_speed(df, z_threshold=z_threshold)

    st.session_state["_overall_delta_df"] = df
    st.session_state["_overall_shape_key"] = shape_key
    return df


# ── Main entry point ──────────────────────────────────────────────────────────

def show(bikes_to_compare) -> None:
    """Render the overall bike comparison analysis."""

    # ── Guard: require loaded data ────────────────────────────────────────────
    efforts: pd.DataFrame | None = st.session_state.get("cleaned_efforts")
    segments: pd.DataFrame | None = st.session_state.get("segments")
    bikes: dict[str, str] = st.session_state.get("bikes", {})
    z_threshold: float = float(st.session_state.get("outlier_z_threshold", 2.0))

    if efforts is None or (hasattr(efforts, "empty") and efforts.empty):
        st.info("👈 Head to **Step 1 — Data Collection** to load your Strava data first.")
        st.stop()

    if segments is None or segments.empty:
        st.warning("No starred segments found.  Star segments on Strava and reload from Step 1.")
        st.stop()

    power_efforts = efforts[efforts["average_watts"].notna()].copy()
    if power_efforts.empty:
        st.warning("No efforts with power data found.")
        st.stop()

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📊 Overall settings")

        _all_bike_names = sorted(
            power_efforts["gear_id"]
            .dropna()
            .map(lambda g: gear_label(g, bikes))
            .unique()
            .tolist()
        )

        if len(_all_bike_names) < 2:
            st.warning("Need at least 2 bikes with power data.")
            st.stop()

        if len(bikes_to_compare) < 2:
            st.warning("Select at least 2 bikes.")
            st.stop()

        ref_bike = st.selectbox(
            "Reference bike (baseline)",
            options=bikes_to_compare,
            index=0,
            help=(
                "The baseline fitness/seasonal trend is fit on this bike only. "
                "Choose the bike with the most efforts and most stable training period."
            ),
        )

        spline_df_choice = st.radio(
            "Spline flexibility (df)",
            options=[3, 5, 7],
            index=1,
            horizontal=True,
            help=(
                "Degrees of freedom for the time-trend spline. "
                "Higher = more flexible fitness trend. "
                "Pick the flattest residual-vs-time result (Exp 2 notebook)."
            ),
        )

        min_efforts = st.number_input(
            "Min efforts per bike per segment",
            min_value=1,
            max_value=20,
            value=3,
            step=1,
            help="Both bikes must have ≥ this many power efforts on a segment.",
        )

        ks_threshold = st.slider(
            "KS power-overlap p-threshold",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
            help=(
                "Segments where power distributions differ significantly "
                "(KS p < threshold) are flagged ⚠️."
            ),
        )

        ref_power_w = st.number_input(
            "Reference power for interpretation (W)",
            min_value=50,
            max_value=600,
            value=200,
            step=10,
            help="Used to convert the delta to seconds-per-km.",
        )

    # ── Build analysis dataset ────────────────────────────────────────────────
    try:
        df = _build_delta_df(power_efforts, segments, bikes, z_threshold)
    except Exception as e:
        st.error(f"Failed to prepare analysis dataset: {e}")
        st.stop()

    df_scope = df[df["bike_name"].isin(bikes_to_compare)].copy()

    # ── Phase 1: Data quality overview ────────────────────────────────────────
    st.subheader("Phase 1 — Data quality")
    with st.expander("Segment coverage & effort counts", expanded=False):
        pivot = (
            df_scope.groupby(["segment_id", "bike_name"])["effort_id"]
            .count()
            .unstack(fill_value=0)
        )
        for b in bikes_to_compare:
            if b not in pivot.columns:
                pivot[b] = 0

        seg_names = segments[["segment_id", "name"]].drop_duplicates("segment_id")
        pivot = pivot.merge(seg_names, left_index=True, right_on="segment_id", how="left")
        pivot = pivot.set_index("name")[bikes_to_compare]

        col_cov, col_meta = st.columns([3, 1])
        with col_cov:
            st.caption("Effort count per segment per bike")
            fig_heat = go.Figure(
                go.Heatmap(
                    z=pivot.values.tolist(),
                    x=pivot.columns.tolist(),
                    y=pivot.index.tolist(),
                    colorscale="Blues",
                    text=pivot.values.tolist(),
                    texttemplate="%{text}",
                    showscale=True,
                )
            )
            fig_heat.update_layout(
                height=max(200, 40 * len(pivot) + 80),
                margin={"l": 180, "r": 20, "t": 20, "b": 40},
                xaxis_title="Bike",
                yaxis_title="Segment",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_heat, width="stretch")

        with col_meta:
            paired = get_paired_segments(df_scope, bikes_to_compare, min_efforts=int(min_efforts))
            st.metric("Total segments", len(pivot))
            st.metric(f"Paired (≥{int(min_efforts)} each)", len(paired))
            if len(paired) < 10:
                st.warning(
                    f"Only **{len(paired)} paired segments** — CIs will be wide. "
                    "Ride more segments, or reduce the min-efforts threshold."
                )

        st.markdown("**Avg power per bike (W)**")
        pwr_summary = (
            df_scope.groupby("bike_name")["average_watts"]
            .describe()[["count", "mean", "std", "min", "50%", "max"]]
            .rename(columns={"50%": "median"})
            .round(1)
        )
        st.dataframe(pwr_summary, width="stretch")

    # ── Phase 2: Baseline model ────────────────────────────────────────────────
    st.divider()
    st.subheader("Phase 2 — Spline baseline model")
    st.caption(
        f"Baseline fit on **{ref_bike}** only (spline df={spline_df_choice}). "
        "Residuals = actual efficiency − predicted trend. "
        "Reference bike residuals should hover near zero."
    )

    paired_segs = get_paired_segments(df_scope, bikes_to_compare, min_efforts=int(min_efforts))
    if not paired_segs:
        st.warning(
            f"No segments with ≥{int(min_efforts)} efforts for all selected bikes. "
            "Adjust the threshold or select different bikes."
        )
        st.stop()

    try:
        with st.spinner("Fitting baseline model…"):
            model = fit_baseline_model(
                df_scope, ref_bike
            )
            df_resid = compute_residuals(df_scope, model)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    except ImportError as e:
        st.error(str(e))
        st.stop()

    # ── Residuals vs time ──────────────────────────────────────────────────────
    resid_plot_df = df_resid.dropna(subset=["residual"]).copy()
    resid_plot_df["_dt"] = pd.to_datetime(
        resid_plot_df["start_date"], errors="coerce", utc=True
    ).dt.tz_convert(None)

    fig_resid = go.Figure()
    for idx, bike in enumerate(bikes_to_compare):
        bdata = resid_plot_df[resid_plot_df["bike_name"] == bike].sort_values("_dt")
        color = _COLOR_SEQ[idx % len(_COLOR_SEQ)]
        if bdata.empty:
            continue
        fig_resid.add_trace(
            go.Scatter(
                x=bdata["_dt"],
                y=bdata["residual"],
                mode="markers",
                name=bike,
                marker={"color": color, "size": 7, "opacity": 0.7},
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d}<br>Residual: %{y:.4f}<extra>" + bike + "</extra>"
                ),
            )
        )
        # 30-day rolling mean
        bdata_idx = bdata.set_index("_dt").sort_index()
        roll = bdata_idx["residual"].rolling("30D", min_periods=3).mean().reset_index()
        if len(roll) >= 3:
            fig_resid.add_trace(
                go.Scatter(
                    x=roll["_dt"],
                    y=roll["residual"],
                    mode="lines",
                    name=f"{bike} 30d avg",
                    line={"color": color, "width": 2.5, "dash": "dot"},
                    hovertemplate="30d avg: %{y:.4f}<extra>" + bike + "</extra>",
                )
            )

    fig_resid.add_hline(y=0, line_dash="dash", line_color="grey", line_width=1)
    fig_resid.update_layout(
        xaxis_title="Date",
        yaxis_title="Residual (speed/W¹⁄³)",
        height=380,
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.3},
    )
    st.plotly_chart(fig_resid, width="stretch")
    st.caption(
        "✅ Good: reference bike residuals near 0 throughout, level shift at bike transition. "
        "⚠️ Residuals trending up during reference period → increase spline df."
    )

    with st.expander("Spline df sensitivity (df = 3 / 5 / 7)", expanded=False):
        st.caption("Pick the df value that gives the flattest within-reference-bike residuals.")
        _fig_sens = go.Figure()
        for _dv, _dash in [(3, "dot"), (5, "solid"), (7, "dash")]:
            try:
                _m = fit_baseline_model(df_scope, ref_bike)
                _dr = compute_residuals(df_scope, _m)
                _rdata = (
                    _dr[_dr["bike_name"] == ref_bike]
                    .dropna(subset=["residual"])
                    .copy()
                )
                _rdata["_dt"] = pd.to_datetime(
                    _rdata["start_date"], errors="coerce", utc=True
                ).dt.tz_convert(None)
                _rdata = _rdata.sort_values("_dt")
                _fig_sens.add_trace(
                    go.Scatter(
                        x=_rdata["_dt"],
                        y=_rdata["residual"],
                        mode="markers",
                        name=f"df={_dv}",
                        marker={"size": 6, "opacity": 0.65},
                        line={"dash": _dash},
                    )
                )
            except Exception:
                pass
        _fig_sens.add_hline(y=0, line_dash="dash", line_color="grey", line_width=1)
        _fig_sens.update_layout(
            xaxis_title="Date",
            yaxis_title=f"{ref_bike} residuals",
            height=300,
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(_fig_sens, width="stretch")

    # ── Phase 3: KS power-overlap filter ──────────────────────────────────────
    st.divider()
    st.subheader("Phase 3 — Power distribution overlap")
    st.caption(
        "KS test checks whether power distributions between bikes are similar on each segment. "
        f"Flagged (⚠️) when KS p < {ks_threshold:.2f} — potential confounding."
    )

    ks_summary = segment_power_overlap_summary(
        df_resid, bikes_to_compare, paired_segs, p_threshold=ks_threshold
    )
    if not ks_summary.empty:
        seg_names_map = (
            dict(zip(segments["segment_id"], segments["name"]))
            if "name" in segments.columns
            else {}
        )
        ks_disp = ks_summary.copy()
        ks_disp["segment"] = ks_summary["segment_id"].map(lambda s: seg_names_map.get(s, str(s)))
        ks_disp["status"] = ks_summary["ks_ok"].map({True: "✅ OK", False: "⚠️ Flagged"})
        ks_disp = ks_disp[["segment", "bike_a", "bike_b", "p_value", "status"]].rename(
            columns={"bike_a": "Bike A", "bike_b": "Bike B",
                     "p_value": "KS p-value", "status": "Power overlap"}
        )
        n_flagged = int((~ks_summary["ks_ok"]).sum())
        if n_flagged:
            st.warning(
                f"**{n_flagged} segment(s)** flagged — power distributions differ between bikes. "
                "Their deltas may reflect pacing differences rather than equipment."
            )
        st.dataframe(ks_disp, width="stretch", hide_index=True)

    # ── Phase 4: Per-segment delta estimation ──────────────────────────────────
    st.divider()
    st.subheader("Phase 4 — Per-segment speed delta")
    st.caption(
        f"OLS: *residual ~ C(bike) + avg_power* per segment.  "
        f"Reference: **{ref_bike}**.  Positive delta = other bike is faster."
    )

    seg_names_map = (
        dict(zip(segments["segment_id"], segments["name"]))
        if "name" in segments.columns
        else {}
    )

    with st.spinner("Estimating per-segment deltas…"):
        deltas_df = per_segment_delta(df_resid, paired_segs, ref_bike, bikes_to_compare)

    if deltas_df.empty:
        st.info("No per-segment deltas could be estimated.  Check that paired segments have enough efforts.")
        st.stop()

    deltas_disp = deltas_df.copy()
    deltas_disp["segment"] = deltas_disp["segment_id"].map(lambda s: seg_names_map.get(s, str(s)))
    deltas_disp["95% CI"] = deltas_disp.apply(
        lambda r: f"[{r['delta'] - 1.96 * r['se']:.4f}, {r['delta'] + 1.96 * r['se']:.4f}]", axis=1
    )
    deltas_disp["delta"] = deltas_disp["delta"].round(5)
    deltas_disp["grade (%)"] = deltas_disp["grade"].round(1)
    deltas_disp["length (m)"] = deltas_disp["length_m"].round(0)
    table_cols = ["segment", "other_bike", "delta", "95% CI", "n_ref", "n_other", "grade (%)", "length (m)"]
    st.dataframe(
        deltas_disp[[c for c in table_cols if c in deltas_disp.columns]].rename(
            columns={"other_bike": "vs bike", "delta": "delta (speed/W¹⁄³)"}
        ),
        width="stretch",
        hide_index=True,
    )

    with st.expander("Delta vs segment grade", expanded=False):
        st.caption(
            "Aero gains → larger deltas on flat segments. "
            "Climbing gains → roughly grade-independent. "
            "Random scatter → noise dominates."
        )
        for idx, other in enumerate([b for b in bikes_to_compare if b != ref_bike]):
            pair_data = deltas_df[deltas_df["other_bike"] == other].dropna(subset=["grade"])
            if pair_data.empty:
                continue
            fig_grade = px.scatter(
                pair_data,
                x="grade",
                y="delta",
                color_discrete_sequence=[_COLOR_SEQ[idx % len(_COLOR_SEQ)]],
                labels={"grade": "Segment grade (%)", "delta": "Delta (speed/W¹⁄³)"},
                title=f"{ref_bike} → {other}",
                trendline="ols",
            )
            fig_grade.add_hline(y=0, line_dash="dash", line_color="grey", line_width=1)
            fig_grade.update_layout(height=320, plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_grade, width="stretch")

    # ── Phase 5: Aggregate summary ─────────────────────────────────────────────
    st.divider()
    st.subheader("Phase 5 — Aggregate summary")

    summary_df = weighted_delta_summary(deltas_df)
    i2_vals = compute_i2(deltas_df)

    # Forest plot
    st.markdown("**Forest plot** — per-segment deltas with 95% CI")
    fig_forest = go.Figure()
    for idx, other in enumerate([b for b in bikes_to_compare if b != ref_bike]):
        pair_key = f"{ref_bike} → {other}"
        pair_data = deltas_df[deltas_df["other_bike"] == other].copy()
        color = _COLOR_SEQ[idx % len(_COLOR_SEQ)]
        seg_labels = pair_data["segment_id"].map(lambda s: seg_names_map.get(s, str(s)))

        fig_forest.add_trace(
            go.Scatter(
                x=pair_data["delta"],
                y=seg_labels,
                mode="markers",
                name=other,
                marker={"color": color, "size": 9},
                error_x={
                    "type": "data",
                    "array": (1.96 * pair_data["se"]).tolist(),
                    "visible": True,
                    "color": color,
                },
                customdata=list(
                    zip(
                        (pair_data["delta"] - 1.96 * pair_data["se"]).tolist(),
                        (pair_data["delta"] + 1.96 * pair_data["se"]).tolist(),
                    )
                ),
                hovertemplate=(
                    "Segment: %{y}<br>Delta: %{x:.5f}<br>"
                    "95% CI: [%{customdata[0]:.5f}, %{customdata[1]:.5f}]<extra>"
                    + other + "</extra>"
                ),
            )
        )
        if pair_key in summary_df["bike_pair"].values:
            pooled_d = float(summary_df[summary_df["bike_pair"] == pair_key]["delta"].iloc[0])
            fig_forest.add_vline(
                x=pooled_d,
                line_dash="dot",
                line_color=color,
                line_width=2,
                annotation_text=f"Pooled {other}",
                annotation_position="top",
            )

    fig_forest.add_vline(x=0, line_dash="dash", line_color="grey", line_width=1.5)
    fig_forest.update_layout(
        xaxis_title="Delta (speed/W¹⁄³) — positive = faster than reference",
        yaxis_title="Segment",
        height=max(350, 35 * len(deltas_df["segment_id"].unique()) + 80),
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.2},
    )
    st.plotly_chart(fig_forest, width="stretch")

    # Weighted summary table
    st.markdown("**Weighted summary** (inverse-variance weighted)")
    _ref_speed_ms = 30.0 / 3.6
    summary_rows = []
    for _, row in summary_df.iterrows():
        pair = str(row["bike_pair"])
        delta = float(row["delta"])
        sec_km = delta_to_sec_per_km(delta, ref_power=float(ref_power_w), ref_speed_ms=_ref_speed_ms)
        speed_gain_kmh = delta * (float(ref_power_w) ** (1.0 / 3.0))
        speed_gain_pct = speed_gain_kmh / 30.0 * 100
        i2 = i2_vals.get(pair, 0.0)
        i2_label = "🟢 Low" if i2 < 0.25 else ("🟡 Moderate" if i2 < 0.75 else "🔴 High")
        summary_rows.append({
            "Comparison": pair,
            "Delta (speed/W¹⁄³)": f"{delta:+.5f}",
            "95% CI": f"[{row['ci_low']:+.5f}, {row['ci_high']:+.5f}]",
            f"Sec/km @ {int(ref_power_w)}W": f"{sec_km:+.1f}",
            f"Speed Δ @ {int(ref_power_w)}W": f"{speed_gain_kmh:+.2f} km/h ({speed_gain_pct:+.1f}%)",
            "Segments": int(row["n_segments"]),
            "I²": f"{i2:.2f} {i2_label}",
        })
    st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

    # Sanity warnings
    for row in summary_df.to_dict("records"):
        speed_gain_pct = abs(float(row["delta"]) * (float(ref_power_w) ** (1.0 / 3.0)) / 30.0)
        if speed_gain_pct > 0.05:
            st.warning(
                "⚠️ At least one estimate exceeds 5% speed gain — above the typical range "
                "for equipment differences. Check for data errors or confounding."
            )
            break

    if any(v > 0.75 for v in i2_vals.values()):
        st.warning(
            "⚠️ I² > 0.75 (high heterogeneity) — results vary substantially across segments. "
            "Decompose by segment type before drawing conclusions. See the Exp 5 notebook."
        )

    # ── Symmetry & transitivity check ─────────────────────────────────────────
    st.divider()
    st.subheader("Symmetry & transitivity check")
    st.caption(
        "Re-runs the pipeline with each other bike as reference. "
        "A robust analysis satisfies |delta(A→B) + delta(B→A)| < 0.01."
    )

    if st.button("Run symmetry check", help="Re-fits the baseline with each other bike as reference."):
        sym_results: list[dict] = []
        for other_ref in [b for b in bikes_to_compare if b != ref_bike]:
            try:
                with st.spinner(f"Fitting baseline on {other_ref}…"):
                    _m2 = fit_baseline_model(df_scope, other_ref)
                    _dr2 = compute_residuals(df_scope, _m2)
                    _d2 = per_segment_delta(_dr2, paired_segs, other_ref, bikes_to_compare)
                    _s2 = weighted_delta_summary(_d2)
            except Exception as e:
                st.warning(f"Symmetry check for {other_ref} failed: {e}")
                continue

            fwd_key = f"{ref_bike} → {other_ref}"
            rev_key = f"{other_ref} → {ref_bike}"
            fwd = summary_df[summary_df["bike_pair"] == fwd_key]["delta"].values
            rev = _s2[_s2["bike_pair"] == rev_key]["delta"].values

            if len(fwd) and len(rev):
                sym_error = float(fwd[0]) + float(rev[0])
                sym_results.append({
                    "Pair": f"{ref_bike} ↔ {other_ref}",
                    f"δ({ref_bike}→{other_ref})": f"{float(fwd[0]):+.5f}",
                    f"δ({other_ref}→{ref_bike})": f"{float(rev[0]):+.5f}",
                    "Sum (≈ 0 ideal)": f"{sym_error:+.5f}",
                    "Symmetric?": "✅" if abs(sym_error) < 0.01 else "⚠️",
                })

            # Transitivity for 3+ bikes
            if len(bikes_to_compare) >= 3:
                others_for_trans = [b for b in bikes_to_compare if b not in (ref_bike, other_ref)]
                for c_bike in others_for_trans[:1]:  # check first triplet
                    d_ac = summary_df[summary_df["bike_pair"] == f"{ref_bike} → {c_bike}"]["delta"].values
                    d_ab = summary_df[summary_df["bike_pair"] == f"{ref_bike} → {other_ref}"]["delta"].values
                    d_bc = _s2[_s2["bike_pair"] == f"{other_ref} → {c_bike}"]["delta"].values
                    if len(d_ac) and len(d_ab) and len(d_bc):
                        trans_err = float(d_ac[0]) - (float(d_ab[0]) + float(d_bc[0]))
                        sym_results.append({
                            "Pair": f"Transitivity: {ref_bike}→{c_bike}",
                            f"δ({ref_bike}→{other_ref})": f"{float(d_ab[0]):+.5f}",
                            f"δ({other_ref}→{c_bike})": f"{float(d_bc[0]):+.5f}",
                            "Sum (≈ 0 ideal)": f"{trans_err:+.5f}",
                            "Symmetric?": "✅" if abs(trans_err) < 0.01 else "⚠️",
                        })

        if sym_results:
            st.dataframe(pd.DataFrame(sym_results), width="stretch", hide_index=True)
        else:
            st.info("No symmetry results could be computed.")

        st.caption(
            "If |sum| >> 0.01, the fitness trend may be absorbing the bike effect. "
            "Consider bridging data (rides on both bikes in the same period), "
            "adjusting the spline df, or using a shorter time window."
        )

    st.caption(
        "💡 **Dig deeper**: see the `notebooks/exp1_*` through `notebooks/exp5_*` notebooks "
        "for step-by-step experiment checkpoints — residual QQ-plots, power overlap details, "
        "grade decompositions, and full symmetry/transitivity tables."
    )

