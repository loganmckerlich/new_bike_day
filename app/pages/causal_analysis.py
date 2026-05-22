"""Causal analysis page for bike speed-per-watt treatment effects."""

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

from src.causal_inference import (
    build_feature_matrix,
    estimate_heterogeneous_effects,
    estimate_treatment_effect,
    get_shap_importances,
    remove_outliers_for_causal_analysis,
)


def _use_metric() -> bool:
    return st.session_state.get("use_metric", True)


def _speed_unit() -> str:
    return "km/h" if _use_metric() else "mph"


def _to_display_speed(kmh: float) -> float:
    """Convert km/h to the currently selected display unit."""
    return kmh if _use_metric() else kmh * 0.621371


def _gear_label(gear_id: str, bikes: dict[str, str]) -> str:
    """Return a display label for a gear id."""
    return bikes.get(str(gear_id), str(gear_id))


def _selection_frame(efforts: pd.DataFrame, old_gear_id: str, new_gear_id: str) -> pd.DataFrame:
    """Filter to selected bikes and create binary treatment indicator."""
    selected = efforts[efforts["gear_id"].astype(str).isin([old_gear_id, new_gear_id])].copy()
    selected["is_new_bike"] = (selected["gear_id"].astype(str) == new_gear_id).astype(int)
    return selected


def _effect_text(
    ate: float,
    low: float,
    high: float,
    comparison_label: str,
    baseline_label: str,
    speed_unit: str = "km/h",
    ate_pct: float | None = None,
    low_pct: float | None = None,
    high_pct: float | None = None,
) -> str:
    """Create plain-English headline interpretation for estimated effect."""
    direction = "faster" if ate >= 0 else "slower"
    pct_clause = ""
    if ate_pct is not None and low_pct is not None and high_pct is not None:
        pct_clause = f" — **{abs(ate_pct):.1f}% {direction}** relative to {baseline_label}'s average pace (95% CI: {low_pct:.1f}% to {high_pct:.1f}%)"
    return (
        "Controlling for wind, temperature, precipitation, road straightness, and terrain, "
        f"**{comparison_label}** is estimated to be **{abs(ate):.2f} {speed_unit} {direction}** "
        f"than **{baseline_label}** "
        f"(95% CI: {low:.2f} to {high:.2f} {speed_unit}){pct_clause}."
    )


def _render_terrain_chart(
    terrain_df: pd.DataFrame, mean_cbrt_watts: float, comparison_label: str, speed_unit: str = "km/h"
) -> None:
    """Render terrain-specific effect chart with confidence intervals."""
    if terrain_df.empty:
        st.info("Not enough terrain-specific coverage to estimate heterogeneous effects yet.")
        return

    terrain_df = terrain_df.copy()
    terrain_df["ate_kmh"] = terrain_df["ate"].apply(lambda v: _to_display_speed(v * mean_cbrt_watts))
    terrain_df["ate_lower_kmh"] = terrain_df["ate_lower"].apply(lambda v: _to_display_speed(v * mean_cbrt_watts))
    terrain_df["ate_upper_kmh"] = terrain_df["ate_upper"].apply(lambda v: _to_display_speed(v * mean_cbrt_watts))
    terrain_df["color"] = terrain_df["ate_kmh"].apply(lambda x: "Faster" if x >= 0 else "Slower")
    terrain_df["err_plus"] = terrain_df["ate_upper_kmh"] - terrain_df["ate_kmh"]
    terrain_df["err_minus"] = terrain_df["ate_kmh"] - terrain_df["ate_lower_kmh"]

    fig = px.bar(
        terrain_df.sort_values("ate_kmh"),
        x="ate_kmh",
        y="segment_type",
        orientation="h",
        color="color",
        color_discrete_map={"Faster": "#16a34a", "Slower": "#dc2626"},
        title=f"Where is {comparison_label} faster?",
    )
    fig.update_traces(
        error_x={"type": "data", "array": terrain_df["err_plus"], "arrayminus": terrain_df["err_minus"]},
        hovertemplate=f"%{{y}}: %{{x:.2f}} {speed_unit}<extra></extra>",
    )
    fig.update_layout(yaxis_title="Segment type", xaxis_title=f"ATE ({speed_unit})", showlegend=False)
    st.plotly_chart(fig, width="stretch")


def _render_control_chart(shap_df: pd.DataFrame) -> None:
    """Render SHAP importance chart with human-readable feature labels."""
    labels = {
        "straightness_index": "Road straightness",
        "headwind_component": "Headwind / tailwind",
        "precipitation_mm": "Precipitation",
        "temp_c": "Temperature",
        "average_grade": "Gradient",
    }

    chart_df = shap_df.copy()
    chart_df["label"] = chart_df["feature"].map(labels).fillna(chart_df["feature"])
    chart_df = chart_df.sort_values("mean_abs_shap", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=chart_df["mean_abs_shap"],
            y=chart_df["label"],
            orientation="h",
            marker_color="#2563eb",
            hovertemplate="%{y}: %{x:.5f}<extra></extra>",
        )
    )
    fig.update_layout(title="What did we control for?", xaxis_title="Mean |SHAP|", yaxis_title="")
    st.plotly_chart(fig, width="stretch")
    st.caption("Weather features are stubbed — importances will update once real data is connected.")


def main() -> None:
    """Render the causal analysis workflow and results."""
    st.title("🧪 Step 4 — Bike Head to Head")
    st.markdown(
        "Estimate the direct speed-per-cbrt-watt impact between two bikes with doubly robust causal inference. "
        "Data has been pre-cleaned in **Step 2 — Data Cleaning**."
    )

    # ── Read shared analysis params from session state ─────────────────────────
    z_threshold: float = float(st.session_state.get("outlier_z_threshold", 2.0))

    efforts: pd.DataFrame | None = st.session_state.get("cleaned_efforts")
    segments: pd.DataFrame | None = st.session_state.get("segments")
    bikes: dict[str, str] = st.session_state.get("bikes", {})

    if st.session_state.get("efforts") is None:
        st.info("👈 Head to **Step 1 — Data Collection** to sign in with Strava and load your data first.")
        st.stop()

    if efforts is None or efforts.empty:
        st.info("👈 Head to **Step 2 — Data Cleaning** to configure and apply data filters first.")
        st.stop()
    if segments is None or segments.empty:
        st.warning("No segment metadata available yet. Reload from Step 1 first.")
        st.stop()

    gear_counts = (
        efforts.dropna(subset=["gear_id"])
        .assign(gear_id=lambda x: x["gear_id"].astype(str))
        .groupby("gear_id", as_index=False)["effort_id"]
        .count()
        .rename(columns={"effort_id": "n_efforts"})
        .sort_values("n_efforts", ascending=False)
    )

    if len(gear_counts) < 2:
        st.warning("Need efforts from at least two bikes to run causal analysis.")
        st.stop()

    gear_options = gear_counts["gear_id"].tolist()
    gear_labels = {g: f"{_gear_label(g, bikes)} ({int(gear_counts.loc[gear_counts['gear_id'] == g, 'n_efforts'].iloc[0])} efforts)" for g in gear_options}

    col_old, col_new = st.columns(2)
    with col_old:
        old_gear_id = st.selectbox("Bike One", options=gear_options, format_func=lambda g: gear_labels[g], index=0)
    with col_new:
        default_new_idx = 1 if len(gear_options) > 1 else 0
        new_gear_id = st.selectbox("Bike Two", options=gear_options, format_func=lambda g: gear_labels[g], index=default_new_idx)

    if old_gear_id == new_gear_id:
        st.warning("Select two different bikes.")
        st.stop()

    baseline_label = _gear_label(old_gear_id, bikes)
    comparison_label = _gear_label(new_gear_id, bikes)
    selected_efforts_raw = _selection_frame(efforts, old_gear_id, new_gear_id)
    selected_efforts_raw = selected_efforts_raw.copy()
    selected_efforts_raw["bike_name"] = (
        selected_efforts_raw["gear_id"]
        .astype(str)
        .map({str(old_gear_id): baseline_label, str(new_gear_id): comparison_label})
        .fillna("Unknown")
    )

    # min_watts and descent filters already applied by Data Cleaning page;
    # still apply per-segment outlier removal here as it is analysis-specific.
    n_treated_raw = int(selected_efforts_raw["is_new_bike"].sum())
    n_control_raw = int((selected_efforts_raw["is_new_bike"] == 0).sum())
    selected_efforts, n_outliers = remove_outliers_for_causal_analysis(
        selected_efforts_raw,
        z_threshold=z_threshold,
        segments_df=segments,
    )
    n_treated_clean = int(selected_efforts["is_new_bike"].sum())
    n_control_clean = int((selected_efforts["is_new_bike"] == 0).sum())
    if n_treated_clean < 30 or n_control_clean < 30:
        st.warning(
            "Guardrail: at least 30 non-outlier efforts are required for each bike before running the model "
            f"({comparison_label}={n_treated_clean}, {baseline_label}={n_control_clean}; "
            f"outliers removed={n_outliers}, z-threshold={z_threshold:.1f})."
        )
        st.stop()

    features = build_feature_matrix(selected_efforts, segments)
    if features.empty:
        st.warning("No valid rows after filtering (requires average_watts >= 50 and complete joins).")
        st.stop()

    if "gear_id" in features.columns:
        features = features.copy()
        features["bike_name"] = (
            features["gear_id"]
            .astype(str)
            .map({str(old_gear_id): baseline_label, str(new_gear_id): comparison_label})
            .fillna("Unknown")
        )

    ate_result = estimate_treatment_effect(features)
    mean_watts = float(features["average_watts"].mean())
    if pd.isna(mean_watts) or mean_watts <= 0:
        mean_watts = 1.0
    mean_cbrt_watts = float(np.cbrt(mean_watts))
    ate_kmh = ate_result["ate"] * mean_cbrt_watts
    ate_low_kmh = ate_result["ate_lower"] * mean_cbrt_watts
    ate_high_kmh = ate_result["ate_upper"] * mean_cbrt_watts
    terrain_effects = estimate_heterogeneous_effects(features)
    shap_importances = get_shap_importances(features)
    speed_unit = _speed_unit()
    ate_disp = _to_display_speed(ate_kmh)
    ate_low_disp = _to_display_speed(ate_low_kmh)
    ate_high_disp = _to_display_speed(ate_high_kmh)

    # Percentage effect relative to baseline bike's observed mean speed
    baseline_speed_mps = features.loc[features["is_new_bike"] == 0, "average_speed_mps"].mean()
    if pd.notna(baseline_speed_mps) and baseline_speed_mps > 0:
        baseline_mean_kmh = baseline_speed_mps * 3.6
        ate_pct = ate_kmh / baseline_mean_kmh * 100
        ate_low_pct = ate_low_kmh / baseline_mean_kmh * 100
        ate_high_pct = ate_high_kmh / baseline_mean_kmh * 100
    else:
        ate_pct = ate_low_pct = ate_high_pct = None

    with st.expander("How this works", expanded=False):
        st.markdown(
            f"### 1) Define the comparison\n"
            f"- **Bike One:** {baseline_label}\n"
            f"- **Bike Two:** {comparison_label}\n"
            "Every effort from Bike Two gets treatment = 1 and every effort from Bike One gets treatment = 0."
        )
        _sample_counts = (
            selected_efforts.assign(
                bike_name=lambda d: d["gear_id"].astype(str).map(
                    {old_gear_id: baseline_label, new_gear_id: comparison_label}
                )
            )
            .groupby("bike_name", as_index=False)["effort_id"]
            .count()
            .rename(columns={"effort_id": "Efforts", "bike_name": "Bike"})
        )
        _fig_counts = px.bar(
            _sample_counts,
            x="Bike",
            y="Efforts",
            color="Bike",
            text="Efforts",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Sample size by bike",
        )
        _fig_counts.update_layout(showlegend=False, yaxis_title="Efforts", xaxis_title="")
        st.plotly_chart(_fig_counts, width="stretch")

        st.markdown("### 2) Remove outlier efforts")
        st.markdown(
            f"Using the same method as **Segment comparison**, we compute speed per ∛watt for each effort "
            f"and remove efforts beyond ±{z_threshold} standard deviations from that bike's segment-level mean."
        )
        st.caption(
            f"Raw efforts: {len(selected_efforts_raw)} • Removed outliers: {n_outliers} • "
            f"Remaining: {len(selected_efforts)}"
        )

        st.markdown("### 3) Build an apples-to-apples feature matrix")
        st.markdown(
            "Each row keeps the measured output (**speed per ∛watt**) plus confounders "
            "(weather, road geometry, gradient, terrain) so we compare similar conditions."
        )
        _preview_cols = [
            "bike_name",
            "is_new_bike",
            "speed_per_cbrt_watt",
            "average_watts",
            "headwind_component",
            "temp_c",
            "average_grade",
            "segment_type",
        ]
        _preview_cols = [c for c in _preview_cols if c in features.columns]
        st.dataframe(features[_preview_cols].head(8), hide_index=True, width="stretch")

        _fig_overlap = px.scatter(
            features,
            x="average_watts",
            y="speed_per_cbrt_watt",
            color="bike_name",
            trendline="lowess",
            labels={"average_watts": "Avg power (W)", "speed_per_cbrt_watt": f"Speed / ∛power ({speed_unit} / W^⅓)", "bike_name": "Bike"},
            title="Observed speed-per-cbrt-watt at similar power outputs",
        )
        _fig_overlap.update_layout(plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(_fig_overlap, width="stretch")
        st.caption(
            "Smoothed lines use LOWESS (Locally Weighted Scatter Plot Smoothing) "
            "to show the average trend between power and speed-per-watt for each bike."
        )

        st.markdown("### 4) Estimate the adjusted effect")
        st.markdown(
            "The doubly robust learner combines a treatment model and an outcome model. "
            "If either model is well specified, the estimate remains consistent."
        )
        _fig_ci = go.Figure()
        _fig_ci.add_trace(
            go.Scatter(
                x=[ate_low_disp, ate_high_disp],
                y=[0, 0],
                mode="lines",
                line={"color": "#475569", "width": 6},
                hovertemplate=f"95% CI: %{{x:.2f}} {speed_unit}<extra></extra>",
                showlegend=False,
            )
        )
        _fig_ci.add_trace(
            go.Scatter(
                x=[ate_disp],
                y=[0],
                mode="markers",
                marker={"size": 14, "color": "#2563eb"},
                hovertemplate=f"ATE: %{{x:.2f}} {speed_unit}<extra></extra>",
                showlegend=False,
            )
        )
        _fig_ci.add_vline(x=0.0, line_dash="dot", line_color="#ef4444")
        _fig_ci.update_layout(
            title=f"Estimated {comparison_label} effect vs {baseline_label}",
            xaxis_title=f"Adjusted effect ({speed_unit})",
            yaxis={"visible": False},
            plot_bgcolor="rgba(0,0,0,0)",
            height=250,
            margin={"l": 20, "r": 20, "t": 60, "b": 20},
        )
        st.plotly_chart(_fig_ci, width="stretch")
        _pct_lines = ""
        if ate_pct is not None:
            _pct_lines = (
                f"- As a percentage of {baseline_label}'s mean speed: **{ate_pct:+.1f}%** "
                f"(95% CI: {ate_low_pct:.1f}% to {ate_high_pct:.1f}%)\n"
            )
        st.markdown(
            f"### 5) Interpret the result\n"
            f"- Point estimate: **{ate_disp:.2f} {speed_unit}**\n"
            f"- 95% interval: **[{ate_low_disp:.2f}, {ate_high_disp:.2f}] {speed_unit}**\n"
            f"{_pct_lines}"
            f"- Read this as: expected speed difference for **{comparison_label}** relative to **{baseline_label}** "
            "at similar effort and route conditions."
        )

    st.subheader("Section 1 — Headline Result")
    st.info("Weather note: weather inputs are currently dummy stub values and will improve when real weather data is integrated.")
    _m1, _m2 = st.columns(2)
    with _m1:
        st.metric(
            f"Speed difference ({comparison_label} vs {baseline_label})",
            f"{ate_disp:.2f} {speed_unit}",
            help=f"95% CI: {ate_low_disp:.2f} to {ate_high_disp:.2f} {speed_unit}",
        )
    with _m2:
        if ate_pct is not None:
            st.metric(
                f"Relative to {baseline_label}'s average pace",
                f"{ate_pct:+.1f}%",
                help=f"95% CI: {ate_low_pct:.1f}% to {ate_high_pct:.1f}%",
            )
    st.markdown(
        _effect_text(
            ate_disp,
            ate_low_disp,
            ate_high_disp,
            comparison_label,
            baseline_label,
            speed_unit=speed_unit,
            ate_pct=ate_pct,
            low_pct=ate_low_pct,
            high_pct=ate_high_pct,
        )
    )
    st.caption(
        f"Model samples: treated={ate_result['n_treated']}, control={ate_result['n_control']} "
        f"(after removing {n_outliers} outlier effort(s), z-threshold={z_threshold:.1f})."
    )

    if ate_low_disp <= 0 <= ate_high_disp:
        st.warning("Confidence interval crosses zero, so this estimate is directionally uncertain.")

    st.subheader("Section 2 — Effect by Terrain")
    _render_terrain_chart(terrain_effects, mean_cbrt_watts=mean_cbrt_watts, comparison_label=comparison_label, speed_unit=speed_unit)

    st.subheader("Section 3 — What Did We Control For")
    _render_control_chart(shap_importances)


main()
