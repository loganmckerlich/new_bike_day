"""Causal analysis page for bike speed-per-watt treatment effects."""

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

from src.causal_inference import (
    build_feature_matrix,
    estimate_heterogeneous_effects,
    estimate_treatment_effect,
    get_shap_importances,
)


def _gear_label(gear_id: str, bikes: dict[str, str]) -> str:
    """Return a display label for a gear id."""
    return bikes.get(str(gear_id), str(gear_id))


def _selection_frame(efforts: pd.DataFrame, old_gear_id: str, new_gear_id: str) -> pd.DataFrame:
    """Filter to selected bikes and create binary treatment indicator."""
    selected = efforts[efforts["gear_id"].astype(str).isin([old_gear_id, new_gear_id])].copy()
    selected["is_new_bike"] = (selected["gear_id"].astype(str) == new_gear_id).astype(int)
    return selected


def _effect_text(ate_kmh: float, low_kmh: float, high_kmh: float, new_label: str) -> str:
    """Create plain-English headline interpretation for estimated effect."""
    direction = "faster" if ate_kmh >= 0 else "slower"
    return (
        "Controlling for wind, temperature, precipitation, road straightness, and terrain, "
        f"**{new_label}** is estimated to be **{abs(ate_kmh):.2f} km/h {direction}** "
        f"(95% CI: {low_kmh:.2f} to {high_kmh:.2f})."
    )


def _render_terrain_chart(terrain_df: pd.DataFrame, mean_watts: float) -> None:
    """Render terrain-specific effect chart with confidence intervals."""
    if terrain_df.empty:
        st.info("Not enough terrain-specific coverage to estimate heterogeneous effects yet.")
        return

    terrain_df = terrain_df.copy()
    terrain_df["ate_kmh"] = terrain_df["ate"] * mean_watts * 3.6
    terrain_df["ate_lower_kmh"] = terrain_df["ate_lower"] * mean_watts * 3.6
    terrain_df["ate_upper_kmh"] = terrain_df["ate_upper"] * mean_watts * 3.6
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
        title="Where is the new bike faster?",
    )
    fig.update_traces(
        error_x={"type": "data", "array": terrain_df["err_plus"], "arrayminus": terrain_df["err_minus"]},
        hovertemplate="%{y}: %{x:.2f} km/h<extra></extra>",
    )
    fig.update_layout(yaxis_title="Segment type", xaxis_title="ATE (km/h)", showlegend=False)
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
    st.title("🧪 Causal analysis")
    st.markdown(
        "Estimate the direct speed-per-watt impact of your new bike with doubly robust causal inference."
    )

    efforts: pd.DataFrame | None = st.session_state.get("efforts")
    segments: pd.DataFrame | None = st.session_state.get("segments")
    bikes: dict[str, str] = st.session_state.get("bikes", {})

    if efforts is None or efforts.empty:
        st.info("👈 Head to the **Home** page to sign in with Strava and load your data first.")
        st.stop()
    if segments is None or segments.empty:
        st.warning("No segment metadata available yet. Reload from Home first.")
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
        old_gear_id = st.selectbox("Old bike", options=gear_options, format_func=lambda g: gear_labels[g], index=0)
    with col_new:
        default_new_idx = 1 if len(gear_options) > 1 else 0
        new_gear_id = st.selectbox("New bike", options=gear_options, format_func=lambda g: gear_labels[g], index=default_new_idx)

    if old_gear_id == new_gear_id:
        st.warning("Select two different bikes.")
        st.stop()

    selected_efforts = _selection_frame(efforts, old_gear_id, new_gear_id)
    n_treated_raw = int(selected_efforts["is_new_bike"].sum())
    n_control_raw = int((selected_efforts["is_new_bike"] == 0).sum())
    if n_treated_raw < 30 or n_control_raw < 30:
        st.warning(
            "Guardrail: at least 30 efforts are required for each bike before running the model "
            f"(new bike={n_treated_raw}, old bike={n_control_raw})."
        )
        st.stop()

    with st.expander("How this works", expanded=False):
        st.markdown(
            "1. We model **speed per watt** so fitness/fatigue effects mediated through power are conditioned out.\n"
            "2. We control for direct confounders (wind, temperature, precipitation, road geometry, gradient, terrain).\n"
            "3. A doubly robust learner combines treatment and outcome models for a more stable causal estimate."
        )

    features = build_feature_matrix(selected_efforts, segments)
    if features.empty:
        st.warning("No valid rows after filtering (requires average_watts >= 50 and complete joins).")
        st.stop()

    ate_result = estimate_treatment_effect(features)
    mean_watts = float(features["average_watts"].mean())
    if pd.isna(mean_watts) or mean_watts <= 0:
        mean_watts = 1.0
    ate_kmh = ate_result["ate"] * mean_watts * 3.6
    ate_low_kmh = ate_result["ate_lower"] * mean_watts * 3.6
    ate_high_kmh = ate_result["ate_upper"] * mean_watts * 3.6
    terrain_effects = estimate_heterogeneous_effects(features)
    shap_importances = get_shap_importances(features)

    st.subheader("Section 1 — Headline Result")
    st.info("Weather note: weather inputs are currently dummy stub values and will improve when real weather data is integrated.")
    st.metric(
        "Average treatment effect (new bike vs old bike)",
        f"{ate_kmh:.2f} km/h",
        help=f"95% CI: {ate_low_kmh:.2f} to {ate_high_kmh:.2f}",
    )
    st.markdown(
        _effect_text(
            ate_kmh,
            ate_low_kmh,
            ate_high_kmh,
            _gear_label(new_gear_id, bikes),
        )
    )
    st.caption(f"Samples: treated={ate_result['n_treated']}, control={ate_result['n_control']}")

    if ate_low_kmh <= 0 <= ate_high_kmh:
        st.warning("Confidence interval crosses zero, so this estimate is directionally uncertain.")

    st.subheader("Section 2 — Effect by Terrain")
    _render_terrain_chart(terrain_effects, mean_watts=mean_watts)

    st.subheader("Section 3 — What Did We Control For")
    _render_control_chart(shap_importances)


main()
