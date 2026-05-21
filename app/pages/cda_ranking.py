"""CdA ranking page — estimate aerodynamic drag per bike from flat segments."""

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

from src.cda import (
    MIN_EFFORTS_PER_BIKE,
    aggregate_cda_by_bike,
    count_impossible_cda,
    estimate_cda,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gear_label(gear_id: str, bikes: dict[str, str]) -> str:
    """Return a display label for a gear_id."""
    return bikes.get(str(gear_id), str(gear_id))


def main() -> None:
    """Render the CdA ranking page."""
    st.title("🌬️ CdA Ranking")
    st.markdown(
        "Estimate the aerodynamic drag coefficient (CdA) per bike from flat segment efforts. "
        "A lower CdA means a more aerodynamic position."
    )

    # ── Weather stub banner ────────────────────────────────────────────────
    st.info(
        "⚠️ Air density is estimated using a default temperature of 18 °C. "
        "CdA estimates will improve once real weather data is connected."
    )

    # ── Section 0 — User Inputs (sidebar) ─────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Mass inputs")
        rider_mass_kg = st.number_input(
            "Rider mass (kg)",
            min_value=30.0,
            max_value=200.0,
            value=75.0,
            step=0.5,
            key="cda_rider_mass_kg",
        )
        bike_mass_kg = st.number_input(
            "Bike mass (kg)",
            min_value=2.0,
            max_value=30.0,
            value=8.0,
            step=0.5,
            key="cda_bike_mass_kg",
        )
        st.caption(
            "These affect the rolling resistance correction. "
            "If unsure, leave as defaults."
        )

    # ── Load session state ─────────────────────────────────────────────────
    efforts: pd.DataFrame | None = st.session_state.get("efforts")
    segments: pd.DataFrame | None = st.session_state.get("segments")
    bikes: dict[str, str] = st.session_state.get("bikes", {})

    if efforts is None or efforts.empty:
        st.info("👈 Head to the **Home** page to sign in with Strava and load your data first.")
        st.stop()
    if segments is None or segments.empty:
        st.warning("No segment metadata available yet. Reload from Home first.")
        st.stop()

    # ── Compute CdA estimates ──────────────────────────────────────────────
    with st.spinner("Estimating CdA from flat segments…"):
        n_impossible = count_impossible_cda(efforts, segments, rider_mass_kg, bike_mass_kg)
        cda_df = estimate_cda(efforts, segments, rider_mass_kg, bike_mass_kg)

    if n_impossible > 0:
        st.warning(
            f"⚠️ {n_impossible} effort(s) produced physically impossible CdA values "
            f"(outside {0.1}–{0.6}) and were removed."
        )

    if cda_df.empty:
        st.warning(
            "No CdA estimates could be computed. "
            "Make sure you have efforts on flat segments (flat_short or flat_long) "
            "with power data."
        )
        st.stop()

    # Attach bike name for display
    cda_df = cda_df.copy()
    cda_df["bike_name"] = (
        cda_df["gear_id"]
        .astype(str)
        .map({str(k): v for k, v in bikes.items()})
        .fillna(cda_df["gear_id"].astype(str))
    )

    # ── Per-bike aggregation ───────────────────────────────────────────────
    agg_df = aggregate_cda_by_bike(cda_df, bikes)

    # Per-bike minimum effort guardrail
    bikes_ok = agg_df[agg_df["n_efforts"] >= MIN_EFFORTS_PER_BIKE]["bike_name"].tolist()
    bikes_insufficient = agg_df[agg_df["n_efforts"] < MIN_EFFORTS_PER_BIKE]["bike_name"].tolist()

    for bike_name in bikes_insufficient:
        st.warning(
            f"Not enough flat segment efforts to estimate CdA for **{bike_name}**. "
            "Try starring more flat segments on Strava."
        )

    agg_display = agg_df[agg_df["bike_name"].isin(bikes_ok)].copy()

    if agg_display.empty:
        st.info(
            "None of your bikes have the minimum 10 flat-segment efforts needed for CdA estimation. "
            "Star more flat segments on Strava and re-load your data."
        )
        st.stop()

    cda_plot_df = cda_df[cda_df["bike_name"].isin(bikes_ok)].copy()

    # ── Section 1 — Ranked Timeline Visual ────────────────────────────────
    st.subheader("Section 1 — Ranked CdA Estimates")
    st.caption(
        "Based on flat segment efforts only, weather-corrected for air density"
    )

    if len(agg_display) == 1:
        st.info(
            f"Only **{agg_display.iloc[0]['bike_name']}** has enough flat efforts "
            "to estimate CdA. Ranking requires at least 2 bikes."
        )

    # Build scatter plot with error bars
    fig_rank = go.Figure()

    # Sort for consistent left-to-right ordering
    agg_sorted = agg_display.sort_values("mean_cda")

    # Marker size scaled by n_efforts (min 12, max 40)
    max_n = max(agg_sorted["n_efforts"].max(), 1)
    min_n = agg_sorted["n_efforts"].min()
    size_range = (12, 40)
    if max_n == min_n:
        sizes = [size_range[1]] * len(agg_sorted)
    else:
        sizes = [
            size_range[0]
            + (n - min_n) / (max_n - min_n) * (size_range[1] - size_range[0])
            for n in agg_sorted["n_efforts"]
        ]

    colors = px.colors.qualitative.Plotly

    for i, (_, row) in enumerate(agg_sorted.iterrows()):
        color = colors[i % len(colors)]
        std_val = row["std_cda"] if not np.isnan(row["std_cda"]) else 0.0
        fig_rank.add_trace(
            go.Scatter(
                x=[row["mean_cda"]],
                y=[0],
                mode="markers+text",
                marker={
                    "size": sizes[i],
                    "color": color,
                    "line": {"width": 2, "color": "white"},
                },
                error_x={
                    "type": "data",
                    "array": [std_val],
                    "arrayminus": [std_val],
                    "visible": True,
                    "color": color,
                    "thickness": 2,
                    "width": 8,
                },
                text=[row["bike_name"]],
                textposition="top center",
                name=row["bike_name"],
                hovertemplate=(
                    f"<b>{row['bike_name']}</b><br>"
                    f"Mean CdA: {row['mean_cda']:.4f}<br>"
                    f"± 1 std: {std_val:.4f}<br>"
                    f"n efforts: {int(row['n_efforts'])}<extra></extra>"
                ),
            )
        )

    fig_rank.update_layout(
        title="Estimated CdA — lower is more aerodynamic",
        xaxis_title="CdA (m²)",
        yaxis_visible=False,
        height=300,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"showgrid": True, "zeroline": False},
        margin={"t": 60, "b": 40, "l": 20, "r": 20},
    )
    st.plotly_chart(fig_rank, use_container_width=True)
    st.caption("Marker size = number of flat efforts. Error bars = ± 1 std.")

    # ── Section 2 — Supporting Detail ─────────────────────────────────────
    st.subheader("Section 2 — Supporting Detail")

    display_cols = ["bike_name", "mean_cda", "median_cda", "std_cda", "n_efforts"]
    st.dataframe(
        agg_display[display_cols].rename(
            columns={
                "bike_name": "Bike",
                "mean_cda": "Mean CdA",
                "median_cda": "Median CdA",
                "std_cda": "Std CdA",
                "n_efforts": "Efforts",
            }
        ).style.format(
            {
                "Mean CdA": "{:.4f}",
                "Median CdA": "{:.4f}",
                "Std CdA": "{:.4f}",
                "Efforts": "{:.0f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("#### Plain English Interpretation")
    for _, row in agg_display.sort_values("mean_cda").iterrows():
        std_str = f" (±{row['std_cda']:.4f})" if not np.isnan(row["std_cda"]) else ""
        st.markdown(
            f"- **{row['bike_name']}** has a mean CdA of **{row['mean_cda']:.4f}**{std_str}, "
            f"estimated from **{int(row['n_efforts'])}** flat efforts."
        )

    # ── Section 3 — Raw Estimates Over Time ────────────────────────────────
    st.subheader("Section 3 — CdA Estimates Over Time")
    st.caption(
        "Individual CdA estimates per effort. Trends may reflect position changes, "
        "new equipment, or weight changes."
    )

    if not cda_plot_df.empty:
        cda_plot_df["start_date"] = pd.to_datetime(cda_plot_df["start_date"], utc=True, errors="coerce")

        fig_ts = px.scatter(
            cda_plot_df,
            x="start_date",
            y="cda_estimate",
            color="bike_name",
            labels={
                "start_date": "Date",
                "cda_estimate": "CdA estimate",
                "bike_name": "Bike",
            },
            title="CdA estimates over time",
            hover_data={"average_watts": True, "average_speed_mps": True},
        )
        fig_ts.update_traces(marker={"size": 7, "opacity": 0.75})
        fig_ts.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="CdA (m²)",
            xaxis_title="Date",
            legend={"orientation": "h", "y": -0.25},
        )
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("No individual CdA estimates to display.")


main()
