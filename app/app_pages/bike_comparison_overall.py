# """Overall bike comparison — XGBoost counterfactual speed pipeline.

# Method
# ------
# 1. Train an XGBoost model to predict speed from power, grade, and seasonal
#    features using only Bike A's efforts.
# 2. Apply that model to Bike B's effort features to get a counterfactual speed
#    ("how fast would Bike A have gone in these conditions?").
# 3. Residual = actual Bike B speed − predicted → positive means B is faster.
# 4. Repeat steps 1–3 with A and B swapped (symmetry check).
# 5. Aggregate both directions: combined = (fwd_mean − rev_mean) / 2.
# """
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.bike_delta import (
    prepare_delta_dataset,
    XGB_FEATURES,
    XGB_WATT_FEATURES,
    fit_xgb_speed_model,
    apply_model_to_bike,
    fit_xgb_watt_model,
    apply_watt_model_to_bike,
    aggregate_paired_delta_bootstrap,
    engineer_features,
    bootstrap_pipeline
)
from src._ui_helpers import use_metric, spd_label, get_available_bikes

from src.utils import navigator, page_guard


_COLOR_A = "#4C72B0"
_COLOR_B = "#DD8452"
_FASTER_COLOR = "#2ca02c"
_SLOWER_COLOR = "#d62728"

_SPEED_DISPLAY_COLS = ["speed_kmh", "predicted_speed_kmh", "speed_residual"]
BOOT_ITERATIONS = 30
# ── Page title ────────────────────────────────────────────────────────────

def overall_comp_inputs():
    available_bikes = get_available_bikes()

    if st.session_state.get("segment_bikes") is not None:
        defaults = st.session_state["segment_bikes"][0:2]
    else:
        defaults = available_bikes[:2]
    bikes_to_compare = st.multiselect(
        "Bikes to compare",
        options=available_bikes,
        default=defaults,
        max_selections=2,
        help="Select 2 bikes to compare.",
        # in future allow this to be more than 2, spider plots already ready for that
    )
    return bikes_to_compare

def _scale_speed_cols(df: pd.DataFrame, scale: float) -> pd.DataFrame:
    """Return a copy of *df* with speed columns multiplied by *scale* for display."""
    if scale == 1.0:
        return df
    out = df.copy()
    for col in _SPEED_DISPLAY_COLS:
        if col in out.columns:
            out[col] = out[col] * scale
    return out


# ── Dataset builder (cached in session state) ─────────────────────────────────
st.cache_data(ttl=3600)
def _build_delta_df(
    efforts: pd.DataFrame,
    segments: pd.DataFrame,
    bikes: dict[str, str],
) -> pd.DataFrame:

    df = prepare_delta_dataset(efforts, segments, bikes)
    df = engineer_features(df)
    return df

def _date_split_bike_df(
    df: pd.DataFrame,
    bike_name: str,
    train_frac: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based 80/20 train/test split for one bike's efforts.

    Sorts the bike's efforts chronologically and reserves the last *1-train_frac*
    fraction as a holdout. Other bikes' rows are untouched in df_train_scope.

    Returns
    -------
    df_train_scope : full df but bike_name rows limited to the training period
    df_holdout     : holdout rows for bike_name only (for out-of-sample stats)
    """
    bike_mask = df["bike_name"] == bike_name
    bike_df = df[bike_mask].sort_values("start_date")
    n = len(bike_df)
    split_idx = max(1, int(n * train_frac))
    train_idx = bike_df.index[:split_idx]
    test_idx = bike_df.index[split_idx:]
    df_train_scope = df.loc[~bike_mask | df.index.isin(train_idx)].copy()
    df_holdout = df.loc[df.index.isin(test_idx)].copy()
    return df_train_scope, df_holdout

def _plot_training_data(
    train_df: pd.DataFrame,
    bike_name: str,
    x_col: str = "average_watts",
    y_col: str = "speed_kmh",
    x_label: str = "Average power (W)",
    y_label: str = "Speed (km/h)",
) -> go.Figure:
    fig = px.scatter(
        train_df.dropna(subset=[x_col, y_col, "average_grade"]),
        x=x_col,
        y=y_col,
        color="average_grade",
        color_continuous_scale="RdYlGn_r",
        labels={
            x_col: x_label,
            y_col: y_label,
            "average_grade": "Grade (%)",
        },
        title=f"{bike_name} — training data",
    )
    fig.update_traces(marker_size=7, marker_opacity=0.7)
    fig.update_layout(height=320, plot_bgcolor="rgba(0,0,0,0)")
    return fig


def _plot_actual_vs_predicted(
    df: pd.DataFrame,
    bike_name: str,
    color: str,
    target_col: str = "speed_kmh",
    pred_col: str = "predicted_speed_kmh",
    unit: str = "km/h",
):
    valid = df.dropna(subset=[target_col, pred_col])
    r2 = r2_score(valid[target_col], valid[pred_col])

    lo = float(valid[[target_col, pred_col]].min().min())
    hi = float(valid[[target_col, pred_col]].max().max())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        mode="lines",
        line={"color": "grey", "dash": "dash", "width": 1.5},
        name="Perfect fit",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=valid[pred_col],
        y=valid[target_col],
        mode="markers",
        marker={"color": color, "size": 7, "opacity": 0.65},
        name=bike_name,
        hovertemplate=f"Predicted: %{{x:.1f}} {unit}<br>Actual: %{{y:.1f}} {unit}<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title=f"Predicted ({unit})",
        yaxis_title=f"Actual ({unit})",
        title=f"Model fit — R² = {r2:.3f}",
        height=320,
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig, r2


def _plot_feature_importance(model, color: str, features: list[str] | None = None, target_label: str = "speed") -> go.Figure:
    feat_list = features if features is not None else XGB_FEATURES
    feat_df = (
        pd.DataFrame({"feature": feat_list, "importance": model.feature_importances_})
        .sort_values("importance")
    )
    fig = go.Figure(go.Bar(
        x=feat_df["importance"],
        y=feat_df["feature"],
        orientation="h",
        marker_color=color,
    ))
    fig.update_layout(
        xaxis_title="Feature importance (gain)",
        yaxis_title="",
        title=f"What drives {target_label} predictions?",
        height=280,
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 120},
    )
    return fig


def _plot_target_vs_input(
    df: pd.DataFrame,
    bike_name: str,
    color: str,
    x_col: str = "average_watts",
    y_col: str = "speed_kmh",
    pred_col: str = "predicted_speed_kmh",
    x_label: str = "Average power (W)",
    y_label: str = "Speed (km/h)",
    pred_fmt: str = ".1f",
) -> go.Figure:
    """Scatter of actual + predicted target vs the primary input variable."""
    valid = df.dropna(subset=[x_col, y_col, pred_col])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=valid[x_col],
        y=valid[y_col],
        mode="markers",
        name="Actual",
        marker={"color": color, "size": 7, "opacity": 0.55},
        hovertemplate=f"{x_label}: %{{x:.0f}}<br>Actual {y_label}: %{{y:{pred_fmt}}}<extra></extra>",
    ))
    sorted_valid = valid.sort_values(x_col)
    fig.add_trace(go.Scatter(
        x=sorted_valid[x_col],
        y=sorted_valid[pred_col],
        mode="markers",
        name="XGB predicted",
        marker={"color": "black", "size": 5, "opacity": 0.35, "symbol": "cross"},
        hovertemplate=f"{x_label}: %{{x:.0f}}<br>Predicted {y_label}: %{{y:{pred_fmt}}}<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        title=f"{y_label} vs {x_label} — {bike_name}",
        height=320,
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.3},
    )
    return fig


def _model_stats(df: pd.DataFrame, target_col: str = "speed_kmh", pred_col: str = "predicted_speed_kmh") -> dict:
    """Compute basic model statistics from a dataframe with actual/predicted values."""
    valid = df.dropna(subset=[target_col, pred_col])
    y, y_hat = valid[target_col].values, valid[pred_col].values
    return {
        "n": len(valid),
        "r2": r2_score(y, y_hat),
        "rmse": mean_squared_error(y, y_hat),
        "mae": mean_absolute_error(y, y_hat),
    }


def _render_model_stats(stats: dict, unit: str = "km/h") -> None:
    """Render a compact 4-metric row for a trained model."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R\u00b2", f"{stats['r2']:.3f}", help="Fraction of variance explained (20% holdout, sorted by date).")
    c2.metric("RMSE", f"{stats['rmse']:.3f} {unit}", help="Root-mean-squared error on held-out efforts.")
    c3.metric("MAE", f"{stats['mae']:.3f} {unit}", help="Mean absolute error on held-out efforts.")
    c4.metric("Efforts (n)", str(stats["n"]), help="Number of held-out efforts used to compute these stats.")


def _render_model_details_expander(
    model,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    bike_name: str,
    color: str,
    eval_label: str,
    train_x_col: str,
    train_y_col: str,
    train_x_lbl: str,
    train_y_lbl: str,
    target_col: str,
    pred_col: str,
    features: list[str],
    target_label: str,
    unit: str,
    expanded: bool = False,
) -> float:
    """Render model viz + stats inside an expander. Returns R²."""
    with st.expander(f"Model details \u2014 {bike_name}", expanded=expanded):
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                _plot_training_data(train_df, bike_name, train_x_col, train_y_col, train_x_lbl, train_y_lbl),
                width="stretch",
            )
            st.caption(
                f"Each dot is one {bike_name} training effort (oldest 80%), coloured by segment grade."
            )
        with col2:
            fig_fit, r2 = _plot_actual_vs_predicted(eval_df, bike_name, color, target_col, pred_col, unit)
            st.plotly_chart(fig_fit, width="stretch")
            st.caption(f"R\u00b2\u00a0= {r2:.3f} ({eval_label}). Points close to the diagonal = good fit.")
        st.plotly_chart(
            _plot_target_vs_input(
                train_df, bike_name, color,
                x_col=train_x_col, y_col=target_col, pred_col=pred_col,
                x_label=train_x_lbl, y_label=train_y_lbl,
            ),
            width="stretch",
        )
        st.caption(
            f"Actual {bike_name} training efforts (coloured) vs the XGB model\u2019s prediction (crosses). "
            "Scatter is real variability from grade, season, and conditions."
        )
        _render_model_stats(_model_stats(eval_df, target_col, pred_col), unit)
        st.plotly_chart(_plot_feature_importance(model, color, features, target_label), width="stretch")
        st.caption(
            f"Which features matter most for predicting {target_label}? "
            "Grade and the primary input (power or speed) typically dominate."
        )
    return r2


def _plot_counterfactual_scatter(
    pred_df: pd.DataFrame,
    bike_a: str,
    bike_b: str,
    target_col: str = "speed_kmh",
    pred_col: str = "predicted_speed_kmh",
    residual_col: str = "speed_residual",
    unit: str = "km/h",
    b_better_label: str = "{b} faster",
    a_better_label: str = "{a} faster",
    title: str = "Counterfactual: how fast would {a} have gone?",
    x_axis_label: str = "Predicted — {a} model ({unit})",
    y_axis_label: str = "Actual — {b} ({unit})",
) -> go.Figure:
    valid = pred_df.dropna(subset=[target_col, pred_col])
    lo = float(valid[[target_col, pred_col]].min().min())
    hi = float(valid[[target_col, pred_col]].max().max())

    b_better = valid[valid[residual_col] >= 0]
    a_better = valid[valid[residual_col] < 0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        mode="lines",
        line={"color": "grey", "dash": "dash", "width": 1.5},
        name="Equal (y = x)",
        hoverinfo="skip",
    ))
    for subset, label, c in [
        (b_better, b_better_label.format(a=bike_a, b=bike_b), _FASTER_COLOR),
        (a_better, a_better_label.format(a=bike_a, b=bike_b), _SLOWER_COLOR),
    ]:
        if subset.empty:
            continue
        fig.add_trace(go.Scatter(
            x=subset[pred_col],
            y=subset[target_col],
            mode="markers",
            name=label,
            marker={"color": c, "size": 7, "opacity": 0.7},
            hovertemplate=(
                f"Predicted ({bike_a} model): %{{x:.1f}} {unit}<br>"
                f"Actual ({bike_b}): %{{y:.1f}} {unit}<br>"
                f"Residual: %{{customdata:.2f}} {unit}<extra></extra>"
            ),
            customdata=subset[residual_col].values,
        ))
    fig.update_layout(
        xaxis_title=x_axis_label.format(a=bike_a, b=bike_b, unit=unit),
        yaxis_title=y_axis_label.format(a=bike_a, b=bike_b, unit=unit),
        title=title.format(a=bike_a, b=bike_b),
        height=340,
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.3},
    )
    return fig


def _plot_residuals(
    pred_df: pd.DataFrame,
    bike_a: str,
    bike_b: str,
    residual_col: str = "speed_residual",
    unit: str = "km/h",
    positive_means: str = "{b} faster",
) -> go.Figure:
    resids = pred_df[residual_col].dropna()
    mean_r = float(resids.mean())
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=resids,
        nbinsx=25,
        marker_color=_FASTER_COLOR if mean_r >= 0 else _SLOWER_COLOR,
        opacity=0.75,
        name="Efforts",
    ))
    fig.add_vline(x=mean_r, line_dash="dash", line_color="black", line_width=2,
                  annotation_text=f"Mean: {mean_r:+.2f} {unit}",
                  annotation_position="top right")
    fig.add_vline(x=0, line_color="grey", line_width=1)
    pos_label = positive_means.format(a=bike_a, b=bike_b)
    fig.update_layout(
        xaxis_title=f"Residual ({unit})  —  positive = {pos_label}",
        yaxis_title="Number of efforts",
        title="Distribution of differences",
        height=300,
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _plot_aggregate(
    summary: dict,
    bike_a: str,
    bike_b: str,
    unit: str = "km/h",
    advantage_label: str = "Speed advantage",
) -> go.Figure:
    """Plot forward, reverse, and combined bike effect estimates with
    bootstrap-derived confidence intervals.

    Expects summary to be the output of aggregate_paired_delta_bootstrap,
    containing fwd_mean, rev_mean, combined, ci_low, ci_high, and the
    underlying bootstrap arrays (fwd_estimates, rev_estimates,
    combined_estimates) attached for per-bar CI computation.
    """
    fwd_estimates = summary["fwd_estimates"]
    rev_estimates = -summary["rev_estimates"]  # negate so positive = B better
    combined_estimates = summary["combined_estimates"]

    fwd = float(np.mean(fwd_estimates))
    rev = float(np.mean(rev_estimates))
    combined = summary["combined"]

    fwd_ci_low, fwd_ci_high = np.percentile(fwd_estimates, [2.5, 97.5])
    rev_ci_low, rev_ci_high = np.percentile(rev_estimates, [2.5, 97.5])
    combined_ci_low, combined_ci_high = summary["ci_low"], summary["ci_high"]

    labels = [
        f"A→B  (model A on {bike_b})",
        f"B→A  (model B on {bike_a}, negated)",
        "Combined (average)",
    ]
    values = [fwd, rev, combined]
    ci_lows = [fwd_ci_low, rev_ci_low, combined_ci_low]
    ci_highs = [fwd_ci_high, rev_ci_high, combined_ci_high]

    colors = [_FASTER_COLOR if v >= 0 else _SLOWER_COLOR for v in values]
    colors[-1] = "#7f7f7f"

    fig = go.Figure()
    for label, value, lo, hi, color in zip(labels, values, ci_lows, ci_highs, colors):
        err_plus = hi - value
        err_minus = value - lo
        fig.add_trace(go.Bar(
            x=[label],
            y=[value],
            marker_color=color,
            error_y={
                "type": "data",
                "symmetric": False,
                "array": [err_plus],
                "arrayminus": [err_minus],
                "visible": True,
            },
            name=label,
            hovertemplate=(
                f"{label}: %{{y:+.2f}} {unit}<br>"
                f"95% CI: [{lo:+.2f}, {hi:+.2f}]<extra></extra>"
            ),
        ))

    fig.add_hline(y=0, line_color="grey", line_width=1)
    fig.update_layout(
        yaxis_title=f"{advantage_label} of {bike_b} over {bike_a} ({unit})",
        showlegend=False,
        height=320,
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Main entry point ──────────────────────────────────────────────────────────

def show(bikes_to_compare: list[str]) -> None:
    """Render the overall bike comparison analysis."""

    efforts = st.session_state.get("cleaned_efforts")
    segments = st.session_state.get("segments")
    bikes: dict[str, str] = st.session_state.get("bikes", {})

    if len(bikes_to_compare) < 2:
        st.warning("Select at least 2 bikes in the sidebar.")
        st.stop()

    # ── Build analysis dataset ────────────────────────────────────────────────
    try:
        df = _build_delta_df(efforts, segments, bikes)
    except Exception as e:
        st.error(f"Failed to prepare analysis dataset: {e}")
        st.stop()

    df_scope = df[df["bike_name"].isin(bikes_to_compare)].copy()

    # ── Assumptions expander (empty) ──────────────────────────────────────────
    with st.expander("Assumptions", expanded=False):
        with open(_REPO_ROOT / "src" / "assumptions.md", "r") as f:
            st.markdown(f.read())

    # ── Mode toggle ───────────────────────────────────────────────────────────
    watt_mode = st.toggle(
        "Watts mode — predict power instead of speed",
        value=False,
        help=(
            "**Speed mode** (default): the model learns power → speed and asks "
            "\"how fast would Bike A have gone in Bike B's conditions?\"\n\n"
            "**Watts mode**: the model learns speed → watts and asks "
            "\"how many watts would Bike A have needed to match Bike B's speed?\" "
            "Positive residual = Bike B needed fewer watts = more efficient."
        ),
    )

    if watt_mode:
        _fit_fn       = fit_xgb_watt_model
        _apply_fn     = apply_watt_model_to_bike
        _features     = XGB_WATT_FEATURES
        _target_col   = "average_watts"
        _pred_col     = "predicted_watts"
        _residual_col = "watts_residual"
        _unit         = "W"
        _mode_str     = "watt"
        # training plot axes: x = speed input, y = watts target
        _train_x_col, _train_y_col   = "speed_kmh", "average_watts"
        _train_x_lbl, _train_y_lbl   = f"Speed ({spd_label()})", "Average power (W)"
        _target_label = "watts"
        _b_better     = "{b} more efficient"
        _a_better     = "{a} more efficient"
        _cf_title     = "Counterfactual: how many watts would {a} have needed?"
        _cf_x_lbl     = "Predicted watts — {a} model ({unit})"
        _cf_y_lbl     = "Actual watts — {b} ({unit})"
        _adv_label    = "Watts advantage (efficiency)"
    else:
        _fit_fn       = fit_xgb_speed_model
        _apply_fn     = apply_model_to_bike
        _features     = XGB_FEATURES
        _target_col   = "speed_kmh"
        _pred_col     = "predicted_speed_kmh"
        _residual_col = "speed_residual"
        _unit         = spd_label()
        _mode_str     = "speed"
        _train_x_col, _train_y_col   = "average_watts", "speed_kmh"
        _train_x_lbl, _train_y_lbl   = "Average power (W)", f"Speed ({spd_label()})"
        _target_label = "speed"
        _b_better     = "{b} faster"
        _a_better     = "{a} faster"
        _cf_title     = "Counterfactual: how fast would {a} have gone?"
        _cf_x_lbl     = "Predicted speed — {a} model ({unit})"
        _cf_y_lbl     = "Actual speed — {b} ({unit})"
        _adv_label    = "Speed advantage"

    # ── Pair selector ─────────────────────────────────────────────────────────
    all_pairs = list(combinations(bikes_to_compare, 2))
    if len(all_pairs) == 1:
        bike_a, bike_b = all_pairs[0]
    else:
        pair_labels = [f"{a}  vs  {b}" for a, b in all_pairs]
        selected_label = st.selectbox("Bike pair to inspect", options=pair_labels, index=0)
        idx_p = pair_labels.index(selected_label)
        bike_a, bike_b = all_pairs[idx_p]

    st.markdown(f"**Comparing: {bike_a}  vs  {bike_b}**")

    # Scale factor: 1.0 in watt mode or metric; km/h→mph otherwise
    _disp_scale: float = 1.0 if (watt_mode or use_metric()) else 0.621371
    _disp = lambda df: _scale_speed_cols(df, _disp_scale)  # noqa: E731

    # ── Step 1: Train model on Bike A ─────────────────────────────────────────
    st.divider()
    st.subheader(f"Step 1 — Train model on {bike_a}")
    st.caption(
        f"An XGBoost model learns the relationship between riding conditions "
        f"(power, grade, time of year, fitness trend) and speed, using only "
        f"**{bike_a}** efforts. This model captures what {bike_a} is capable of "
        "in any given conditions."
    )

    df_train_scope_a, df_test_a = _date_split_bike_df(df_scope, bike_a)
    with st.spinner(f"Training XGBoost on {bike_a}\u2026"):
        model_a = _fit_fn(df_train_scope_a, bike_a, str(_mode_str))


    train_a = _apply_fn(model_a, df_train_scope_a, bike_a)
    holdout_a = _apply_fn(model_a, df_test_a, bike_a)
    _eval_a = holdout_a if not holdout_a.empty else train_a
    _eval_a_label = "holdout" if not holdout_a.empty else "in-sample \u2014 too few efforts to hold out"


    r2_a = _render_model_details_expander(
        model_a, _disp(train_a), _disp(_eval_a), bike_a, _COLOR_A, _eval_a_label,
        _train_x_col, _train_y_col, _train_x_lbl, _train_y_lbl,
        _target_col, _pred_col, _features, _target_label, _unit, expanded=True,
    )
    st.info(
        f"**Step 1 in plain terms:** We've taught the model what {bike_a} is capable of — "
        f"it explains **{r2_a:.0%}** of the variation in {bike_a}'s {_target_label}. "
        f"We'll use this as a {bike_a} baseline and ask: what would {bike_a} have done on {bike_b}'s efforts?"
    ) 

    # ── Step 2: Apply model A to Bike B ───────────────────────────────────────
    st.divider()
    if watt_mode:
        st.subheader(f"Step 2 — Predict counterfactual watts for {bike_b}")
        st.caption(
            f"For every {bike_b} effort, we ask: **how many watts would {bike_a} "
            f"have needed to achieve the same speed?** Positive residual = {bike_b} "
            "used fewer watts = more efficient."
        )
    else:
        st.subheader(f"Step 2 — Predict counterfactual speed for {bike_b}")
        st.caption(
            f"We now ask: for every {bike_b} effort, **what speed would {bike_a} have "
            f"achieved in the same conditions?** The {bike_a} model answers this — treating "
            f"the effort's power, grade, and season as inputs."
        )

    with st.spinner("Retraining for bootstrapping confidence intervals and increased model coverage\u2026"):
        ab_bootstrap_results = bootstrap_pipeline(
            df_scope,
            train_bike=bike_a,
            target_bike=bike_b,
            n_iterations=BOOT_ITERATIONS,
            random_state=42,
            apply_fn = _apply_fn,
            fit_fn = _fit_fn,
            label = _residual_col,
            predicted_col=_pred_col,
            target_col=_target_col,
        )

    # this is the per effort avg from the boot model, has target and prediction and residual for each effort
    pred_ab = ab_bootstrap_results['effort_residuals']
    if pred_ab.empty:
        st.warning(f"No usable {bike_b} efforts after filtering. Cannot compute counterfactual.")
        st.stop()
    pred_ab_d = _disp(pred_ab)

    st.plotly_chart(_plot_counterfactual_scatter(
        pred_ab_d, bike_a, bike_b,
        target_col=_target_col, pred_col=_pred_col, residual_col=_residual_col, unit=_unit,
        b_better_label=_b_better, a_better_label=_a_better,
        title=_cf_title, x_axis_label=_cf_x_lbl, y_axis_label=_cf_y_lbl,
    ), width='stretch')
    if watt_mode:
        st.caption(
            f"X-axis: watts {bike_a}'s model predicts for each condition. "
            f"Y-axis: watts {bike_b} actually used. "
            f"**Green (above diagonal) = {bike_b} used fewer watts = more efficient.**"
        )
    else:
        st.caption(
            f"X-axis: the speed the {bike_a} model predicts for each set of conditions. "
            f"Y-axis: what {bike_b} actually achieved. "
            f"**Green points (above diagonal) = {bike_b} was faster than {bike_a}'s model expects.**"
        )
    if watt_mode:
        st.info(
            f"**Step 2 in plain terms:** For each {bike_b} effort, we asked — "
            f"*how many watts would {bike_a} have needed to go the same speed?* "
            f"Points above the line mean {bike_b} used fewer watts, i.e. it's more efficient."
        )
    else:
        st.info(
            f"**Step 2 in plain terms:** For each {bike_b} effort, we asked — "
            f"*how fast would {bike_a} have gone in these exact conditions?* "
            f"Points above the line mean {bike_b} was faster than {bike_a} would have been."
        )

    # ── Step 3: Residuals (A→B direction) ────────────────────────────────────
    st.divider()
    if watt_mode:
        st.subheader(f"Step 3 — Watts difference: {bike_a} model → {bike_b} efforts")
        st.caption(
            f"Residual = predicted watts (what {bike_a} would need) − actual watts ({bike_b} used). "
            f"Positive = {bike_b} is more efficient."
        )
    else:
        st.subheader(f"Step 3 — Speed difference: {bike_a} model → {bike_b} efforts")
        st.caption(
            "The residual = actual speed − predicted speed. It quantifies how much faster "
            f"(or slower) {bike_b} is relative to what {bike_a}'s model expects."
        )

    mean_ab = float(pred_ab_d[_residual_col].mean())
    n_ab = int(pred_ab_d[_residual_col].notna().sum())

    col_m1, col_m2 = st.columns([1, 2])
    with col_m1:
        if watt_mode:
            sign = "more efficient" if mean_ab >= 0 else "less efficient"
            st.metric(
                label=f"{bike_b} is",
                value=f"{abs(mean_ab):.1f} W {sign}",
                delta=f"{mean_ab:+.1f} W  (n\u00a0= {n_ab} efforts)",
                delta_color="normal",
            )
            st.caption(f"than {bike_a} model")
        else:
            sign = "faster" if mean_ab >= 0 else "slower"
            st.metric(
                label=f"{bike_b} is",
                value=f"{abs(mean_ab):.2f} {_unit} {sign}",
                delta=f"{mean_ab:+.2f} {_unit}  (n\u00a0= {n_ab} efforts)",
                delta_color="normal",
            )
            st.caption(f"than {bike_a} model")
    with col_m2:
        st.plotly_chart(_plot_residuals(pred_ab_d, bike_a, bike_b, _residual_col, _unit, _b_better), width='stretch')
    st.caption(
        f"Mean residual = {mean_ab:+.2f} {_unit}. "
        "This is the **A\u2192B direction**: positive means B is better. "
        "Spread reflects effort-to-effort variability."
    )
    if watt_mode:
        # watts_residual = predicted_a_watts − actual_b_watts  (see apply_watt_model_to_bike)
        # Step 2 asked: "how many watts would bike_a need to go bike_b's speed?"
        # mean_ab > 0 → bike_a's model predicts MORE watts than bike_b used → bike_b is more efficient
        # mean_ab < 0 → bike_a's model predicts FEWER watts than bike_b used → bike_a is more efficient
        # Subject must be bike_a (the model) to directly answer the Step 2 question.
        _ab_direction = "more" if mean_ab >= 0 else "fewer"
        st.info(
            f"**Step 3 in plain terms:** In general, for the same efforts, {bike_a} would have required "
            f"**{abs(mean_ab):.1f} W {_ab_direction}** than {bike_b} used to go the same speed — "
            f"based on {n_ab} matched efforts."
        )
    else:
        _ab_direction = "faster" if mean_ab >= 0 else "slower"
        st.info(
            f"**Step 3 in plain terms:** In general, if you did an effort on {bike_b} "
            f"instead of {bike_a}, you would've gone **{abs(mean_ab):.2f} {_unit} {_ab_direction}** — "
            f"based on {n_ab} matched efforts."
        )

    # ── Halftime summary ─────────────────────────────────────────────────────
    st.markdown("---")
    if watt_mode:
        st.info(
            f"**Where we are so far:** We trained a model on {bike_a}'s efforts that, "
            f"given speed, grade, and conditions, predicts how many watts {bike_a} typically requires. "
            f"We then fed {bike_b}'s actual effort conditions into that model — "
            f"so for each effort we have: *how many watts the model thinks {bike_a} would've needed* "
            f"vs *how many watts {bike_b} actually used*. "
            f"The mean of those differences ({mean_ab:+.1f} W) is our first estimate of the efficiency gap.\n\n"
            f"**Now we do it in reverse** — train on {bike_b}, feed in {bike_a}'s conditions — "
            f"to get an independent second estimate and check that both directions agree."
        )
    else:
        st.info(
            f"**Where we are so far:** We trained a model on {bike_a}'s efforts that, "
            f"given power, grade, and conditions, predicts the speed {bike_a} typically produces. "
            f"We then fed {bike_b}'s actual effort conditions into that model — "
            f"so for each effort we have: *how fast the model thinks {bike_a} would've gone* "
            f"vs *how fast {bike_b} actually went*. "
            f"The mean of those differences ({mean_ab:+.2f} {_unit}) is our first estimate of the speed gap.\n\n"
            f"**Now we do it in reverse** — train on {bike_b}, feed in {bike_a}'s conditions — "
            f"to get an independent second estimate and check that both directions agree."
        )

    # ── Step 4: Reverse (train B, apply to A) ────────────────────────────────
    st.divider()
    st.subheader(f"Step 4 \u2014 Reverse: train on {bike_b}, apply to {bike_a}")
    st.caption(
        f"We repeat steps 1\u20133 with {bike_b} as the training bike and apply the model "
        f"to {bike_a}'s efforts. Consistent and opposite results give greater confidence."
    )

    df_train_scope_b, df_test_b = _date_split_bike_df(df_scope, bike_b)
    with st.spinner(f"Training XGBoost on {bike_b}\u2026"):
        model_b = _fit_fn(df_train_scope_b, bike_b, str(_mode_str))

    train_b = _apply_fn(model_b, df_train_scope_b, bike_b)
    holdout_b = _apply_fn(model_b, df_test_b, bike_b)
    _eval_b = holdout_b if not holdout_b.empty else train_b
    _eval_b_label = "holdout" if not holdout_b.empty else "in-sample \u2014 too few efforts to hold out"

    r2_b = _render_model_details_expander(
        model_b, _disp(train_b), _disp(_eval_b), bike_b, _COLOR_B, _eval_b_label,
        _train_x_col, _train_y_col, _train_x_lbl, _train_y_lbl,
        _target_col, _pred_col, _features, _target_label, _unit, expanded=False,
    )


    with st.spinner("Retraining for bootstrapping confidence intervals and increased model coverage\u2026"):
        ba_bootstrap_results = bootstrap_pipeline(
            df_scope,
            train_bike=bike_b,
            target_bike=bike_a,
            n_iterations=BOOT_ITERATIONS,
            random_state=42,
            apply_fn = _apply_fn,
            fit_fn = _fit_fn,
            label = _residual_col,
            predicted_col=_pred_col,
            target_col=_target_col,
        )


    pred_ba = ba_bootstrap_results['effort_residuals']
    if pred_ba.empty:
        st.warning(f"No usable {bike_a} efforts after filtering. Cannot compute reverse counterfactual.")
        st.stop()
    pred_ba_d = _disp(pred_ba)

    st.plotly_chart(_plot_counterfactual_scatter(
        pred_ba_d, bike_b, bike_a,
        target_col=_target_col, pred_col=_pred_col, residual_col=_residual_col, unit=_unit,
        b_better_label=_a_better, a_better_label=_b_better,
        title=_cf_title.replace("{a}", bike_b).replace("{b}", bike_a),
        x_axis_label=_cf_x_lbl, y_axis_label=_cf_y_lbl,
    ), width='stretch')
    st.caption(
        f"Same logic as Step 2, but {bike_b}'s model is applied to {bike_a}'s efforts."
    )

    mean_ba = float(pred_ba_d[_residual_col].mean())
    n_ba = int(pred_ba_d[_residual_col].notna().sum())

    col_m3, col_m4 = st.columns([1, 2])
    with col_m3:
        if watt_mode:
            sign_ba = "more efficient" if mean_ba >= 0 else "less efficient"
            st.metric(
                label=f"{bike_a} is",
                value=f"{abs(mean_ba):.1f} W {sign_ba}",
                delta=f"{mean_ba:+.1f} W  (n\u00a0= {n_ba} efforts)",
                delta_color="normal",
            )
            st.caption(f"than {bike_b} model")
        else:
            sign_ba = "faster" if mean_ba >= 0 else "slower"
            st.metric(
                label=f"{bike_a}",
                value=f"{abs(mean_ba):.2f} {_unit} {sign_ba}",
                delta=f"{mean_ba:+.2f} {_unit}  (n\u00a0= {n_ba} efforts)",
                delta_color="normal",
            )
            st.caption(f"than {bike_b} model")
    with col_m4:
        st.plotly_chart(_plot_residuals(pred_ba_d, bike_b, bike_a, _residual_col, _unit, _a_better), width='stretch')
    st.caption(
        f"Mean residual = {mean_ba:+.2f} {_unit} (positive = {bike_a} better). "
        f"Negated to match direction: {-mean_ba:+.2f} {_unit} advantage for {bike_b}."
    )
    if watt_mode:
        # watts_residual = predicted_b_watts − actual_a_watts  (see apply_watt_model_to_bike)
        # Step 4 asked: "how many watts would bike_b need to go bike_a's speed?"
        # mean_ba > 0 → bike_b's model predicts MORE watts than bike_a used → bike_a is more efficient
        # mean_ba < 0 → bike_b's model predicts FEWER watts than bike_a used → bike_b is more efficient
        # Subject must be bike_b (the model) to directly answer the Step 4 question.
        _ba_direction = "more" if mean_ba >= 0 else "fewer"
        st.info(
            f"**Step 4 in plain terms:** Flipping it around — for the same efforts, {bike_b} would have required "
            f"**{abs(mean_ba):.1f} W {_ba_direction}** than {bike_a} used to go the same speed — "
            f"based on {n_ba} matched efforts."
        )
    else:
        _ba_direction = "faster" if mean_ba <= 0 else "slower"
        st.info(
            f"**Step 4 in plain terms:** Flipping it around — if you did a {bike_a} effort "
            f"on {bike_b} instead, you would've gone **{abs(mean_ba):.2f} {_unit} {_ba_direction}** — "
            f"based on {n_ba} matched efforts."
        )

    # ── Step 5: Aggregate ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("Step 5 \u2014 Aggregate")
    st.caption(
        "We combine both directions into a single estimate **weighted by effort count**: "
        "**combined\u00a0= (fwd\u202fmean\u00a0\u00d7\u00a0n\\_fwd \u2212 rev\u202fmean\u00a0\u00d7\u00a0n\\_rev) / (n\\_fwd\u00a0+\u00a0n\\_rev)**. "
        "The direction with more efforts carries more weight."
    )
    st.caption("""
    In previous steps models are trained with TT split, here they are retrained on full bike a/b because they are only used to predict on other bike.
    Also bootstrapping is performed on this rerun for confidence intervals""")

    summary = aggregate_paired_delta_bootstrap(ab_bootstrap_results, ba_bootstrap_results)

    # Bootstrap residuals are always in km/h (training data is never scaled).
    # Scale all summary values to the display unit so metrics, CIs, and the
    # aggregate plot are consistent with _unit. Watt mode is unaffected
    # because _disp_scale is 1.0 in that case.
    if _disp_scale != 1.0:
        for _k in ("combined", "ci_low", "ci_high", "fwd_mean", "rev_mean", "symmetry_gap"):
            summary[_k] = summary[_k] * _disp_scale
        for _k in ("fwd_estimates", "rev_estimates", "combined_estimates"):
            summary[_k] = summary[_k] * _disp_scale

    combined = summary["combined"]
    winner = bike_b if combined >= 0 else bike_a
    loser = bike_a if combined >= 0 else bike_b
    advantage = abs(combined)

    ci_str = ""
    if not np.isnan(summary["ci_low"]):
        ci_str = f"  (95%\u00a0CI: {summary['ci_low']:+.2f} to {summary['ci_high']:+.2f}\u00a0{_unit})"

    if watt_mode:
        st.metric(
            label="Overall efficiency advantage",
            value=f"{loser} uses {advantage:.1f} W more than {winner}",
            delta=f"{combined:+.1f} W{ci_str}",
            delta_color="normal",
        )
    else:
        st.metric(
            label="Overall speed advantage",
            value=f"{winner} is {advantage:.2f} {_unit} faster than {loser}",
            delta=f"{combined:+.2f} {_unit}{ci_str}",
            delta_color="normal",
        )

    st.plotly_chart(_plot_aggregate(summary, bike_a, bike_b, _unit, _adv_label), width='stretch')

    better_word = "more efficient" if watt_mode else "faster"
    st.caption(
        f"**A\u2192B**: apply {bike_a}'s model to {bike_b}'s efforts \u2014 positive means {bike_b} {better_word}. "
        f"**B\u2192A (negated)**: apply {bike_b}'s model to {bike_a}'s efforts, negated so positive still means {bike_b} {better_word}. "
        "**Combined**: average of both directions \u2014 our best symmetric estimate."
    )

    fwd_sign = np.sign(summary["fwd_mean"])
    rev_sign_as_b_adv = -np.sign(summary["rev_mean"])
    if fwd_sign == rev_sign_as_b_adv:
        st.success(f"Both directions agree on which bike is {better_word}, increasing confidence in the result.")
    else:
        st.warning(
            f"The two directions disagree on which bike is {better_word}. "
            "This may indicate high variability, few efforts, or conditions the model cannot fully control."
        )
    if watt_mode:
        st.info(
            f"**Step 5 in plain terms:** Putting it all together — **{winner}** is on average "
            f"**{advantage:.1f} W more efficient** than {loser} in equivalent riding conditions."
            + (f" (95% CI: {abs(summary['ci_low']):.1f}–{abs(summary['ci_high']):.1f} W)" if not np.isnan(summary['ci_low']) else "")
        )
    else:
        st.info(
            f"**Step 5 in plain terms:** Putting it all together — **{winner}** is on average "
            f"**{advantage:.2f}\u00a0{_unit} faster** than {loser} in equivalent riding conditions."
            + (f" (95%\u00a0CI: {abs(summary['ci_low']):.2f}\u2013{abs(summary['ci_high']):.2f}\u00a0{_unit})" if not np.isnan(summary['ci_low']) else "")
        )

    with st.expander("How reliable is this?"):
        validate(ab_bootstrap_results, ba_bootstrap_results, bike_a, bike_b)


def build_summary(
    ab_boot: dict,
    ba_boot: dict,
) -> dict:
    """Build a summary dict for validate() from two bootstrap_pipeline outputs.

    Parameters
    ----------
    ab_boot:
        Output of bootstrap_pipeline(df, train_bike=A, target_bike=B).
    ba_boot:
        Output of bootstrap_pipeline(df, train_bike=B, target_bike=A).

    Returns
    -------
    dict with keys: combined, ci_low, ci_high, fwd_mean, rev_mean,
    symmetry_gap, n_fwd, n_rev.
    """
    fwd_mean = ab_boot["mean_residual"]
    rev_mean = ba_boot["mean_residual"]

    # combined effect: average of both directions
    # negate rev so positive = bike_b faster in both cases
    combined = (fwd_mean - rev_mean) / 2

    # build combined bootstrap distribution from pooled residuals
    # by pairing fwd and rev residuals index-wise, truncated to
    # the shorter length to keep arrays aligned
    fwd_residuals = np.array(ab_boot["per_iteration_mean_resid"])
    rev_residuals = np.array(ba_boot["per_iteration_mean_resid"])
    n_paired = min(len(fwd_residuals), len(rev_residuals))
    combined_estimates = (
        fwd_residuals[:n_paired] - rev_residuals[:n_paired]
    ) / 2

    ci_low = float(np.percentile(combined_estimates, 2.5))
    ci_high = float(np.percentile(combined_estimates, 97.5))
    symmetry_gap = abs(fwd_mean + rev_mean)

    return {
        "combined": combined,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "fwd_mean": fwd_mean,
        "rev_mean": rev_mean,
        "fwd_estimates": fwd_residuals,
        "rev_estimates": rev_residuals,
        "combined_estimates": combined_estimates,
        "symmetry_gap": symmetry_gap,
        "n_fwd": ab_boot["n_successful"],
        "n_rev": ba_boot["n_successful"],
    }

def validate(ab_bootstrap_results, ba_bootstrap_results, bike_a, bike_b) -> None:
    """
    Visualize residual distributions for both prediction directions (A→B and B→A)
    to assess whether the causal estimate is trustworthy.

    A well-specified model should produce residuals that are roughly symmetric
    around zero, since systematic skew suggests the model is picking up bias
    rather than random variability. The means of the two distributions should
    also be near-mirror images of each other (the "symmetry check") — if training
    on A and predicting B gives a mean residual of +0.4, training on B and
    predicting A should give roughly -0.4. Large divergence between the two
    suggests the model isn't fully controlling for conditions.

    Parameters
    ----------
    ab_bootstrap_results:
        Output of bootstrap_pipeline(df, train_bike=A, target_bike=B).
    ba_bootstrap_results:
        Output of bootstrap_pipeline(df, train_bike=B, target_bike=A).
    bike_a:
        Name of bike A.
    bike_b:
        Name of bike B.
    summary:
        Output of aggregate_paired_delta_bootstrap — provides combined,
        ci_low, ci_high for the effect size vs CI width check.
    """
    st.subheader("Residual distribution")
    st.markdown(
        "These histograms show the gap between actual and predicted speed "
        "for each direction of the model. A trustworthy result looks like a roughly "
        "symmetric bell shape centered away from zero — that center point is the "
        "estimated bike effect, and the spread around it reflects natural ride-to-ride "
        "variability rather than something the model is missing."
    )
    summary = build_summary(ab_bootstrap_results, ba_bootstrap_results)
    ab_boot_residuals = ab_bootstrap_results["boot_residuals"]
    ba_boot_residuals = ba_bootstrap_results["boot_residuals"]
    ab_mean = ab_bootstrap_results["mean_residual"]
    ba_mean = ba_bootstrap_results["mean_residual"]
    symmetry_gap = summary["symmetry_gap"]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=ab_boot_residuals,
        name=f"{bike_a} → {bike_b}",
        opacity=0.6,
        marker_color="#2ca02c",
        nbinsx=40,
    ))
    fig.add_trace(go.Histogram(
        x=ba_boot_residuals,
        name=f"{bike_b} → {bike_a}",
        opacity=0.6,
        marker_color="#ff7f0e",
        nbinsx=40,
    ))

    fig.add_vline(
        x=ab_mean,
        line_dash="dash",
        line_color="#2ca02c",
        annotation_text=f"{bike_a}→{bike_b} mean: {ab_mean:.3f}",
        annotation_position="top",
    )
    fig.add_vline(
        x=ba_mean,
        line_dash="dash",
        line_color="#ff7f0e",
        annotation_text=f"{bike_b}→{bike_a} mean: {ba_mean:.3f}",
        annotation_position="bottom",
    )
    fig.add_vline(x=0, line_color="gray", line_width=1)

    fig.update_layout(
        barmode="overlay",
        xaxis_title="Residual (actual − predicted speed km/h)",
        yaxis_title="Count",
        legend_title="Direction",
        height=420,
        margin=dict(t=40, b=40),
    )

    st.plotly_chart(fig, width='stretch')

    st.markdown(
        f"**{bike_a}→{bike_b} mean residual:** {ab_mean:.3f}  \n"
        f"**{bike_b}→{bike_a} mean residual:** {ba_mean:.3f}  \n"
        f"**Symmetry gap:** {symmetry_gap:.3f} "
        f"(how far the two means are from being exact opposites — closer to 0 is better)  \n"
        f"**Combined effect size:** {summary['combined']:.3f} "
        f"(the estimated effect of the bike)."
    )
    st.info("""
        These being near exact opposites is good because it is like saying 
        "Bike A is X faster than bike B, and bike B is Y slower than bike A"
        Inherently, in real life X=Y so in our experiment X being close to Y is good
    """)
    effect_size = abs(summary["combined"])

    # in a perfect world symmetry gap = 0
    # effect size is my mean combined estimate (effect of bike) - center of my distribution
    if symmetry_gap < 0.2 * effect_size:
        st.success(f"🟢 Symmetric — estimates are within 20% this indicates a strong relationship.")
    elif symmetry_gap < 0.50 * effect_size:
        st.warning(f"🟡 Moderate — some divergence between directions, interpret with care. (20%-50%)")
    else:
        st.error(f"🔴 Asymmetric — the two directions disagree significantly. Treat the result with caution. (>50%)")

    st.markdown(
        "If these means don't roughly mirror each other, it may indicate a bug, "
        "insufficient data, or that the model isn't fully controlling for conditions." \
        "Because the effect size is generally quite small, I was generous with what % we consider Symmetric and Moderate"
    )

    # --- Effect size vs CI width ---
    st.subheader("Confidence interval width")
    st.markdown(
        "A statistically detectable effect is only meaningful if the uncertainty "
        "around it is small relative to the effect itself. A confidence interval "
        "wider than the effect means we can't reliably distinguish signal from noise."
    )

    combined = summary["combined"]
    ci_low = summary["ci_low"]
    ci_high = summary["ci_high"]
    ci_width = ci_high - ci_low
    effect_size = abs(combined)
    ci_to_effect = ci_width / effect_size if effect_size > 0 else float("inf")
    ci_crosses_zero = ci_low < 0 < ci_high

    faster_bike = bike_b if combined > 0 else bike_a
    effect_label = f"{effect_size:.2f} km/h"
    ci_label = f"[{ci_low:+.2f}, {ci_high:+.2f}] km/h"

    col1, col2, col3 = st.columns(3)
    col1.metric("Estimated effect", f"{combined:+.2f} km/h")
    col2.metric("95% CI", ci_label)
    col3.metric("CI width / effect size", f"{ci_to_effect:.1f}×")

    if ci_crosses_zero:
        st.error(
            f"🔴 The confidence interval crosses zero {ci_label}. "
            f"The data does not clearly favor either bike — the effect could be zero. "
            f"More rides on comparable segments are needed to reach a conclusion."
        )
    elif ci_to_effect > 2.0:
        st.warning(
            f"🟡 The estimated effect is {effect_label} in favor of {faster_bike}, "
            f"but the confidence interval is {ci_to_effect:.1f}× wider than the effect itself. "
            f"The direction is consistent but the magnitude is uncertain — ride more segments to narrow this down."
        )
    elif ci_to_effect > 1.0:
        st.warning(
            f"🟡 Moderate confidence. {faster_bike} appears faster by {effect_label} "
            f"and the CI does not cross zero, but uncertainty is still meaningful relative to the effect size. "
            f"Treat the magnitude with some caution."
        )
    else:
        st.success(
            f"🟢 Strong confidence. {faster_bike} is estimated to be {effect_label} faster "
            f"and the confidence interval is tight relative to the effect size {ci_label}. "
            f"This is a reliable result."
        )


def main() -> None:
    st.title("📊 Step 4 — Head to Head Bike Comparison")
    st.markdown(
        "Filters and cleaning are already applied (configured in **Step 2 — Data Cleaning**). "
        "Select bikes and segments below to compare performance."
    )

    page_guard("bike_comparison_overall")

    bikes_to_compare = overall_comp_inputs()

    show(bikes_to_compare)



navigator("bike_comparison_overall1")
main()
navigator("bike_comparison_overall2")