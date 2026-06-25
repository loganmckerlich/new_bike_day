
"""Bike speed delta estimation — segment-based statistical pipeline.

Implements the 5-phase guide from the Bike Speed Delta Estimation PDF:

    Phase 1: Data preparation and feature engineering.
    Phase 2: Spline baseline model fit on reference bike only, projected onto all.
    Phase 3: Power distribution overlap check per segment (KS test).
    Phase 4: Per-segment OLS regression for delta estimation.
    Phase 5: Aggregate with inverse-variance weighting and report.

Key design decision: the baseline is fit on one reference bike only to avoid
absorbing the bike effect into the fitness/seasonal trend.

Data field mapping (Strava → PDF terminology)
----------------------------------------------
avg_speed        → distance / moving_time * 3.6 (km/h)
avg_power        → average_watts
ride_id          → effort_id
bike_id          → gear_id mapped via bikes dict → bike_name
timestamp        → start_date (parsed to naive UTC datetime)
elevation_gain   → total_elevation_gain (from segments)
average_grade    → average_grade (from segments)
air_temp         → optional; merged from activities if present
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import statsmodels.formula.api as smf
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import streamlit as st
from src.analytics import compute_speed_per_watt

__all__ = [
    "prepare_delta_dataset",
    "get_paired_segments",
    "fit_baseline_model",
    "compute_residuals",
    "power_overlap_ok",
    "segment_power_overlap_summary",
    "per_segment_delta",
    "weighted_delta_summary",
    "compute_i2",
    "delta_to_sec_per_km",
    # XGBoost counterfactual pipeline
    "XGB_FEATURES",
    "fit_xgb_speed_model",
    "apply_model_to_bike",
    "aggregate_paired_delta",
    # XGBoost watts-efficiency pipeline
    "XGB_WATT_FEATURES",
    "fit_xgb_watt_model",
    "apply_watt_model_to_bike",
]


# ── Phase 1 — Data preparation ─────────────────────────────────────────────────

def prepare_delta_dataset(
    efforts_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    bikes_dict: dict[str, str],
    min_watts: float = 0,
) -> pd.DataFrame:
    """Prepare a flat analysis DataFrame for the delta pipeline.

    Merges segment metadata, derives speed and efficiency metrics, and
    engineers the temporal features needed by the spline baseline model.

    Parameters
    ----------
    efforts_df:
        Cleaned efforts with at minimum: effort_id, segment_id, gear_id,
        start_date, moving_time, average_watts.
    segments_df:
        Segments with at minimum: segment_id, distance, average_grade.
    bikes_dict:
        Mapping of gear_id → human-readable bike name (from Strava API).
    min_watts:
        Drop efforts below this average power threshold (0 = keep all).

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame ready for Phases 2–5.  New columns added:
        ``bike_name``, ``speed_kmh``, ``speed_per_cbrt_watt``,
        ``ride_index`` (days since first effort), ``doy_sin``, ``doy_cos``,
        ``average_grade`` (merged from segments_df).
    """
    required = {"effort_id", "segment_id", "gear_id", "start_date", "moving_time", "average_watts"}
    missing = required - set(efforts_df.columns)
    if missing:
        raise ValueError(f"efforts_df is missing required columns: {missing}")

    df = efforts_df.copy()

    # ── Map gear_id → bike name ────────────────────────────────────────────────
    df["bike_name"] = df["gear_id"].map(
        lambda g: bikes_dict.get(str(g), str(g)) if g is not None else "Unknown"
    )

    df = df[df["bike_name"] != 'nan'].copy()  # Drop efforts with missing gear_id

    # ── Merge segment distance + grade + type ─────────────────────────────────
    # Only bring over columns that are not already present in df (e.g. because
    # data_cleaning.py already merged them from segments).  Re-merging a column
    # that exists in both frames would produce _x / _y suffixes and lose the
    # original column name.
    seg_cols = ["segment_id"]
    for col in ("distance", "average_grade", "maximum_grade", "segment_type", "segment_type_detail", "climb_category", "hazardous", "total_elevation_gain"):
        if col in segments_df.columns and col not in df.columns:
            seg_cols.append(col)
    if len(seg_cols) > 1:
        seg_meta = segments_df[seg_cols].drop_duplicates("segment_id")
        df = df.merge(seg_meta, on="segment_id", how="left")

    # ── Derive speed_kmh (segment distance / effort moving time) ──────────────
    if "speed_kmh" not in df.columns:
        if "distance" in df.columns:
            safe_time = df["moving_time"].replace(0, np.nan)
            df["speed_kmh"] = (df["distance"] / safe_time * 3.6).where(
                safe_time.notna() & df["distance"].notna()
            )
        else:
            df["speed_kmh"] = np.nan

    # ── Power and speed filters ────────────────────────────────────────────────
    if min_watts > 0:
        df = df[df["average_watts"] >= min_watts].copy()

    # Drop rows that are unusable for the model
    df = df[df["average_watts"].notna() & df["speed_kmh"].notna()].copy()

    # ── Primary outcome variable: speed / power^(1/3) ─────────────────────────
    # Reuses the existing compute_speed_per_watt function which produces
    # speed_per_cbrt_watt = speed_kmh / average_watts^(1/3)
    df = compute_speed_per_watt(df)

    # ── Date features for baseline model ──────────────────────────────────────
    # Parse to UTC-normalised naive datetime so arithmetic is straightforward.
    ts = pd.to_datetime(df["start_date"], errors="coerce", utc=True).dt.tz_convert(None)
    df["_ts"] = ts
    t_min = df["_ts"].min()
    df["ride_index"] = (df["_ts"] - t_min).dt.days.astype(float)
    df["doy_sin"] = np.sin(2 * np.pi * df["_ts"].dt.dayofyear / 365.0)
    df["doy_cos"] = np.cos(2 * np.pi * df["_ts"].dt.dayofyear / 365.0)
    df = df.drop(columns=["_ts"])

    return df.reset_index(drop=True)


def get_paired_segments(
    df: pd.DataFrame,
    bikes: list[str],
    min_efforts: int = 3,
) -> list[int]:
    """Return segment IDs where ALL listed bikes have ≥ min_efforts efforts.

    Parameters
    ----------
    df:
        Output from :func:`prepare_delta_dataset`.
    bikes:
        Bike names to require coverage for.
    min_efforts:
        Minimum number of efforts per bike per segment.
    """
    scope = df[df["bike_name"].isin(bikes)]
    seg_counts = (
        scope.groupby(["segment_id", "bike_name"])["effort_id"]
        .count()
        .unstack(fill_value=0)
    )
    for b in bikes:
        if b not in seg_counts.columns:
            seg_counts[b] = 0
    valid_mask = (seg_counts[bikes] >= min_efforts).all(axis=1)
    return seg_counts[valid_mask].index.tolist()


# ── Phase 2 — Baseline model ───────────────────────────────────────────────────

def fit_baseline_model(
    df: pd.DataFrame,
    ref_bike_name: str,
) -> tuple[Any, Any]:
    """

    The baseline captures seasonal
    variation (doy_sin, doy_cos) while controlling for power (average_watts).
    Fitting on only the reference bike prevents the trend from absorbing the
    bike effect, which would underestimate the speed delta.

    Parameters
    ----------
    df:
        Enriched dataset from :func:`prepare_delta_dataset`.
    ref_bike_name:
        Bike name used as the temporal reference.  Choose the bike with the
        most efforts and most stable fitness period.

    Returns
    -------
    model : sklearn.ensemble.GradientBoostingRegressor
        Fitted regression model.
    """

    bike_df = df[df["bike_name"] == ref_bike_name].dropna(
        subset=["ride_index", "average_watts", "doy_sin", "doy_cos", "speed_per_cbrt_watt"]
    ).copy()

    X = bike_df[["ride_index", "average_watts", "doy_sin", "doy_cos"]]
    y = bike_df["speed_per_cbrt_watt"].values

    model = GradientBoostingRegressor().fit(X, y)
    return model


def compute_residuals(
    df: pd.DataFrame,
    model: Any,
) -> pd.DataFrame:
    """Project the baseline model onto all bikes and compute residuals.

    Residuals = actual speed_per_cbrt_watt − predicted by the fitness/seasonal
    baseline.  A positive residual for bike B means it was faster than the
    reference baseline predicts, controlling for power and time.

    Parameters
    ----------
    df:
        Full enriched dataset (all bikes) from :func:`prepare_delta_dataset`.
    model:
        Fitted baseline model from :func:`fit_baseline_model`.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``predicted`` and ``residual`` columns added.
        Rows with missing features remain NaN in those columns.
    """

    out = df.copy()
    out["predicted"] = np.nan
    out["residual"] = np.nan


    try:
        # ride_index is used to try to account for fitness over time
        preds = model.predict(out[["ride_index", "average_watts", "doy_sin", "doy_cos"]])
    except Exception as e:
        warnings.warn(f"Residual computation failed: {e}")
        return out

    out["predicted"] = preds
    out["residual"] = (
        out["speed_per_cbrt_watt"] - out["predicted"]
    )
    return out


# ── Phase 3 — Power distribution overlap check ────────────────────────────────

def power_overlap_ok(
    seg_df: pd.DataFrame,
    bike_a: str,
    bike_b: str,
    p_threshold: float = 0.05,
) -> bool:
    """Check whether power distributions are similar enough to compare bikes.

    Uses a two-sample Kolmogorov-Smirnov test on ``average_watts``.
    Returns True when the distributions are NOT significantly different
    (p > p_threshold), meaning a like-for-like comparison is possible.

    Parameters
    ----------
    seg_df:
        Efforts for a single segment (may include multiple bikes).
    bike_a, bike_b:
        Bike names to compare.
    p_threshold:
        KS test p-value threshold. Default 0.05.
    """
    a = seg_df[seg_df["bike_name"] == bike_a]["average_watts"].dropna()
    b = seg_df[seg_df["bike_name"] == bike_b]["average_watts"].dropna()
    if len(a) < 3 or len(b) < 3:
        return False
    _, p = ks_2samp(a, b)
    return bool(p > p_threshold)


def segment_power_overlap_summary(
    df: pd.DataFrame,
    bikes: list[str],
    segment_ids: list[int],
    p_threshold: float = 0.05,
) -> pd.DataFrame:
    """Return a DataFrame with KS power-overlap status for each segment × pair.

    Columns: segment_id, bike_a, bike_b, ks_ok (bool), p_value (float).
    """
    pairs = [
        (bikes[i], bikes[j])
        for i in range(len(bikes))
        for j in range(i + 1, len(bikes))
    ]
    records: list[dict] = []
    for seg_id in segment_ids:
        seg_df = df[df["segment_id"] == seg_id]
        for bike_a, bike_b in pairs:
            a = seg_df[seg_df["bike_name"] == bike_a]["average_watts"].dropna()
            b = seg_df[seg_df["bike_name"] == bike_b]["average_watts"].dropna()
            if len(a) < 3 or len(b) < 3:
                records.append(
                    {"segment_id": seg_id, "bike_a": bike_a, "bike_b": bike_b,
                     "ks_ok": False, "p_value": np.nan}
                )
                continue
            _, p = ks_2samp(a, b)
            records.append(
                {"segment_id": seg_id, "bike_a": bike_a, "bike_b": bike_b,
                 "ks_ok": bool(p > p_threshold), "p_value": round(float(p), 4)}
            )
    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=["segment_id", "bike_a", "bike_b", "ks_ok", "p_value"]
    )


# ── Phase 4 — Per-segment delta estimation ────────────────────────────────────

def per_segment_delta(
    df: pd.DataFrame,
    paired_segments: list[int],
    ref_bike: str,
    bikes: list[str],
) -> pd.DataFrame:
    """Run per-segment OLS to estimate speed_per_cbrt_watt delta vs reference.

    For each paired segment, fits:
        ``residual ~ C(bike_name) + average_watts``

    Extracts the coefficient and standard error for each non-reference bike.
    This controls for any remaining within-segment power imbalance beyond
    what the global baseline already removed.

    Parameters
    ----------
    df:
        Dataset with a ``residual`` column from :func:`compute_residuals`.
    paired_segments:
        Segment IDs where all bikes have sufficient efforts.
    ref_bike:
        Reference bike name (intercept category in the OLS).
    bikes:
        All bike names being compared (including ref_bike).

    Returns
    -------
    pd.DataFrame
        One row per (segment × non-reference bike). Columns:
        segment_id, ref_bike, other_bike, bike_pair, delta, se,
        weight (1/se²), n_ref, n_other, grade, length_m, paired.
    """
    other_bikes = [b for b in bikes if b != ref_bike]
    if not other_bikes:
        return pd.DataFrame()

    scope = df[
        df["segment_id"].isin(paired_segments) & df["residual"].notna()
    ].copy()

    records: list[dict] = []
    for seg_id, seg_df in scope.groupby("segment_id"):
        if seg_df["bike_name"].nunique() < 2:
            continue
        if ref_bike not in seg_df["bike_name"].values:
            continue

        # Force C(bike_name) reference level to ref_bike via category ordering
        seg_df = seg_df.copy()
        all_present = [ref_bike] + [b for b in other_bikes if b in seg_df["bike_name"].values]
        seg_df["bike_name"] = pd.Categorical(seg_df["bike_name"], categories=all_present)

        try:
            ols_model = smf.ols("residual ~ C(bike_name) + average_watts", data=seg_df).fit()
        except Exception as exc:
            warnings.warn(f"Segment {seg_id} OLS failed: {exc}")
            continue

        grade = float(seg_df["average_grade"].mean()) if "average_grade" in seg_df.columns else np.nan
        length = float(seg_df["distance"].iloc[0]) if "distance" in seg_df.columns else np.nan

        for other in other_bikes:
            if other not in seg_df["bike_name"].values:
                continue
            coef_name = f"C(bike_name)[T.{other}]"
            if coef_name not in ols_model.params:
                continue
            delta = float(ols_model.params[coef_name])
            se = float(ols_model.bse[coef_name])
            if se <= 0 or np.isnan(se):
                continue
            records.append({
                "segment_id": seg_id,
                "ref_bike": ref_bike,
                "other_bike": other,
                "bike_pair": f"{ref_bike} → {other}",
                "delta": delta,
                "se": se,
                "weight": 1.0 / se ** 2,
                "n_ref": int((seg_df["bike_name"] == ref_bike).sum()),
                "n_other": int((seg_df["bike_name"] == other).sum()),
                "grade": grade,
                "length_m": length,
                "paired": True,
            })

    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=["segment_id", "ref_bike", "other_bike", "bike_pair",
                 "delta", "se", "weight", "n_ref", "n_other", "grade", "length_m", "paired"]
    )


# ── Phase 5 — Aggregate and report ────────────────────────────────────────────

def weighted_delta_summary(deltas_df: pd.DataFrame) -> pd.DataFrame:
    """Compute inverse-variance weighted delta per bike pair (meta-analytic standard).

    Weight = 1/SE².  Segments with tighter estimates carry more weight.

    Parameters
    ----------
    deltas_df:
        Output from :func:`per_segment_delta`.

    Returns
    -------
    pd.DataFrame
        One row per bike_pair with columns:
        bike_pair, ref_bike, other_bike, delta, se, ci_low, ci_high, n_segments.
    """
    if deltas_df.empty:
        return pd.DataFrame(
            columns=["bike_pair", "ref_bike", "other_bike",
                     "delta", "se", "ci_low", "ci_high", "n_segments"]
        )

    records: list[dict] = []
    for pair, grp in deltas_df.groupby("bike_pair"):
        w = grp["weight"].values
        d = grp["delta"].values
        estimate = float((w * d).sum() / w.sum())
        se = float(np.sqrt(1.0 / w.sum()))
        records.append({
            "bike_pair": pair,
            "ref_bike": grp["ref_bike"].iloc[0],
            "other_bike": grp["other_bike"].iloc[0],
            "delta": estimate,
            "se": se,
            "ci_low": estimate - 1.96 * se,
            "ci_high": estimate + 1.96 * se,
            "n_segments": len(grp),
        })
    return pd.DataFrame(records)


def compute_i2(deltas_df: pd.DataFrame) -> dict[str, float]:
    """Compute the I² heterogeneity statistic per bike pair.

    I² = max(0, (Q − (k−1)) / Q) where k is the number of segments and Q is
    the Cochran Q-statistic (weighted sum of squared deviations from the pooled
    estimate).

    I² < 0.25  →  low heterogeneity (results are consistent)
    I² 0.25–0.75  →  moderate heterogeneity
    I² > 0.75  →  high heterogeneity (decompose by segment type before reporting)

    Returns
    -------
    dict mapping bike_pair → I² value in [0, 1].
    """
    result: dict[str, float] = {}
    for pair, grp in deltas_df.groupby("bike_pair"):
        w = grp["weight"].values
        d = grp["delta"].values
        if len(d) < 2:
            result[str(pair)] = 0.0
            continue
        pooled = float((w * d).sum() / w.sum())
        Q = float(np.sum(w * (d - pooled) ** 2))
        k = len(d)
        i2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
        result[str(pair)] = round(i2, 3)
    return result


def delta_to_sec_per_km(
    delta: float,
    ref_power: float = 200.0,
    ref_speed_ms: float | None = None,
) -> float:
    """Convert a speed_per_cbrt_watt delta to seconds-per-km at a reference power.

    In aerodynamics speed ∝ P^(1/3), so the speed_per_cbrt_watt metric has
    units of km/h / W^(1/3).  To convert to a human-readable speed difference:

        delta_speed_kmh = delta * ref_power^(1/3)

    Parameters
    ----------
    delta:
        Estimated delta in speed_per_cbrt_watt units (km/h per W^(1/3)).
        Positive = the other bike is faster than the reference.
    ref_power:
        Reference watts for interpretation (default 200 W — typical endurance
        riding effort).
    ref_speed_ms:
        Reference speed in m/s.  If None, approximated as 30 km/h (8.33 m/s),
        which is a typical recreational road-cycling pace at ~200 W.

    Returns
    -------
    float
        Seconds per km saved (positive = faster, negative = slower).
    """
    delta_speed_kmh = delta * (ref_power ** (1.0 / 3.0))
    delta_speed_ms = delta_speed_kmh / 3.6

    if ref_speed_ms is None:
        ref_speed_ms = 30.0 / 3.6  # 30 km/h as m/s

    new_speed_ms = ref_speed_ms + delta_speed_ms
    if new_speed_ms <= 0:
        return 0.0

    t_base = 1000.0 / ref_speed_ms      # seconds to cover 1 km at base speed
    t_new = 1000.0 / new_speed_ms        # seconds to cover 1 km at new speed
    return float(t_base - t_new)         # positive = time saved = faster


# ── XGBoost counterfactual pipeline ───────────────────────────────────────────

XGB_FEATURES: list[str] = [
    'average_watts',
    'average_grade',
    'maximum_grade',
    'doy_sin',
    'doy_cos',
    'log_watts',
    'watts_per_grade',
    'distance_km',
    'heartrate',
    'effort_count',
    'segtype_detail_sprint_uphill',
    'woy_cos',
    'month_sin',
    'month_cos',
    'cbrt_watts',
    'woy_sin',
    'segtype_ascent',
    'segtype_detail_sprint_flat',
    'segtype_detail_sprint_downhill'
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add candidate features to the prepared dataset.
    Edit freely — comment out or add columns as you experiment.
    """
    out = df.copy()
    ts = pd.to_datetime(out["start_date"], errors="coerce", utc=True).dt.tz_convert(None)

    # ── Power transforms ──────────────────────────────────────────────────────
    out["log_watts"] = np.log1p(out["average_watts"])
    out["cbrt_watts"] = np.cbrt(out["average_watts"])

    # Speed transforms
    out["log_speed"] = np.log1p(out["speed_kmh"])
    out["cbrt_speed"] = np.cbrt(out["speed_kmh"])
    
    # ── Interaction: power efficiency on a slope ──────────────────────────────
    safe_grade = out["average_grade"].replace(0, np.nan)
    out["watts_per_grade"] = out["average_watts"] / safe_grade.abs()

    # ── Segment length ────────────────────────────────────────────────────────
    if "distance" in out.columns:
        out["distance_km"] = out["distance"] / 1000.0

    # ── Pacing: elapsed vs moving time ───────────────────────────────────────
    if "elapsed_time" in out.columns and "moving_time" in out.columns:
        safe_moving = out["moving_time"].replace(0, np.nan)
        out["elapsed_ratio"] = out["elapsed_time"] / safe_moving

    # ── Heart rate (when available) ───────────────────────────────────────────
    if "average_heartrate" in out.columns:
        out["heartrate"] = out["average_heartrate"]

    # ── Finer cyclical seasonality (week of year) ─────────────────────────────
    woy = ts.dt.isocalendar().week.astype(float)
    out["woy_sin"] = np.sin(2 * np.pi * woy / 52.0)
    out["woy_cos"] = np.cos(2 * np.pi * woy / 52.0)

    # ── Cumulative effort count as fitness ramp proxy ─────────────────────────
    out = out.sort_values("start_date")
    out["effort_count"] = np.arange(1, len(out) + 1, dtype=float)

    # ── Month encoded cyclically ──────────────────────────────────────────────
    month = ts.dt.month.astype(float)
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    dummies = pd.get_dummies(out["segment_type_detail"], prefix="segtype_detail", dummy_na=True, dtype=float)
    out = pd.concat([out, dummies], axis=1)

    dummies = pd.get_dummies(out["segment_type"], prefix="segtype", dummy_na=True, dtype=float)
    out = pd.concat([out, dummies], axis=1)
    
    return out.reset_index(drop=True)

@st.cache_resource(ttl=3600)
def fit_xgb_speed_model(df: pd.DataFrame, bike_name: str, cache_key: str = None) -> xgb.XGBRegressor:
    """Train an XGBoost regressor on one bike's efforts to predict speed_kmh.

    Features: average_watts, average_grade, doy_sin, doy_cos, ride_index.
    Target: speed_kmh.

    Parameters
    ----------
    df:
        Prepared dataset from :func:`prepare_delta_dataset` (all bikes).
    bike_name:
        Name of the bike to train on.

    Returns
    -------
    xgb.XGBRegressor
        Fitted model.

    Raises
    ------
    ValueError
        When fewer than 5 usable efforts exist for the bike.
    """
    bike_df = (
        df[df["bike_name"] == bike_name]
        .dropna(subset=XGB_FEATURES + ["speed_kmh"])
        .copy()
    )

    if len(bike_df) < 5:
        raise ValueError(
            f"Not enough efforts for {bike_name!r} (need ≥5, got {len(bike_df)})."
        )

    X = bike_df[XGB_FEATURES]
    y = bike_df["speed_kmh"].values

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y)
    return model

def apply_model_to_bike(
    model: xgb.XGBRegressor,
    df: pd.DataFrame,
    target_bike: str,
) -> pd.DataFrame:
    """Apply a speed model trained on source bike to another bike's efforts.

    For each effort on *target_bike*, the model predicts the speed that the
    source bike would have achieved under the same conditions (power, grade,
    season, fitness level).  The residual is the raw speed advantage of
    *target_bike* over what the source bike's model expects.

    Added columns
    -------------
    predicted_speed_kmh:
        Model's counterfactual speed for these conditions.
    speed_residual:
        ``speed_kmh - predicted_speed_kmh``.
        Positive → target_bike was faster than the source model predicts.

    Parameters
    ----------
    model:
        Fitted :class:`xgb.XGBRegressor` from :func:`fit_xgb_speed_model`.
    df:
        Full prepared dataset from :func:`prepare_delta_dataset`.
    target_bike:
        Bike name to apply the model to.

    Returns
    -------
    pd.DataFrame
        Filtered to *target_bike* efforts with new prediction columns.
    """
    out = (
        df[df["bike_name"] == target_bike]
        .dropna(subset=XGB_FEATURES + ["speed_kmh"])
        .copy()
    )

    if out.empty:
        return out

    out["predicted_speed_kmh"] = model.predict(out[XGB_FEATURES])
    out["speed_residual"] = out["speed_kmh"] - out["predicted_speed_kmh"]
    return out

def bootstrap_pipeline(
    df: pd.DataFrame,
    train_bike: str,
    target_bike: str,
    n_iterations: int = 1000,
    random_state: int = 42,
    apply_fn = apply_model_to_bike,
    fit_fn = fit_xgb_speed_model,
    label: str = "speed_residual",
    predicted_col: str = "predicted_speed_kmh",
    target_col: str = "speed_kmh",
) -> dict:

    rng = np.random.default_rng(random_state)
    residuals = []
    iteration_mean_resid = []
    all_results = pd.DataFrame()
    n_skipped = 0

    for _ in range(n_iterations):
        boot_sample = df.sample(
            n=len(df), replace=True, random_state=rng.integers(0, 1_000_000)
        )
        try:
            model = fit_fn(boot_sample, train_bike)
            result = apply_fn(model, boot_sample, target_bike)
        except ValueError:
            n_skipped += 1
            result = pd.DataFrame()
            continue

        if result.empty:
            n_skipped += 1
            continue

        all_results = pd.concat([all_results, result], ignore_index=True)

        iteration_mean_resid.append(result[label].mean())
        residuals.extend(result[label].tolist())

    residuals = np.array(residuals)

    effort_level_agg_residual = all_results.groupby("effort_id")[[predicted_col,target_col,label]].mean().reset_index()

    return {
        "full_result": result,
        "boot_residuals": residuals,
        "effort_residuals": effort_level_agg_residual,
        "mean_residual": residuals.mean(),
        "ci_lower": np.percentile(residuals, 2.5),
        "ci_upper": np.percentile(residuals, 97.5),
        "n_successful": len(residuals),
        "n_skipped": n_skipped,
        "per_iteration_mean_resid":iteration_mean_resid
    }

def aggregate_paired_delta_bootstrap(
    fwd_boot: dict,
    rev_boot: dict,
) -> dict[str, float]:
    """Combine forward and reverse bootstrap results into a single delta
    with a percentile-based confidence interval.

    Forward (A→B): fwd_boot["residuals"] = bootstrap distribution of
        mean(actual_B_speed − model_A_predicted). Positive → B is faster.
    Reverse (B→A): rev_boot["residuals"] = bootstrap distribution of
        mean(actual_A_speed − model_B_predicted). Positive → A is faster.

    For each bootstrap iteration i, the combined estimate is
    (fwd_residuals[i] - rev_residuals[i]) / 2. A positive combined value
    means B is faster than A. The confidence interval comes from the
    percentiles of this combined distribution rather than a normal
    approximation, since the bootstrap already captures the empirical
    sampling distribution directly.

    Parameters
    ----------
    fwd_boot:
        Output of bootstrap_pipeline(df, train_bike=A, target_bike=B).
    rev_boot:
        Output of bootstrap_pipeline(df, train_bike=B, target_bike=A).

    Returns
    -------
    dict with keys: fwd_mean, rev_mean, combined, ci_low, ci_high,
    fwd_estimates, rev_estimates, combined_estimates, n_fwd, n_rev,
    symmetry_gap.
    """
    fwd_residuals = np.array(fwd_boot["per_iteration_mean_resid"])
    rev_residuals = np.array(rev_boot["per_iteration_mean_resid"])

    fwd_mean = fwd_boot["mean_residual"]
    rev_mean = rev_boot["mean_residual"]

    # pair iterations index-wise to preserve correlation structure;
    # truncate to the shorter length if iteration counts differ
    # (e.g. due to skipped iterations in one direction)
    n_paired = min(len(fwd_residuals), len(rev_residuals))
    fwd_residuals_paired = fwd_residuals[:n_paired]
    rev_residuals_paired = rev_residuals[:n_paired]
    combined_estimates = (fwd_residuals_paired - rev_residuals_paired) / 2

    combined = float(combined_estimates.mean())
    ci_low = float(np.percentile(combined_estimates, 2.5))
    ci_high = float(np.percentile(combined_estimates, 97.5))

    symmetry_gap = abs(fwd_mean + rev_mean)

    return {
        "fwd_mean": fwd_mean,
        "rev_mean": rev_mean,
        "combined": combined,
        "fwd_estimates": fwd_residuals_paired,
        "rev_estimates": rev_residuals_paired,
        "combined_estimates": combined_estimates,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_fwd": fwd_boot["n_successful"],
        "n_rev": rev_boot["n_successful"],
        "symmetry_gap": symmetry_gap,
    }
# ── XGBoost watts-efficiency counterfactual pipeline ──────────────────────────

# add some transformations of speed ie cbrt or sqrt
XGB_WATT_FEATURES: list[str] = [
    "speed_kmh",
    'average_grade',
    'maximum_grade',
    'doy_sin',
    'doy_cos',
    'distance_km',
    'heartrate',
    'effort_count',
    'segtype_detail_sprint_uphill',
    'woy_cos',
    'month_sin',
    'month_cos',
    'woy_sin',
    'segtype_ascent',
    'segtype_detail_sprint_flat',
    'segtype_detail_sprint_downhill',
    'log_speed',
    'cbrt_speed'
]

@st.cache_resource(ttl=3600)
def fit_xgb_watt_model(df: pd.DataFrame, bike_name: str, cache_key: str = None) -> xgb.XGBRegressor:
    """Train an XGBoost regressor on one bike's efforts to predict average_watts.

    The inverse of :func:`fit_xgb_speed_model`: given the speed achieved and
    riding conditions, predict how many watts were required.  Used to answer
    "how many watts would Bike A have needed to achieve Bike B's speed?"

    Features: speed_kmh, average_grade, doy_sin, doy_cos, ride_index.
    Target:   average_watts.

    A positive watt residual (predicted_A_watts > actual_B_watts) means Bike B
    achieves the same speed with fewer watts → Bike B is more efficient.
    """
    bike_df = (
        df[df["bike_name"] == bike_name]
        .dropna(subset=XGB_WATT_FEATURES + ["average_watts"])
        .copy()
    )

    if len(bike_df) < 5:
        raise ValueError(
            f"Not enough efforts for {bike_name!r} (need ≥5, got {len(bike_df)})."
        )

    X = bike_df[XGB_WATT_FEATURES]
    y = bike_df["average_watts"].values

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y)
    return model

def apply_watt_model_to_bike(
    model: xgb.XGBRegressor,
    df: pd.DataFrame,
    target_bike: str,
) -> pd.DataFrame:
    """Apply a watt model trained on source bike to another bike's efforts.

    For each *target_bike* effort, predicts how many watts the source bike
    would have needed to achieve the same speed under the same conditions.

    Added columns
    -------------
    predicted_watts:
        Watts the source bike's model predicts for these conditions.
    watts_residual:
        ``predicted_watts - average_watts``.
        Positive → target_bike used fewer watts at the same speed (more efficient).
    """
    out = (
        df[df["bike_name"] == target_bike]
        .dropna(subset=XGB_WATT_FEATURES + ["average_watts"])
        .copy()
    )

    if out.empty:
        return out

    out["predicted_watts"] = model.predict(out[XGB_WATT_FEATURES])
    # positive residual = target used fewer watts than source would have → target more efficient
    out["watts_residual"] = out["predicted_watts"] - out["average_watts"]
    return out
