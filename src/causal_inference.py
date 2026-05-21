"""Causal inference utilities for bike-vs-bike speed-per-watt analysis."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from src.analytics import compute_speed_per_watt, filter_outliers_by_power_speed
from src.weather import get_weather_for_efforts

_REQUIRED_COVARIATES: tuple[str, ...] = (
    "straightness_index",
    "headwind_component",
    "precipitation_mm",
    "average_grade",
    "temp_c",
)


def remove_outliers_for_causal_analysis(
    efforts_df: pd.DataFrame,
    z_threshold: float = 2.0,
    segments_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, int]:
    """Remove speed-per-cbrt-watt outliers using segment-comparison methodology."""
    required_cols = {"segment_id", "average_watts"}
    if efforts_df.empty or not required_cols.issubset(efforts_df.columns):
        return efforts_df.copy(), 0

    df = efforts_df.copy()

    # Derive speed_kmh from segment distance + moving_time when not already present.
    if "speed_kmh" not in df.columns and segments_df is not None and not segments_df.empty:
        dist_col = next((c for c in ("distance_m", "distance") if c in segments_df.columns), None)
        if dist_col is not None:
            seg_dist = segments_df[["segment_id", dist_col]].drop_duplicates("segment_id")
            df = df.merge(seg_dist, on="segment_id", how="left", suffixes=("", "_seg"))
            safe_time = df["moving_time"].replace(0, np.nan)
            df["speed_kmh"] = (df[dist_col] / safe_time * 3.6).where(safe_time.notna() & df[dist_col].notna())

    if "speed_kmh" not in df.columns or df["speed_kmh"].isna().all():
        return efforts_df.copy(), 0

    with_spw = compute_speed_per_watt(df)
    filtered, annotated = filter_outliers_by_power_speed(with_spw, z_threshold=z_threshold)
    n_outliers = int(annotated["is_outlier"].sum()) if "is_outlier" in annotated.columns else 0
    cleaned = filtered.drop(columns=["is_outlier", "z_score", "speed_per_cbrt_watt"], errors="ignore")
    # Drop any extra distance column added by the merge above if it wasn't in the original.
    if dist_col is not None and dist_col not in efforts_df.columns:
        cleaned = cleaned.drop(columns=[dist_col], errors="ignore")
    return cleaned, n_outliers


def _safe_float(value: Any) -> float | None:
    """Convert a value to float when possible."""
    try:
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _segment_bearing_deg(start_lat: float, start_lng: float, end_lat: float, end_lng: float) -> float:
    """Compute forward azimuth in degrees from start to end point."""
    lat1 = math.radians(start_lat)
    lat2 = math.radians(end_lat)
    dlon = math.radians(end_lng - start_lng)
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _covariate_columns(df: pd.DataFrame) -> list[str]:
    """Return covariate columns consumed by the causal models."""
    return [c for c in df.columns if c in _REQUIRED_COVARIATES or c.startswith("segment_type_")]


def build_feature_matrix(efforts_df: pd.DataFrame, segments_df: pd.DataFrame) -> pd.DataFrame:
    """Build model-ready features for causal speed-per-watt estimation.

    Args:
        efforts_df: Efforts table with per-attempt data and an ``is_new_bike`` column.
        segments_df: Segment metadata including distance/grade/coordinates/type.

    Returns:
        DataFrame containing outcome, treatment, and engineered covariates.
    """
    if efforts_df.empty:
        return pd.DataFrame()

    efforts = get_weather_for_efforts(efforts_df, segments_df)
    segment_cols = [
        c
        for c in (
            "segment_id",
            "distance",
            "distance_m",
            "average_grade",
            "segment_type",
            "start_lat",
            "start_lng",
            "end_lat",
            "end_lng",
            "start_latlng",
            "end_latlng",
        )
        if c in segments_df.columns
    ]
    merged = efforts.merge(segments_df[segment_cols], on="segment_id", how="left")

    merged = merged[merged["average_watts"].notna() & (merged["average_watts"] >= 50)].copy()
    if merged.empty:
        return merged

    if "distance_m" not in merged.columns:
        merged["distance_m"] = merged.get("distance")

    if "average_speed_mps" not in merged.columns:
        moving_time = merged.get("moving_time", pd.Series(np.nan, index=merged.index)).replace(0, np.nan)
        merged["average_speed_mps"] = merged["distance_m"] / moving_time

    merged["speed_per_cbrt_watt"] = merged["average_speed_mps"] * 3.6 / np.cbrt(merged["average_watts"])

    def _resolve_point(row: pd.Series, prefix: str) -> tuple[float | None, float | None]:
        lat = _safe_float(row.get(f"{prefix}_lat"))
        lng = _safe_float(row.get(f"{prefix}_lng"))
        if lat is not None and lng is not None:
            return lat, lng
        latlng = row.get(f"{prefix}_latlng")
        if isinstance(latlng, (list, tuple)) and len(latlng) >= 2:
            return _safe_float(latlng[0]), _safe_float(latlng[1])
        return None, None

    straightness_vals: list[float] = []
    bearings: list[float | None] = []
    for row in merged.to_dict(orient="records"):
        row_s = pd.Series(row)
        start = _resolve_point(row_s, "start")
        end = _resolve_point(row_s, "end")
        distance_m = _safe_float(row_s.get("distance_m"))
        if None in start or None in end or not distance_m or distance_m <= 0:
            straightness_vals.append(np.nan)
            bearings.append(None)
            continue

        crow = geodesic((start[0], start[1]), (end[0], end[1])).meters
        straightness_vals.append(float(np.clip(crow / distance_m, 0.0, 1.0)))
        bearings.append(_segment_bearing_deg(start[0], start[1], end[0], end[1]))

    merged["straightness_index"] = straightness_vals
    merged["segment_bearing_deg"] = bearings
    merged["segment_bearing_deg"] = pd.to_numeric(merged["segment_bearing_deg"], errors="coerce")
    merged["wind_speed_kph"] = pd.to_numeric(merged["wind_speed_kph"], errors="coerce")
    merged["wind_direction_deg"] = pd.to_numeric(merged["wind_direction_deg"], errors="coerce")
    merged["headwind_component"] = merged["wind_speed_kph"] * np.cos(
        np.radians(merged["wind_direction_deg"] - merged["segment_bearing_deg"])
    )

    merged["straightness_index"] = merged["straightness_index"].fillna(1.0)
    merged["headwind_component"] = merged["headwind_component"].fillna(0.0)
    merged["average_grade"] = merged["average_grade"].fillna(0.0)
    merged["temp_c"] = merged["temp_c"].fillna(float(merged["temp_c"].median() if merged["temp_c"].notna().any() else 18.0))
    merged["precipitation_mm"] = merged["precipitation_mm"].fillna(0.0)

    segment_type = merged.get("segment_type", pd.Series("flat", index=merged.index)).fillna("flat")
    segment_dummies = pd.get_dummies(segment_type, prefix="segment_type")
    for label in ("flat", "ascent", "descent", "sprint"):
        col = f"segment_type_{label}"
        if col not in segment_dummies.columns:
            segment_dummies[col] = 0

    merged = pd.concat([merged, segment_dummies], axis=1)
    merged["is_new_bike"] = merged.get("is_new_bike", 0).fillna(0).astype(int)
    return merged[merged["speed_per_cbrt_watt"].notna()].reset_index(drop=True)


def _dr_pseudo_outcomes(
    y: np.ndarray,
    t: np.ndarray,
    X: np.ndarray,
    prop_model: LogisticRegression,
    out_model_0: GradientBoostingRegressor,
    out_model_1: GradientBoostingRegressor,
) -> np.ndarray:
    """Compute doubly-robust pseudo-outcomes for each unit.

    tau_i = (mu1 - mu0) + T*(Y - mu1)/e - (1-T)*(Y - mu0)/(1-e)
    """
    e = np.clip(prop_model.predict_proba(X)[:, 1], 0.01, 0.99)
    mu0 = out_model_0.predict(X)
    mu1 = out_model_1.predict(X)
    return (mu1 - mu0) + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)


def estimate_treatment_effect(df: pd.DataFrame) -> dict[str, float | int]:
    """Estimate average treatment effect (new bike vs old bike) via doubly-robust estimation.

    Implements the DR-Learner manually using sklearn:
    1. Fit propensity model P(T=1|X) with LogisticRegression.
    2. Fit separate outcome models E[Y|T=0,X] and E[Y|T=1,X] with GBR.
    3. Compute DR pseudo-outcomes per unit.
    4. Fit a final LinearRegression on pseudo-outcomes to get conditional effects.

    Args:
        df: Feature matrix produced by :func:`build_feature_matrix`.

    Returns:
        Dictionary with ATE summary and confidence interval bounds.
    """
    x_cols = _covariate_columns(df)
    X = df[x_cols].astype(float).to_numpy()
    y = df["speed_per_cbrt_watt"].astype(float).to_numpy()
    t = df["is_new_bike"].astype(int).to_numpy()

    prop_model = LogisticRegression(max_iter=1000)
    prop_model.fit(X, t)

    mask0, mask1 = t == 0, t == 1
    out_model_0 = GradientBoostingRegressor(random_state=42)
    out_model_1 = GradientBoostingRegressor(random_state=42)
    out_model_0.fit(X[mask0], y[mask0])
    out_model_1.fit(X[mask1], y[mask1])

    pseudo = _dr_pseudo_outcomes(y, t, X, prop_model, out_model_0, out_model_1)

    final = LinearRegression()
    final.fit(X, pseudo)
    effects = final.predict(X)

    ate = float(np.mean(effects))
    se = float(np.std(effects, ddof=1) / np.sqrt(len(effects))) if len(effects) > 1 else 0.0
    margin = 1.96 * se

    return {
        "ate": ate,
        "ate_lower": ate - margin,
        "ate_upper": ate + margin,
        "n_treated": int(mask1.sum()),
        "n_control": int(mask0.sum()),
    }


def estimate_heterogeneous_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate terrain-specific treatment effects via causal forest (T-Learner).

    Fits separate RandomForestRegressors on control and treated units; the
    individual treatment effect for each observation is the difference in
    their predicted potential outcomes: tau(x) = mu_1(x) - mu_0(x).

    Args:
        df: Feature matrix produced by :func:`build_feature_matrix`.

    Returns:
        DataFrame with one row per segment type and ATE confidence bounds.
    """
    x_cols = _covariate_columns(df)
    X = df[x_cols].astype(float).to_numpy()
    y = df["speed_per_cbrt_watt"].astype(float).to_numpy()
    t = df["is_new_bike"].astype(int).to_numpy()

    mask0, mask1 = t == 0, t == 1
    rf0 = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=42)
    rf1 = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=42)
    rf0.fit(X[mask0], y[mask0])
    rf1.fit(X[mask1], y[mask1])

    all_effects = rf1.predict(X) - rf0.predict(X)

    rows: list[dict[str, float | str]] = []
    for seg_type in ("flat", "ascent", "descent", "sprint"):
        mask = df.get(f"segment_type_{seg_type}", pd.Series(0, index=df.index)).astype(int).to_numpy() == 1
        if not mask.any():
            continue

        seg_effects = all_effects[mask]
        ate = float(seg_effects.mean())
        se = float(seg_effects.std(ddof=1) / np.sqrt(len(seg_effects))) if len(seg_effects) > 1 else 0.0
        margin = 1.96 * se
        rows.append(
            {
                "segment_type": seg_type,
                "ate": ate,
                "ate_lower": ate - margin,
                "ate_upper": ate + margin,
            }
        )

    return pd.DataFrame(rows)


def get_shap_importances(df: pd.DataFrame) -> pd.DataFrame:
    """Compute feature importance rankings with SHAP values.

    Args:
        df: Feature matrix produced by :func:`build_feature_matrix`.

    Returns:
        DataFrame with columns ``feature`` and ``mean_abs_shap``.
    """
    import shap

    x_cols = _covariate_columns(df)
    X = df[x_cols].astype(float)
    y = df["speed_per_cbrt_watt"].astype(float)

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    values = np.asarray(shap_values, dtype=float)

    return (
        pd.DataFrame(
            {
                "feature": x_cols,
                "mean_abs_shap": np.abs(values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
