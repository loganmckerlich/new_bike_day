"""CdA (drag coefficient × frontal area) estimation from flat segment efforts.

Physics background
------------------
The cycling power equation on flat ground:

    P = 0.5 × CdA × ρ × v³ + Crr × m × g × v

Rearranging to solve for CdA:

    CdA = (P - Crr × m × g × v) / (0.5 × ρ × v³)

Where:
    P   = power in watts (average_watts)
    v   = speed in m/s (average_speed_mps)
    ρ   = air density in kg/m³, derived from temperature:
          ρ ≈ 1.225 × (273.15 / (273.15 + temp_c))
    Crr = rolling resistance coefficient (0.004)
    m   = total system mass in kg (rider + bike)
    g   = 9.81 m/s²

CdA estimates are only valid on flat segments (flat_short and flat_long).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.weather import get_weather_for_efforts

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_G: float = 9.81          # gravitational acceleration (m/s²)
_CRR: float = 0.004       # rolling resistance coefficient
_RHO_STD: float = 1.225   # standard air density at 0 °C (kg/m³)
_T0: float = 273.15        # 0 °C in Kelvin

# CdA physical plausibility bounds
_CDA_MIN: float = 0.1
_CDA_MAX: float = 0.6

# Flat segment type filter
_FLAT_TYPES: frozenset[str] = frozenset({"flat_short", "flat_long"})

# Minimum efforts required per bike to report an estimate
MIN_EFFORTS_PER_BIKE: int = 10


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_cda(
    efforts: pd.DataFrame,
    segments: pd.DataFrame,
    rider_mass_kg: float,
    bike_mass_kg: float,
) -> pd.DataFrame:
    """Estimate CdA per effort from flat segment data.

    Args:
        efforts: Efforts DataFrame with at least ``effort_id``, ``gear_id``,
            ``segment_id``, ``start_date``, ``average_watts``,
            ``average_speed_mps`` columns.
        segments: Segment metadata DataFrame with at least ``segment_id`` and
            ``segment_type_detail`` columns.
        rider_mass_kg: Rider mass in kilograms.
        bike_mass_kg: Bike mass in kilograms.

    Returns:
        DataFrame with columns:
        ``effort_id``, ``gear_id``, ``segment_id``, ``start_date``,
        ``cda_estimate``, ``average_watts``, ``average_speed_mps``, ``temp_c``.
        Rows with physically impossible CdA values (outside 0.1–0.6) and
        statistical outliers (outside mean ± 2 std) are removed.
    """
    if efforts.empty or segments.empty:
        return pd.DataFrame(
            columns=[
                "effort_id",
                "gear_id",
                "segment_id",
                "start_date",
                "cda_estimate",
                "average_watts",
                "average_speed_mps",
                "temp_c",
            ]
        )

    total_mass_kg = rider_mass_kg + bike_mass_kg

    # Merge efforts with segment metadata to get segment_type_detail
    seg_cols = [c for c in ("segment_id", "segment_type_detail") if c in segments.columns]
    merged = efforts.merge(segments[seg_cols], on="segment_id", how="left")

    # Filter to flat segments only
    flat = merged[merged["segment_type_detail"].isin(_FLAT_TYPES)].copy()
    if flat.empty:
        return pd.DataFrame(
            columns=[
                "effort_id",
                "gear_id",
                "segment_id",
                "start_date",
                "cda_estimate",
                "average_watts",
                "average_speed_mps",
                "temp_c",
            ]
        )

    # Enrich with weather for air density
    flat = get_weather_for_efforts(flat, segments)

    # Fill missing temp with default (18 °C) — weather is currently stubbed
    if "temp_c" not in flat.columns:
        flat["temp_c"] = 18.0
    else:
        flat["temp_c"] = flat["temp_c"].fillna(18.0)

    # Compute air density from temperature
    flat["rho"] = _RHO_STD * (_T0 / (_T0 + flat["temp_c"]))

    # Ensure required columns exist and are numeric
    for col in ("average_watts", "average_speed_mps"):
        flat[col] = pd.to_numeric(flat[col], errors="coerce")

    # Drop rows missing power or speed
    flat = flat.dropna(subset=["average_watts", "average_speed_mps"])
    flat = flat[(flat["average_watts"] > 0) & (flat["average_speed_mps"] > 0)]

    if flat.empty:
        return pd.DataFrame(
            columns=[
                "effort_id",
                "gear_id",
                "segment_id",
                "start_date",
                "cda_estimate",
                "average_watts",
                "average_speed_mps",
                "temp_c",
            ]
        )

    v = flat["average_speed_mps"]
    p = flat["average_watts"]
    rho = flat["rho"]

    # CdA = (P - Crr × m × g × v) / (0.5 × ρ × v³)
    rolling_power = _CRR * total_mass_kg * _G * v
    drag_denominator = 0.5 * rho * v**3
    flat["cda_estimate"] = (p - rolling_power) / drag_denominator

    result = flat[
        [
            "effort_id",
            "gear_id",
            "segment_id",
            "start_date",
            "cda_estimate",
            "average_watts",
            "average_speed_mps",
            "temp_c",
        ]
    ].copy()

    # Remove statistical outliers: mean ± 2 std across all estimates
    cda = result["cda_estimate"]
    if len(cda) >= 3:
        mean_cda = cda.mean()
        std_cda = cda.std(ddof=1)
        result = result[np.abs(cda - mean_cda) <= 2 * std_cda].copy()

    # Drop physically impossible values
    result = result[
        (result["cda_estimate"] >= _CDA_MIN) & (result["cda_estimate"] <= _CDA_MAX)
    ].copy()

    return result.reset_index(drop=True)


def count_impossible_cda(
    efforts: pd.DataFrame,
    segments: pd.DataFrame,
    rider_mass_kg: float,
    bike_mass_kg: float,
) -> int:
    """Count efforts that would produce physically impossible CdA values.

    Args:
        efforts: Efforts DataFrame (same as passed to :func:`estimate_cda`).
        segments: Segment metadata DataFrame.
        rider_mass_kg: Rider mass in kilograms.
        bike_mass_kg: Bike mass in kilograms.

    Returns:
        Number of efforts whose raw CdA estimate falls outside [0.1, 0.6].
    """
    if efforts.empty or segments.empty:
        return 0

    total_mass_kg = rider_mass_kg + bike_mass_kg

    seg_cols = [c for c in ("segment_id", "segment_type_detail") if c in segments.columns]
    merged = efforts.merge(segments[seg_cols], on="segment_id", how="left")
    flat = merged[merged["segment_type_detail"].isin(_FLAT_TYPES)].copy()
    if flat.empty:
        return 0

    flat = get_weather_for_efforts(flat, segments)
    if "temp_c" not in flat.columns:
        flat["temp_c"] = 18.0
    else:
        flat["temp_c"] = flat["temp_c"].fillna(18.0)

    flat["rho"] = _RHO_STD * (_T0 / (_T0 + flat["temp_c"]))

    for col in ("average_watts", "average_speed_mps"):
        flat[col] = pd.to_numeric(flat[col], errors="coerce")

    flat = flat.dropna(subset=["average_watts", "average_speed_mps"])
    flat = flat[(flat["average_watts"] > 0) & (flat["average_speed_mps"] > 0)]
    if flat.empty:
        return 0

    v = flat["average_speed_mps"]
    p = flat["average_watts"]
    rho = flat["rho"]

    rolling_power = _CRR * total_mass_kg * _G * v
    drag_denominator = 0.5 * rho * v**3
    raw_cda = (p - rolling_power) / drag_denominator

    return int(((raw_cda < _CDA_MIN) | (raw_cda > _CDA_MAX)).sum())


def aggregate_cda_by_bike(
    cda_df: pd.DataFrame,
    gear_map: dict[str, str],
) -> pd.DataFrame:
    """Aggregate per-effort CdA estimates to per-bike statistics.

    Args:
        cda_df: Output of :func:`estimate_cda`.
        gear_map: Mapping of gear_id → bike display name.

    Returns:
        DataFrame with columns:
        ``bike_name``, ``mean_cda``, ``median_cda``, ``std_cda``, ``n_efforts``.
        Sorted ascending by ``mean_cda`` (lower CdA = more aerodynamic).
    """
    if cda_df.empty:
        return pd.DataFrame(
            columns=["bike_name", "mean_cda", "median_cda", "std_cda", "n_efforts"]
        )

    agg = (
        cda_df.groupby("gear_id")["cda_estimate"]
        .agg(
            mean_cda="mean",
            median_cda="median",
            std_cda="std",
            n_efforts="count",
        )
        .reset_index()
    )

    agg["bike_name"] = agg["gear_id"].astype(str).map(
        {str(k): v for k, v in gear_map.items()}
    ).fillna(agg["gear_id"].astype(str))

    agg = agg[["bike_name", "mean_cda", "median_cda", "std_cda", "n_efforts"]].sort_values(
        "mean_cda", ascending=True
    )

    return agg.reset_index(drop=True)
