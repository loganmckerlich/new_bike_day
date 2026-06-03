"""Power-normalised bike speed analytics.

Core question: which bike is faster, controlling for rider effort?

Methodology
-----------
1.  ``compute_speed_per_watt``  –  derive speed / power^(1/3) for every effort.
    In aerodynamics, speed scales as the cube root of power (v ∝ P^(1/3)),
    so this ratio is approximately constant for a given bike and conditions.
2.  ``filter_outliers_by_power_speed``  –  per-segment z-score on
    speed_per_cbrt_watt removes efforts where speed doesn't match power
    (drafting, strong wind, …).
3.  ``power_normalized_profile``  –  per segment-type × bike mean
    speed_per_cbrt_watt used to populate the efficiency spider chart.
4.  ``outlier_detection_frames``  –  helper that returns the raw, annotated, and
    filtered DataFrames needed by the step-by-step visual explainer.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "compute_speed_per_watt",
    "filter_outliers_by_power_speed",
    "power_normalized_profile",
    "mean_profile_by_segment_type",
    "outlier_detection_frames",
    "apply_min_watts_filter",
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_speed_per_watt(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``speed_per_cbrt_watt`` column to *df* and return a copy.

    ``speed_per_cbrt_watt = speed_kmh / power^(1/3)``

    In aerodynamics, speed scales as the cube root of power (v ∝ P^(1/3)),
    so this ratio is approximately constant for a given bike and conditions.
    It is a more physically correct efficiency metric than the linear speed/W.

    Rows where either value is missing or zero watts are left as NaN.
    """
    out = df.copy()
    safe_watts = out["average_watts"].replace(0, np.nan)
    out["speed_per_cbrt_watt"] = out["speed_kmh"] / np.cbrt(safe_watts)
    return out


def filter_outliers_by_power_speed(
    df: pd.DataFrame,
    z_threshold: float = 2.0,
    min_efforts: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag and remove efforts whose speed/power^(1/3) ratio is anomalous.

    For each non-descent segment independently:
    -   Compute z-score of ``speed_per_cbrt_watt``.
    -   Mark any effort with ``|z| > z_threshold`` as an outlier.
    -   Segments with fewer than *min_efforts* valid rows are left unfiltered.
    -   Descent efforts are never outlier-filtered.

    Parameters
    ----------
    df:
        DataFrame that already has a ``speed_per_cbrt_watt`` column
        (produced by :func:`compute_speed_per_watt`).
    z_threshold:
        Number of standard deviations from the segment mean beyond which
        an effort is considered anomalous.
    min_efforts:
        Minimum number of valid-power efforts in a segment before outlier
        detection is applied.  Segments below this threshold keep all rows.

    Returns
    -------
    filtered : pd.DataFrame
        Copy of *df* with outliers removed and an ``is_outlier`` bool column
        added for reference.
    annotated : pd.DataFrame
        Full copy of *df* with ``is_outlier`` and ``z_score`` columns added
        (useful for the explainer visualisation).
    """
    out = df.copy()
    out["is_outlier"] = False
    out["z_score"] = np.nan

    valid = out["speed_per_cbrt_watt"].notna() & out["average_watts"].notna()
    if "segment_type" in out.columns:
        valid &= out["segment_type"] != "descent"

    group_cols = ["segment_id", "bike_name"]
    for _, grp_idx in out[valid].groupby(group_cols).groups.items():
        grp = out.loc[grp_idx, "speed_per_cbrt_watt"]
        if len(grp) < min_efforts:
            continue
        mean_spw = grp.mean()
        std_spw = grp.std(ddof=1)
        if std_spw == 0 or np.isnan(std_spw):
            continue
        z = (grp - mean_spw) / std_spw
        out.loc[grp_idx, "z_score"] = z.values
        out.loc[grp_idx[np.abs(z.values) > z_threshold], "is_outlier"] = True

    annotated = out.copy()
    filtered = out[~out["is_outlier"]].copy()
    return filtered, annotated


def power_normalized_profile(
    efforts: pd.DataFrame,
    bikes: Sequence[str],
    segment_types: Sequence[str],
    valid_segment_ids: Sequence[int],
    segment_type_col: str = "segment_type",
) -> dict[str, list[float]]:
    """Compute per-type mean speed/power^(1/3) for the efficiency spider chart.

    Parameters
    ----------
    efforts:
        Filtered efforts (outliers removed) with ``speed_per_cbrt_watt``,
        ``bike_name``, segment type column, and ``segment_id`` columns.
    bikes:
        Ordered list of bike names to include.
    segment_types:
        Ordered list of segment type labels (e.g. ``["sprint", "flat", …]``).
    valid_segment_ids:
        Segment IDs where all selected bikes meet the minimum sample size.
    segment_type_col:
        Column used to group segments (``segment_type`` or ``segment_type_detail``).

    Returns
    -------
    dict mapping bike_name → list of mean speed_per_cbrt_watt values, one per
    segment type.  Missing types get 0.0.
    """
    return mean_profile_by_segment_type(
        efforts,
        bikes,
        segment_types,
        valid_segment_ids,
        value_col="speed_per_cbrt_watt",
        segment_type_col=segment_type_col,
    )


def mean_profile_by_segment_type(
    efforts: pd.DataFrame,
    bikes: Sequence[str],
    segment_types: Sequence[str],
    valid_segment_ids: Sequence[int],
    value_col: str,
    segment_type_col: str = "segment_type",
) -> dict[str, list[float]]:
    """Compute a fixed-order bike profile across segment types/subtypes.

    For each bike, values are aggregated in two steps:
    1. mean per segment_id within each segment type/subtype
    2. mean of those per-segment means per segment type/subtype

    A template DataFrame with all provided ``segment_types`` is left-joined
    with each bike's aggregated data so missing categories are filled with 0.
    """
    scope = efforts[efforts["segment_id"].isin(valid_segment_ids)].copy()
    template = pd.DataFrame({segment_type_col: list(segment_types)})
    profile: dict[str, list[float]] = {}

    for bike in bikes:
        bike_eff = scope[(scope["bike_name"] == bike) & scope[value_col].notna()].copy()
        if bike_eff.empty:
            profile[bike] = [0.0] * len(segment_types)
            continue

        per_seg = (
            bike_eff.groupby(["segment_id", segment_type_col], as_index=False)[value_col]
            .mean()
        )
        per_type = per_seg.groupby(segment_type_col, as_index=False)[value_col].mean()

        complete = template.merge(per_type, on=segment_type_col, how="left")
        complete[value_col] = complete[value_col].fillna(0.0).astype(float)
        profile[bike] = complete[value_col].tolist()

    return profile


def outlier_detection_frames(
    efforts: pd.DataFrame,
    segment_id: int,
    z_threshold: float = 2.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return three DataFrames used by the step-by-step visual explainer.

    Parameters
    ----------
    efforts:
        Full ``watt_efforts`` DataFrame (with ``speed_per_cbrt_watt`` already
        computed).
    segment_id:
        The segment to illustrate.
    z_threshold:
        Outlier threshold passed to :func:`filter_outliers_by_power_speed`.

    Returns
    -------
    raw : pd.DataFrame
        All efforts for the segment (no outlier column).
    annotated : pd.DataFrame
        All efforts with ``is_outlier`` and ``z_score`` columns.
    filtered : pd.DataFrame
        Efforts after outlier removal.
    """
    seg_efforts = efforts[efforts["segment_id"] == segment_id].copy()
    if "speed_per_cbrt_watt" not in seg_efforts.columns:
        seg_efforts = compute_speed_per_watt(seg_efforts)
    filtered, annotated = filter_outliers_by_power_speed(seg_efforts, z_threshold=z_threshold)
    raw = seg_efforts.copy()
    return raw, annotated, filtered


def apply_min_watts_filter(
    df: pd.DataFrame,
    min_watts: int,
    *,
    descents_exempt: bool = False,
    segments_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Filter efforts below a minimum power threshold.

    Parameters
    ----------
    df:
        Efforts DataFrame. Must have an ``average_watts`` column.  If
        ``descents_exempt`` is True, the DataFrame should also have a
        ``segment_type`` column (or it can be joined from *segments_df*).
    min_watts:
        Minimum average watts required to keep an effort.  Pass ``0`` (or a
        negative value) to skip filtering entirely.
    descents_exempt:
        When True, efforts on descent segments bypass the watts threshold.
    segments_df:
        Optional segments DataFrame used to join ``segment_type`` onto *df*
        when the column is not already present.

    Returns
    -------
    pd.DataFrame
        Filtered copy of *df*.
    """
    if min_watts <= 0 or "average_watts" not in df.columns:
        return df.copy()

    if descents_exempt:
        # Use a separate variable for the potentially-merged frame so it is
        # clear when we are working on a copy versus the original df.
        type_frame = df
        if "segment_type" not in type_frame.columns and segments_df is not None and not segments_df.empty:
            _seg_types = segments_df[["segment_id", "segment_type"]].drop_duplicates("segment_id")
            # Merge produces a new aligned frame; original df is unchanged.
            type_frame = df.merge(_seg_types, on="segment_id", how="left")
        if "segment_type" in type_frame.columns:
            is_descent = type_frame["segment_type"] == "descent"
            mask = (type_frame["average_watts"] >= min_watts) | is_descent
            # When type_frame diverges from df (merge added columns) use
            # mask.values for positional alignment back onto df.
            return df[mask.values].copy() if type_frame is not df else df[mask].copy()

    return df[df["average_watts"] >= min_watts].copy()

# Force module reload for Streamlit Cloud cache
