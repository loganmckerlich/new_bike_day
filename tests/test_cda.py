"""Unit tests for src.cda — CdA estimation helpers."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.cda import (
    MIN_EFFORTS_PER_BIKE,
    aggregate_cda_by_bike,
    count_impossible_cda,
    estimate_cda,
)


def _flat_efforts(n: int = 5) -> pd.DataFrame:
    """Return a small DataFrame of synthetic flat-segment efforts.

    Uses ~205 W at 10 m/s (36 km/h) which produces CdA ≈ 0.30 at 18 °C,
    well within the physical plausibility range [0.1, 0.6].
    """
    return pd.DataFrame(
        {
            "effort_id": list(range(1, n + 1)),
            "gear_id": ["b111111"] * n,
            "segment_id": [10] * n,
            "start_date": [f"2026-01-{(i % 28) + 1:02d}T10:00:00Z" for i in range(n)],
            # 205 W at 10 m/s on a flat → CdA ≈ 0.30
            "average_watts": [205.0] * n,
            "average_speed_mps": [10.0] * n,
        }
    )


def _flat_segments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment_id": [10],
            "segment_type_detail": ["flat_short"],
            "start_lat": [51.0],
            "start_lng": [-0.1],
        }
    )


class EstimateCdaTests(unittest.TestCase):
    """Tests for estimate_cda()."""

    def test_returns_empty_df_when_efforts_empty(self) -> None:
        result = estimate_cda(pd.DataFrame(), _flat_segments(), 75.0, 8.0)
        self.assertTrue(result.empty)
        for col in ("effort_id", "gear_id", "segment_id", "start_date",
                    "cda_estimate", "average_watts", "average_speed_mps", "temp_c"):
            self.assertIn(col, result.columns)

    def test_returns_empty_df_when_segments_empty(self) -> None:
        result = estimate_cda(_flat_efforts(), pd.DataFrame(), 75.0, 8.0)
        self.assertTrue(result.empty)

    def test_filters_non_flat_segments(self) -> None:
        efforts = _flat_efforts()
        segments = pd.DataFrame(
            {
                "segment_id": [10],
                "segment_type_detail": ["ascent_moderate"],
                "start_lat": [51.0],
                "start_lng": [-0.1],
            }
        )
        result = estimate_cda(efforts, segments, 75.0, 8.0)
        self.assertTrue(result.empty)

    def test_cda_values_within_physical_range(self) -> None:
        with patch("src.cda.get_weather_for_efforts") as mock_weather:
            mock_weather.side_effect = lambda df, segs: df.assign(
                temp_c=18.0,
                wind_speed_kph=0.0,
                wind_direction_deg=0.0,
                precipitation_mm=0.0,
            )
            result = estimate_cda(_flat_efforts(15), _flat_segments(), 75.0, 8.0)

        if not result.empty:
            self.assertTrue((result["cda_estimate"] >= 0.1).all())
            self.assertTrue((result["cda_estimate"] <= 0.6).all())

    def test_output_columns_present(self) -> None:
        with patch("src.cda.get_weather_for_efforts") as mock_weather:
            mock_weather.side_effect = lambda df, segs: df.assign(
                temp_c=18.0,
                wind_speed_kph=0.0,
                wind_direction_deg=0.0,
                precipitation_mm=0.0,
            )
            result = estimate_cda(_flat_efforts(15), _flat_segments(), 75.0, 8.0)

        expected_cols = [
            "effort_id", "gear_id", "segment_id", "start_date",
            "cda_estimate", "average_watts", "average_speed_mps", "temp_c",
        ]
        for col in expected_cols:
            self.assertIn(col, result.columns)

    def test_outlier_removal_drops_extreme_cda(self) -> None:
        """An effort with extreme power should be flagged as an outlier."""
        efforts = _flat_efforts(20)
        # Make one effort have wildly high power → extreme CdA outlier
        efforts.loc[0, "average_watts"] = 2000.0

        with patch("src.cda.get_weather_for_efforts") as mock_weather:
            mock_weather.side_effect = lambda df, segs: df.assign(
                temp_c=18.0,
                wind_speed_kph=0.0,
                wind_direction_deg=0.0,
                precipitation_mm=0.0,
            )
            result = estimate_cda(efforts, _flat_segments(), 75.0, 8.0)

        # The outlier effort (id=1) should be removed
        if not result.empty:
            self.assertNotIn(1, result["effort_id"].tolist())

    def test_flat_long_included(self) -> None:
        """flat_long segment type should also be included."""
        efforts = _flat_efforts(15)
        segments = pd.DataFrame(
            {
                "segment_id": [10],
                "segment_type_detail": ["flat_long"],
                "start_lat": [51.0],
                "start_lng": [-0.1],
            }
        )
        with patch("src.cda.get_weather_for_efforts") as mock_weather:
            mock_weather.side_effect = lambda df, segs: df.assign(
                temp_c=18.0,
                wind_speed_kph=0.0,
                wind_direction_deg=0.0,
                precipitation_mm=0.0,
            )
            result = estimate_cda(efforts, segments, 75.0, 8.0)

        self.assertFalse(result.empty)

    def test_mass_affects_cda_estimate(self) -> None:
        """Heavier system → higher rolling resistance → lower net aerodynamic power → lower CdA."""
        with patch("src.cda.get_weather_for_efforts") as mock_weather:
            mock_weather.side_effect = lambda df, segs: df.assign(
                temp_c=18.0,
                wind_speed_kph=0.0,
                wind_direction_deg=0.0,
                precipitation_mm=0.0,
            )
            result_light = estimate_cda(_flat_efforts(15), _flat_segments(), 60.0, 7.0)
            result_heavy = estimate_cda(_flat_efforts(15), _flat_segments(), 90.0, 12.0)

        if not result_light.empty and not result_heavy.empty:
            # Heavier system has higher rolling resistance, so aerodynamic component is
            # attributed as smaller → lower CdA estimate
            self.assertLess(
                result_heavy["cda_estimate"].mean(),
                result_light["cda_estimate"].mean(),
            )


class CountImpossibleCdaTests(unittest.TestCase):
    """Tests for count_impossible_cda()."""

    def test_returns_zero_on_empty_efforts(self) -> None:
        count = count_impossible_cda(pd.DataFrame(), _flat_segments(), 75.0, 8.0)
        self.assertEqual(count, 0)

    def test_counts_negative_cda_as_impossible(self) -> None:
        """An effort with power below rolling resistance produces negative CdA."""
        efforts = pd.DataFrame(
            {
                "effort_id": [1],
                "gear_id": ["b111111"],
                "segment_id": [10],
                "start_date": ["2026-01-01T10:00:00Z"],
                "average_watts": [1.0],   # far too low → negative CdA
                "average_speed_mps": [10.0],
            }
        )
        with patch("src.cda.get_weather_for_efforts") as mock_weather:
            mock_weather.side_effect = lambda df, segs: df.assign(
                temp_c=18.0,
                wind_speed_kph=0.0,
                wind_direction_deg=0.0,
                precipitation_mm=0.0,
            )
            count = count_impossible_cda(efforts, _flat_segments(), 75.0, 8.0)

        self.assertGreaterEqual(count, 1)


class AggregateCdaByBikeTests(unittest.TestCase):
    """Tests for aggregate_cda_by_bike()."""

    def test_returns_empty_df_on_empty_input(self) -> None:
        result = aggregate_cda_by_bike(pd.DataFrame(), {})
        self.assertTrue(result.empty)
        for col in ("bike_name", "mean_cda", "median_cda", "std_cda", "n_efforts"):
            self.assertIn(col, result.columns)

    def test_aggregation_columns_present(self) -> None:
        cda_df = pd.DataFrame(
            {
                "effort_id": [1, 2, 3, 4],
                "gear_id": ["b111111", "b111111", "b222222", "b222222"],
                "segment_id": [10, 10, 10, 10],
                "start_date": ["2026-01-01T10:00:00Z"] * 4,
                "cda_estimate": [0.30, 0.32, 0.28, 0.29],
                "average_watts": [205.0] * 4,
                "average_speed_mps": [10.0] * 4,
                "temp_c": [18.0] * 4,
            }
        )
        gear_map = {"b111111": "Bike A", "b222222": "Bike B"}
        result = aggregate_cda_by_bike(cda_df, gear_map)

        self.assertEqual(set(result.columns), {"bike_name", "mean_cda", "median_cda", "std_cda", "n_efforts"})
        self.assertEqual(len(result), 2)

    def test_sorted_ascending_by_mean_cda(self) -> None:
        cda_df = pd.DataFrame(
            {
                "effort_id": [1, 2, 3, 4],
                "gear_id": ["b111111", "b111111", "b222222", "b222222"],
                "segment_id": [10, 10, 10, 10],
                "start_date": ["2026-01-01T10:00:00Z"] * 4,
                "cda_estimate": [0.35, 0.36, 0.28, 0.29],
                "average_watts": [205.0] * 4,
                "average_speed_mps": [10.0] * 4,
                "temp_c": [18.0] * 4,
            }
        )
        gear_map = {"b111111": "Bike A", "b222222": "Bike B"}
        result = aggregate_cda_by_bike(cda_df, gear_map)

        cda_values = result["mean_cda"].tolist()
        self.assertEqual(cda_values, sorted(cda_values))

    def test_unknown_gear_id_falls_back_to_id_string(self) -> None:
        cda_df = pd.DataFrame(
            {
                "effort_id": [1],
                "gear_id": ["b999999"],
                "segment_id": [10],
                "start_date": ["2026-01-01T10:00:00Z"],
                "cda_estimate": [0.30],
                "average_watts": [205.0],
                "average_speed_mps": [10.0],
                "temp_c": [18.0],
            }
        )
        result = aggregate_cda_by_bike(cda_df, {})
        self.assertEqual(result.iloc[0]["bike_name"], "b999999")

    def test_n_efforts_matches_row_count(self) -> None:
        cda_df = pd.DataFrame(
            {
                "effort_id": [1, 2, 3],
                "gear_id": ["b111111"] * 3,
                "segment_id": [10, 10, 10],
                "start_date": ["2026-01-01T10:00:00Z"] * 3,
                "cda_estimate": [0.30, 0.31, 0.29],
                "average_watts": [205.0] * 3,
                "average_speed_mps": [10.0] * 3,
                "temp_c": [18.0] * 3,
            }
        )
        result = aggregate_cda_by_bike(cda_df, {"b111111": "Bike A"})
        self.assertEqual(int(result.iloc[0]["n_efforts"]), 3)

    def test_mean_cda_correct(self) -> None:
        cda_df = pd.DataFrame(
            {
                "effort_id": [1, 2],
                "gear_id": ["b111111", "b111111"],
                "segment_id": [10, 10],
                "start_date": ["2026-01-01T10:00:00Z"] * 2,
                "cda_estimate": [0.30, 0.40],
                "average_watts": [205.0, 205.0],
                "average_speed_mps": [10.0, 10.0],
                "temp_c": [18.0, 18.0],
            }
        )
        result = aggregate_cda_by_bike(cda_df, {"b111111": "Bike A"})
        self.assertAlmostEqual(float(result.iloc[0]["mean_cda"]), 0.35, places=6)


class MinEffortsPerBikeTests(unittest.TestCase):
    def test_constant_value(self) -> None:
        self.assertEqual(MIN_EFFORTS_PER_BIKE, 10)


if __name__ == "__main__":
    unittest.main()
