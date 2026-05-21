"""Unit tests for src.analytics helpers."""

from __future__ import annotations

import unittest

import pandas as pd

from src.analytics import apply_min_watts_filter, mean_profile_by_segment_type, power_normalized_profile


class PowerNormalizedProfileTests(unittest.TestCase):
    def test_apply_min_watts_filter_is_public_and_descent_aware(self) -> None:
        efforts = pd.DataFrame(
            {
                "segment_id": [1, 2, 3],
                "average_watts": [80, 120, 90],
                "segment_type": ["descent", "flat", "ascent"],
            }
        )

        out = apply_min_watts_filter(efforts, min_watts=100, descents_exempt=True)

        self.assertEqual(out["segment_id"].tolist(), [1, 2])

    def test_supports_custom_segment_type_column(self) -> None:
        efforts = pd.DataFrame(
            {
                "segment_id": [1, 1, 2, 2],
                "bike_name": ["Bike A", "Bike B", "Bike A", "Bike B"],
                "segment_type": ["flat", "flat", "ascent", "ascent"],
                "segment_type_detail": ["flat_short", "flat_short", "ascent_steep", "ascent_steep"],
                "speed_per_cbrt_watt": [0.10, 0.11, 0.08, 0.09],
            }
        )

        profile = power_normalized_profile(
            efforts,
            bikes=["Bike A", "Bike B"],
            segment_types=["flat_short", "ascent_steep"],
            valid_segment_ids=[1, 2],
            segment_type_col="segment_type_detail",
        )

        self.assertEqual(profile["Bike A"], [0.10, 0.08])
        self.assertEqual(profile["Bike B"], [0.11, 0.09])

    def test_mean_profile_fills_missing_types_in_fixed_order(self) -> None:
        efforts = pd.DataFrame(
            {
                "segment_id": [1, 1, 2, 2],
                "bike_name": ["Bike A", "Bike B", "Bike A", "Bike B"],
                "segment_type_detail": ["flat_short", "flat_short", "ascent_steep", "ascent_steep"],
                "speed_kmh": [30.0, 31.0, 20.0, 21.0],
            }
        )

        profile = mean_profile_by_segment_type(
            efforts,
            bikes=["Bike A", "Bike B"],
            segment_types=["flat_short", "flat_long", "ascent_steep"],
            valid_segment_ids=[1, 2],
            value_col="speed_kmh",
            segment_type_col="segment_type_detail",
        )

        self.assertEqual(profile["Bike A"], [30.0, 0.0, 20.0])
        self.assertEqual(profile["Bike B"], [31.0, 0.0, 21.0])


if __name__ == "__main__":
    unittest.main()
