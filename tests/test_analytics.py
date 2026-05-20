"""Unit tests for src.analytics helpers."""

from __future__ import annotations

import unittest

import pandas as pd

from src.analytics import power_normalized_profile


class PowerNormalizedProfileTests(unittest.TestCase):
    def test_supports_custom_segment_type_column(self) -> None:
        efforts = pd.DataFrame(
            {
                "segment_id": [1, 1, 2, 2],
                "bike_name": ["Bike A", "Bike B", "Bike A", "Bike B"],
                "segment_type": ["flat", "flat", "ascent", "ascent"],
                "segment_type_detail": ["flat_short", "flat_short", "ascent_steep", "ascent_steep"],
                "speed_per_watt": [0.10, 0.11, 0.08, 0.09],
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


if __name__ == "__main__":
    unittest.main()
