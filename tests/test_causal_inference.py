"""Unit tests for causal feature engineering."""

from __future__ import annotations

import unittest

import pandas as pd

from src.causal_inference import build_feature_matrix, remove_outliers_for_causal_analysis


class BuildFeatureMatrixTests(unittest.TestCase):
    def test_build_feature_matrix_filters_low_watts_and_adds_covariates(self) -> None:
        efforts = pd.DataFrame(
            {
                "effort_id": [1, 2],
                "segment_id": [100, 100],
                "start_date": ["2026-01-01T10:00:00Z", "2026-01-02T10:00:00Z"],
                "moving_time": [100, 100],
                "average_watts": [200.0, 40.0],
                "gear_id": ["old", "new"],
                "is_new_bike": [0, 1],
            }
        )
        segments = pd.DataFrame(
            {
                "segment_id": [100],
                "distance": [1000.0],
                "average_grade": [3.0],
                "segment_type": ["ascent"],
                "start_lat": [51.0],
                "start_lng": [-0.1],
                "end_lat": [51.01],
                "end_lng": [-0.09],
            }
        )

        out = build_feature_matrix(efforts, segments)

        self.assertEqual(len(out), 1)
        self.assertIn("speed_per_watt", out.columns)
        self.assertIn("straightness_index", out.columns)
        self.assertIn("headwind_component", out.columns)
        self.assertIn("precipitation_mm", out.columns)
        self.assertIn("temp_c", out.columns)
        self.assertIn("segment_type_ascent", out.columns)
        self.assertEqual(int(out.iloc[0]["is_new_bike"]), 0)

    def test_remove_outliers_filters_by_speed_per_watt_zscore(self) -> None:
        efforts = pd.DataFrame(
            {
                "effort_id": [1, 2, 3, 4, 5],
                "segment_id": [10, 10, 10, 10, 10],
                "bike_name": ["Bike A"] * 5,
                "speed_kmh": [30.0, 31.0, 29.0, 30.0, 60.0],
                "average_watts": [200.0, 200.0, 200.0, 200.0, 200.0],
                "is_new_bike": [0, 0, 0, 0, 0],
            }
        )

        cleaned, n_outliers = remove_outliers_for_causal_analysis(efforts, z_threshold=1.5)

        self.assertEqual(n_outliers, 1)
        self.assertEqual(len(cleaned), 4)
        self.assertNotIn("is_outlier", cleaned.columns)
        self.assertNotIn("z_score", cleaned.columns)

    def test_remove_outliers_returns_input_when_columns_missing(self) -> None:
        efforts = pd.DataFrame({"segment_id": [1], "average_watts": [200.0]})
        cleaned, n_outliers = remove_outliers_for_causal_analysis(efforts)
        self.assertEqual(n_outliers, 0)
        pd.testing.assert_frame_equal(cleaned, efforts)


if __name__ == "__main__":
    unittest.main()
