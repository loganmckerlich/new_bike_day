"""Tests for src.analytics — covering the first-load dtype bug."""

import unittest

import numpy as np
import pandas as pd

from src.analytics import compute_speed_per_watt
from src.database import _EFFORTS_COLS


class ComputeSpeedPerWattTest(unittest.TestCase):
    def _make_efforts(self, dtype="float64") -> pd.DataFrame:
        """Return a minimal efforts DataFrame with the given column dtype."""
        rows = [
            {"average_watts": 200.0, "speed_kmh": 30.0},
            {"average_watts": 150.0, "speed_kmh": 25.0},
            {"average_watts": 0.0, "speed_kmh": 5.0},  # zero watts → NaN
            {"average_watts": None, "speed_kmh": None},  # missing → NaN
        ]
        df = pd.DataFrame(rows)
        if dtype == "object":
            df = df.astype(object)
        return df

    def test_float64_dtype_produces_correct_values(self) -> None:
        df = self._make_efforts("float64")
        result = compute_speed_per_watt(df)
        self.assertIn("speed_per_cbrt_watt", result.columns)
        self.assertAlmostEqual(result["speed_per_cbrt_watt"].iloc[0], 30.0 / np.cbrt(200.0))
        self.assertAlmostEqual(result["speed_per_cbrt_watt"].iloc[1], 25.0 / np.cbrt(150.0))
        self.assertTrue(pd.isna(result["speed_per_cbrt_watt"].iloc[2]))  # zero watts
        self.assertTrue(pd.isna(result["speed_per_cbrt_watt"].iloc[3]))  # missing

    def test_object_dtype_does_not_crash(self) -> None:
        """Simulates first-load: pd.concat of an empty DataFrame with fresh API data
        produces object dtype; compute_speed_per_watt must not raise TypeError."""
    def test_object_dtype_does_not_crash(self) -> None:
        """Simulates first-load: pd.concat of an empty DataFrame with fresh API data
        produces object dtype; compute_speed_per_watt must not raise TypeError."""
        empty = pd.DataFrame(columns=_EFFORTS_COLS)  # all object dtype
        window = pd.DataFrame([
            {
                "athlete_id": None, "effort_id": "1", "segment_id": 123,
                "activity_id": "456", "gear_id": "b1", "start_date": "2024-01-01",
                "elapsed_time": 65, "moving_time": 60,
                "average_watts": 200.0, "average_heartrate": 150.0,
            },
            {
                "athlete_id": None, "effort_id": "2", "segment_id": 123,
                "activity_id": "457", "gear_id": "b1", "start_date": "2024-01-02",
                "elapsed_time": 70, "moving_time": 65,
                "average_watts": 180.0, "average_heartrate": 155.0,
            },
        ])
        df = pd.concat([empty, window], ignore_index=True)
        self.assertEqual(df["average_watts"].dtype, object)  # confirm object dtype after concat

        # speed_kmh is added after the concat in the cleaning page (via _compute_speed_kmh),
        # which uses moving_time (also object dtype), so speed_kmh is object too.
        df["speed_kmh"] = 500.0 / df["moving_time"] * 3.6
        self.assertEqual(df["speed_kmh"].dtype, object)      # confirm object dtype

        result = compute_speed_per_watt(df)

        self.assertFalse(result["speed_per_cbrt_watt"].isna().all())
        self.assertAlmostEqual(
            float(result["speed_per_cbrt_watt"].iloc[0]),
            float(df["speed_kmh"].iloc[0]) / np.cbrt(200.0),
        )
        self.assertAlmostEqual(
            float(result["speed_per_cbrt_watt"].iloc[1]),
            float(df["speed_kmh"].iloc[1]) / np.cbrt(180.0),
        )


if __name__ == "__main__":
    unittest.main()
