"""Unit tests for weather enrichment helpers."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src.weather import get_weather, get_weather_for_efforts


class WeatherHelpersTests(unittest.TestCase):
    def test_get_weather_stub_returns_expected_shape(self) -> None:
        result = get_weather(51.0, -0.1, "2026-05-20")
        self.assertEqual(
            set(result.keys()),
            {"temp_c", "wind_speed_kph", "wind_direction_deg", "precipitation_mm"},
        )

    def test_get_weather_for_efforts_calls_unique_segment_date_pairs(self) -> None:
        efforts = pd.DataFrame(
            {
                "effort_id": [1, 2, 3],
                "segment_id": [10, 10, 11],
                "start_date": [
                    "2026-01-01T10:00:00Z",
                    "2026-01-01T11:00:00Z",
                    "2026-01-02T11:00:00Z",
                ],
            }
        )
        segments = pd.DataFrame(
            {
                "segment_id": [10, 11],
                "start_lat": [51.1, 51.2],
                "start_lng": [-0.1, -0.2],
            }
        )

        with patch("src.weather.get_weather") as weather_mock:
            weather_mock.return_value = {
                "temp_c": 18.0,
                "wind_speed_kph": 15.0,
                "wind_direction_deg": 180.0,
                "precipitation_mm": 0.0,
            }
            out = get_weather_for_efforts(efforts, segments)

        self.assertEqual(weather_mock.call_count, 2)
        for col in ("temp_c", "wind_speed_kph", "wind_direction_deg", "precipitation_mm"):
            self.assertIn(col, out.columns)
            self.assertTrue(out[col].notna().all())


if __name__ == "__main__":
    unittest.main()
