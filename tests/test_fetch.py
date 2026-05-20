"""Unit tests for src/fetch.py — activity filtering and gear_id resolution."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.fetch import (
    _BIKE_SPORT_TYPES,
    _classify_segment,
    get_athlete_activities,
    ingest_all,
)


def _mock_response(json_data: object, status_code: int = 200) -> MagicMock:
    """Return a mock requests.Response with the given JSON payload."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data
    mock.raise_for_status = MagicMock()
    return mock


class BikeTypesTests(unittest.TestCase):
    def test_common_types_included(self) -> None:
        for sport_type in ("Ride", "VirtualRide", "MountainBikeRide", "GravelRide", "EBikeRide"):
            self.assertIn(sport_type, _BIKE_SPORT_TYPES, f"{sport_type} should be a bike type")

    def test_non_bike_types_excluded(self) -> None:
        for sport_type in ("Run", "Swim", "Walk", "Hike", "AlpineSki"):
            self.assertNotIn(sport_type, _BIKE_SPORT_TYPES, f"{sport_type} should not be a bike type")


class SegmentClassificationTests(unittest.TestCase):
    def test_sprint_checked_first(self) -> None:
        seg_type, seg_detail = _classify_segment(320.0, 12.0)
        self.assertEqual(seg_type, "sprint")
        self.assertEqual(seg_detail, "sprint_uphill")

    def test_flat_split_short_and_long(self) -> None:
        self.assertEqual(_classify_segment(800.0, 0.0), ("flat", "flat_short"))
        self.assertEqual(_classify_segment(1800.0, 0.0), ("flat", "flat_long"))

    def test_ascent_and_descent_subtypes(self) -> None:
        self.assertEqual(_classify_segment(2000.0, 3.0), ("ascent", "ascent_shallow"))
        self.assertEqual(_classify_segment(2000.0, 7.0), ("ascent", "ascent_moderate"))
        self.assertEqual(_classify_segment(2000.0, 10.0), ("ascent", "ascent_steep"))
        self.assertEqual(_classify_segment(2000.0, -2.0), ("descent", "descent_gentle"))
        self.assertEqual(_classify_segment(2000.0, -6.0), ("descent", "descent_steep"))


class GetAthleteActivitiesTests(unittest.TestCase):
    def _sample_activities(self) -> list[dict]:
        return [
            # Valid: road bike ride with power
            {"id": 1, "sport_type": "Ride", "average_watts": 200.0, "device_watts": True, "gear_id": "b111", "name": "Morning Ride", "start_date": "2024-01-01T08:00:00Z"},
            # Valid: virtual ride with power
            {"id": 2, "sport_type": "VirtualRide", "average_watts": 185.0, "device_watts": False, "gear_id": "b222", "name": "Zwift", "start_date": "2024-01-02T08:00:00Z"},
            # Invalid: run — wrong sport type
            {"id": 3, "sport_type": "Run", "average_watts": 0.0, "device_watts": False, "gear_id": None, "name": "Easy Run", "start_date": "2024-01-03T08:00:00Z"},
            # Invalid: ride but no power
            {"id": 4, "sport_type": "Ride", "average_watts": 0.0, "device_watts": False, "gear_id": "b111", "name": "No Power Ride", "start_date": "2024-01-04T08:00:00Z"},
            # Valid: device_watts true even with average_watts missing
            {"id": 5, "sport_type": "GravelRide", "average_watts": None, "device_watts": True, "gear_id": "b333", "name": "Gravel", "start_date": "2024-01-05T08:00:00Z"},
        ]

    @patch("src.fetch.requests.get")
    def test_filters_to_bike_rides_with_power(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = [
            _mock_response(self._sample_activities()),
            _mock_response([]),  # empty second page → stop
        ]
        result = get_athlete_activities("token")
        # Only IDs 1, 2, 5 should pass
        self.assertEqual(set(result.keys()), {1, 2, 5})

    @patch("src.fetch.requests.get")
    def test_gear_id_is_preserved(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = [
            _mock_response(self._sample_activities()),
            _mock_response([]),
        ]
        result = get_athlete_activities("token")
        self.assertEqual(result[1]["gear_id"], "b111")
        self.assertEqual(result[2]["gear_id"], "b222")
        self.assertEqual(result[5]["gear_id"], "b333")

    @patch("src.fetch.requests.get")
    def test_max_activities_limits_fetch(self, mock_get: MagicMock) -> None:
        page1 = [
            {
                "id": i,
                "sport_type": "Ride",
                "average_watts": 200.0,
                "device_watts": True,
                "gear_id": "b1",
                "name": f"Ride {i}",
                "start_date": "2024-01-01T00:00:00Z",
            }
            for i in range(1, 4)
        ]
        mock_get.return_value = _mock_response(page1)
        # max_activities=3 should fetch exactly one batch of 3 and stop
        result = get_athlete_activities("token", max_activities=3)
        self.assertEqual(len(result), 3)
        # Only one HTTP request should have been made
        self.assertEqual(mock_get.call_count, 1)

    @patch("src.fetch.requests.get")
    def test_falls_back_to_type_field(self, mock_get: MagicMock) -> None:
        """Older Strava API responses use 'type' instead of 'sport_type'."""
        activities = [
            {"id": 10, "type": "Ride", "sport_type": None, "average_watts": 250.0, "device_watts": False, "gear_id": "b999", "name": "Old API Ride", "start_date": "2024-01-06T00:00:00Z"},
        ]
        mock_get.side_effect = [_mock_response(activities), _mock_response([])]
        result = get_athlete_activities("token")
        self.assertIn(10, result)


class IngestAllGearResolutionTests(unittest.TestCase):
    """Test that ingest_all resolves gear_id from activities, not from segment efforts."""

    @patch("src.fetch.time.sleep")
    @patch("src.fetch.requests.get")
    def test_gear_id_resolved_from_activities(self, mock_get: MagicMock, _sleep: MagicMock) -> None:
        """gear_id on effort rows should come from the activities lookup, not the effort object."""
        athlete_resp = _mock_response({"bikes": [{"id": "b111", "name": "Trek Domane"}]})

        activities_page1 = [
            {"id": 42, "sport_type": "Ride", "average_watts": 210.0, "device_watts": True,
             "gear_id": "b111", "name": "Morning Ride", "start_date": "2024-03-01T07:00:00Z"},
        ]

        starred_segs = [
            {"id": 99, "name": "Test Hill", "distance": 1000.0, "average_grade": 5.0,
             "climb_category": 1, "total_elevation_gain": 50.0, "start_latlng": [51.0, -1.0]},
        ]

        segment_efforts = [
            # Note: no gear_id on the embedded activity — only the id
            {"id": 777, "activity": {"id": 42, "resource_state": 1},
             "start_date": "2024-03-01T07:30:00Z", "elapsed_time": 120,
             "moving_time": 118, "average_watts": 220.0, "average_heartrate": 155.0},
        ]

        # Each call returns < per_page items so pagination stops after one page each.
        mock_get.side_effect = [
            athlete_resp,                       # GET /athlete
            _mock_response(activities_page1),   # GET /athlete/activities
            _mock_response(starred_segs),       # GET /segments/starred
            _mock_response(segment_efforts),    # GET /segment_efforts
        ]

        result = ingest_all("token")

        efforts_df: pd.DataFrame = result["efforts"]
        self.assertFalse(efforts_df.empty, "efforts should not be empty")
        self.assertIn("gear_id", efforts_df.columns)
        self.assertEqual(efforts_df.iloc[0]["gear_id"], "b111")

    @patch("src.fetch.time.sleep")
    @patch("src.fetch.requests.get")
    def test_effort_unmatched_activity_gets_none_gear_id(self, mock_get: MagicMock, _sleep: MagicMock) -> None:
        """An effort whose activity_id doesn't match any power ride gets gear_id=None."""
        mock_get.side_effect = [
            _mock_response({"bikes": []}),      # GET /athlete
            _mock_response([]),                 # GET /athlete/activities (no power rides)
            _mock_response([
                {"id": 99, "name": "Seg", "distance": 500.0, "average_grade": 1.0,
                 "climb_category": 0, "total_elevation_gain": 5.0, "start_latlng": [51.0, -1.0]},
            ]),                                 # GET /segments/starred
            _mock_response([
                {"id": 888, "activity": {"id": 999, "resource_state": 1},
                 "start_date": "2024-03-01T09:00:00Z", "elapsed_time": 60,
                 "moving_time": 58, "average_watts": 180.0, "average_heartrate": 145.0},
            ]),                                 # GET /segment_efforts
        ]

        result = ingest_all("token")
        efforts_df: pd.DataFrame = result["efforts"]
        self.assertFalse(efforts_df.empty)
        self.assertIsNone(efforts_df.iloc[0]["gear_id"])


class IngestAllDevModeTests(unittest.TestCase):
    """Integration tests for ingest_all(dev=True) — no mocking, reads real JSON fixtures."""

    def setUp(self) -> None:
        self.result = ingest_all(access_token="", dev=True)

    # --- bikes ---

    def test_bikes_is_dict(self) -> None:
        self.assertIsInstance(self.result["bikes"], dict)

    def test_bikes_contains_expected_gear_ids(self) -> None:
        bikes = self.result["bikes"]
        self.assertIn("b111111", bikes)
        self.assertIn("b222222", bikes)
        self.assertEqual(bikes["b111111"], "Trek Domane SL5")
        self.assertEqual(bikes["b222222"], "Canyon Grail CF SLX")

    # --- segments ---

    def test_segments_is_dataframe(self) -> None:
        self.assertIsInstance(self.result["segments"], pd.DataFrame)

    def test_segments_has_expected_columns(self) -> None:
        expected = {"segment_id", "name", "distance", "average_grade", "climb_category",
                    "total_elevation_gain", "start_lat", "start_lng", "segment_type", "segment_type_detail"}
        self.assertTrue(expected.issubset(set(self.result["segments"].columns)))

    def test_segments_row_count(self) -> None:
        self.assertEqual(len(self.result["segments"]), 3)

    def test_segment_classification(self) -> None:
        segs = self.result["segments"].set_index("name")
        self.assertEqual(segs.loc["Box Hill", "segment_type"], "ascent")
        self.assertEqual(segs.loc["Box Hill", "segment_type_detail"], "ascent_moderate")
        self.assertEqual(segs.loc["Town Sprint", "segment_type"], "sprint")
        self.assertEqual(segs.loc["Town Sprint", "segment_type_detail"], "sprint_flat")
        self.assertEqual(segs.loc["River Road Flat", "segment_type"], "flat")
        self.assertEqual(segs.loc["River Road Flat", "segment_type_detail"], "flat_long")
        self.assertEqual(segs.loc["North Downs Descent", "segment_type"], "descent")
        self.assertEqual(segs.loc["North Downs Descent", "segment_type_detail"], "descent_steep")

    # --- efforts ---

    def test_efforts_is_dataframe(self) -> None:
        self.assertIsInstance(self.result["efforts"], pd.DataFrame)

    def test_efforts_has_gear_id_column(self) -> None:
        self.assertIn("gear_id", self.result["efforts"].columns)

    def test_efforts_row_count(self) -> None:
        self.assertEqual(len(self.result["efforts"]), 7)

    def test_gear_id_resolved_from_activities(self) -> None:
        efforts = self.result["efforts"]
        # activity 300001/300003/300005 → b111111; 300008 → b222222
        b1_efforts = efforts[efforts["gear_id"] == "b111111"]
        b2_efforts = efforts[efforts["gear_id"] == "b222222"]
        self.assertEqual(len(b1_efforts), 4)
        self.assertEqual(len(b2_efforts), 3)

    def test_efforts_no_raw_api_keys(self) -> None:
        """Ensure parsed column names (not raw API keys like 'id') are used."""
        self.assertIn("effort_id", self.result["efforts"].columns)
        self.assertNotIn("id", self.result["efforts"].columns)


if __name__ == "__main__":
    unittest.main()
