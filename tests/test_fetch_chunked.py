import unittest

import pandas as pd

from src.fetch import ingest_window


class _Response:
    def __init__(self, data, headers=None):
        self._data = data
        self.headers = headers or {}

    def json(self):
        return self._data

    def raise_for_status(self):
        return None


class _Http:
    def __init__(self, activities, segment_responses):
        self.activities = activities
        self.segment_responses = segment_responses

    def get(self, url, headers=None, params=None, timeout=None):
        if url.endswith("/athlete/activities"):
            return _Response(self.activities, headers={"X-RateLimit-Limit": "200,2000", "X-RateLimit-Usage": "10,100"})
        segment_id = int(params.get("segment_id"))
        response_data, response_headers = self.segment_responses[segment_id]
        return _Response(response_data, headers=response_headers)


class ChunkedIngestTests(unittest.TestCase):
    def test_ingest_window_joins_gear_from_activity_summary_and_filters_unmatched_efforts(self):
        http = _Http(
            activities=[
                {"id": 1, "gear_id": "g1", "sport_type": "Ride", "start_date": "2026-01-01T00:00:00Z"},
                {"id": 2, "gear_id": "g2", "sport_type": "Run", "start_date": "2026-01-01T00:00:00Z"},
            ],
            segment_responses={
                10: (
                    [
                        {"id": 100, "activity": {"id": 1}, "start_date": "2026-01-02T00:00:00Z"},
                        {"id": 101, "activity": {"id": 2}, "start_date": "2026-01-02T00:00:00Z"},
                    ],
                    {"X-RateLimit-Limit": "200,2000", "X-RateLimit-Usage": "11,101"},
                )
            },
        )
        segments = pd.DataFrame([{"segment_id": 10}])
        result = ingest_window("token", segments, pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-31"), _http=http)
        self.assertTrue(result["complete"])
        self.assertEqual(len(result["efforts"]), 1)
        self.assertEqual(result["efforts"].iloc[0]["gear_id"], "g1")

    def test_ingest_window_discards_partial_window_when_rate_limit_hits_mid_window(self):
        http = _Http(
            activities=[{"id": 1, "gear_id": "g1", "sport_type": "Ride", "start_date": "2026-01-01T00:00:00Z"}],
            segment_responses={
                10: (
                    [{"id": 100, "activity": {"id": 1}, "start_date": "2026-01-02T00:00:00Z"}],
                    {"X-RateLimit-Limit": "200,2000", "X-RateLimit-Usage": "180,100"},
                ),
                11: (
                    [{"id": 101, "activity": {"id": 1}, "start_date": "2026-01-03T00:00:00Z"}],
                    {"X-RateLimit-Limit": "200,2000", "X-RateLimit-Usage": "181,101"},
                ),
            },
        )
        segments = pd.DataFrame([{"segment_id": 10}, {"segment_id": 11}])
        result = ingest_window("token", segments, pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-31"), _http=http)
        self.assertFalse(result["complete"])
        self.assertTrue(result["mid_window_rate_limit"])
        self.assertTrue(result["efforts"].empty)


if __name__ == "__main__":
    unittest.main()
