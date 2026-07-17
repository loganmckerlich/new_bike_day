import unittest
from unittest.mock import patch

import pandas as pd

import src.database as db


class Response:
    def __init__(self, data):
        self.data = data


class RPC:
    def __init__(self, data):
        self._data = data

    def execute(self):
        return Response(self._data)


class FakeQuery:
    def __init__(self, store, table_name):
        self.store = store
        self.table_name = table_name
        self._filters = []
        self._order = None
        self._limit = None
        self._payload = None
        self._delete = False

    def select(self, *args):
        return self

    def eq(self, column, value):
        self._filters.append((column, value))
        return self

    def order(self, column, desc=False):
        self._order = (column, desc)
        return self

    def limit(self, value):
        self._limit = value
        return self

    def delete(self):
        self._delete = True
        return self

    def upsert(self, payload, on_conflict=None):
        self._payload = payload
        self._on_conflict = on_conflict
        return self

    def _rows_for_athlete(self):
        if not self._filters:
            return []
        athlete_id = self._filters[0][1]
        return [row for row in self.store[self.table_name] if row.get("athlete_id") == athlete_id]

    def _delete_matching_rows(self):
        rows = self._rows_for_athlete()
        athlete_id = self._filters[0][1] if self._filters else None
        if athlete_id is None:
            return []
        self.store[self.table_name] = [row for row in self.store[self.table_name] if row.get("athlete_id") != athlete_id]
        return rows

    def execute(self):
        if self._delete and self.table_name in self.store:
            return Response(self._delete_matching_rows())
        if self.table_name == "users" and self._payload is not None:
            athlete_id = self._payload.get("athlete_id")
            self.store[self.table_name] = [
                row for row in self.store[self.table_name] if row.get("athlete_id") != athlete_id
            ]
            self.store[self.table_name].append(self._payload)
            return Response([self._payload])
        if self.table_name == "users" and self._filters:
            rows = self._rows_for_athlete()
            if self._limit == 1:
                rows = rows[:1]
            return Response(rows)
        if self.table_name == "users" and self._order is not None:
            rows = list(self.store[self.table_name])
            rows.sort(key=lambda row: row.get("last_accessed") or "")
            return Response(rows)
        if self.table_name in self.store and self._payload is not None:
            payload = self._payload
            if isinstance(payload, list):
                self.store[self.table_name].extend(payload)
                self.store.setdefault("_upsert_calls", []).append((self.table_name, len(payload)))
            else:
                self.store[self.table_name].append(payload)
                self.store.setdefault("_upsert_calls", []).append((self.table_name, 1))
            return Response(payload if isinstance(payload, list) else [payload])
        if self.table_name in self.store and self._filters:
            rows = self._rows_for_athlete()
            return Response(rows)
        return Response([])


class FakeClient:
    def __init__(self):
        self.store = {
            "starred_segments": [],
            "segment_efforts": [],
            "bikes": [],
            "athlete_ftp": [],
            "athlete_tokens": [],
            "users": [],
        }

    def table(self, name):
        return FakeQuery(self.store, name)

    def rpc(self, name):
        return RPC([{"get_db_size_mb": 500.0}])


class DatabaseTests(unittest.TestCase):
    def test_touch_user_creates_row_with_athlete_scope(self) -> None:
        client = FakeClient()
        with patch.object(db, "supabase", client):
            db.touch_user(42)
            self.assertEqual(client.store["users"][0]["athlete_id"], "42")
            self.assertIn("last_accessed", client.store["users"][0])

    def test_cleanup_if_needed_deletes_legacy_user_data_in_order(self) -> None:
        client = FakeClient()
        client.store["users"].append({"athlete_id": "1", "last_accessed": "2024-01-01T00:00:00+00:00"})
        client.store["segment_efforts"].append({"athlete_id": "1", "effort_id": "99"})
        client.store["starred_segments"].append({"athlete_id": "1", "segment_id": "77"})
        client.store["bikes"].append({"athlete_id": "1", "gear_id": "g1"})
        client.store["athlete_ftp"].append({"athlete_id": "1", "ftp": 250})
        client.store["athlete_tokens"].append({"athlete_id": "1", "access_token": "t"})

        with patch.object(db, "supabase", client), patch.object(db, "get_db_size_mb", side_effect=[500.0, 400.0]):
            db.cleanup_if_needed("2")

        self.assertEqual(client.store["segment_efforts"], [])
        self.assertEqual(client.store["starred_segments"], [])
        self.assertEqual(client.store["bikes"], [])
        self.assertEqual(client.store["athlete_ftp"], [])
        self.assertEqual(client.store["athlete_tokens"], [])
        self.assertEqual(client.store["users"], [])

    def test_save_and_load_user_ingest_dates(self) -> None:
        client = FakeClient()
        with patch.object(db, "supabase", client):
            db.save_user_ingest_dates(
                42,
                last_ingested_date="2026-01-31T00:00:00+00:00",
                oldest_ingested_date="2025-01-01T00:00:00+00:00",
            )
            last_ingested, oldest_ingested = db.load_user_ingest_dates(42)
        self.assertEqual(last_ingested, "2026-01-31T00:00:00+00:00")
        self.assertEqual(oldest_ingested, "2025-01-01T00:00:00+00:00")

    def test_save_efforts_batches_large_datasets(self) -> None:
        """save_efforts must split >_UPSERT_BATCH_SIZE records into multiple calls."""
        n = db._UPSERT_BATCH_SIZE + 10
        df = pd.DataFrame(
            [
                {
                    "effort_id": str(i),
                    "segment_id": "1",
                    "activity_id": str(i),
                    "gear_id": "g1",
                    "start_date": "2026-01-01T00:00:00Z",
                    "elapsed_time": 60,
                    "moving_time": 55,
                    "average_watts": 200.0,
                    "average_heartrate": 150.0,
                }
                for i in range(n)
            ]
        )
        client = FakeClient()
        with patch.object(db, "supabase", client):
            db.save_efforts(df, 42)

        upsert_calls = [c for c in client.store.get("_upsert_calls", []) if c[0] == "segment_efforts"]
        self.assertEqual(len(upsert_calls), 2, "Expected 2 batches for n > _UPSERT_BATCH_SIZE")
        self.assertEqual(upsert_calls[0][1], db._UPSERT_BATCH_SIZE)
        self.assertEqual(upsert_calls[1][1], 10)
        # All records made it into the store
        self.assertEqual(len(client.store["segment_efforts"]), n)


if __name__ == "__main__":
    unittest.main()
