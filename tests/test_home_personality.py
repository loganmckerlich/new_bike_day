"""Tests for home page personality helpers."""

from __future__ import annotations

import unittest

from src.home_personality import load_dev_athlete_profile


class HomePersonalityTests(unittest.TestCase):
    def test_load_dev_athlete_profile_reads_expected_fields(self) -> None:
        profile = load_dev_athlete_profile()
        self.assertEqual(profile["first_name"], "Dev")
        self.assertEqual(profile["full_name"], "Dev User")
        self.assertEqual(profile["primary_bike"], "Trek Domane SL5")


if __name__ == "__main__":
    unittest.main()
