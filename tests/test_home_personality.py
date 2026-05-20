"""Tests for home page personality helpers."""

from __future__ import annotations

import unittest

from src.home_personality import build_cheeky_conclusion, load_dev_athlete_profile


class HomePersonalityTests(unittest.TestCase):
    def test_load_dev_athlete_profile_reads_expected_fields(self) -> None:
        profile = load_dev_athlete_profile()
        self.assertEqual(profile["first_name"], "Dev")
        self.assertEqual(profile["full_name"], "Dev User")
        self.assertEqual(profile["primary_bike"], "Trek Domane SL5")

    def test_build_cheeky_conclusion_uses_takeaway_or_fallback(self) -> None:
        custom = build_cheeky_conclusion(
            athlete_name="Dev",
            legs_status="Crispy",
            vibe="Chaos, but fun",
            takeaway="The bike had legs, I had opinions.",
        )
        self.assertIn("Dev says:", custom)
        self.assertIn("The bike had legs, I had opinions.", custom)

        fallback = build_cheeky_conclusion(
            athlete_name=None,
            legs_status="Spicy",
            vibe="Smooth and smug",
            takeaway="   ",
        )
        self.assertIn("The rider says:", fallback)
        self.assertIn("I respected the watts and feared the climbs.", fallback)

        empty_name = build_cheeky_conclusion(
            athlete_name="   ",
            legs_status="Fresh-ish",
            vibe="Smooth and smug",
            takeaway="Still smiling somehow.",
        )
        self.assertIn("The rider says:", empty_name)


if __name__ == "__main__":
    unittest.main()
