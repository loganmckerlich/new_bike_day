import unittest

from src.auth import is_strava_capacity_error


class AuthErrorTests(unittest.TestCase):
    def test_is_strava_capacity_error_matches_capacity_messages(self) -> None:
        self.assertTrue(is_strava_capacity_error("Application has reached athlete capacity"))
        self.assertTrue(is_strava_capacity_error("Maximum users reached for this app"))

    def test_is_strava_capacity_error_ignores_unrelated_messages(self) -> None:
        self.assertFalse(is_strava_capacity_error("invalid redirect uri"))
        self.assertFalse(is_strava_capacity_error(None))


if __name__ == "__main__":
    unittest.main()
