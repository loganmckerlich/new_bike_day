import unittest

from src.auth import (
    _oauth_error_details,
    format_oauth_failure_message,
    is_strava_capacity_error,
)


class AuthErrorTests(unittest.TestCase):
    def test_is_strava_capacity_error_matches_capacity_messages(self) -> None:
        self.assertTrue(is_strava_capacity_error("Application has reached athlete capacity"))
        self.assertTrue(is_strava_capacity_error("Maximum users reached for this app"))

    def test_is_strava_capacity_error_ignores_unrelated_messages(self) -> None:
        self.assertFalse(is_strava_capacity_error("invalid redirect uri"))
        self.assertFalse(is_strava_capacity_error("Rate limit exceeded for token endpoint"))
        self.assertFalse(is_strava_capacity_error(None))

    def test_oauth_error_details_prefers_human_readable_nested_errors(self) -> None:
        payload = {"errors": [{"resource": "Application", "message": "Athlete capacity reached"}]}
        self.assertEqual(_oauth_error_details(payload), "Athlete capacity reached")

    def test_oauth_error_details_prefers_nested_message_over_top_level_error(self) -> None:
        payload = {
            "error": "authorization_failed",
            "errors": [{"message": "Athlete capacity reached"}],
        }
        self.assertEqual(_oauth_error_details(payload), "Athlete capacity reached")

    def test_format_oauth_failure_message_for_capacity(self) -> None:
        msg = format_oauth_failure_message("application has reached athlete capacity")
        self.assertIn("maximum allowed users", msg)
        self.assertIn("Choose Logan sample data below", msg)

    def test_format_oauth_failure_message_for_generic_error(self) -> None:
        msg = format_oauth_failure_message("invalid redirect uri")
        self.assertIn("Strava sign-in failed", msg)
        self.assertIn("Choose Logan sample data below", msg)

    def test_format_oauth_failure_message_handles_missing_details(self) -> None:
        msg = format_oauth_failure_message(None)
        self.assertIn("unknown error", msg)


if __name__ == "__main__":
    unittest.main()
