"""Unit tests for Streamlit app helpers."""

import unittest

from app.streamlit_app import _normalized_redirect_uri


class RedirectUriNormalizationTests(unittest.TestCase):
    def test_empty_or_whitespace_falls_back_to_localhost_root(self) -> None:
        self.assertEqual(_normalized_redirect_uri(""), "http://localhost:8501/")
        self.assertEqual(_normalized_redirect_uri("   "), "http://localhost:8501/")

    def test_origin_without_path_gets_root_path(self) -> None:
        self.assertEqual(_normalized_redirect_uri("http://localhost:8501"), "http://localhost:8501/")
        self.assertEqual(_normalized_redirect_uri("https://example.com"), "https://example.com/")

    def test_existing_path_query_and_fragment_are_preserved(self) -> None:
        self.assertEqual(_normalized_redirect_uri("https://example.com/callback"), "https://example.com/callback")
        self.assertEqual(
            _normalized_redirect_uri("https://example.com?x=1#frag"),
            "https://example.com/?x=1#frag",
        )


if __name__ == "__main__":
    unittest.main()
