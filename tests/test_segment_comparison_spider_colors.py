"""Tests for spider plot color handling in segment comparison page."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.plot_colors import to_rgba


class SegmentComparisonSpiderColorTests(unittest.TestCase):
    def test_accepts_plotly_rgb_colors(self) -> None:
        self.assertEqual(to_rgba("rgb(102,194,165)", 0.2), "rgba(102,194,165,0.2)")

    def test_accepts_hex_colors(self) -> None:
        self.assertEqual(to_rgba("#66c2a5", 0.2), "rgba(102,194,165,0.2)")


if __name__ == "__main__":
    unittest.main()
