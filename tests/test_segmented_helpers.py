import unittest

import pandas as pd

from app.app_pages.bike_comparison_segmented import _highlight_best_value, _straightness_index


class SegmentExplorerHelpersTest(unittest.TestCase):
    def test_highlight_best_value_marks_the_maximum(self) -> None:
        series = pd.Series([1, 3, 2])

        styles = _highlight_best_value(series)

        self.assertEqual(styles[0], "")
        self.assertEqual(styles[1], "background-color: #dcfce7; color: #166534; font-weight: 600")
        self.assertEqual(styles[2], "")

    def test_straightness_index_is_bound_between_zero_and_one(self) -> None:
        geo = {"polyline_points": [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)]}

        index = _straightness_index(geo)

        self.assertIsNotNone(index)
        self.assertGreaterEqual(index, 0.0)
        self.assertLessEqual(index, 1.0)


if __name__ == "__main__":
    unittest.main()
