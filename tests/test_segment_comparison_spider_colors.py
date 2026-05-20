"""Tests for spider plot color handling in segment comparison page."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from plotly.colors import hex_to_rgb, unlabel_rgb


def _load_rgba_helper():
    page_path = Path(__file__).resolve().parents[1] / "app" / "app_pages" / "segment_comparison.py"
    source = page_path.read_text(encoding="utf-8")
    module_ast = ast.parse(source)
    rgba_node = next(
        node for node in module_ast.body if isinstance(node, ast.FunctionDef) and node.name == "_rgba"
    )
    helper_module = ast.Module(body=[rgba_node], type_ignores=[])
    ast.fix_missing_locations(helper_module)
    namespace = {"hex_to_rgb": hex_to_rgb, "unlabel_rgb": unlabel_rgb}
    exec(compile(helper_module, str(page_path), "exec"), namespace)
    return namespace["_rgba"]


class SegmentComparisonSpiderColorTests(unittest.TestCase):
    def test_accepts_plotly_rgb_colors(self) -> None:
        rgba = _load_rgba_helper()
        self.assertEqual(rgba("rgb(102,194,165)", 0.2), "rgba(102.0,194.0,165.0,0.2)")

    def test_accepts_hex_colors(self) -> None:
        rgba = _load_rgba_helper()
        self.assertEqual(rgba("#66c2a5", 0.2), "rgba(102,194,165,0.2)")


if __name__ == "__main__":
    unittest.main()
