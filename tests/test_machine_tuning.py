"""Unit tests for machine_tuning (run: make test-thresholds). Standard library only."""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import machine_tuning as mt  # noqa: E402


class PickCandidateTest(unittest.TestCase):
    def test_geometric_mean_over_cases(self):
        table = {
            "a": {"x": 1.0, "y": 4.0},  # 1.0 and 2.0 of best -> 1.414
            "b": {"x": 1.5, "y": 2.0},  # 1.5 and 1.0 of best -> 1.225
        }
        chosen, costs = mt.pick_candidate(table)
        self.assertEqual(chosen, "b")
        self.assertAlmostEqual(costs["a"], math.sqrt(2))
        self.assertAlmostEqual(costs["b"], math.sqrt(1.5))

    def test_default_kept_within_tolerance(self):
        table = {16: {"SS": 1.000, "MKL-S": 1.000}, 0: {"SS": 0.995, "MKL-S": 1.0}}
        self.assertEqual(mt.pick_candidate(table, prefer=16, tolerance=0.01)[0], 16)
        self.assertEqual(mt.pick_candidate(table, prefer=16, tolerance=0.0)[0], 0)
        table[0] = {"SS": 0.9, "MKL-S": 0.95}
        self.assertEqual(mt.pick_candidate(table, prefer=16, tolerance=0.01)[0], 0)

    def test_cases_must_match(self):
        with self.assertRaises(ValueError):
            mt.pick_candidate({"a": {"x": 1.0}, "b": {"y": 1.0}})
        with self.assertRaises(ValueError):
            mt.pick_candidate({})


class FormatTest(unittest.TestCase):
    def test_all_keys_in_library_format(self):
        text = mt.format_machine_params(mt.DEFAULTS, ["measured here"])
        self.assertEqual(text.splitlines()[0], "# measured here")
        self.assertIn("result_base_offset_sparse: 16\n", text)
        self.assertIn("result_base_offset_dense: 0\n", text)
        self.assertEqual(len(text.splitlines()), 1 + len(mt.KEYS))

    def test_missing_key(self):
        with self.assertRaises(ValueError):
            mt.format_machine_params({"result_base_offset_sparse": 16})


if __name__ == "__main__":
    unittest.main()
