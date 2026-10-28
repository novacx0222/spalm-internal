"""Unit tests for machine_tuning (run: make test-thresholds). Standard library only."""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import machine_tuning as mt  # noqa: E402


class AutoSliceLengthTest(unittest.TestCase):
    def test_matches_the_library_rule(self):
        # encoder-1025: uniform 1% (5.1 entries per 512-key subrow) -> 2048,
        # uniform 5% (25.6 entries) and full subrows stay at 512.
        self.assertEqual(mt.auto_slice_length(512, 0.01, 16000, 16, 2048), 2048)
        self.assertEqual(mt.auto_slice_length(512, 0.05, 16000, 16, 2048), 512)
        self.assertEqual(mt.auto_slice_length(512, 1.0, 16000, 16, 2048), 512)

    def test_limits(self):
        self.assertEqual(mt.auto_slice_length(512, 0.001, 16000, 16, 1024), 1024)
        # Never past K once K is reached; never shorter than configured.
        self.assertEqual(mt.auto_slice_length(512, 0.001, 700, 16, 4096), 1024)
        self.assertEqual(mt.auto_slice_length(512, 0.001, 300, 16, 4096), 512)
        self.assertEqual(mt.auto_slice_length(512, 0.001, 16000, 1, 4096), 1024)


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


class FitSliceRuleTest(unittest.TestCase):
    LENGTHS = (256, 512, 1024, 2048, 4096)

    @classmethod
    def times(cls, best_by_density, fixed=None):
        """Each case is fastest at its given length, 10% slower per doubling.

        fixed: {density: best length} of cases at the base length (SS).
        """
        cases = {}
        for automatic, table in ((True, best_by_density), (False, fixed or {})):
            for density, best in table.items():
                cases[(automatic, density)] = (density, {
                    L: 1.0 + 0.1 * abs(math.log2(L / best))
                    for L in cls.LENGTHS}, automatic)
        return cases

    def test_recovers_the_rule_behind_the_measurements(self):
        # 2 and 4 entries per 512-key subrow -> 2048, 8 -> 1024, 16+ -> 512,
        # SS best at 512: base 512, target 16, maximum 2048.
        times = self.times({2 / 512: 2048, 4 / 512: 2048, 8 / 512: 1024,
                            16 / 512: 512, 64 / 512: 512}, {0.01: 512})
        (base, target, max_length), costs = mt.fit_slice_rule(
            times, 16000, (256, 512, 1024), (1, 2, 4, 8, 16, 32, 64),
            (512, 1024, 2048, 4096))
        self.assertEqual((base, target, max_length), (512, 16, 2048))
        self.assertAlmostEqual(costs[(512, 16, 2048)], 1.0)
        self.assertGreater(costs[(256, 1, 512)], 1.0)
        self.assertNotIn((1024, 16, 512), costs)  # max below base

    def test_longer_base(self):
        # Dense subrows and SS both best at 1024: the base moves.
        times = self.times({2 / 512: 2048, 64 / 512: 1024}, {0.01: 1024})
        (base, target, max_length), _ = mt.fit_slice_rule(
            times, 16000, (512, 1024), (1, 16, 64), (1024, 2048))
        self.assertEqual((base, target, max_length), (1024, 16, 2048))

    def test_longer_slices_never_help(self):
        times = self.times({2 / 512: 512, 64 / 512: 512})
        (base, target, max_length), _ = mt.fit_slice_rule(
            times, 16000, (512,), (1, 16), (512, 2048))
        # Ties go to the smaller base, target, then maximum.
        self.assertEqual((base, target, max_length), (512, 1, 512))

    def test_unmeasured_lengths_are_skipped(self):
        times = {"c": (2 / 512, {512: 2.0, 1024: 1.0}, True)}
        (base, target, max_length), costs = mt.fit_slice_rule(
            times, 16000, (512,), (16,), (1024, 2048))
        self.assertEqual((base, target, max_length), (512, 16, 1024))
        self.assertNotIn((512, 16, 2048), costs)
        with self.assertRaises(ValueError):
            mt.fit_slice_rule(times, 16000, (512,), (16,), (4096,))


class FormatTest(unittest.TestCase):
    def test_all_keys_in_library_format(self):
        text = mt.format_machine_params(mt.DEFAULTS, ["measured here"])
        self.assertEqual(text.splitlines()[0], "# measured here")
        self.assertIn("slice_length_max: 2048\n", text)
        self.assertIn("slice_length: 512\n", text)
        self.assertIn("tile_size_l3_ds: 2304\n", text)
        self.assertIn("slice_target_entries_dd: 16\n", text)
        self.assertIn("result_base_offset_sparse: 16\n", text)
        self.assertEqual(len(text.splitlines()), 1 + len(mt.KEYS))

    def test_missing_key(self):
        params = dict(mt.DEFAULTS)
        del params["tile_size_l3"]
        with self.assertRaises(ValueError):
            mt.format_machine_params(params)


if __name__ == "__main__":
    unittest.main()
