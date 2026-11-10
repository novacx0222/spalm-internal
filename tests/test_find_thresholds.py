"""Unit tests for spalm_timing and spalm_calibration (run: make
test-thresholds). Standard library only."""

import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import spalm_calibration as calib  # noqa: E402
from spalm_timing import (DENSE_MM_LABEL, SPARSE_MM_LABEL,  # noqa: E402
                          parse_r_block, parse_slice_length, parse_time)

# One call's output as printed by print_time ("%s %.6f ms").
OUTPUT = """SPALM MATMUL
[1-slice-subrow-csr] 0.012000 ms
[slice-prepare-total] 1.500000 ms
[sparse-sparse-s-transpose-time] 2.000000 ms
[sparse-sparse-sparse-mm-time] 12.345678 ms
[sparse-sparse-total-sparse-mm-time] 14.345678 ms
"""


class ParseTimeTest(unittest.TestCase):
    def test_exact_label_in_seconds(self):
        self.assertAlmostEqual(parse_time(OUTPUT, SPARSE_MM_LABEL), 0.012345678)

    def test_label_is_not_a_substring_match(self):
        # "[sparse-sparse-total-sparse-mm-time]" must not satisfy the kernel label.
        text = "[sparse-sparse-total-sparse-mm-time] 3.000000 ms\n"
        self.assertIsNone(parse_time(text, SPARSE_MM_LABEL))

    def test_missing_label(self):
        self.assertIsNone(parse_time(OUTPUT, DENSE_MM_LABEL))

    def test_old_unbracketed_format_is_not_accepted(self):
        self.assertIsNone(parse_time("Sparse MM time: 0.0123s\n", SPARSE_MM_LABEL))

    def test_duplicate_label_is_an_error(self):
        text = f"{DENSE_MM_LABEL} 1.0 ms\n{DENSE_MM_LABEL} 2.0 ms\n"
        with self.assertRaises(ValueError):
            parse_time(text, DENSE_MM_LABEL)

    def test_invalid_values_are_errors(self):
        for value in ("0.000000", "-1.0", "nan", "inf", "abc"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_time(f"{DENSE_MM_LABEL} {value} ms\n", DENSE_MM_LABEL)

    def test_sub_microsecond_values(self):
        self.assertAlmostEqual(parse_time(f"{DENSE_MM_LABEL} 0.000123 ms\n", DENSE_MM_LABEL),
                               1.23e-7)


class ParseSliceLengthTest(unittest.TestCase):
    LINE = ("[slice-length] rule=counts configured=512 chosen=2048 sides=RS "
            "entries_per_subrow=1.2,2.3,4.1 cost=9e+09,6e+09,4e+09 "
            "cost0=9.12346e+09")

    def test_counted(self):
        self.assertEqual(parse_slice_length(f"x\n{self.LINE}\ny\n"),
                         (512, 2048, "RS", 9.12346e9))

    def test_rule_did_not_run(self):
        self.assertIsNone(parse_slice_length("Slice length: 512\n"))

    def test_not_counted_is_an_error(self):
        text = ("[slice-length] rule=counts configured=512 chosen=512 sides=R "
                "cost0_bound=1.5e+08 (not counted)\n")
        with self.assertRaises(ValueError):
            parse_slice_length(text)

    def test_density_rule_is_ignored(self):
        text = ("[slice-length] rule=density configured=512 chosen=2048 "
                "entries_per_subrow=1 -> 4\n")
        self.assertIsNone(parse_slice_length(text))

    def test_repeated_line_is_an_error(self):
        with self.assertRaises(ValueError):
            parse_slice_length(f"{self.LINE}\n{self.LINE}\n")


class ParseRBlockTest(unittest.TestCase):
    def test_line(self):
        text = ("[dense-packing-buffers] r_bytes=12583040 r_huge=1 "
                "s_block_bytes=65664 s_shared_huge=0\n")
        self.assertEqual(parse_r_block(text), (12583040, True))
        self.assertEqual(parse_r_block(text.replace("r_huge=1", "r_huge=0")),
                         (12583040, False))

    def test_missing_or_repeated(self):
        line = "[dense-packing-buffers] r_bytes=1 r_huge=0 s_block_bytes=0\n"
        for text in ("", line * 2):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_r_block(text)


class CalibrationFileTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "cpu" / "double" / "calibration.txt"

    def tearDown(self):
        self.dir.cleanup()

    def test_new_file_gets_header_and_values(self):
        calib.update_calibration(self.path, {"asl_min_cost": 5e9}, "asl run")
        text = self.path.read_text()
        self.assertTrue(text.startswith("# "))
        self.assertIn("asl_min_cost: 5e+09  # asl run\n", text)
        self.assertEqual(calib.read_calibration(self.path),
                         {"asl_min_cost": "5e+09"})

    def test_update_keeps_other_lines_and_replaces_in_place(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("# mine\nss_page_huge_entries: 2097152  # page\n"
                             "asl_min_cost: 1  # old\nasl_min_cost: 2\n")
        calib.update_calibration(
            self.path, {"asl_min_cost": 3e9, "r_block_huge_bytes": -1}, "new")
        self.assertEqual(self.path.read_text().splitlines(), [
            "# mine",
            "ss_page_huge_entries: 2097152  # page",
            "asl_min_cost: 3e+09  # new",
            "r_block_huge_bytes: -1  # new",
        ])

    def test_rejects_unknown_key_and_bad_values(self):
        with self.assertRaises(ValueError):
            calib.update_calibration(self.path, {"slice_length": 1024}, "x")
        with self.assertRaises(ValueError):
            calib.update_calibration(self.path, {"asl_min_cost": math.inf}, "x")
        with self.assertRaises(ValueError):
            calib.update_calibration(self.path, {"asl_min_cost": 1}, "a\nb")
        self.assertFalse(self.path.exists())

    def test_format_value(self):
        self.assertEqual(calib.format_value(8388608), "8388608")
        self.assertEqual(calib.format_value(3e9), "3e+09")
        self.assertEqual(calib.format_value(calib.NEVER_MIN_COST), "1e+300")
        self.assertEqual(calib.format_value(1234.5678), "1234.57")


class ChooseMinCostTest(unittest.TestCase):
    def test_cut_between_slower_and_faster_cells(self):
        cells = [(1e8, 1.0, 1.5), (5e8, 1.0, 1.2), (4e9, 1.0, 0.8),
                 (2e10, 1.0, 0.6)]
        value, lengthened = calib.choose_min_cost(cells)
        self.assertAlmostEqual(value, math.sqrt(5e8 * 4e9))
        self.assertEqual(lengthened, {2, 3})

    def test_every_cell_faster_takes_the_lowest_cost(self):
        value, lengthened = calib.choose_min_cost([(3e9, 1, 0.9), (1e9, 1, 0.5)])
        self.assertEqual(value, 1e9)
        self.assertEqual(lengthened, {0, 1})

    def test_no_cell_faster_never_lengthens(self):
        value, lengthened = calib.choose_min_cost([(3e9, 1, 1.1), (1e9, 1, 1.0)])
        self.assertEqual(value, calib.NEVER_MIN_COST)
        self.assertEqual(lengthened, set())

    def test_slow_cell_just_below_fast_ones_stays_at_base(self):
        # The cut goes between 2e9 (1.1x longer) and 3e9 (0.7x): including
        # the 2e9 cell would add log(1.1) to the sum.
        cells = [(1e9, 1, 1.3), (2e9, 1, 1.1), (3e9, 1, 0.7), (4e9, 1, 0.7)]
        value, lengthened = calib.choose_min_cost(cells)
        self.assertAlmostEqual(value, math.sqrt(2e9 * 3e9))
        self.assertEqual(lengthened, {2, 3})

    def test_equal_costs_are_not_split(self):
        cells = [(1e9, 1, 0.5), (1e9, 1, 2.0), (5e9, 1, 0.9)]
        value, lengthened = calib.choose_min_cost(cells)
        self.assertAlmostEqual(value, math.sqrt(1e9 * 5e9))
        self.assertEqual(lengthened, {2})

    def test_tie_goes_to_the_higher_cut(self):
        cells = [(1e9, 1, 1.0), (5e9, 1, 0.8)]
        value, lengthened = calib.choose_min_cost(cells)
        self.assertAlmostEqual(value, math.sqrt(1e9 * 5e9))
        self.assertEqual(lengthened, {1})

    def test_zero_cost_below_the_cut(self):
        value, _ = calib.choose_min_cost([(0, 1, 1.2), (4e9, 1, 0.8)])
        self.assertEqual(value, 2e9)

    def test_bad_cells(self):
        for cells in ([], [(1e9, 0, 1)], [(-1, 1, 1)], [(1e9, 1, math.nan)]):
            with self.subTest(cells=cells), self.assertRaises(ValueError):
                calib.choose_min_cost(cells)


class HugeBreakEvenTest(unittest.TestCase):
    MIB = calib.MIB

    def line(self, a_ms, b_ms_per_mib, sizes_mib):
        return [(int(s * self.MIB), (a_ms + b_ms_per_mib * s) / 1e3)
                for s in sizes_mib]

    def test_zero_of_the_line_rounded_up(self):
        # +0.5 ms flat, -0.06 ms per MiB: zero at 8.33 MiB -> 9 MiB.
        points = self.line(0.5, -0.06, [2, 4, 6, 9, 12, 24, 48])
        value, (a, b) = calib.huge_break_even(points)
        self.assertEqual(value, 9 * self.MIB)
        self.assertAlmostEqual(a, 0.5e-3)
        self.assertAlmostEqual(b, -0.06e-3)

    def test_blocks_below_2_mib_are_ignored(self):
        points = self.line(0.5, -0.06, [2, 4, 12]) + [(self.MIB, 1.0)]
        self.assertEqual(calib.huge_break_even(points)[0], 9 * self.MIB)

    def test_always_pays(self):
        points = self.line(-0.1, -0.05, [2, 8, 16])
        self.assertEqual(calib.huge_break_even(points)[0], 2 * self.MIB)

    def test_never_pays(self):
        self.assertEqual(
            calib.huge_break_even(self.line(0.5, 0.01, [2, 8, 16]))[0], -1)
        # Falling, but still above zero at the largest block measured.
        self.assertEqual(
            calib.huge_break_even(self.line(0.5, -0.01, [2, 8, 16]))[0], -1)

    def test_needs_two_sizes(self):
        with self.assertRaises(ValueError):
            calib.huge_break_even([(4 * self.MIB, 0.1), (4 * self.MIB, 0.2)])
        with self.assertRaises(ValueError):
            calib.huge_break_even([(self.MIB, 0.1), (4 * self.MIB, 0.2)])


class MedianTest(unittest.TestCase):
    def test_median(self):
        self.assertEqual(calib.median([3, 1, 2]), 2)
        self.assertEqual(calib.median([4, 1, 3, 2]), 2.5)
        with self.assertRaises(ValueError):
            calib.median([])


if __name__ == "__main__":
    unittest.main()
