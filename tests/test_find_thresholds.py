"""Unit tests for spalm_timing (run: make test-thresholds). Standard library only."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spalm_timing import DENSE_MM_LABEL, SPARSE_MM_LABEL, parse_time  # noqa: E402

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


if __name__ == "__main__":
    unittest.main()
