"""Parse SPALM native timing lines (standard library only).

The library prints one line per measured stage, "<label> <value> ms", where
<label> is "[stage-name]" (see print_time in include/util.h), for example:

    [sparse-sparse-sparse-mm-time] 12.345678 ms
"""

import math
import re

# Kernel-only times used for calibration. The dense label is shared by the
# dense-dense, right-inner (DS) and left-inner (SD) modes.
SPARSE_MM_LABEL = "[sparse-sparse-sparse-mm-time]"
DENSE_MM_LABEL = "[dense-dense-dense-mm-time]"
# Whole multiplication, including the packing buffers' allocation and free
# (r_block_huge_bytes stage), and the same plus slice preparation (asl stage:
# the slice length changes both).
MM_TOTAL_LABEL = "[mm-total]"
SLICE_MM_TOTAL_LABEL = "[slice-mm-total]"

_LINE = re.compile(r"\s*(\[[^\]\s]+\])\s+(\S+)\s+ms\s*")


def parse_time(text, label):
    """Return the duration in seconds of the single line for `label`.

    Returns None if the label does not occur. A repeated label, a malformed
    value, or a negative/zero/nonfinite duration raises ValueError: such
    output must never be used as a measurement.
    """
    values = []
    for line in text.splitlines():
        match = _LINE.fullmatch(line)
        if not match or match.group(1) != label:
            continue
        try:
            milliseconds = float(match.group(2))
        except ValueError:
            raise ValueError(f"Malformed duration for {label}: {line!r}") from None
        if not math.isfinite(milliseconds) or milliseconds <= 0:
            raise ValueError(f"Invalid duration for {label}: {line!r}")
        values.append(milliseconds / 1000.0)
    if len(values) > 1:
        raise ValueError(f"{label} reported {len(values)} times in one call")
    return values[0] if values else None


_SLICE_LENGTH_LINE = re.compile(
    r"\[slice-length\] rule=counts configured=(\d+) chosen=(\d+) "
    r"sides=(\S+)(.*)")
_BUFFERS_LINE = re.compile(
    r"\[dense-packing-buffers\] r_bytes=(\d+) r_huge=([01])(?: .*)?")


def _single(text, pattern, what):
    matches = [m for m in map(pattern.fullmatch, text.splitlines()) if m]
    if len(matches) > 1:
        raise ValueError(f"{what} reported {len(matches)} times in one call")
    return matches[0] if matches else None


def parse_slice_length(text):
    """(configured, chosen, sides, cost0) of the counts rule's line, or None
    when the rule did not run (K below twice the configured length, only SS
    slice pairs, or a slice_length key). A line without cost0 (a gate above
    the input's cost bound: "(not counted)") raises ValueError."""
    match = _single(text, _SLICE_LENGTH_LINE, "[slice-length]")
    if match is None:
        return None
    configured, chosen, sides, rest = match.groups()
    cost0 = re.search(r" cost0=(\S+)", rest)
    if cost0 is None:
        raise ValueError(f"slice length not counted: {match.group(0)!r}")
    value = float(cost0.group(1))
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid cost0: {match.group(0)!r}")
    return int(configured), int(chosen), sides, value


def parse_r_block(text):
    """(r_bytes, r_huge) of the dense path's packing-buffers line."""
    match = _single(text, _BUFFERS_LINE, "[dense-packing-buffers]")
    if match is None:
        raise ValueError("no [dense-packing-buffers] line")
    return int(match.group(1)), match.group(2) == "1"
