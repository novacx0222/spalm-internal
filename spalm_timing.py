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


def _float_list(rest, name, line):
    match = re.search(rf" {name}=(\S+)", rest)
    if match is None:
        raise ValueError(f"no {name}= in {line!r}")
    try:
        values = [float(x) for x in match.group(1).split(",")]
    except ValueError:
        raise ValueError(f"malformed {name}= in {line!r}") from None
    if not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError(f"invalid {name}= in {line!r}")
    return values


def parse_slice_length_levels(text):
    """The counts rule's line with its per-length lists (v34), or None when
    the rule did not run: a dict with configured, chosen, sides, cost0,
    per_subrow, mult and packed (one value per length tried, the configured
    first) and pack_value_entries. A line without them ("(not counted)", or
    a library before v34) raises ValueError."""
    match = _single(text, _SLICE_LENGTH_LINE, "[slice-length]")
    if match is None:
        return None
    configured, chosen, sides, rest = match.groups()
    line = match.group(0)
    if " cost0=" not in rest:
        raise ValueError(f"slice length not counted: {line!r}")
    out = {"configured": int(configured), "chosen": int(chosen),
           "sides": sides,
           "cost0": _float_list(rest, "cost0", line)[0],
           "per_subrow": _float_list(rest, "entries_per_subrow", line),
           "mult": _float_list(rest, "mult", line),
           "packed": _float_list(rest, "packed", line),
           "pack_value_entries":
               _float_list(rest, "pack_value_entries", line)[0]}
    levels = len(out["mult"])
    if len(out["packed"]) != levels or len(out["per_subrow"]) != levels:
        raise ValueError(f"lists of different lengths in {line!r}")
    if out["chosen"] not in [out["configured"] << l for l in range(levels)]:
        raise ValueError(f"chosen is not a length tried: {line!r}")
    return out


_CONF_NUMBER = r"\[conf\] .* {key}=(\S+)"


def parse_conf_number(text, key):
    """A numeric setting of the call's "[conf]" line."""
    matches = re.findall(_CONF_NUMBER.format(key=re.escape(key)), text)
    if len(matches) != 1:
        raise ValueError(f"{key}: {len(matches)} [conf] values")
    return float(matches[0])


def parse_r_block(text):
    """(r_bytes, r_huge) of the dense path's packing-buffers line."""
    match = _single(text, _BUFFERS_LINE, "[dense-packing-buffers]")
    if match is None:
        raise ValueError("no [dense-packing-buffers] line")
    return int(match.group(1)), match.group(2) == "1"
