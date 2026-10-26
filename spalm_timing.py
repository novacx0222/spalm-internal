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
