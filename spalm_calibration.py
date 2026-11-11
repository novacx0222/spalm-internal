"""Per-machine defaults measured on synthetic inputs (standard library only).

calibration.txt (parameters/<platform>/<type>/calibration.txt, next to
tile_sizes.txt) holds "key: value" lines that the library takes as the
defaults of asl_min_cost, asl_pack_value_entries, r_block_huge_bytes and
ss_page_huge_entries; a key given to spalm() wins over the file
(slice_decision.h). find_thresholds.py writes the first three,
calibrate_run.py --page-sweep the last. Each writer replaces only its own
lines and keeps the rest of the file.

No SuiteSparse or grid input is involved: every value here comes from
matrices the scripts generate on the spot.
"""

import math
import os
import sys
from pathlib import Path

CALIBRATED_KEYS = ("asl_min_cost", "asl_pack_value_entries",
                   "r_block_huge_bytes", "ss_page_huge_entries")

HEADER = (
    "# Per-machine defaults measured on synthetic inputs; a key given to\n"
    "# spalm() (spalm.conf) wins over this file. Written by find_thresholds.py\n"
    "# (asl_min_cost, asl_pack_value_entries, r_block_huge_bytes) and\n"
    "# calibrate_run.py --page-sweep (ss_page_huge_entries); each replaces\n"
    "# only its own lines.\n"
)

# asl_min_cost when no synthetic input ran faster at the longer length: no
# modelled cost reaches it, so the rule never lengthens.
NEVER_MIN_COST = 1e300

MIB = 1 << 20
# Packing buffers below this are never mapped (packing_buffers.h).
HUGE_BYTES = 2 * MIB


def format_value(value):
    """An int as is; a float as the library parses it (%.6g, 5e+09)."""
    if isinstance(value, bool):
        raise TypeError("not a number")
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        raise ValueError(f"not finite: {value}")
    return f"{value:.6g}"


def _line_key(line):
    """The key of a "key: value" line, or None (comment, blank, other)."""
    body = line.split("#", 1)[0]
    if ":" not in body:
        return None
    key = body.split(":", 1)[0].strip()
    return key or None


def read_calibration(path):
    """{key: value text} of the file, {} if it does not exist."""
    path = Path(path)
    if not path.exists():
        return {}
    values = {}
    for line in path.read_text().splitlines():
        key = _line_key(line)
        if key is not None:
            values[key] = line.split("#", 1)[0].split(":", 1)[1].strip()
    return values


def update_calibration(path, values, note):
    """Set each key of `values` in the file, as "key: value  # note".

    Other lines stay as they are; a key already there is replaced in place
    (a repeated line of it is dropped), a new one is appended. A new file
    starts with HEADER. Written to a temporary file and renamed.
    """
    for key in values:
        if key not in CALIBRATED_KEYS:
            raise ValueError(f"not a calibrated key: {key}")
    if "\n" in note:
        raise ValueError("note must be one line")
    path = Path(path)
    lines = path.read_text().splitlines() if path.exists() else HEADER.splitlines()
    out, done = [], set()
    for line in lines:
        key = _line_key(line)
        if key in values:
            if key not in done:
                out.append(f"{key}: {format_value(values[key])}  # {note}")
                done.add(key)
            continue
        out.append(line)
    for key, value in values.items():
        if key not in done:
            out.append(f"{key}: {format_value(value)}  # {note}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(out) + "\n")
    os.replace(tmp, path)


def choose_min_cost(cells):
    """asl_min_cost from cells (cost0, t_base, t_long).

    cost0 is the rule's modelled cost at the configured length, t_base and
    t_long the measured times at the configured length and at the length the
    rule chose without a gate. The gate lengthens every cell with cost0 at or
    above it. Of the cuts between distinct cost0 values, the one minimising
    the sum of log(t_long / t_base) over the lengthened cells wins (the
    geometric mean over all cells); a tie goes to the higher cut, which
    lengthens less.

    Returns (min_cost, lengthened): the value lies halfway (geometric mean)
    between the last cell kept at the base length and the first lengthened
    one; just below the lowest cost0 when every cell is lengthened (nothing
    below it was measured; LOWEST_CUT_MARGIN); NEVER_MIN_COST when none is. `lengthened` is the set of cell
    indices (into `cells`) the value lengthens.
    """
    if not cells:
        raise ValueError("no cells")
    for cost0, base, long_ in cells:
        if not (cost0 >= 0 and base > 0 and long_ > 0):
            raise ValueError(f"bad cell {(cost0, base, long_)}")
    order = sorted(range(len(cells)), key=lambda i: cells[i][0])
    n = len(order)
    suffix = [0.0] * (n + 1)
    for j in range(n - 1, -1, -1):
        cost0, base, long_ = cells[order[j]]
        suffix[j] = suffix[j + 1] + math.log(long_ / base)
    best = n
    for j in range(n - 1, -1, -1):
        if j > 0 and cells[order[j - 1]][0] == cells[order[j]][0]:
            continue  # no threshold separates equal costs
        if suffix[j] < suffix[best] - 1e-12:
            best = j
    if best == n:
        return NEVER_MIN_COST, set()
    return _cut_value([cells[i][0] for i in order], best), set(order[best:])


# Weights tried for asl_pack_value_entries (multiplied entries per packed
# value): 0 (packing left out, as before v34) and powers of two around the
# range where the synthetic clustered inputs change their decision (0.5 to
# 4).
PACK_WEIGHTS = (0.0, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
# A lengthened cell may be at most this much slower than at the configured
# length (measurement noise); a cut that lengthens a slower one is unsafe.
SAFE_RATIO = 1.05


def rule_level(per_subrow, mult, packed, weight, subrow_write_entries,
               min_cost_ratio):
    """The level slice_length_rule::choose() picks without a gate, from the
    per-length lists of its log line, for another packing weight: double
    while the subrows hold fewer than subrow_write_entries entries and the
    cost (multiplication + weight x packed values) drops to min_cost_ratio
    of the previous length's or below."""
    raw = [m + weight * p for m, p in zip(mult, packed)]
    level = 0
    while (level + 1 < len(raw) and per_subrow[level] < subrow_write_entries
           and raw[level + 1] <= min_cost_ratio * raw[level]):
        level += 1
    return level


# The cut below the lowest cost lies this fraction under it: the library
# logs cost0 and find_thresholds.py writes asl_min_cost with 6 significant
# digits each, so a cut at the logged cost could round to just above the
# exact cost of that cell, which would then not be lengthened.
LOWEST_CUT_MARGIN = 1e-4


def _cut_value(costs, j):
    """asl_min_cost that lengthens sorted costs[j:] and not costs[:j]."""
    if j == len(costs):
        return NEVER_MIN_COST
    if j == 0:
        return costs[0] * (1 - LOWEST_CUT_MARGIN)
    prev, lo = costs[j - 1], costs[j]
    return math.sqrt(prev * lo) if prev > 0 else lo / 2


def choose_pack_and_gate(cells, subrow_write_entries, min_cost_ratio,
                         weights=PACK_WEIGHTS, safe_ratio=SAFE_RATIO):
    """asl_pack_value_entries and asl_min_cost, chosen together.

    cells: dicts with cost0, per_subrow, mult, packed (the rule's per-length
    lists) and times {level: seconds}, measured at level 0 and every level
    up to the longest that any of `weights` picks. A level beyond the
    measured ones is replaced by the longest measured (`capped` counts
    them; find_thresholds.py measures them all).

    For each weight the rule's level per cell is replayed (rule_level), and
    for each gate (a cut between distinct cost0 values, or never) the cells
    at or above it that the rule lengthens are lengthened. A gate is safe
    when none of those is more than safe_ratio slower than at level 0. Of
    the safe (weight, gate) pairs, the lowest sum of log(time / level-0
    time) over all cells wins (the geometric mean); at one weight a tie goes
    to the higher gate. Weights that tie: 0 if it is among them (no evidence
    for packing), otherwise the middle one of the tied weights.

    Returns a dict: weight, min_cost, levels (per cell, 0 if not
    lengthened), and per_weight, one row per weight (weight, min_cost,
    score, lengthened, slower: cells the rule alone would lengthen that are
    more than safe_ratio slower, capped).
    """
    if not cells:
        raise ValueError("no cells")
    for c in cells:
        if not c["times"].get(0, 0) > 0:
            raise ValueError(f"no level-0 time: {c}")
        if any(not t > 0 for t in c["times"].values()):
            raise ValueError(f"bad time: {c}")
    order = sorted(range(len(cells)), key=lambda i: cells[i]["cost0"])
    costs = [cells[i]["cost0"] for i in order]
    rows = []
    for w in weights:
        capped = 0
        log_ratio = []
        for i in order:
            c = cells[i]
            level = rule_level(c["per_subrow"], c["mult"], c["packed"], w,
                               subrow_write_entries, min_cost_ratio)
            top = max(c["times"])
            if level > top:
                capped += 1
                level = top
            log_ratio.append((level, math.log(c["times"][level] /
                                              c["times"][0])))
        slower = sum(1 for level, x in log_ratio
                     if level > 0 and x > math.log(safe_ratio))
        # Scan the cuts from the top: cells j.. are lengthened.
        best_j, best_score = len(order), 0.0
        score, safe = 0.0, True
        for j in range(len(order) - 1, -1, -1):
            level, x = log_ratio[j]
            if level > 0:
                score += x
                safe = safe and x <= math.log(safe_ratio)
            if not safe:
                break
            if j > 0 and costs[j - 1] == costs[j]:
                continue  # no threshold separates equal costs
            if score < best_score - 1e-12:
                best_j, best_score = j, score
        levels = [0] * len(cells)
        for j in range(best_j, len(order)):
            levels[order[j]] = log_ratio[j][0]
        rows.append({"weight": w, "min_cost": _cut_value(costs, best_j),
                     "score": best_score, "levels": levels,
                     "lengthened": sum(1 for x in levels if x > 0),
                     "slower": slower, "capped": capped})
    best = min(r["score"] for r in rows)
    tied = [r for r in rows if r["score"] <= best + 1e-12]
    if any(r["weight"] == 0 for r in tied):
        pick = next(r for r in tied if r["weight"] == 0)
    else:
        pick = tied[(len(tied) - 1) // 2]
    return {"weight": pick["weight"], "min_cost": pick["min_cost"],
            "levels": pick["levels"],
            "per_weight": [{k: v for k, v in r.items() if k != "levels"}
                           for r in rows]}


def fit_line(points):
    """Least-squares (a, b) of y = a + b * x over (x, y) points."""
    n = len(points)
    if n < 2:
        raise ValueError("need two points")
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    sxx = sum((x - mx) ** 2 for x, _ in points)
    if sxx == 0:
        raise ValueError("need two different x")
    b = sum((x - mx) * (y - my) for x, y in points) / sxx
    return my - b * mx, b


def huge_break_even(points):
    """r_block_huge_bytes from (block bytes, t_huge - t_never) points.

    Only blocks of at least 2 MiB can be mapped, so only those count. A line
    delta = a + b * MiB is fitted; from its zero on (rounded up to a whole
    MiB) the mapping saves time. Returns (bytes, (a, b)): 2 MiB when the line
    is at or below zero already at the smallest measured block, -1 (never)
    when it stays above zero up to the largest one.
    """
    pts = [(b / MIB, d) for b, d in points if b >= HUGE_BYTES]
    a, b = fit_line(pts)
    lo = min(x for x, _ in pts)
    hi = max(x for x, _ in pts)
    if a + b * lo <= 0:
        return HUGE_BYTES, (a, b)
    if b >= 0 or a + b * hi > 0:
        return -1, (a, b)
    return max(HUGE_BYTES, int(math.ceil(-a / b)) * MIB), (a, b)


# glibc's mallopt parameter (malloc.h), and the threshold that
# fix_mmap_threshold keeps: glibc's initial one.
M_MMAP_THRESHOLD = -3
FRESH_MMAP_THRESHOLD = 128 * 1024


def fix_mmap_threshold(threshold=FRESH_MMAP_THRESHOLD):
    """Keep glibc's mmap threshold at `threshold` for the rest of the process,
    so that every buffer of that size or more is a mapping of its own.

    glibc raises the threshold, up to 32 MiB, whenever a mapped block is
    freed. In a process that calls spalm() many times, later buffers below
    that size then come from heap memory an earlier call already touched: no
    page faults, nothing to unmap. A DuckDB query calls spalm() once in its
    process and gets fresh mappings: in v26/v27 the R blocks of 2 to 13 MiB
    on 4 KiB pages were unmapped after each call (about 60 us per MiB), and
    huge pages paid from about 8 MiB. Measured with reused memory, the
    mapping looked like it never paid, and longer slices looked cheaper
    than they are. Setting the threshold also turns glibc's adjustment off.

    Returns True when set, False when there is no glibc (not Linux), where
    nothing is changed."""
    if not sys.platform.startswith("linux"):
        return False
    import ctypes
    try:
        mallopt = ctypes.CDLL(None).mallopt
    except AttributeError:
        return False
    mallopt.argtypes = (ctypes.c_int, ctypes.c_int)
    mallopt.restype = ctypes.c_int
    if mallopt(M_MMAP_THRESHOLD, threshold) != 1:
        raise OSError(f"mallopt(M_MMAP_THRESHOLD, {threshold}) failed")
    return True


def median(values):
    s = sorted(values)
    if not s:
        raise ValueError("no values")
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2
