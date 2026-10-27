"""Choose machine parameters from measurements (standard library only).

find_thresholds.py measures the native library on the machine that will run
it and uses these functions to pick, from those measurements:

  - tile_size_l2 / tile_size_l3: the dense-path (SD, DS, DD) tile sizes,
    one pair for all three (calls mixing them) and one per algorithm
    (tile_size_l2_sd, ...: a call whose only dense-path algorithm it is);
  - slice_length: the base slice length (every method);
  - slice_target_entries / slice_length_max: the automatic slice length rule
    for SD, DS and DD (the library doubles the base slice length, up to
    slice_length_max keys and at most K, while the sparse side's subrows
    would hold fewer than slice_target_entries entries on average);
  - result_base_offset_sparse / result_base_offset_dense: where the caller
    starts a huge-page dense result, past the 2 MiB boundary.

The result is written to parameters/<platform>/<float|double>/
machine_params.txt, which the library reads (see parse_machine_params in
src/slice_decision.cpp). DEFAULTS are the values used without the file
(slice_length and the tile sizes from tile_sizes.txt).
"""

import math

DEFAULTS = {
    "slice_length": 512,
    "tile_size_l2": 128,
    "tile_size_l3": 2304,
    "tile_size_l2_sd": 128,
    "tile_size_l3_sd": 2304,
    "tile_size_l2_ds": 128,
    "tile_size_l3_ds": 2304,
    "tile_size_l2_dd": 128,
    "tile_size_l3_dd": 2304,
    "slice_target_entries": 16,
    "slice_length_max": 2048,
    "result_base_offset_sparse": 16,
    "result_base_offset_dense": 0,
}

KEYS = tuple(DEFAULTS)


def auto_slice_length(configured, density, K, target_entries, max_length):
    """The library's automatic slice length (choose_slice_length in spalm.cpp).

    density is the stored entries per key of the sparse side's nonempty rows.
    """
    length = configured
    while (length < max_length and length < K
           and density * min(length, K) < target_entries):
        length *= 2
    return length


def relative_cost(times, best):
    """Geometric mean of times[case] / best[case] over the cases of `times`."""
    logs = [math.log(times[case] / best[case]) for case in times]
    return math.exp(sum(logs) / len(logs))


def best_per_case(table):
    """table: {candidate: {case: seconds}} -> {case: fastest seconds}."""
    best = {}
    for times in table.values():
        for case, t in times.items():
            best[case] = min(t, best.get(case, math.inf))
    return best


def pick_candidate(table, prefer=None, tolerance=0.0):
    """Candidate with the lowest relative cost over all cases.

    Every candidate must have been measured on the same cases. When `prefer`
    is a candidate whose cost is within `tolerance` (relative) of the lowest,
    `prefer` is returned instead: a difference that small is noise. Returns
    (candidate, {candidate: cost}).
    """
    cases = None
    for candidate, times in table.items():
        if cases is None:
            cases = set(times)
        elif set(times) != cases:
            raise ValueError(f"{candidate!r} was measured on other cases")
    if not cases:
        raise ValueError("no measurements")
    best = best_per_case(table)
    costs = {c: relative_cost(times, best) for c, times in table.items()}
    chosen = min(costs, key=lambda c: (costs[c], repr(c)))
    if prefer in costs and costs[prefer] <= costs[chosen] * (1 + tolerance):
        chosen = prefer
    return chosen, costs


def fit_slice_rule(times, K, bases, targets, maxima):
    """Fit (slice_length, slice_target_entries, slice_length_max).

    times: {case: (density, {length: seconds}, automatic)}, where a case is
    any key (e.g. (method, density)), density is the sparse side's entries
    per key, and automatic tells whether the method follows the automatic
    rule (SD, DS, DD) or always runs at the base length (SS). Each candidate
    picks one length per case; the candidate whose choices are, as a
    geometric mean, closest to each case's fastest measured length wins.
    Candidates with max < base, or that would pick an unmeasured length, are
    skipped. Ties go to the smaller base, target, then maximum.
    Returns ((base, target, max), {candidate: cost}).
    """
    costs = {}
    for base in sorted(bases):
        for target in sorted(targets):
            for max_length in sorted(maxima):
                if max_length < base:
                    continue
                logs = []
                for density, measured, automatic in times.values():
                    length = (auto_slice_length(base, density, K, target,
                                                max_length)
                              if automatic else base)
                    if length not in measured:
                        break
                    logs.append(
                        math.log(measured[length] / min(measured.values())))
                else:
                    if logs:
                        costs[(base, target, max_length)] = math.exp(
                            sum(logs) / len(logs))
    if not costs:
        raise ValueError("no candidate rule chooses only measured lengths")
    chosen = min(costs, key=lambda c: (costs[c], c))
    return chosen, costs


def format_machine_params(params, comments=()):
    """machine_params.txt text: '# comment' lines, then 'key: value' lines."""
    missing = [k for k in KEYS if k not in params]
    if missing:
        raise ValueError(f"missing machine parameters: {missing}")
    lines = [f"# {c}" for c in comments]
    lines += [f"{k}: {int(params[k])}" for k in KEYS]
    return "\n".join(lines) + "\n"
