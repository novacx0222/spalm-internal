"""Choose machine parameters from measurements (standard library only).

find_thresholds.py measures the native library on the machine that will run
it and uses these functions to pick, from those measurements:

  - result_base_offset_sparse / result_base_offset_dense: where the caller
    starts a huge-page dense result, past the 2 MiB boundary, for the
    sparse-output backends (SS, MKL sparse) and the others.

The result is written to parameters/<platform>/<float|double>/
machine_params.txt, which the library reads (see parse_machine_params in
src/slice_decision.cpp). DEFAULTS are the values used without the file.
"""

import math

DEFAULTS = {
    "result_base_offset_sparse": 16,
    "result_base_offset_dense": 0,
}

KEYS = tuple(DEFAULTS)


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


def format_machine_params(params, comments=()):
    """machine_params.txt text: '# comment' lines, then 'key: value' lines."""
    missing = [k for k in KEYS if k not in params]
    if missing:
        raise ValueError(f"missing machine parameters: {missing}")
    lines = [f"# {c}" for c in comments]
    lines += [f"{k}: {int(params[k])}" for k in KEYS]
    return "\n".join(lines) + "\n"
