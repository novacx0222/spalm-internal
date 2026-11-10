#pragma once

// Slice length chosen from counts of the inputs (auto_slice_length = 1).
//
// SD multiplies each nonempty subrow of R (one row within one slice column)
// with every packed S slice of that column: per R entry, one packed row per
// S slice, and per R subrow, one read-modify-write of its 16 outputs per S
// slice. DS is the same with the sides swapped (packed R slices, S
// subrows). Per slice column c, with n_S(c) the nonempty S slices:
//
//     SD cost(c) = n_S(c) * (entries_R(c) + kSubrowWriteEntries * subrows_R(c))
//
// Doubling the slice length merges subrows (fewer writes) but also merges
// the packed slices of two columns, so every entry is multiplied with the
// slices of both: when S is spread thinly over the keys (most SuiteSparse
// inputs), n_S(c) nearly doubles and the doubling adds work instead of
// saving it. The rule therefore doubles only while
//   - the sparse side's subrows hold fewer than kSubrowWriteEntries entries
//     on average (beyond that, encoder-1025 measured the configured 512 to be
//     fastest: the saved writes no longer pay for longer packed panels), and
//   - the modelled cost drops by at least 1 - kMinCostRatio.
// DD and mixed tables sum the SD and DS costs. All counts are exact (one
// pass over each CSR).
//
// Two limits come from measurements (v25 SuiteSparse run, 8488C, 30
// threads, 170 matrices at 512 and at the lengths the rules chose; the
// model's costs recomputed exactly). Only these depend on the machine:
//   - No length beyond K (levels_within): a longer panel only pads a single
//     slice column.
//   - The modelled cost at the configured length must reach kMinCost: there,
//     about 250 ms of SD or DS kernel, or 120 ms of DD (whose cost counts
//     both sides). Below it, of the 37 measured cells the
//     rule would have lengthened, 30 got slower (1.40x slice-mm in gmean)
//     and one faster: the packing buffers grow with the length, and on
//     medium graph-like inputs the kernels ran 1.2-1.9x slower where the
//     model predicted 0.55-0.75 (probably the longer packed panels no longer
//     fit in L2). Above it, 11 of 13 got faster and none slower (0.72x).
// That run still had the packing buffers on 4 KiB pages, which cost 2-7 ms
// more per call at 2048 than at 512 (packing_buffers.h). Without that cost
// medium graph-like inputs still ran slower at 2048 (pajek_fa, gnutella05:
// 1.1-1.4x library time), and a few just below kMinCost ran faster
// (priebel_192bit DD and SD: 0.96-0.97x).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace slice_length_rule {

// A subrow write costs about as much as this many multiplied entries, and
// subrows holding at least this many entries are not lengthened.
inline constexpr double kSubrowWriteEntries = 16.0;
// A doubling must cut the modelled cost to at most this fraction.
inline constexpr double kMinCostRatio = 0.9;
// Smallest modelled cost at the configured length that is lengthened at
// all (see the top of this file). In the v25 run the cells above it were
// all at least 3.3e9; the largest below, 2.4e9 (priebel_192bit DD), ran
// the same at 2048 as at 512.
inline constexpr double kMinCost = 3e9;
// Lengths tried: base, 2 * base and 4 * base. Also the capacity of the
// count tables below: params::max_levels may only lower it.
inline constexpr int kMaxLevels = 3;

// The constants above, overridable per call (spalm.conf keys asl_*). The
// defaults are the constants.
struct params {
    double subrow_write_entries = kSubrowWriteEntries;
    double min_cost_ratio = kMinCostRatio;
    double min_cost = kMinCost;
    int max_levels = kMaxLevels; // 1 to kMaxLevels
};

// Levels to count: base << level stays within K (at least one level, at
// most max_levels, clamped to [1, kMaxLevels]).
inline int levels_within(
    const int base, const int K, const int max_levels = kMaxLevels
) {
    const int cap = std::clamp(max_levels, 1, kMaxLevels);
    int levels = 1;
    while (levels < cap && base > 0 &&
           (static_cast<long long>(base) << levels) <= K)
        ++levels;
    return levels;
}

// Per slice column, for slice length base << level: stored entries,
// nonempty subrows and nonempty slices (slice_height rows each).
struct column_counts {
    int levels = 0;
    std::vector<long long> entries[kMaxLevels];
    std::vector<long long> subrows[kMaxLevels];
    std::vector<long long> slices[kMaxLevels];
};

inline int column_count(const int K, const long long length) {
    return static_cast<int>((K + length - 1) / length);
}

// One pass over a CSR with sorted keys in [0, K). Threads take contiguous
// ranges of slices; each counts into its own arrays (bounded to 64 MiB in
// all), which are summed at the end.
inline column_counts count_columns(
    const int *row_ptrs, const int *col_idxs, const int rows, const int K,
    const int base, const int levels, const int slice_height, const int threads
) {
    column_counts out;
    out.levels = std::clamp(levels, 1, kMaxLevels);
    int cols[kMaxLevels] = {};
    int offset[kMaxLevels] = {};
    int total = 0;
    for (int l = 0; l < out.levels; ++l) {
        cols[l] = column_count(K, static_cast<long long>(base) << l);
        offset[l] = total;
        total += cols[l];
        out.entries[l].assign(cols[l], 0);
        out.subrows[l].assign(cols[l], 0);
        out.slices[l].assign(cols[l], 0);
    }
    if (rows <= 0 || total == 0) return out;
    const int slice_count = (rows + slice_height - 1) / slice_height;
    // Per thread: entries, subrows and slices as int (each at most nnz,
    // which fits an int CSR) plus the last slice seen per column.
    const size_t thread_bytes = static_cast<size_t>(total) * 4 * sizeof(int);
    const size_t budget = size_t(64) << 20;
    const int team = static_cast<int>(std::max<size_t>(
        1, std::min<size_t>(
               {static_cast<size_t>(std::max(threads, 1)),
                static_cast<size_t>(slice_count),
                std::max<size_t>(1, budget / thread_bytes)}
           )
    ));
    std::vector<std::vector<int>> local(team);
#pragma omp parallel num_threads(team)
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();
#else
        const int tid = 0;
        const int nt = 1;
#endif
        std::vector<int> &cnt = local[tid];
        cnt.assign(static_cast<size_t>(total) * 3, 0);
        std::vector<int> last_slice(total, -1);
        const int s0 = static_cast<int>((long long) slice_count * tid / nt);
        const int s1 =
            static_cast<int>((long long) slice_count * (tid + 1) / nt);
        for (int s = s0; s < s1; ++s) {
            const int r_end = std::min(rows, (s + 1) * slice_height);
            for (int r = s * slice_height; r < r_end; ++r) {
                int prev[kMaxLevels] = {-1, -1, -1};
                for (int k = row_ptrs[r]; k < row_ptrs[r + 1]; ++k) {
                    const int key = col_idxs[k];
                    // Out-of-range keys are reported by the slicing later.
                    if (key < 0 || key >= K) continue;
                    // floor(key / (base << l)) == floor(key / base) >> l
                    const int c0 = key / base;
                    for (int l = 0; l < out.levels; ++l) {
                        const int c = c0 >> l;
                        const int i = offset[l] + c;
                        ++cnt[i];
                        if (c != prev[l]) {
                            prev[l] = c;
                            ++cnt[total + i];
                        }
                        if (last_slice[i] != s) {
                            last_slice[i] = s;
                            ++cnt[2 * total + i];
                        }
                    }
                }
            }
        }
    }
    for (const auto &cnt : local) {
        if (cnt.empty()) continue;
        for (int l = 0; l < out.levels; ++l) {
            for (int c = 0; c < cols[l]; ++c) {
                const int i = offset[l] + c;
                out.entries[l][c] += cnt[i];
                out.subrows[l][c] += cnt[total + i];
                out.slices[l][c] += cnt[2 * total + i];
            }
        }
    }
    return out;
}

// Modelled cost of multiplying the subrows of `sparse` with the packed
// slices of `packed` at one level (see the top of this file).
inline double side_cost(
    const column_counts &sparse, const column_counts &packed, const int level,
    const double subrow_write_entries = kSubrowWriteEntries
) {
    double cost = 0;
    const auto &e = sparse.entries[level];
    const auto &sr = sparse.subrows[level];
    const auto &sl = packed.slices[level];
    for (size_t c = 0; c < e.size(); ++c)
        cost += static_cast<double>(sl[c]) *
                (static_cast<double>(e[c]) + subrow_write_entries * sr[c]);
    return cost;
}

// An upper bound of the level-0 cost from sizes alone: per sparse-side
// entry, at most one packed row and one subrow write per slice of the other
// side (a subrow holds at least one entry).
inline double cost0_bound(
    const long long r_entries, const int r_rows, const int r_height,
    const long long s_entries, const int s_rows, const int s_height,
    const bool r_sparse, const bool s_sparse,
    const double subrow_write_entries = kSubrowWriteEntries
) {
    const double r_slices = r_height > 0 ? (r_rows + r_height - 1.0) / r_height : 0;
    const double s_slices = s_height > 0 ? (s_rows + s_height - 1.0) / s_height : 0;
    const double per_entry = 1 + subrow_write_entries;
    return (r_sparse ? static_cast<double>(r_entries) * per_entry * std::ceil(s_slices) : 0.0) +
           (s_sparse ? static_cast<double>(s_entries) * per_entry * std::ceil(r_slices) : 0.0);
}

inline double entries_per_subrow(const column_counts &x, const int level) {
    long long e = 0, s = 0;
    for (size_t c = 0; c < x.entries[level].size(); ++c) {
        e += x.entries[level][c];
        s += x.subrows[level][c];
    }
    return s ? static_cast<double>(e) / static_cast<double>(s) : 0.0;
}

struct decision {
    int level = 0;
    double cost0 = 0;                      // modelled cost at level 0
    double cost[kMaxLevels] = {};          // relative to level 0
    double per_subrow[kMaxLevels] = {};    // entries per nonempty subrow
};

// r_sparse: R's subrows are multiplied with packed S (SD, DD); s_sparse:
// S's subrows with packed R (DS, DD). Both count tables need the same
// levels. Below p.min_cost at level 0 nothing is lengthened.
inline decision choose(
    const column_counts &r, const column_counts &s, const bool r_sparse,
    const bool s_sparse, const params &p
) {
    decision d;
    const int levels = std::min(
        {r.levels, s.levels, std::clamp(p.max_levels, 1, kMaxLevels)}
    );
    double raw[kMaxLevels] = {};
    for (int l = 0; l < levels; ++l) {
        raw[l] = (r_sparse ? side_cost(r, s, l, p.subrow_write_entries) : 0.0) +
                 (s_sparse ? side_cost(s, r, l, p.subrow_write_entries) : 0.0);
        double e = 1e300;
        if (r_sparse) e = std::min(e, entries_per_subrow(r, l));
        if (s_sparse) e = std::min(e, entries_per_subrow(s, l));
        d.per_subrow[l] = (r_sparse || s_sparse) ? e : 0.0;
    }
    d.cost0 = raw[0];
    for (int l = 0; l < levels; ++l)
        d.cost[l] = raw[0] > 0 ? raw[l] / raw[0] : 1.0;
    if ((!r_sparse && !s_sparse) || raw[0] < p.min_cost) return d;
    while (d.level + 1 < levels &&
           d.per_subrow[d.level] < p.subrow_write_entries &&
           raw[d.level + 1] <= p.min_cost_ratio * raw[d.level])
        ++d.level;
    return d;
}

// The default constants, with only min_cost given.
inline decision choose(
    const column_counts &r, const column_counts &s, const bool r_sparse,
    const bool s_sparse, const double min_cost = kMinCost
) {
    params p;
    p.min_cost = min_cost;
    return choose(r, s, r_sparse, s_sparse, p);
}

} // namespace slice_length_rule
