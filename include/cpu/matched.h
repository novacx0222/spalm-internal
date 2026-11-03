#ifndef MATCHED_H
#define MATCHED_H

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>

#include "ss_mat.h"

// The pairs that matched, for spalm_matched: bit j of row i, in matched_words(N)
// words per row, is set when some key has an entry in row i of R and in row j of
// S, whatever the values. A sum of zero is then told apart from a pair without a
// match, which the dense result cannot do. Each kernel marks the pairs that it
// multiplies, as the SS kernel finds them and the dense path by slice pairs.

inline size_t matched_words(int N) {
    return (static_cast<size_t>(N) + 63) / 64;
}

// SS: marks the result columns of one key's entries of S in a row that only this
// thread writes. Columns in one word, which ascending lists give in runs, take
// one OR.
inline void mark_columns(const int *columns, int count, uint64_t *row) {
    int c = 0;
    while (c < count) {
        const int word = columns[c] >> 6;
        uint64_t bits = 0;
        do {
            bits |= uint64_t(1) << (columns[c] & 63);
        } while (++c < count && columns[c] >> 6 == word);
        row[word] |= bits;
    }
}

// Dense path: marks the cells of the slice pairs that a task multiplies, by
// the subrows that share a key of the slice column, whichever kernel runs.
// A worker's buffer holds a mask per key of the slice column: over the R
// subrows of the R slice at hand (rows_of), and over the S subrows of each S
// slice of the task (cols_of). A pair is scanned from its sparse side, as its
// kernel is: DS pairs over the S entries with rows_of, SD and DD pairs over
// the R entries with cols_of; a subrow stops once it matches the whole other
// slice. Tasks of other S slices share the words of a row, so a word is read
// first and ORed atomically only when it lacks bits. The buffer is all zero
// between tasks.
template <typename T> class slice_marker {
  public:
    // Masks (uint32_t) of a worker for tasks of up to s_slices S slices
    static size_t buffer_size(int sl, int s_slices) {
        return static_cast<size_t>(sl) * (s_slices + 1);
    }

    // by_r: the task may have SD or DD pairs; by_s: DS pairs
    slice_marker(
        uint64_t *matched_p, int N, uint32_t *buffer, const ss_matrix<T> &R,
        const ss_matrix<T> &S, int k_offset_p, int S_start_p, int S_end_p,
        bool by_r_p, bool by_s_p
    )
        : matched(matched_p), words(matched_words(N)), R_ssm(R), S_ssm(S),
          sl(R.sl), k_offset(k_offset_p), S_start(S_start_p), S_end(S_end_p),
          by_r(by_r_p), by_s(by_s_p), rows_of(buffer), cols_of(buffer + sl) {
        if (by_r)
            for (int S_si = S_start; S_si < S_end; ++S_si)
                set_masks(S_ssm, S_si, cols_of_slice(S_si), true);
    }

    ~slice_marker() {
        if (by_r)
            for (int S_si = S_start; S_si < S_end; ++S_si)
                set_masks(S_ssm, S_si, cols_of_slice(S_si), false);
    }

    // Before and after the S slices of an R slice
    void load(int R_si) {
        if (by_s) set_masks(R_ssm, R_si, rows_of, true);
    }
    void clear(int R_si) {
        if (by_s) set_masks(R_ssm, R_si, rows_of, false);
    }

    void mark(int R_si, int S_si, bool over_s) {
        const slice &R_s = R_ssm.s_csc_mat.vals[R_si];
        const slice &S_s = S_ssm.s_csc_mat.vals[S_si];
        // hits[t]: the S subrows that R subrow t shares a key with
        uint32_t hits[32];
        if (over_s) {
            std::fill(hits, hits + R_s.len, 0);
            for (int j = 0; j < S_s.len; ++j)
                for (uint32_t h = scan(S_ssm, S_s.start + j, rows_of, R_s.len);
                     h; h &= h - 1)
                    hits[__builtin_ctz(h)] |= uint32_t(1) << j;
        } else {
            const uint32_t *cols = cols_of_slice(S_si);
            for (int t = 0; t < R_s.len; ++t)
                hits[t] = scan(R_ssm, R_s.start + t, cols, S_s.len);
        }
        const int *R_rows = &R_ssm.sr_csc_mat.row_idxs[R_s.start];
        const int *S_rows = &S_ssm.sr_csc_mat.row_idxs[S_s.start];
        for (int t = 0; t < R_s.len; ++t) {
            uint32_t h = hits[t];
            if (!h) continue;
            uint64_t *row = matched + static_cast<size_t>(R_rows[t]) * words;
            if (S_s.continuous) {
                // Subrow j is column S_rows[0] + j: at most two words
                const int shift = S_rows[0] & 63;
                uint64_t *word = row + (S_rows[0] >> 6);
                set_bits(word, uint64_t(h) << shift);
                if (shift) set_bits(word + 1, uint64_t(h) >> (64 - shift));
                continue;
            }
            // Columns ascend with j: the bits of one word go in together
            while (h) {
                int j = __builtin_ctz(h);
                const int word = S_rows[j] >> 6;
                uint64_t bits = 0;
                do {
                    bits |= uint64_t(1) << (S_rows[j] & 63);
                    h &= h - 1;
                } while (h && S_rows[j = __builtin_ctz(h)] >> 6 == word);
                set_bits(row + word, bits);
            }
        }
    }

  private:
    uint32_t *cols_of_slice(int S_si) const {
        return cols_of + static_cast<size_t>(S_si - S_start) * sl;
    }

    // Sets (or clears) bit t of the masks of subrow t's keys
    void set_masks(
        const ss_matrix<T> &ssm, int si, uint32_t *masks, bool set
    ) const {
        const slice &s = ssm.s_csc_mat.vals[si];
        assert(s.len <= 32);
        for (int t = 0; t < s.len; ++t) {
            const subrow &sr = ssm.sr_csc_mat.vals[s.start + t];
            for (int e = sr.start; e < sr.start + sr.len; ++e) {
                uint32_t &mask = masks[ssm.csr_mat.col_idxs[e] - k_offset];
                mask = set ? mask | uint32_t(1) << t : 0;
            }
        }
    }

    // The OR of the masks of a subrow's keys, up to all `others` subrows
    uint32_t scan(
        const ss_matrix<T> &ssm, int sr_idx, const uint32_t *masks, int others
    ) const {
        const uint32_t all = static_cast<uint32_t>((uint64_t(1) << others) - 1);
        const subrow &sr = ssm.sr_csc_mat.vals[sr_idx];
        const int *keys = ssm.csr_mat.col_idxs;
        uint32_t h = 0;
        for (int e = sr.start; e < sr.start + sr.len && h != all; ++e)
            h |= masks[keys[e] - k_offset];
        return h;
    }

    static void set_bits(uint64_t *word, uint64_t bits) {
        if (bits && (__atomic_load_n(word, __ATOMIC_RELAXED) & bits) != bits)
            __atomic_fetch_or(word, bits, __ATOMIC_RELAXED);
    }

    uint64_t *matched;
    size_t words;
    const ss_matrix<T> &R_ssm, &S_ssm;
    int sl, k_offset, S_start, S_end;
    bool by_r, by_s;
    uint32_t *rows_of, *cols_of;
};

#endif
