#ifndef SLICE_DECISION_H
#define SLICE_DECISION_H

#include <climits>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

enum MM_ALG {
    SPARSE_SPARSE,
    DENSE_DENSE_RIGHT_INNER_KERNEL,
    DENSE_DENSE_LEFT_INNER_KERNEL,
    DENSE_DENSE
};

// Boundary arrays use exclusive upper-bound semantics (compared with <).
// A value of 0 means "never" (no bucket satisfies b < 0).
extern int num_buckets;
extern std::unique_ptr<uint8_t[]> ss_upper;
extern std::unique_ptr<uint8_t[]> ri_upper;
extern int left_dense_threshold;
extern unsigned long long dense_mm_threshold;
extern bool fallback;
extern bool no_features;
// Set by the FORCE_MKL_DENSE / FORCE_MKL_SPARSE init methods.
extern bool force_mkl_dense;
extern bool force_mkl_sparse;
inline constexpr int DEFAULT_BUCKET_LOG_CURVATURE = 50;
inline bool use_log_buckets = false;
// ~10% of buckets fall below 1% density
inline int bucket_log_curvature = DEFAULT_BUCKET_LOG_CURVATURE;

// Parse a bucket method string ("linear_N" or "log_N_cC") and set the globals
// above.  An empty or null string defaults to "linear_100". Every call sets
// both globals, so no setting leaks from a previous call. Returns false for an
// unknown method or a non-positive curvature (log(1 + 0) would divide by 0).
inline bool parse_bucket_method(const char *bucket_method) {
    use_log_buckets = false;
    bucket_log_curvature = DEFAULT_BUCKET_LOG_CURVATURE;
    if (bucket_method == nullptr || bucket_method[0] == '\0' ||
        std::strncmp(bucket_method, "linear", 6) == 0) {
        return true;
    }
    if (std::strncmp(bucket_method, "log", 3) != 0) return false;
    use_log_buckets = true;
    const char *c_pos = std::strstr(bucket_method, "_c");
    if (c_pos != nullptr) bucket_log_curvature = std::atoi(c_pos + 2);
    return bucket_log_curvature > 0;
}

inline int get_bucket(const int nnz, const int slice_ele_num) {
    if (nnz == 0) return 0;
    float density = (float) nnz / slice_ele_num;
    if (use_log_buckets) {
        const float C = (float) bucket_log_curvature;
        float normalized = std::log(1.0f + density * C) / std::log(1.0f + C);
        return (int) std::round(normalized * num_buckets);
    }
    return (int) std::round(density * num_buckets);
}

// Estimated nnz for a slice of `slice_ele_num` elements at bucket index `b`.
inline int bucket_to_nnz(const int b, const int slice_ele_num) {
    float density;
    if (use_log_buckets) {
        const float C = (float) bucket_log_curvature;
        float normalized = (float) b / num_buckets;
        density = (std::exp(normalized * std::log(1.0f + C)) - 1.0f) / C;
    } else {
        density = (float) b / num_buckets;
    }
    int ret = (int) (density * slice_ele_num);
    return ret > 0 ? ret : 1;
}

int get_total_sparse_buckets();

inline int
left_density_to_sparse_bucket(const int nnz, const int slice_ele_num) {
    int bucket_idx = get_bucket(nnz, slice_ele_num);
    return ss_upper[bucket_idx];
}

void set_to_all_dense();

MM_ALG get_alg(const int left_bucket, const int right_bucket);

bool init_decision_thresholds(
    const char *thres_exec_method, const int type, const char *platform
);

void free_bucket_info();

bool read_tile_sizes(
    const int type, const char *platform, int &slice_height, int &slice_length,
    int &tile_size_l2, int &tile_size_l3
);

// Machine-specific parameters, measured by find_thresholds.py on the machine
// that runs the library and saved to
// parameters/<platform>/<float|double>/machine_params.txt (not in git). The
// defaults are the values used before the file existed (inner-v11 and
// outer-v5): without the file, nothing changes.
struct machine_params {
    // Override tile_sizes.txt when > 0.
    int slice_length = 0;
    int tile_size_l2 = 0;
    int tile_size_l3 = 0;
    // Tile sizes of SD, DS and DD (index MACHINE_SD/DS/DD), used instead of
    // the two above when that algorithm is the only dense-path algorithm of
    // a call (a forced method); 0: unset (keys tile_size_l2_sd, ...).
    int method_tile_size_l2[3] = {};
    int method_tile_size_l3[3] = {};
    // Automatic slice length: double the configured length, up to
    // slice_length_max keys, while the sparse side's subrows would hold
    // fewer than slice_target_entries entries on average.
    int slice_target_entries = 16;
    int slice_length_max = 2048;
    // Bytes past a 2 MiB boundary where the caller starts a huge-page dense
    // result, for the sparse-output backends (SS, MKL sparse) and the rest.
    int result_base_offset_sparse = 16;
    int result_base_offset_dense = 0;
    // The file that was read, "" when the defaults are used.
    std::string path;
};

enum { MACHINE_SD = 0, MACHINE_DS = 1, MACHINE_DD = 2 };

// "key: value" lines, '#' comments. A missing file leaves the defaults and
// returns true; an unreadable value, an unknown key or a value out of range
// returns false.
bool parse_machine_params(const std::string &path, machine_params &params);

bool read_machine_params(
    const int type, const char *platform, machine_params &params
);

#endif
