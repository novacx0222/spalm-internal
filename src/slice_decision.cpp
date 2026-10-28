#include "slice_decision.h"

#include <stdexcept>

#include "path_utils.h"

int num_buckets;
std::unique_ptr<uint8_t[]> ss_upper;
std::unique_ptr<uint8_t[]> ri_upper;
int left_dense_threshold;
unsigned long long dense_mm_threshold;
bool fallback = true;
bool no_features = false;
bool force_mkl_dense = false;
bool force_mkl_sparse = false;

int get_total_sparse_buckets() { return ss_upper[0]; }

void set_to_all_dense() {
    std::fill(ss_upper.get(), ss_upper.get() + num_buckets + 1, (uint8_t) 0);
}

MM_ALG get_alg(const int left_bucket, const int right_bucket) {
    if (right_bucket < ss_upper[left_bucket]) return SPARSE_SPARSE;
    if (right_bucket < ri_upper[left_bucket])
        return DENSE_DENSE_RIGHT_INNER_KERNEL;
    if (left_bucket >= left_dense_threshold) return DENSE_DENSE;
    return DENSE_DENSE_LEFT_INNER_KERNEL;
}

static constexpr int DEFAULT_NUM_BUCKETS = 100;

static void init_arrays(int nb) {
    num_buckets = nb;
    ss_upper = std::make_unique_for_overwrite<uint8_t[]>(num_buckets + 1);
    ri_upper = std::make_unique_for_overwrite<uint8_t[]>(num_buckets + 1);
}

bool init_decision_thresholds(
    const char *thres_exec_method, const int type, const char *platform
) {
    // Reset before the early-return modes too; otherwise a previous
    // "no_features" or forced-MKL call leaks into a later call.
    no_features = false;
    force_mkl_dense = false;
    force_mkl_sparse = false;
    // Forced MKL backends skip slicing and the cost model, so they need no
    // calibrated thresholds; the arrays are only filled to stay well defined.
    const bool force_dense = strcmp(thres_exec_method, "FORCE_MKL_DENSE") == 0;
    if (force_dense || strcmp(thres_exec_method, "FORCE_MKL_SPARSE") == 0) {
        init_arrays(DEFAULT_NUM_BUCKETS);
        std::fill(
            ss_upper.get(), ss_upper.get() + num_buckets + 1, (uint8_t) 0
        );
        std::fill(
            ri_upper.get(), ri_upper.get() + num_buckets + 1, (uint8_t) 0
        );
        left_dense_threshold = 0;
        dense_mm_threshold = 0;
        fallback = true;
        force_mkl_dense = force_dense;
        force_mkl_sparse = !force_dense;
        return true;
    }
    if (strcmp(thres_exec_method, "sparse") == 0) {
        init_arrays(DEFAULT_NUM_BUCKETS);
        std::fill(
            ss_upper.get(), ss_upper.get() + num_buckets + 1,
            (uint8_t) (num_buckets + 1)
        );
        std::fill(
            ri_upper.get(), ri_upper.get() + num_buckets + 1, (uint8_t) 0
        );
        left_dense_threshold = 0;
        dense_mm_threshold = 0;
        fallback = false;
        return true;
    } else if (strcmp(thres_exec_method, "dense_bm_inner_kernel") == 0) {
        init_arrays(DEFAULT_NUM_BUCKETS);
        std::fill(
            ss_upper.get(), ss_upper.get() + num_buckets + 1, (uint8_t) 0
        );
        std::fill(
            ri_upper.get(), ri_upper.get() + num_buckets + 1, (uint8_t) 0
        );
        left_dense_threshold = 0;
        dense_mm_threshold = 0;
        fallback = false;
        return true;
    } else if (
        strcmp(thres_exec_method, "dense_right_spmm_inner_kernel") == 0
    ) {
        init_arrays(DEFAULT_NUM_BUCKETS);
        std::fill(
            ss_upper.get(), ss_upper.get() + num_buckets + 1, (uint8_t) 0
        );
        std::fill(
            ri_upper.get(), ri_upper.get() + num_buckets + 1,
            (uint8_t) (num_buckets + 1)
        );
        left_dense_threshold = num_buckets + 1;
        dense_mm_threshold = 0;
        fallback = false;
        return true;
    } else if (strcmp(thres_exec_method, "dense_left_spmm_inner_kernel") == 0) {
        init_arrays(DEFAULT_NUM_BUCKETS);
        std::fill(
            ss_upper.get(), ss_upper.get() + num_buckets + 1, (uint8_t) 0
        );
        std::fill(
            ri_upper.get(), ri_upper.get() + num_buckets + 1, (uint8_t) 0
        );
        left_dense_threshold = num_buckets + 1;
        dense_mm_threshold = 0;
        fallback = false;
        return true;
    }

    const std::string params_base = resolve_path_near_lib(
        reinterpret_cast<void *>(&init_decision_thresholds), "../parameters",
        "parameters"
    );
    const std::string type_dir = (type == 0) ? "/float" : "/double";
    const std::string bucket_dir =
        use_log_buckets ? ("log_" + std::to_string(DEFAULT_NUM_BUCKETS) + "_c" +
                           std::to_string(bucket_log_curvature))
                        : ("linear_" + std::to_string(DEFAULT_NUM_BUCKETS));
    const std::string thres_path = params_base + "/" + platform + type_dir +
                                   "/" + bucket_dir +
                                   "/decision_thresholds.txt";
    std::ifstream inputFile(thres_path);
    if (inputFile.is_open()) {
        std::cout << "Loaded thresholds from: " << bucket_dir << std::endl;
    }

    if (!inputFile.is_open()) {
        std::cerr << "Error: decision threshold file not found: " << thres_path
                  << "\n  (use_log_buckets=" << use_log_buckets
                  << ", bucket_log_curvature=" << bucket_log_curvature
                  << ", num_buckets=" << DEFAULT_NUM_BUCKETS << ")\n";
        return false;
    }

    // find_thresholds.py writes num_buckets first, then num_buckets + 1
    // values for each array. Reject anything else instead of writing past the
    // arrays or leaving entries uninitialized. Bucket indices and thresholds
    // up to num_buckets + 1 are stored as uint8_t, hence the 254 limit.
    auto parse_error = [&](const char *what) {
        std::cerr << "Error: invalid decision threshold file " << thres_path
                  << ": " << what << std::endl;
        return false;
    };
    bool have_num_buckets = false, have_left = false, have_dense = false;
    int ss_count = -1, ri_count = -1;
    std::string line;
    while (std::getline(inputFile, line)) {
        size_t colon_pos = line.find(':');
        if (colon_pos == std::string::npos || colon_pos + 1 >= line.length()) {
            continue;
        }
        std::string key = line.substr(0, colon_pos);
        std::string value_str = line.substr(colon_pos + 1);
        std::istringstream iss(value_str);

        if (key == "num_buckets") {
            int nb;
            if (have_num_buckets || !(iss >> nb) || nb < 1 || nb > 254) {
                return parse_error("num_buckets must appear once, in [1, 254]");
            }
            init_arrays(nb);
            have_num_buckets = true;
        } else if (key == "ss_upper" || key == "ri_upper") {
            if (!have_num_buckets) {
                return parse_error("num_buckets must precede the arrays");
            }
            const bool ss = key == "ss_upper";
            uint8_t *values = ss ? ss_upper.get() : ri_upper.get();
            int val, cnt = 0;
            while (iss >> val) {
                if (cnt > num_buckets || val < 0 || val > num_buckets + 1) {
                    return parse_error("threshold array value out of range");
                }
                values[cnt++] = static_cast<uint8_t>(val);
            }
            (ss ? ss_count : ri_count) = cnt;
        } else if (key == "left_dense_threshold") {
            have_left = static_cast<bool>(iss >> left_dense_threshold);
        } else if (key == "dense_mm_threshold") {
            have_dense = static_cast<bool>(iss >> dense_mm_threshold);
        }
    }
    inputFile.close();
    if (!have_num_buckets || ss_count != num_buckets + 1 ||
        ri_count != num_buckets + 1 || !have_left || !have_dense) {
        return parse_error(
            "expected num_buckets, ss_upper and ri_upper (num_buckets + 1 "
            "values each), left_dense_threshold and dense_mm_threshold"
        );
    }
    fallback = true;
    no_features = false;

    if (strcmp(thres_exec_method, "dense") == 0) {
        dense_mm_threshold = 0;
        fallback = false;
        set_to_all_dense();
    } else if (strcmp(thres_exec_method, "no_fallback") == 0) {
        fallback = false;
    } else if (strcmp(thres_exec_method, "no_threshold") == 0) {
        dense_mm_threshold = 0;
    } else if (strcmp(thres_exec_method, "no_features") == 0) {
        no_features = true;
    }

    return true;
}

void free_bucket_info() {
    ss_upper.reset();
    ri_upper.reset();
}

bool read_tile_sizes(
    const int type, const char *platform, int &slice_height, int &slice_length,
    int &tile_size_l2, int &tile_size_l3
) {
    const std::string params_base = resolve_path_near_lib(
        reinterpret_cast<void *>(&read_tile_sizes), "../parameters",
        "parameters"
    );
    const std::string type_dir = (type == 0) ? "/float" : "/double";
    const std::string tile_path =
        params_base + "/" + platform + type_dir + "/tile_sizes.txt";
    std::ifstream file(tile_path);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open tile_sizes.txt." << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        size_t colon_pos = line.find(':');

        if (colon_pos != std::string::npos && colon_pos + 1 < line.length()) {
            std::string key = line.substr(0, colon_pos);
            std::string value_str = line.substr(colon_pos + 1);

            std::stringstream ss(value_str);
            ss >> value_str;

            try {
                if (key == "slice_length") {
                    slice_length = std::stoi(value_str);
                } else if (key == "tile_size_l2") {
                    tile_size_l2 = std::stoi(value_str);
                } else if (key == "tile_size_l3") {
                    tile_size_l3 = std::stoi(value_str);
                } else if (key == "slice_height") {
                    slice_height = std::stoi(value_str);
                }
            } catch (const std::logic_error &e) {
                // std::stoi throws invalid_argument or out_of_range.
                std::cerr << "Invalid number format for key '" << key
                          << "': " << value_str << std::endl;
                return false;
            }
        }
    }

    file.close();
    return true;
}

bool parse_machine_params(const std::string &path, machine_params &params) {
    std::ifstream file(path);
    if (!file.is_open()) return true;
    auto error = [&](const std::string &what) {
        std::cerr << "Error: invalid machine parameter file " << path << ": "
                  << what << std::endl;
        return false;
    };
    std::string line;
    while (std::getline(file, line)) {
        const size_t start = line.find_first_not_of(" \t\r");
        if (start == std::string::npos || line[start] == '#') continue;
        const size_t colon = line.find(':');
        if (colon == std::string::npos) return error("expected key: value");
        std::string key = line.substr(start, colon - start);
        key.erase(key.find_last_not_of(" \t") + 1);
        std::istringstream iss(line.substr(colon + 1));
        long long value;
        std::string rest;
        if (!(iss >> value) || (iss >> rest)) {
            return error("'" + key + "' needs one integer");
        }
        if (key == "result_base_offset_sparse" ||
            key == "result_base_offset_dense") {
            // The caller's allocation accepts offsets below 4096 that are a
            // multiple of the value size; 8 covers float and double.
            if (value < 0 || value >= 4096 || value % 8 != 0) {
                return error(key + " must be a multiple of 8 in [0, 4096)");
            }
            (key == "result_base_offset_sparse"
                 ? params.result_base_offset_sparse
                 : params.result_base_offset_dense) = static_cast<int>(value);
        } else {
            params.ignored += (params.ignored.empty() ? "" : " ") + key;
        }
    }
    params.path = path;
    return true;
}

bool read_machine_params(
    const int type, const char *platform, machine_params &params
) {
    const std::string params_base = resolve_path_near_lib(
        reinterpret_cast<void *>(&read_machine_params), "../parameters",
        "parameters"
    );
    const std::string type_dir = (type == 0) ? "/float" : "/double";
    return parse_machine_params(
        params_base + "/" + platform + type_dir + "/machine_params.txt", params
    );
}
