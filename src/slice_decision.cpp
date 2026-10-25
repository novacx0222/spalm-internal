#include "slice_decision.h"

#include "path_utils.h"

int num_buckets;
std::unique_ptr<uint8_t[]> ss_upper;
std::unique_ptr<uint8_t[]> ri_upper;
int left_dense_threshold;
unsigned long long dense_mm_threshold;
bool fallback = true;
bool no_features = false;

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
            iss >> nb;
            init_arrays(nb);
        } else if (key == "ss_upper") {
            int val, cnt = 0;
            while (iss >> val) {
                ss_upper[cnt++] = static_cast<uint8_t>(val);
            }
        } else if (key == "ri_upper") {
            int val, cnt = 0;
            while (iss >> val) {
                ri_upper[cnt++] = static_cast<uint8_t>(val);
            }
        } else if (key == "left_dense_threshold") {
            iss >> left_dense_threshold;
        } else if (key == "dense_mm_threshold") {
            iss >> dense_mm_threshold;
        }
    }
    inputFile.close();
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
            } catch (const std::invalid_argument &e) {
                std::cerr << "Invalid number format for key '" << key
                          << "': " << value_str << std::endl;
            }
        }
    }

    file.close();
    return true;
}
