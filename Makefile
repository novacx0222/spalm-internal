# MKL is needed to build the libraries, not to run the portable tests.
TEST_GOALS := test test-dispatch test-preparation test-thresholds
ifneq ($(filter-out $(TEST_GOALS),$(or $(MAKECMDGOALS),default)),)
ifndef ONEAPI_ROOT
$(error ONEAPI_ROOT is not set. Please 'source set_intel_vars.sh'.)
endif
endif

CC = g++
CFLAGS = -shared -fPIC -mavx -march=native --std=c++20
# GNU OpenMP for the kernels and MKL (libmkl_gnu_thread), as DuckDB uses
# libgomp: MKL's Intel threading layer loaded into such a process resolves
# omp_get_thread_num() to libgomp, so every MKL thread acts as thread 0.
OMP_LIB = -lgomp
MKL_LINK_TOOL = $(ONEAPI_ROOT)/mkl/latest/bin/mkl_link_tool --compiler=gnu_c --openmp=gomp
OPT_LEVEL ?= -O3
SRC_DIR = src
CPU_SRC_DIR = $(SRC_DIR)/cpu
BIN_DIR = bin

SPALM_OBJS = $(BIN_DIR)/spalm.o $(BIN_DIR)/sparse_sparse_matmul.o $(BIN_DIR)/dense_dense_matmul.o $(BIN_DIR)/slice_decision.o $(BIN_DIR)/kernel.o

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/spalm.so: OPT_LEVEL = -O3
$(BIN_DIR)/spalm.so: $(BIN_DIR)/run_baseline.so $(SPALM_OBJS) | $(BIN_DIR)
	$(MKL_LINK_TOOL) $(CC) $(CFLAGS) -g $(OPT_LEVEL) $(SPALM_OBJS) $(OMP_LIB) -o $(BIN_DIR)/spalm.so
spalm: $(BIN_DIR)/spalm.so

debug: OPT_LEVEL = -O0
debug: clean $(BIN_DIR) $(SPALM_OBJS)
	$(MKL_LINK_TOOL) $(CC) $(CFLAGS) -g $(OPT_LEVEL) $(SPALM_OBJS) $(OMP_LIB) -o $(BIN_DIR)/spalm.so
.PHONY: baseline spalm debug clean


$(BIN_DIR)/sparse_sparse_matmul.o: $(CPU_SRC_DIR)/sparse_sparse_matmul.cpp include/ss_mat.h | $(BIN_DIR)
	$(CC) $(CFLAGS) -g $(OPT_LEVEL) -Iinclude -fopenmp -c $< -o $@

$(BIN_DIR)/dense_dense_matmul.o: $(CPU_SRC_DIR)/dense_dense_matmul.cpp include/ss_mat.h | $(BIN_DIR)
	$(CC) $(CFLAGS) -g $(OPT_LEVEL) -Iinclude -fopenmp -c $< -o $@

$(BIN_DIR)/slice_decision.o: $(SRC_DIR)/slice_decision.cpp include/slice_decision.h | $(BIN_DIR)
	$(CC) $(CFLAGS) -g $(OPT_LEVEL) -Iinclude -fopenmp -c $< -o $@

$(BIN_DIR)/kernel.o: $(CPU_SRC_DIR)/kernel.cpp include/ss_mat.h | $(BIN_DIR)
	$(CC) $(CFLAGS) -g $(OPT_LEVEL) -Iinclude -fopenmp -c $< -o $@

$(BIN_DIR)/spalm.o: $(CPU_SRC_DIR)/spalm.cpp include/ss_mat.h | $(BIN_DIR)
	$(ONEAPI_ROOT)/mkl/latest/bin/mkl_link_tool $(CC) $(CFLAGS) -g $(OPT_LEVEL) -Iinclude -fopenmp -c $< -o $@

$(BIN_DIR)/run_baseline.so: $(SRC_DIR)/run_baseline.cpp | $(BIN_DIR)
	$(MKL_LINK_TOOL) $(CC) $(CFLAGS) -g -O3 -Iinclude $(SRC_DIR)/run_baseline.cpp $(OMP_LIB) -o $(BIN_DIR)/run_baseline.so
baseline: $(BIN_DIR)/run_baseline.so


clean:
	rm -f $(BIN_DIR)/*.o $(BIN_DIR)/*.so
	rmdir $(BIN_DIR) 2>/dev/null || true

# Header dependencies, so header-only changes rebuild the objects using them.
$(SPALM_OBJS): include/util.h include/slice_decision.h
$(BIN_DIR)/spalm.o: include/method_dispatch.h include/csr_preparation.h \
	include/cpu/execution_options.h include/cpu/dense_dense_matmul.h \
	include/cpu/sparse_sparse_matmul.h
$(BIN_DIR)/sparse_sparse_matmul.o: include/csr_preparation.h \
	include/cpu/sparse_sparse_matmul.h
$(BIN_DIR)/dense_dense_matmul.o: include/cpu/dense_dense_matmul.h include/cpu/kernel.h
$(BIN_DIR)/run_baseline.so: include/util.h

# Portable tests: no AVX-512, MKL or built libraries; run from the repo root.
# Needs a compiler with OpenMP, e.g. on macOS: make test CXX=g++-16
TEST_CXXFLAGS ?= -std=c++20 -O1 -g -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer
TEST_BIN_DIR ?= /tmp/spalm-tests
PYTHON ?= python3
.PHONY: test test-dispatch test-preparation test-thresholds
test: test-dispatch test-preparation test-thresholds

# Forced methods, backend selection and timing output.
test-dispatch:
	mkdir -p $(TEST_BIN_DIR)
	$(CXX) $(CPPFLAGS) $(TEST_CXXFLAGS) -Iinclude tests/timing_dispatch_test.cpp $(LDFLAGS) -o $(TEST_BIN_DIR)/timing_dispatch_test
	$(CXX) $(CPPFLAGS) $(TEST_CXXFLAGS) -Iinclude tests/slice_timing_test.cpp src/slice_decision.cpp $(LDFLAGS) -o $(TEST_BIN_DIR)/slice_timing_test
	$(CXX) $(CPPFLAGS) $(TEST_CXXFLAGS) -Itests/stubs -Iinclude tests/dispatch_test.cpp src/slice_decision.cpp $(LDFLAGS) -o $(TEST_BIN_DIR)/dispatch_test
	$(TEST_BIN_DIR)/timing_dispatch_test
	$(TEST_BIN_DIR)/slice_timing_test
	$(TEST_BIN_DIR)/dispatch_test

# CSR preparation: memory bounds, validation, zero pruning, ownership.
test-preparation:
	mkdir -p $(TEST_BIN_DIR)
	$(CXX) $(CPPFLAGS) $(TEST_CXXFLAGS) -Iinclude tests/csr_memory_test.cpp src/slice_decision.cpp $(LDFLAGS) -o $(TEST_BIN_DIR)/csr_memory_test
	$(CXX) $(CPPFLAGS) $(TEST_CXXFLAGS) -Itests/stubs -Iinclude tests/preparation_dispatch_test.cpp src/slice_decision.cpp $(LDFLAGS) -o $(TEST_BIN_DIR)/preparation_dispatch_test
	$(CXX) $(CPPFLAGS) $(TEST_CXXFLAGS) -Iinclude tests/key_compaction_test.cpp src/slice_decision.cpp $(LDFLAGS) -o $(TEST_BIN_DIR)/key_compaction_test
	$(TEST_BIN_DIR)/csr_memory_test
	$(TEST_BIN_DIR)/preparation_dispatch_test
	$(TEST_BIN_DIR)/key_compaction_test
	# A smaller OpenMP team than requested must still process every row.
	OMP_THREAD_LIMIT=2 $(TEST_BIN_DIR)/csr_memory_test
	OMP_THREAD_LIMIT=2 $(TEST_BIN_DIR)/preparation_dispatch_test
	OMP_THREAD_LIMIT=2 $(TEST_BIN_DIR)/key_compaction_test

# Timing-label parser of find_thresholds.py (standard library only).
test-thresholds:
	$(PYTHON) -B -m unittest discover -s tests -p 'test_find_thresholds.py' -v
