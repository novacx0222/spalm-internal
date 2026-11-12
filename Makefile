# MKL is needed to build the libraries, not to run the portable tests.
TEST_GOALS := test test-dispatch test-preparation test-sparse-output test-huge-pages test-thresholds test-slice-length test-kernels-native
ifneq ($(filter-out $(TEST_GOALS),$(or $(MAKECMDGOALS),default)),)
ifndef ONEAPI_ROOT
$(error ONEAPI_ROOT is not set. Please 'source set_intel_vars.sh'.)
endif
endif

CC = g++
CFLAGS = -shared -fPIC -mavx -march=native --std=c++20
# OpenMP runtime of the kernels and of MKL; DuckDB must load the same one
# (SPALM_OMP_RUNTIME in duckdb-spalm/CMakeLists.txt).
# - iomp5 (default): MKL's Intel threading layer (libmkl_intel_thread) and
#   libiomp5, which also serves the GOMP_* calls that g++ -fopenmp emits.
# - gomp: libmkl_gnu_thread and libgomp, as in m0-m3.
# Never mix them: with libgomp in DuckDB and libiomp5 here, MKL's
# omp_get_thread_num() resolved to libgomp, every MKL thread acted as
# thread 0, MKL dense ran ~20x slower and MKL sparse crashed (05149e6).
# Objects do not depend on this choice: make clean before switching.
OMP_RUNTIME ?= iomp5
ifeq ($(OMP_RUNTIME),iomp5)
IOMP5_DIR ?= $(ONEAPI_ROOT)/compiler/latest/lib
OMP_LIB = -L$(IOMP5_DIR) -liomp5
MKL_LINK_TOOL = $(ONEAPI_ROOT)/mkl/latest/bin/mkl_link_tool --compiler=gnu_c --openmp=iomp5
else ifeq ($(OMP_RUNTIME),gomp)
OMP_LIB = -lgomp
MKL_LINK_TOOL = $(ONEAPI_ROOT)/mkl/latest/bin/mkl_link_tool --compiler=gnu_c --openmp=gomp
else
$(error OMP_RUNTIME must be iomp5 or gomp, not '$(OMP_RUNTIME)')
endif
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
	include/cpu/sparse_sparse_matmul.h include/spalm_sparse_result.h \
	include/huge_pages.h
$(BIN_DIR)/sparse_sparse_matmul.o: include/csr_preparation.h \
	include/cpu/sparse_sparse_matmul.h include/cpu/ss_sparse_output.h \
	include/spalm_sparse_result.h include/huge_pages.h
$(BIN_DIR)/dense_dense_matmul.o: include/cpu/dense_dense_matmul.h include/cpu/kernel.h \
	include/packing_buffers.h include/huge_pages.h
$(BIN_DIR)/run_baseline.so: include/util.h include/huge_pages.h

# Portable tests: no AVX-512, MKL or built libraries; run from the repo root.
# Needs a compiler with OpenMP, e.g. on macOS: make test CXX=g++-16
TEST_CXXFLAGS ?= -std=c++20 -O1 -g -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer
TEST_BIN_DIR ?= /tmp/spalm-tests
PYTHON ?= python3
.PHONY: test test-dispatch test-preparation test-sparse-output test-huge-pages test-thresholds test-slice-length
test: test-dispatch test-preparation test-sparse-output test-huge-pages test-thresholds test-slice-length

# Forced methods, backend selection and timing output; the
# log_candidate_work counts (candidate_work.h) against brute force.
test-dispatch:
	mkdir -p $(TEST_BIN_DIR)
	$(CXX) $(CPPFLAGS) $(TEST_CXXFLAGS) -Iinclude tests/timing_dispatch_test.cpp $(LDFLAGS) -o $(TEST_BIN_DIR)/timing_dispatch_test
	$(CXX) $(CPPFLAGS) $(TEST_CXXFLAGS) -Iinclude tests/slice_timing_test.cpp src/slice_decision.cpp $(LDFLAGS) -o $(TEST_BIN_DIR)/slice_timing_test
	$(CXX) $(CPPFLAGS) $(TEST_CXXFLAGS) -Itests/stubs -Iinclude tests/dispatch_test.cpp src/slice_decision.cpp $(LDFLAGS) -o $(TEST_BIN_DIR)/dispatch_test
	$(CXX) $(CPPFLAGS) $(TEST_CXXFLAGS) -Iinclude tests/candidate_work_test.cpp src/slice_decision.cpp $(LDFLAGS) -o $(TEST_BIN_DIR)/candidate_work_test
	$(TEST_BIN_DIR)/timing_dispatch_test
	$(TEST_BIN_DIR)/slice_timing_test
	$(TEST_BIN_DIR)/dispatch_test
	$(TEST_BIN_DIR)/candidate_work_test

# Actual AVX-512 kernels of bin/spalm.so (build-libs first) against a scalar
# reference, for every forced SPALM method and execution switch. x86 only.
.PHONY: test-kernels-native
test-kernels-native:
	mkdir -p $(TEST_BIN_DIR)
	$(CXX) -std=c++20 -O1 -g -Iinclude tests/native_kernel_test.cpp -ldl -o $(TEST_BIN_DIR)/native_kernel_test
	$(TEST_BIN_DIR)/native_kernel_test $(BIN_DIR)/spalm.so

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

# Sparse output (spalm_sparse): the SS row loop and its pages with a scalar
# kernel, and the C API for SS and MKL sp2m against the dense result.
test-sparse-output:
	mkdir -p $(TEST_BIN_DIR)
	$(CXX) $(CPPFLAGS) $(TEST_CXXFLAGS) -Itests/stubs -Iinclude tests/sparse_output_test.cpp src/slice_decision.cpp $(LDFLAGS) -o $(TEST_BIN_DIR)/sparse_output_test
	$(TEST_BIN_DIR)/sparse_output_test
	OMP_THREAD_LIMIT=2 $(TEST_BIN_DIR)/sparse_output_test

# Huge-page allocations (huge_pages.h): owned arrays, MKL's allocator and the
# packing buffers of the dense-path kernels (packing_buffers.h).
# On Linux also checks that large blocks are 2 MiB aligned and THP eligible.
test-huge-pages:
	mkdir -p $(TEST_BIN_DIR)
	$(CXX) $(CPPFLAGS) $(TEST_CXXFLAGS) -Iinclude tests/huge_pages_test.cpp $(LDFLAGS) -o $(TEST_BIN_DIR)/huge_pages_test
	$(TEST_BIN_DIR)/huge_pages_test

# Slice length from counts (slice_length_rule.h): counts against a reference
# and the decision on uniform, banded and thinly spread inputs.
test-slice-length:
	mkdir -p $(TEST_BIN_DIR)
	$(CXX) $(CPPFLAGS) $(TEST_CXXFLAGS) -Iinclude tests/slice_length_rule_test.cpp $(LDFLAGS) -o $(TEST_BIN_DIR)/slice_length_rule_test
	$(TEST_BIN_DIR)/slice_length_rule_test
	OMP_THREAD_LIMIT=2 $(TEST_BIN_DIR)/slice_length_rule_test

# Timing-label parser of find_thresholds.py (standard library only).
test-thresholds:
	$(PYTHON) -B -m unittest discover -s tests -p 'test_find_thresholds.py' -v
