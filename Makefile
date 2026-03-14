ifndef ONEAPI_ROOT
$(error ONEAPI_ROOT is not set. Please 'source set_intel_vars.sh'.)
endif

CC = g++
CFLAGS = -shared -fPIC -mavx -march=native --std=c++20
CPU_VENDOR := $(shell grep -m1 'vendor_id' /proc/cpuinfo | awk '{print $$3}')
ifeq ($(CPU_VENDOR),GenuineIntel)
    OMP_LIB = -liomp5
else
    OMP_LIB = -lgomp
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
	$(ONEAPI_ROOT)/mkl/latest/bin/mkl_link_tool $(CC) $(CFLAGS) -g $(OPT_LEVEL) $(SPALM_OBJS) $(OMP_LIB) -o $(BIN_DIR)/spalm.so
spalm: $(BIN_DIR)/spalm.so

debug: OPT_LEVEL = -O0
debug: clean $(BIN_DIR) $(SPALM_OBJS)
	$(ONEAPI_ROOT)/mkl/latest/bin/mkl_link_tool $(CC) $(CFLAGS) -g $(OPT_LEVEL) $(SPALM_OBJS) $(OMP_LIB) -o $(BIN_DIR)/spalm.so
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
	$(ONEAPI_ROOT)/mkl/latest/bin/mkl_link_tool $(CC) $(CFLAGS) -g -O3 -Iinclude $(SRC_DIR)/run_baseline.cpp $(OMP_LIB) -o $(BIN_DIR)/run_baseline.so
baseline: $(BIN_DIR)/run_baseline.so


clean:
	rm -f $(BIN_DIR)/*.o $(BIN_DIR)/*.so
	rmdir $(BIN_DIR) 2>/dev/null || true
