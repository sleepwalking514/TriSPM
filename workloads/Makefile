# ============================================================
# TriSPM Workloads — Build & Run Triton kernels on gem5
#
# Usage:
#   make vector_add          # build one kernel
#   make all                 # build all kernels
#
#   make run-vector_add      # run on gem5 (must build first)
#   make run-all             # run all kernels on gem5
#
#   make clean               # remove all build artifacts
#
# To add a new kernel, create workloads/<name>/ with:
#   kernel.py   — Triton kernel definition
#   harness.c   — C test driver
#   config.sh   — BLOCK_SIZE, SIZE, etc.
# Then add <name> to KERNELS below.
# ============================================================

KERNELS := vector_add matmul layer_norm

.PHONY: all clean run-all $(KERNELS) $(addprefix run-,$(KERNELS))

all: $(KERNELS)

run-all: $(addprefix run-,$(KERNELS))

# Build a kernel
$(KERNELS):
	@./scripts/build_kernel.sh $@

# Run on gem5 (default: cache baseline). Build first with `make <kernel>`.
$(addprefix run-,$(KERNELS)):
	@./scripts/run_gem5.sh $(subst run-,,$@) --cache_baseline

clean:
	rm -rf $(addsuffix /build,$(KERNELS))
	@echo "Cleaned all build directories."
