# ============================================================
# TriSPM Workloads — Build & Run Triton kernels on gem5
#
# Usage:
#   make vector_add          # build one kernel
#   make run-vector_add      # build + run on gem5
#   make all                 # build all kernels
#   make run-all             # build + run all kernels
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

# Build + run on gem5 (default: cache baseline)
$(addprefix run-,$(KERNELS)):
	@./scripts/build_kernel.sh $(subst run-,,$@)
	@echo ""
	@./scripts/run_gem5.sh $(subst run-,,$@) --cache_baseline

clean:
	rm -rf $(addsuffix /build,$(KERNELS))
	@echo "Cleaned all build directories."
