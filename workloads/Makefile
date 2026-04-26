# ============================================================
# TriSPM Workloads — Build & Run Triton kernels on gem5
#
# Build targets:
#   make <kernel>           # SPM-enabled build      → build/
#   make <kernel>-nospm     # SPM-disabled build     → build_nospm/
#   make all                # SPM build for all kernels
#
# Run targets:
#   make run-<kernel>       # cache_baseline (uses build_nospm/)
#   make run-<kernel>-spm   # SPM mode       (uses build/)
#   make run-all            # cache_baseline for all kernels
#
#   make clean              # remove all build directories
#
# To add a new kernel, create workloads/<name>/ with:
#   kernel.py   — Triton kernel definition
#   harness.c   — C test driver
#   config.sh   — BLOCK_SIZE, SIZE, etc.
# Then add <name> to KERNELS below.
# ============================================================

KERNELS := vector_add matmul layer_norm

NOSPM_TARGETS := $(addsuffix -nospm,$(KERNELS))
RUN_TARGETS   := $(addprefix run-,$(KERNELS))
RUN_SPM_TARGETS := $(addsuffix -spm,$(RUN_TARGETS))

.PHONY: all clean run-all $(KERNELS) $(NOSPM_TARGETS) \
        $(RUN_TARGETS) $(RUN_SPM_TARGETS)

all: $(KERNELS)

run-all: $(RUN_TARGETS)

# Build SPM-enabled kernel.
$(KERNELS):
	@./scripts/build_kernel.sh $@

# Build SPM-disabled kernel for cache-baseline runs.
$(NOSPM_TARGETS):
	@./scripts/build_kernel.sh $(@:-nospm=) --no-spm

# Run on gem5 cache_baseline using the no-SPM binary.
$(RUN_TARGETS): run-%: %-nospm
	@./scripts/run_gem5.sh $* --no-spm --cache_baseline

# Run on gem5 SPM mode using the SPM-enabled binary.
$(RUN_SPM_TARGETS): run-%-spm: %
	@./scripts/run_gem5.sh $*

clean:
	rm -rf $(addsuffix /build,$(KERNELS)) $(addsuffix /build_nospm,$(KERNELS))
	@echo "Cleaned all build directories."
