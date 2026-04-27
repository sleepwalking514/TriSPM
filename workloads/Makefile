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
#   make matmul-stats       # compare named matmul stats files
#   make run-matmul-compare N=256 BS=32
#   make run-matmul-steady-compare N=256 BS=32
#   make run-matmul-size-sweep
#   make run-matmul-steady-size-sweep
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
N ?= 256
BS ?= 32
MATMUL_SWEEP_SIZES ?= 128 256
MATMUL_STEADY_WARMUP ?= 2
MATMUL_STEADY_MEASURE ?= 5
MATMUL_TAG := n$(N)-bs$(BS)
MATMUL_STEADY_TAG := steady-$(MATMUL_TAG)
MATMUL_SINGLE_ENV := MATMUL_M=$(N) MATMUL_N=$(N) MATMUL_K=$(N) \
                     MATMUL_BLOCK_SIZE_M=$(BS) \
                     MATMUL_BLOCK_SIZE_N=$(BS) \
                     MATMUL_BLOCK_SIZE_K=$(BS)

NOSPM_TARGETS := $(addsuffix -nospm,$(KERNELS))
RUN_TARGETS   := $(addprefix run-,$(KERNELS))
RUN_SPM_TARGETS := $(addsuffix -spm,$(RUN_TARGETS))

.PHONY: all clean run-all matmul-stats run-matmul-compare \
        run-matmul-steady-compare run-matmul-size-sweep \
        run-matmul-steady-size-sweep \
        $(KERNELS) $(NOSPM_TARGETS) $(RUN_TARGETS) $(RUN_SPM_TARGETS)

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

matmul-stats:
	@python3 ./scripts/compare_matmul_stats.py

run-matmul-compare:
	@echo "===== matmul cold-cache N=$(N) BS=$(BS) ====="
	@$(MATMUL_SINGLE_ENV) \
	  KERNEL_BUILD_SUFFIX=-$(MATMUL_TAG) STATS_SUFFIX=-$(MATMUL_TAG) \
	  $(MAKE) run-matmul-spm
	@$(MATMUL_SINGLE_ENV) \
	  KERNEL_BUILD_SUFFIX=-$(MATMUL_TAG) STATS_SUFFIX=-$(MATMUL_TAG) \
	  $(MAKE) run-matmul
	@python3 ./scripts/compare_matmul_stats.py \
	  --spm m5out/spm-matmul-$(MATMUL_TAG)-stats.txt \
	  --cache m5out/cache-matmul-$(MATMUL_TAG)-stats.txt

run-matmul-steady-compare:
	@echo "===== matmul steady-state N=$(N) BS=$(BS) ====="
	@$(MATMUL_SINGLE_ENV) \
	  MATMUL_WARMUP_ITERS=$(MATMUL_STEADY_WARMUP) \
	  MATMUL_MEASURE_ITERS=$(MATMUL_STEADY_MEASURE) \
	  MATMUL_FLUSH_BEFORE_ROI=0 \
	  KERNEL_BUILD_SUFFIX=-$(MATMUL_STEADY_TAG) STATS_SUFFIX=-$(MATMUL_STEADY_TAG) \
	  $(MAKE) run-matmul-spm
	@$(MATMUL_SINGLE_ENV) \
	  MATMUL_WARMUP_ITERS=$(MATMUL_STEADY_WARMUP) \
	  MATMUL_MEASURE_ITERS=$(MATMUL_STEADY_MEASURE) \
	  MATMUL_FLUSH_BEFORE_ROI=0 \
	  KERNEL_BUILD_SUFFIX=-$(MATMUL_STEADY_TAG) STATS_SUFFIX=-$(MATMUL_STEADY_TAG) \
	  $(MAKE) run-matmul
	@python3 ./scripts/compare_matmul_stats.py \
	  --spm m5out/spm-matmul-$(MATMUL_STEADY_TAG)-stats.txt \
	  --cache m5out/cache-matmul-$(MATMUL_STEADY_TAG)-stats.txt \
	  --measure-iters $(MATMUL_STEADY_MEASURE)

run-matmul-size-sweep:
	@for size in $(MATMUL_SWEEP_SIZES); do \
	  tag=n$${size}-bs$(BS); \
	  echo "===== matmul cold-cache N=$${size} BS=$(BS) ====="; \
	  MATMUL_M=$$size MATMUL_N=$$size MATMUL_K=$$size \
	    MATMUL_BLOCK_SIZE_M=$(BS) MATMUL_BLOCK_SIZE_N=$(BS) MATMUL_BLOCK_SIZE_K=$(BS) \
	    KERNEL_BUILD_SUFFIX=-$${tag} STATS_SUFFIX=-$${tag} \
	    $(MAKE) run-matmul-spm; \
	  MATMUL_M=$$size MATMUL_N=$$size MATMUL_K=$$size \
	    MATMUL_BLOCK_SIZE_M=$(BS) MATMUL_BLOCK_SIZE_N=$(BS) MATMUL_BLOCK_SIZE_K=$(BS) \
	    KERNEL_BUILD_SUFFIX=-$${tag} STATS_SUFFIX=-$${tag} \
	    $(MAKE) run-matmul; \
	  python3 ./scripts/compare_matmul_stats.py \
	    --spm m5out/spm-matmul-$${tag}-stats.txt \
	    --cache m5out/cache-matmul-$${tag}-stats.txt; \
	done

run-matmul-steady-size-sweep:
	@for size in $(MATMUL_SWEEP_SIZES); do \
	  tag=steady-n$${size}-bs$(BS); \
	  echo "===== matmul steady-state N=$${size} BS=$(BS) ====="; \
	  MATMUL_M=$$size MATMUL_N=$$size MATMUL_K=$$size \
	    MATMUL_BLOCK_SIZE_M=$(BS) MATMUL_BLOCK_SIZE_N=$(BS) MATMUL_BLOCK_SIZE_K=$(BS) \
	    MATMUL_WARMUP_ITERS=$(MATMUL_STEADY_WARMUP) \
	    MATMUL_MEASURE_ITERS=$(MATMUL_STEADY_MEASURE) \
	    MATMUL_FLUSH_BEFORE_ROI=0 \
	    KERNEL_BUILD_SUFFIX=-$${tag} STATS_SUFFIX=-$${tag} \
	    $(MAKE) run-matmul-spm; \
	  MATMUL_M=$$size MATMUL_N=$$size MATMUL_K=$$size \
	    MATMUL_BLOCK_SIZE_M=$(BS) MATMUL_BLOCK_SIZE_N=$(BS) MATMUL_BLOCK_SIZE_K=$(BS) \
	    MATMUL_WARMUP_ITERS=$(MATMUL_STEADY_WARMUP) \
	    MATMUL_MEASURE_ITERS=$(MATMUL_STEADY_MEASURE) \
	    MATMUL_FLUSH_BEFORE_ROI=0 \
	    KERNEL_BUILD_SUFFIX=-$${tag} STATS_SUFFIX=-$${tag} \
	    $(MAKE) run-matmul; \
	  python3 ./scripts/compare_matmul_stats.py \
	    --spm m5out/spm-matmul-$${tag}-stats.txt \
	    --cache m5out/cache-matmul-$${tag}-stats.txt \
	    --measure-iters $(MATMUL_STEADY_MEASURE); \
	done

clean:
	rm -rf $(addsuffix /build*,$(KERNELS)) $(addsuffix /build_nospm*,$(KERNELS))
	@echo "Cleaned all build directories."
