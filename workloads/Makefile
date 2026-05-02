# ============================================================
# TriSPM Workloads — thin wrappers around scripts/run_experiment.py.
#
# All real logic (manifest loading, env export, sweeps, presets) lives
# in scripts/run_experiment.py. Build/run paths are defined in
# scripts/trispm_paths.py.
#
# Common targets:
#   make build-<kernel>         build SPM + cache binaries (default tag)
#   make run-<kernel>           SPM run only
#   make cmp-<kernel>           SPM + cache + saved compare table
#   make cmp-<kernel> PRESET=steady
#   make cmp-<kernel> SET="CHECK_RESULT=0"
#   make cmp-matmul SET="N=512 M=512 K=512 BLOCK_SIZE_M=64"
#   make verify-<kernel>          check LLIR for SPM markers + tier JSON
#   make verify-graph GRAPH=layer_norm_qkv
#   make verify                   verify all kernels (continues past failures)
#   make sweep-matmul SWEEP=size [PRESET=steady] [SET="CHECK_RESULT=0"]
#
# Compare output:
#   workloads/m5out/<kernel>/<tag>/compare.txt   SPM-vs-cache delta
#   workloads/m5out/<kernel>/<tag>/spm_stats.txt SPM-only signals (DMA, banks)
#
# <tag> is rendered from each kernel's experiment.toml [kernel].tag_template.
# Slashes in the template become subdirectories — matmul uses
# "{M}x{N}x{K}/{bsM}x{bsN}x{bsK}" so its tree is shape/tile/. Build dirs
# stay flat (slashes collapse to '-').
#
# matmul sweep=size:
#   values come from matmul/experiment.toml [sweeps.size].
#   It mirrors N into M and K, then tags runs as <MxNxK>/<bsMxbsNxbsK>.
#   With PRESET=steady, tags are prefixed as steady-<MxNxK>/<bsMxbsNxbsK>.
#
#   make all                    build all kernels (spm + cache)
#   make clean                  rm -rf workloads/build
# ============================================================

KERNELS := vector_add matmul layer_norm activation residual_add softmax

PRESET ?=
SWEEP  ?=
SET    ?=
GRAPH  ?= layer_norm_qkv
ENV    ?=
EXPECT_SPM ?=
EXPECT_TIER_JSON ?=
EXPECT_PROMOTION_SOURCE ?=

DRIVER := python3 ./scripts/run_experiment.py
GRAPH_DRIVER := python3 ./scripts/graph_placement.py

# Translate make-vars to driver flags.
PRESET_FLAG := $(if $(PRESET),--preset $(PRESET),)
SET_FLAG    := $(foreach kv,$(SET),--set $(kv))
ENV_FLAG    := $(foreach kv,$(ENV),--env $(kv))
EXPECT_SPM_FLAG := $(if $(EXPECT_SPM),--expect-spm $(EXPECT_SPM),)
EXPECT_TIER_JSON_FLAG := $(if $(EXPECT_TIER_JSON),--expect-tier-json $(EXPECT_TIER_JSON),)
EXPECT_PROMOTION_SOURCE_FLAG := $(if $(EXPECT_PROMOTION_SOURCE),--expect-promotion-source "$(EXPECT_PROMOTION_SOURCE)",)
EXTRA_FLAGS := $(ENV_FLAG) $(EXPECT_SPM_FLAG) $(EXPECT_TIER_JSON_FLAG) $(EXPECT_PROMOTION_SOURCE_FLAG)

BUILD_TARGETS  := $(addprefix build-,$(KERNELS))
RUN_TARGETS    := $(addprefix run-,$(KERNELS))
CMP_TARGETS    := $(addprefix cmp-,$(KERNELS))
SWEEP_TARGETS  := $(addprefix sweep-,$(KERNELS))
VERIFY_TARGETS := $(addprefix verify-,$(KERNELS))

.PHONY: all clean verify plan-graph build-graph verify-graph $(BUILD_TARGETS) $(RUN_TARGETS) $(CMP_TARGETS) $(SWEEP_TARGETS) $(VERIFY_TARGETS)

all: $(BUILD_TARGETS)

# Build both spm + cache binaries; no gem5 run.
$(BUILD_TARGETS): build-%:
	@$(DRIVER) $* --mode build $(PRESET_FLAG) $(SET_FLAG) $(EXTRA_FLAGS)

# SPM-only run (build + run).
$(RUN_TARGETS): run-%:
	@$(DRIVER) $* --mode spm $(PRESET_FLAG) $(SET_FLAG) $(EXTRA_FLAGS)

# Full comparison: build + run both modes + delta table.
$(CMP_TARGETS): cmp-%:
	@$(DRIVER) $* --mode compare $(PRESET_FLAG) $(SET_FLAG) $(EXTRA_FLAGS)

# Manifest-defined sweep, e.g. `make sweep-matmul SWEEP=size PRESET=steady`.
$(SWEEP_TARGETS): sweep-%:
	@test -n "$(SWEEP)" || { echo "SWEEP=<name> required (see kernel's experiment.toml [sweeps.*])"; exit 2; }
	@$(DRIVER) $* --mode compare --sweep $(SWEEP) $(PRESET_FLAG) $(SET_FLAG) $(EXTRA_FLAGS)

# Verify SPM markers in LLIR + tier JSON + launcher (build-only, no gem5).
$(VERIFY_TARGETS): verify-%:
	@$(DRIVER) $* --mode verify $(PRESET_FLAG) $(SET_FLAG) $(EXTRA_FLAGS)

plan-graph:
	@$(GRAPH_DRIVER) $(GRAPH) --mode plan

build-graph:
	@$(GRAPH_DRIVER) $(GRAPH) --mode build

verify-graph:
	@$(GRAPH_DRIVER) $(GRAPH) --mode verify

# Verify all kernels (continues past individual failures, reports at end).
verify:
	@fail=0; \
	for k in $(KERNELS); do \
		$(DRIVER) $$k --mode verify $(PRESET_FLAG) $(SET_FLAG) $(EXTRA_FLAGS) || fail=$$((fail + 1)); \
	done; \
	echo ""; \
	if [ $$fail -gt 0 ]; then \
		echo "$$fail kernel(s) failed verify."; exit 1; \
	else \
		echo "All kernels passed verify."; \
	fi

clean:
	rm -rf build
	@echo "Removed workloads/build/."
