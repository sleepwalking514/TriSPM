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

KERNELS := vector_add matmul layer_norm

PRESET ?=
SWEEP  ?=
SET    ?=

DRIVER := python3 ./scripts/run_experiment.py

# Translate make-vars to driver flags.
PRESET_FLAG := $(if $(PRESET),--preset $(PRESET),)
SET_FLAG    := $(foreach kv,$(SET),--set $(kv))

BUILD_TARGETS := $(addprefix build-,$(KERNELS))
RUN_TARGETS   := $(addprefix run-,$(KERNELS))
CMP_TARGETS   := $(addprefix cmp-,$(KERNELS))
SWEEP_TARGETS := $(addprefix sweep-,$(KERNELS))

.PHONY: all clean $(BUILD_TARGETS) $(RUN_TARGETS) $(CMP_TARGETS) $(SWEEP_TARGETS)

all: $(BUILD_TARGETS)

# Build both spm + cache binaries; no gem5 run.
$(BUILD_TARGETS): build-%:
	@$(DRIVER) $* --mode build $(PRESET_FLAG) $(SET_FLAG)

# SPM-only run (build + run).
$(RUN_TARGETS): run-%:
	@$(DRIVER) $* --mode spm $(PRESET_FLAG) $(SET_FLAG)

# Full comparison: build + run both modes + delta table.
$(CMP_TARGETS): cmp-%:
	@$(DRIVER) $* --mode compare $(PRESET_FLAG) $(SET_FLAG)

# Manifest-defined sweep, e.g. `make sweep-matmul SWEEP=size PRESET=steady`.
$(SWEEP_TARGETS): sweep-%:
	@test -n "$(SWEEP)" || { echo "SWEEP=<name> required (see kernel's experiment.toml [sweeps.*])"; exit 2; }
	@$(DRIVER) $* --mode compare --sweep $(SWEEP) $(PRESET_FLAG) $(SET_FLAG)

clean:
	rm -rf build
	@echo "Removed workloads/build/."
