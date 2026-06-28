#!/bin/bash
# ============================================================
# Build a Triton kernel for RISC-V.
#
# Usage:  ./scripts/build_kernel.sh <kernel> --mode {spm,cache} [--tag TAG]
# Example:
#   ./scripts/build_kernel.sh matmul --mode spm
#   ./scripts/build_kernel.sh matmul --mode cache --tag n256-bs32
#
# --mode cache: skip ConvertMemoryToSPM pass, build cache-baseline binary.
# --tag:        artifact label rendered from the kernel's experiment.toml
#               tag_template (slashes are flattened to '-' in build dirs).
#               Build dir is workloads/build/<kernel>/<mode>-<flat-tag>/
#               (single source of truth lives in scripts/trispm_paths.py).
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

KERNEL=""
MODE=""
TAG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --tag)  TAG="$2";  shift 2 ;;
        --) shift; break ;;
        -*) echo "unknown flag: $1" >&2; exit 2 ;;
        *)  if [ -z "$KERNEL" ]; then KERNEL="$1"; shift
            else echo "unexpected arg: $1" >&2; exit 2; fi ;;
    esac
done

[ -n "$KERNEL" ] || { echo "Usage: $0 <kernel> --mode {spm,cache} [--tag TAG]" >&2; exit 2; }
[ "$MODE" = "spm" ] || [ "$MODE" = "cache" ] || { echo "--mode must be spm or cache" >&2; exit 2; }
[ -n "$TAG" ] || { echo "--tag is required (driver renders it from experiment.toml)" >&2; exit 2; }

KERNEL_DIR="$TRISPM_ROOT/workloads/kernels/$KERNEL"
[ -d "$KERNEL_DIR" ] || { echo "ERROR: kernel directory not found: $KERNEL_DIR" >&2; exit 1; }

if [ -z "${KERNEL_CFLAGS:-}" ]; then
    echo "ERROR: KERNEL_CFLAGS is not set; build through scripts/run_experiment.py so experiment.toml params are exported." >&2
    exit 1
fi

BUILD_DIR="$(python3 "$SCRIPT_DIR/trispm_paths.py" build_dir "$KERNEL" "$MODE" --tag "$TAG")"

if [ "$MODE" = "cache" ]; then
    export TRITON_DISABLE_SPM=1
    # Triton's compile cache key omits TRITON_DISABLE_SPM, so spm and cache
    # builds collide unless we point them at separate caches.
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR_NOSPM:-$HOME/.triton/cache_nospm}"
    echo "(SPM pass disabled — cache-baseline build)"
else
    export TRITON_ENABLE_SPM_ROW_RESIDENT_REDUCTIONS="${TRITON_ENABLE_SPM_ROW_RESIDENT_REDUCTIONS:-1}"
    export TRITON_ENABLE_SPM_PROMOTION_PROFITABILITY="${TRITON_ENABLE_SPM_PROMOTION_PROFITABILITY:-1}"
    export TRITON_SPM_ROW_RESIDENT_PRODUCER_PASS="${TRITON_SPM_ROW_RESIDENT_PRODUCER_PASS:-auto}"
    export TRITON_SPM_ROW_RESIDENT_MAX_BYTES="${TRITON_SPM_ROW_RESIDENT_MAX_BYTES:-65536}"
    export TRITON_SPM_PROMOTION_REPORT="${TRITON_SPM_PROMOTION_REPORT:-1}"
    export TRITON_SPM_SOFTMAX_INTERNAL_ROW_BLOCK="${TRITON_SPM_SOFTMAX_INTERNAL_ROW_BLOCK:-${SOFTMAX_SPM_INTERNAL_ROW_BLOCK:-1}}"
    export TRITON_SPM_SOFTMAX_ROW_BLOCK="${TRITON_SPM_SOFTMAX_ROW_BLOCK:-${SOFTMAX_SPM_ROW_BLOCK:-2}}"
    export TRITON_SPM_SOFTMAX_ROW_GROUP_BLOCKS="${TRITON_SPM_SOFTMAX_ROW_GROUP_BLOCKS:-${SOFTMAX_SPM_ROW_GROUP_BLOCKS:-8}}"
    export TRITON_SPM_SOFTMAX_CACHE_EXP="${TRITON_SPM_SOFTMAX_CACHE_EXP:-1}"
    export TRITON_SPM_LAYERNORM_INTERNAL_ROW_BLOCK="${TRITON_SPM_LAYERNORM_INTERNAL_ROW_BLOCK:-${SPM_INTERNAL_ROW_BLOCK:-${LAYERNORM_SPM_INTERNAL_ROW_BLOCK:-1}}}"
    export TRITON_SPM_LAYERNORM_ROW_BLOCK="${TRITON_SPM_LAYERNORM_ROW_BLOCK:-${SPM_ROW_BLOCK:-${LAYERNORM_SPM_ROW_BLOCK:-2}}}"
    export TRITON_SPM_LAYERNORM_ROW_GROUP_BLOCKS="${TRITON_SPM_LAYERNORM_ROW_GROUP_BLOCKS:-${SPM_ROW_GROUP_BLOCKS:-${LAYERNORM_SPM_ROW_GROUP_BLOCKS:-8}}}"
    export TRITON_SPM_RMSNORM_INTERNAL_ROW_BLOCK="${TRITON_SPM_RMSNORM_INTERNAL_ROW_BLOCK:-${SPM_INTERNAL_ROW_BLOCK:-${RMSNORM_SPM_INTERNAL_ROW_BLOCK:-1}}}"
    export TRITON_SPM_RMSNORM_ROW_BLOCK="${TRITON_SPM_RMSNORM_ROW_BLOCK:-${SPM_ROW_BLOCK:-${RMSNORM_SPM_ROW_BLOCK:-2}}}"
    export TRITON_SPM_RMSNORM_ROW_GROUP_BLOCKS="${TRITON_SPM_RMSNORM_ROW_GROUP_BLOCKS:-${SPM_ROW_GROUP_BLOCKS:-${RMSNORM_SPM_ROW_GROUP_BLOCKS:-8}}}"

    # The cache key also omits SPM policy env vars, so each policy variant needs
    # a separate compile cache.
    if [ "${TRITON_ENABLE_SPM_ROW_RESIDENT_REDUCTIONS:-0}" = "1" ]; then
        SPM_CACHE_DIR="${TRITON_CACHE_DIR_SPM_ROW_RESIDENT:-$HOME/.triton/cache_spm_row_resident}"
    else
        SPM_CACHE_DIR="${TRITON_CACHE_DIR_SPM_NOREDUCE:-$HOME/.triton/cache_spm_noreduce}"
    fi
    if [ -n "${TRITON_SPM_ROW_RESIDENT_MAX_BYTES:-}" ]; then
        ROW_KEY="$(printf '%s' "$TRITON_SPM_ROW_RESIDENT_MAX_BYTES" | tr -cs '[:alnum:]_.-' '_')"
        SPM_CACHE_DIR="${SPM_CACHE_DIR}_rowbytes_${ROW_KEY}"
    fi
    if [ -n "${TRITON_SPM_ROW_RESIDENT_PRODUCER_PASS:-}" ]; then
        PRODUCER_KEY="$(printf '%s' "$TRITON_SPM_ROW_RESIDENT_PRODUCER_PASS" | tr -cs '[:alnum:]_.-' '_')"
        SPM_CACHE_DIR="${SPM_CACHE_DIR}_producer_${PRODUCER_KEY}"
    fi
    if [ "${TRITON_SPM_SOFTMAX_INTERNAL_ROW_BLOCK:-0}" = "1" ]; then
        SPM_CACHE_DIR="${SPM_CACHE_DIR}_softmax_internal_row_block"
        SOFTMAX_RB_KEY="$(printf '%s' "${TRITON_SPM_SOFTMAX_ROW_BLOCK:-${SOFTMAX_SPM_ROW_BLOCK:-default}}" | tr -cs '[:alnum:]_.-' '_')"
        SOFTMAX_RG_KEY="$(printf '%s' "${TRITON_SPM_SOFTMAX_ROW_GROUP_BLOCKS:-${SOFTMAX_SPM_ROW_GROUP_BLOCKS:-default}}" | tr -cs '[:alnum:]_.-' '_')"
        SPM_CACHE_DIR="${SPM_CACHE_DIR}_rb_${SOFTMAX_RB_KEY}_rg_${SOFTMAX_RG_KEY}"
    fi
    if [ "${TRITON_SPM_SOFTMAX_CACHE_EXP:-0}" = "1" ]; then
        SPM_CACHE_DIR="${SPM_CACHE_DIR}_softmax_cache_exp"
    fi
    if [ "${TRITON_SPM_LAYERNORM_CACHE_CENTERED:-0}" = "1" ]; then
        SPM_CACHE_DIR="${SPM_CACHE_DIR}_layernorm_cache_centered"
    fi
    if [ "${TRITON_SPM_LAYERNORM_INTERNAL_ROW_BLOCK:-0}" = "1" ]; then
        SPM_CACHE_DIR="${SPM_CACHE_DIR}_layernorm_internal_row_block"
        LAYERNORM_RB_KEY="$(printf '%s' "${TRITON_SPM_LAYERNORM_ROW_BLOCK:-${SPM_ROW_BLOCK:-default}}" | tr -cs '[:alnum:]_.-' '_')"
        LAYERNORM_RG_KEY="$(printf '%s' "${TRITON_SPM_LAYERNORM_ROW_GROUP_BLOCKS:-${SPM_ROW_GROUP_BLOCKS:-default}}" | tr -cs '[:alnum:]_.-' '_')"
        SPM_CACHE_DIR="${SPM_CACHE_DIR}_rb_${LAYERNORM_RB_KEY}_rg_${LAYERNORM_RG_KEY}"
    fi
    if [ "${TRITON_SPM_RMSNORM_INTERNAL_ROW_BLOCK:-0}" = "1" ]; then
        SPM_CACHE_DIR="${SPM_CACHE_DIR}_rmsnorm_internal_row_block"
        RMSNORM_RB_KEY="$(printf '%s' "${TRITON_SPM_RMSNORM_ROW_BLOCK:-${SPM_ROW_BLOCK:-default}}" | tr -cs '[:alnum:]_.-' '_')"
        RMSNORM_RG_KEY="$(printf '%s' "${TRITON_SPM_RMSNORM_ROW_GROUP_BLOCKS:-${SPM_ROW_GROUP_BLOCKS:-default}}" | tr -cs '[:alnum:]_.-' '_')"
        SPM_CACHE_DIR="${SPM_CACHE_DIR}_rb_${RMSNORM_RB_KEY}_rg_${RMSNORM_RG_KEY}"
    fi
    if [ -n "${TRITON_SPM_ATTENTION_Q_RESIDENT:-}" ]; then
        ATTN_Q_KEY="$(printf '%s' "$TRITON_SPM_ATTENTION_Q_RESIDENT" | tr -cs '[:alnum:]_.-' '_')"
        SPM_CACHE_DIR="${SPM_CACHE_DIR}_attn_q_${ATTN_Q_KEY}"
    fi
    if [ -n "${TRITON_SPM_ATTENTION_Q_RESIDENT_MIN_TRIPS:-}" ]; then
        ATTN_Q_TRIPS_KEY="$(printf '%s' "$TRITON_SPM_ATTENTION_Q_RESIDENT_MIN_TRIPS" | tr -cs '[:alnum:]_.-' '_')"
        SPM_CACHE_DIR="${SPM_CACHE_DIR}_attn_q_mintrips_${ATTN_Q_TRIPS_KEY}"
    fi
    if [ -n "${TRITON_SPM_ATTENTION_Q_RESIDENT_MIN_USES:-}" ]; then
        ATTN_Q_USES_KEY="$(printf '%s' "$TRITON_SPM_ATTENTION_Q_RESIDENT_MIN_USES" | tr -cs '[:alnum:]_.-' '_')"
        SPM_CACHE_DIR="${SPM_CACHE_DIR}_attn_q_minuses_${ATTN_Q_USES_KEY}"
    fi
    if [ -n "${TRITON_SPM_ATTENTION_Q_RESIDENT_MAX_BYTES:-}" ]; then
        ATTN_Q_BYTES_KEY="$(printf '%s' "$TRITON_SPM_ATTENTION_Q_RESIDENT_MAX_BYTES" | tr -cs '[:alnum:]_.-' '_')"
        SPM_CACHE_DIR="${SPM_CACHE_DIR}_attn_q_maxbytes_${ATTN_Q_BYTES_KEY}"
    fi
    if [ "${TRITON_SPM_ATTENTION_QK_TILE:-0}" = "1" ]; then
        SPM_CACHE_DIR="${SPM_CACHE_DIR}_attn_qk_tile"
    fi
    if [ "${TRITON_SPM_ATTENTION_PV_GENERATED_TILE:-0}" = "1" ]; then
        SPM_CACHE_DIR="${SPM_CACHE_DIR}_attn_pv_generated_tile"
    fi
    if [ "${TRITON_SPM_ATTENTION_KV_STREAM:-0}" = "1" ]; then
        SPM_CACHE_DIR="${SPM_CACHE_DIR}_attn_kv_stream"
    fi
    if [ "${TRITON_SPM_ATTENTION_KV_STREAM_STAGE_Q:-0}" = "1" ]; then
        SPM_CACHE_DIR="${SPM_CACHE_DIR}_attn_kv_stage_q"
    fi
    if [ "${TRITON_SPM_PAGED_KV_DECODE:-1}" != "0" ]; then
        SPM_CACHE_DIR="${SPM_CACHE_DIR}_paged_kv_decode"
    fi
    if [ "${TRITON_SPM_PAGED_KV_DECODE:-1}" != "0" ] && \
       [ "${TRITON_SPM_PAGED_KV_DECODE_DOUBLE_BUFFER:-1}" != "0" ]; then
        SPM_CACHE_DIR="${SPM_CACHE_DIR}_paged_kv_double_buffer"
    fi
    if [ -n "${TRITON_SPM_PAGED_KV_DECODE_MIN_BYTES:-}" ]; then
        PKV_MINB_KEY="$(printf '%s' "$TRITON_SPM_PAGED_KV_DECODE_MIN_BYTES" | tr -cs '[:alnum:]_.-' '_')"
        SPM_CACHE_DIR="${SPM_CACHE_DIR}_paged_kv_minbytes_${PKV_MINB_KEY}"
    fi
    if [ -n "${TRITON_SPM_PAGED_KV_DECODE_MAX_DESCRIPTORS:-}" ]; then
        PKV_MAXD_KEY="$(printf '%s' "$TRITON_SPM_PAGED_KV_DECODE_MAX_DESCRIPTORS" | tr -cs '[:alnum:]_.-' '_')"
        SPM_CACHE_DIR="${SPM_CACHE_DIR}_paged_kv_maxdesc_${PKV_MAXD_KEY}"
    fi
    if [ "${TRITON_USE_XSPM_INSN:-0}" = "1" ]; then
        SPM_CACHE_DIR="${SPM_CACHE_DIR}_xspm_insn"
    fi
    if [ "${TRITON_ENABLE_SPM_PROMOTION_PROFITABILITY:-0}" = "1" ]; then
        SPM_CACHE_DIR="${SPM_CACHE_DIR}_promotion_profitability"
    fi
    GENERIC_AFFINE_MINB_KEY="$(printf '%s' "${TRITON_SPM_GENERIC_AFFINE_TILE_MIN_BYTES:-64}" | tr -cs '[:alnum:]_.-' '_')"
    SPM_CACHE_DIR="${SPM_CACHE_DIR}_generic_affine_tile_minbytes_${GENERIC_AFFINE_MINB_KEY}"
    if [ "${TRITON_SPM_PROMOTION_REPORT:-0}" = "1" ]; then
        SPM_CACHE_DIR="${SPM_CACHE_DIR}_promotion_report"
    fi
    MICRO_M_KEY="$(printf '%s' "${TRITON_MICRO_M:-8}" | tr -cs '[:alnum:]_.-' '_')"
    WINDOW_K_KEY="$(printf '%s' "${TRITON_SPM_WINDOW_K:-8}" | tr -cs '[:alnum:]_.-' '_')"
    SPM_CACHE_DIR="${SPM_CACHE_DIR}_micro_m_${MICRO_M_KEY}_window_k_${WINDOW_K_KEY}"
    CACHE_KEY="$(printf '%s' "$SPM_CACHE_DIR" | sha1sum | awk '{print $1}')"
    export TRITON_CACHE_DIR="${HOME}/.triton/trispm_${CACHE_KEY}"
    export TRISPM_VERBOSE_CACHE_DIR="$SPM_CACHE_DIR"
fi

mkdir -p "$BUILD_DIR"
export KERNEL_AUX_FILE_DIR="$BUILD_DIR"
# AOT sidecars are pass side effects, not cached artifacts.  Force compilation
# so every tag gets matching promotion JSON and launcher allocation code.
export TRITON_ALWAYS_COMPILE=1

echo "===== [1/3] Triton kernel → LLVM IR ====="
python3 "$KERNEL_DIR/kernel.py" 2>"$BUILD_DIR/triton_stderr.log"
if [ ! -f "$BUILD_DIR/${KERNEL}.llir" ]; then
    echo "ERROR: Triton did not produce $BUILD_DIR/${KERNEL}.llir" >&2
    echo "       See $BUILD_DIR/triton_stderr.log" >&2
    exit 1
fi
echo "  → $BUILD_DIR/${KERNEL}.llir"

echo "===== [2/3] LLVM IR → RISC-V assembly ====="
$LLC $LLC_FLAGS "$BUILD_DIR/${KERNEL}.llir" -o "$BUILD_DIR/${KERNEL}.s"
echo "  → $BUILD_DIR/${KERNEL}.s"

echo "===== [3/3] Link with harness → RISC-V binary ====="
XSPM_CFLAG=()
if [ "${TRITON_USE_XSPM_INSN:-0}" = "1" ]; then
    XSPM_CFLAG=(-DUSE_XSPM_INSN)
fi
REAL_HW_CFLAG=()
REAL_HW_SRC=()
if [ "${TRISPM_REAL_HW:-0}" = "1" ]; then
    REAL_HW_CFLAG=(-DTRISPM_REAL_HW -DSPM_BASE="${TRITON_SPM_BASE:-0x40000000}")
    REAL_HW_SRC=("$TRISPM_ROOT/simulator/src/scratchpad_mem/libspm_real_hw.c")
fi
$CLANG $CLANG_FLAGS \
    $KERNEL_CFLAGS \
    "${XSPM_CFLAG[@]}" \
    "${REAL_HW_CFLAG[@]}" \
    -I"$BUILD_DIR" \
    -I"$TRISPM_ROOT/simulator/src/scratchpad_mem" \
    "$BUILD_DIR/${KERNEL}.s" \
    "$BUILD_DIR/${KERNEL}_launcher.c" \
    "$KERNEL_DIR/harness.c" \
    "${REAL_HW_SRC[@]}" \
    -lm \
    -o "$BUILD_DIR/${KERNEL}_test"
echo "  → $BUILD_DIR/${KERNEL}_test"

echo ""
echo "Build complete: $BUILD_DIR/${KERNEL}_test"
