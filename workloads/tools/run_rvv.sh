#!/bin/bash
# Run a compiled TriSPM RISC-V binary on the real RVV laptop.
#
# Usage:
#   ./tools/run_rvv.sh <kernel> --mode {cache,spm} --tag TAG [--remote rvv]
#
# Build first with TRISPM_REAL_HW=1 for SPM real-hardware runs:
#   TRISPM_REAL_HW=1 TRITON_SPM_SIZE=131072 SPM_SIZE_BYTES=131072 \
#     ./scripts/build_kernel.sh matmul --mode spm --tag ...
#
# Optional environment:
#   RVV_HOST=rvv
#   RVV_CPU=0
#   RVV_DIR=/tmp/trispm-rvv
#   RVV_USE_PERF=1
#   RVV_PERF_EVENTS=cycles,instructions
#   RVV_SSH_OPTS="-o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=1"
#   RVV_SUDO=1
#   RVV_SUDO_PASSWORD=...
#   TRISPM_REAL_HW_DMA=cpu|udma
#   TRISPM_REAL_HW_UDMA_ALLOC_ARGS=0,1
# ============================================================
set -euo pipefail

SPM_SIZE_BYTES_WAS_SET="${SPM_SIZE_BYTES+x}"
TRITON_SPM_SIZE_WAS_SET="${TRITON_SPM_SIZE+x}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS_DIR="$SCRIPT_DIR/../scripts"
source "$SCRIPT_DIR/../env.sh"

parse_size_bytes() {
    local value="$1"
    if ! [[ "$value" =~ ^(0[xX][0-9a-fA-F]+|[0-9]+)$ ]]; then
        return 1
    fi
    printf '%d' "$((value))"
}

KERNEL=""
MODE=""
TAG=""
REMOTE="${RVV_HOST:-rvv}"
REMOTE_DIR="${RVV_DIR:-/tmp/trispm-rvv}"
CPU="${RVV_CPU:-0}"
USE_PERF="${RVV_USE_PERF:-1}"
PERF_EVENTS="${RVV_PERF_EVENTS:-cycles,instructions}"
SSH_OPTS="${RVV_SSH_OPTS:--o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=1}"

while [ $# -gt 0 ]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        --remote) REMOTE="$2"; shift 2 ;;
        --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
        --cpu) CPU="$2"; shift 2 ;;
        --) shift; break ;;
        -*) echo "unknown flag: $1" >&2; exit 2 ;;
        *) if [ -z "$KERNEL" ]; then KERNEL="$1"; shift
           else echo "unexpected arg: $1" >&2; exit 2; fi ;;
    esac
done

[ -n "$KERNEL" ] || { echo "Usage: $0 <kernel> --mode {cache,spm} --tag TAG" >&2; exit 2; }
[ "$MODE" = "spm" ] || [ "$MODE" = "cache" ] || { echo "--mode must be spm or cache" >&2; exit 2; }
[ -n "$TAG" ] || { echo "--tag is required" >&2; exit 2; }

BINARY="$(python3 "$SCRIPTS_DIR/trispm_paths.py" binary "$KERNEL" "$MODE" --tag "$TAG")"
OUT_DIR="$(python3 "$SCRIPTS_DIR/trispm_paths.py" m5out_dir "$KERNEL" "$MODE" --tag "$TAG")"
RUN_LOG="$OUT_DIR/rvv-run.log"
PERF_LOG="$OUT_DIR/rvv-perf.txt"

[ -f "$BINARY" ] || { echo "ERROR: binary not found: $BINARY" >&2; exit 1; }
mkdir -p "$OUT_DIR"

REMOTE_BIN="$REMOTE_DIR/$(basename "$BINARY")"

REMOTE_SPM_SIZE="${SPM_SIZE_BYTES:-${TRITON_SPM_SIZE:-131072}}"
if [ "$MODE" = "spm" ] && [ -z "$SPM_SIZE_BYTES_WAS_SET" ] && [ -z "$TRITON_SPM_SIZE_WAS_SET" ]; then
    REMOTE_SPM_SIZE=131072
fi

if [ "$MODE" = "spm" ]; then
    TCM_GRANULE=131072
    TCM_TOTAL=524288
    if ! REMOTE_SPM_SIZE_DEC="$(parse_size_bytes "$REMOTE_SPM_SIZE")"; then
        echo "ERROR: SPM_SIZE_BYTES must be decimal or hex bytes for real TCM runs: $REMOTE_SPM_SIZE" >&2
        exit 2
    fi
    if (( REMOTE_SPM_SIZE_DEC == 0 ||
          REMOTE_SPM_SIZE_DEC % TCM_GRANULE != 0 ||
          REMOTE_SPM_SIZE_DEC > TCM_TOTAL )); then
        echo "ERROR: real TCM SPM_SIZE_BYTES=$REMOTE_SPM_SIZE_DEC is unsafe; use a nonzero multiple of 131072 and <= 524288" >&2
        exit 2
    fi
    REMOTE_SPM_SIZE="$REMOTE_SPM_SIZE_DEC"
fi

echo "===== Copying to $REMOTE ====="
read -r -a SSH_OPTS_ARR <<<"$SSH_OPTS"
ssh "${SSH_OPTS_ARR[@]}" "$REMOTE" "mkdir -p '$REMOTE_DIR'"
scp "${SSH_OPTS_ARR[@]}" "$BINARY" "$REMOTE:$REMOTE_BIN" >/dev/null

REMOTE_ENV=(
    "SPM_SIZE_BYTES=$REMOTE_SPM_SIZE"
    "TRISPM_REAL_HW_DMA=${TRISPM_REAL_HW_DMA:-cpu}"
)
if [ -n "${TRISPM_REAL_HW_TCM_DEVICE:-}" ]; then
    REMOTE_ENV+=("TRISPM_REAL_HW_TCM_DEVICE=$TRISPM_REAL_HW_TCM_DEVICE")
fi
if [ -n "${TRISPM_REAL_HW_UDMA_DEVICE:-}" ]; then
    REMOTE_ENV+=("TRISPM_REAL_HW_UDMA_DEVICE=$TRISPM_REAL_HW_UDMA_DEVICE")
fi
if [ -n "${TRISPM_REAL_HW_UDMA_ALLOC_ARGS:-}" ]; then
    REMOTE_ENV+=("TRISPM_REAL_HW_UDMA_ALLOC_ARGS=$TRISPM_REAL_HW_UDMA_ALLOC_ARGS")
fi

ENV_CMD=""
for kv in "${REMOTE_ENV[@]}"; do
    ENV_CMD+="$(printf '%q' "$kv") "
done

if [ "$USE_PERF" = "0" ]; then
    RUN_CMD="cd '$REMOTE_DIR' && chmod +x '$REMOTE_BIN' && env $ENV_CMD taskset -c '$CPU' '$REMOTE_BIN'"
else
    RUN_CMD="cd '$REMOTE_DIR' && chmod +x '$REMOTE_BIN' && env $ENV_CMD taskset -c '$CPU' perf stat -e '$PERF_EVENTS' '$REMOTE_BIN'"
fi
if [ "${RVV_SUDO:-0}" = "1" ]; then
    RUN_CMD="sudo -S sh -c $(printf '%q' "$RUN_CMD")"
fi

echo "===== Running on $REMOTE ====="
echo "  binary: $REMOTE_BIN"
echo "  cpu:    $CPU"
echo "  env:    ${REMOTE_ENV[*]}"
echo "  log:    $RUN_LOG"

if [ "${RVV_SUDO:-0}" = "1" ]; then
    if [ -z "${RVV_SUDO_PASSWORD:-}" ]; then
        echo "ERROR: RVV_SUDO=1 requires RVV_SUDO_PASSWORD in the environment" >&2
        exit 1
    fi
    ssh "${SSH_OPTS_ARR[@]}" "$REMOTE" "$RUN_CMD" <<<"$RVV_SUDO_PASSWORD" 2>&1 | tee "$RUN_LOG"
else
    ssh "${SSH_OPTS_ARR[@]}" "$REMOTE" "$RUN_CMD" 2>&1 | tee "$RUN_LOG"
fi

cp "$RUN_LOG" "$PERF_LOG"
echo "RVV run log:  $RUN_LOG"
echo "RVV perf log: $PERF_LOG"
