#!/bin/bash
# Run Phase 3.5 baseline suites.
#
# Usage:
#   ./scripts/phase35_baseline.sh verify   # build-only policy checks
#   ./scripts/phase35_baseline.sh smoke    # short gem5 compares
#   ./scripts/phase35_baseline.sh full     # full P0 baseline set
#
# Outputs:
#   workloads/m5out/<kernel>/<tag>/compare.txt
#   workloads/m5out/<kernel>/<tag>/compare.csv
#   workloads/m5out/<kernel>/<tag>/spm_stats.txt
#   workloads/m5out/<kernel>/<tag>/spm_stats.csv
#   workloads/m5out/<kernel>/<tag>/artifacts.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKLOADS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DRIVER="$SCRIPT_DIR/run_experiment.py"

suite="${1:-full}"

run_cmd() {
    echo
    echo "===== $* ====="
    "$@"
}

run_verify() {
    run_cmd make -C "$WORKLOADS_DIR" verify-layer_norm
    run_cmd make -C "$WORKLOADS_DIR" verify-softmax
    run_cmd python3 "$DRIVER" softmax --mode verify --preset phase35-large-row
    run_cmd python3 "$DRIVER" softmax --mode verify \
        --preset phase35-row-resident-large-row \
        --expect-spm false \
        --expect-tier-json empty \
        --expect-rejection-source "Softmax x row" \
        --expect-rejection-reason unsupported_reduction_residency_plan \
        --expect-residency-plan "Softmax x row"
    run_cmd python3 "$DRIVER" layer_norm --mode verify \
        --preset phase35-row-resident-small \
        --expect-spm true \
        --expect-tier-json empty \
        --expect-promotion-source "LayerNorm x row" \
        --expect-residency-plan "LayerNorm x row"
    run_cmd python3 "$DRIVER" layer_norm --mode verify \
        --preset phase35-row-resident-large \
        --expect-spm true \
        --expect-tier-json empty \
        --expect-promotion-source "LayerNorm x row" \
        --expect-residency-plan "LayerNorm x row"
    run_cmd python3 "$DRIVER" layer_norm --mode verify \
        --preset phase35-d3-small \
        --expect-spm false \
        --expect-tier-json empty \
        --expect-rejection-reason insufficient_row_work \
        --expect-residency-plan "LayerNorm x row"
    run_cmd python3 "$DRIVER" layer_norm --mode verify \
        --preset phase35-d3-large \
        --expect-spm true \
        --expect-tier-json empty \
        --expect-promotion-source "LayerNorm x row" \
        --expect-residency-plan "LayerNorm x row"
}

run_smoke() {
    run_cmd python3 "$DRIVER" layer_norm --mode compare \
        --preset phase35-row-resident-small
    run_cmd python3 "$DRIVER" softmax --mode compare \
        --preset phase35-smoke
}

run_full() {
    run_verify
    run_smoke
    run_cmd python3 "$DRIVER" layer_norm --mode compare \
        --preset phase35-small
    run_cmd python3 "$DRIVER" layer_norm --mode compare \
        --preset phase35-large
    run_cmd python3 "$DRIVER" layer_norm --mode compare \
        --preset phase35-row-resident-large
    run_cmd python3 "$DRIVER" layer_norm --mode compare \
        --preset phase35-d3-small
    run_cmd python3 "$DRIVER" layer_norm --mode compare \
        --preset phase35-d3-large
    run_cmd python3 "$DRIVER" softmax --mode compare \
        --preset phase35-large-row
}

case "$suite" in
    verify)
        run_verify
        ;;
    smoke)
        run_smoke
        ;;
    full)
        run_full
        ;;
    *)
        echo "Usage: $0 [verify|smoke|full]" >&2
        exit 2
        ;;
esac
