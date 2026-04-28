# Per-kernel C-build glue for matmul.
# Sourced by build_kernel.sh after run_experiment.py exports MATMUL_*
# env vars from experiment.toml [params]. Defaults live in the manifest;
# this file only maps those env vars into -D macros for harness.c.

KERNEL_CFLAGS="-DM=$MATMUL_M -DN=$MATMUL_N -DK=$MATMUL_K \
               -DBLOCK_SIZE_M=$MATMUL_BLOCK_SIZE_M \
               -DBLOCK_SIZE_N=$MATMUL_BLOCK_SIZE_N \
               -DBLOCK_SIZE_K=$MATMUL_BLOCK_SIZE_K \
               -DMATMUL_WARMUP_ITERS=$MATMUL_WARMUP_ITERS \
               -DMATMUL_MEASURE_ITERS=$MATMUL_MEASURE_ITERS \
               -DMATMUL_FLUSH_BEFORE_ROI=$MATMUL_FLUSH_BEFORE_ROI \
               -DMATMUL_CHECK_RESULT=$MATMUL_CHECK_RESULT"
