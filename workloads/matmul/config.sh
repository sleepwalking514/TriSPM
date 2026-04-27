# Per-kernel configuration for matmul.
# C[M,N] = A[M,K] @ B[K,N], tiled with GROUP_SIZE_M grouping.
# Sourced by build_kernel.sh.
#
# The MATMUL_* env vars let sweep targets build multiple variants without
# editing this file. Defaults preserve the 256x256 cold-cache smoke test.

M="${MATMUL_M:-256}"
N="${MATMUL_N:-256}"
K="${MATMUL_K:-256}"
BLOCK_SIZE_M="${MATMUL_BLOCK_SIZE_M:-32}"
BLOCK_SIZE_N="${MATMUL_BLOCK_SIZE_N:-32}"
BLOCK_SIZE_K="${MATMUL_BLOCK_SIZE_K:-32}"
GROUP_SIZE_M="${MATMUL_GROUP_SIZE_M:-4}"
MATMUL_WARMUP_ITERS="${MATMUL_WARMUP_ITERS:-0}"
MATMUL_MEASURE_ITERS="${MATMUL_MEASURE_ITERS:-1}"
MATMUL_FLUSH_BEFORE_ROI="${MATMUL_FLUSH_BEFORE_ROI:-1}"
MATMUL_CHECK_RESULT="${MATMUL_CHECK_RESULT:-1}"

KERNEL_CFLAGS="-DM=$M -DN=$N -DK=$K \
               -DBLOCK_SIZE_M=$BLOCK_SIZE_M \
               -DBLOCK_SIZE_N=$BLOCK_SIZE_N \
               -DBLOCK_SIZE_K=$BLOCK_SIZE_K \
               -DMATMUL_WARMUP_ITERS=$MATMUL_WARMUP_ITERS \
               -DMATMUL_MEASURE_ITERS=$MATMUL_MEASURE_ITERS \
               -DMATMUL_FLUSH_BEFORE_ROI=$MATMUL_FLUSH_BEFORE_ROI \
               -DMATMUL_CHECK_RESULT=$MATMUL_CHECK_RESULT"
