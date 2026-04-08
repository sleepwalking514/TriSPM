# Per-kernel configuration for matmul.
# C[M,N] = A[M,K] @ B[K,N], tiled with GROUP_SIZE_M grouping.
# Sourced by build_kernel.sh.

M=64
N=64
K=64
BLOCK_SIZE_M=16
BLOCK_SIZE_N=16
BLOCK_SIZE_K=16
GROUP_SIZE_M=4

KERNEL_CFLAGS="-DM=$M -DN=$N -DK=$K \
               -DBLOCK_SIZE_M=$BLOCK_SIZE_M \
               -DBLOCK_SIZE_N=$BLOCK_SIZE_N \
               -DBLOCK_SIZE_K=$BLOCK_SIZE_K"
