# Per-kernel configuration for vector_add.
# Sourced by build_kernel.sh.

BLOCK_SIZE=64       # Elements per tile. Must match kernel.py.
                    # VLEN=256 → 8 floats/vreg, LMUL=8 → 64 floats/vgroup.
SIZE=4096           # Total array size for the test harness.

KERNEL_CFLAGS="-DBLOCK_SIZE=$BLOCK_SIZE -DSIZE=$SIZE"
