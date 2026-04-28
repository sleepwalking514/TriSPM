# Per-kernel C-build glue for vector_add.
# Sourced by build_kernel.sh after run_experiment.py exports SIZE / BLOCK_SIZE /
# CHECK_RESULT from experiment.toml [params] (env_prefix is empty for this
# kernel). Defaults live in the manifest.

KERNEL_CFLAGS="-DBLOCK_SIZE=$BLOCK_SIZE -DSIZE=$SIZE -DCHECK_RESULT=$CHECK_RESULT"
