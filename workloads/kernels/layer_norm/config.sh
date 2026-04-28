# Per-kernel C-build glue for layer_norm.
# Per-row normalization over N features.
# Sourced by build_kernel.sh after run_experiment.py exports M_SIZE / N_SIZE /
# CHECK_RESULT from experiment.toml [params] (env_prefix is empty). Defaults
# live in the manifest.

KERNEL_CFLAGS="-DM_SIZE=$M_SIZE -DN_SIZE=$N_SIZE -DCHECK_RESULT=$CHECK_RESULT"
