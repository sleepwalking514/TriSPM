# Per-kernel configuration for layer_norm.
# Per-row normalization over N features.
# Sourced by build_kernel.sh.

M_SIZE="${M_SIZE:-32}"  # rows (batch * seq_len)
N_SIZE="${N_SIZE:-64}"  # features per row
CHECK_RESULT="${CHECK_RESULT:-1}"

KERNEL_CFLAGS="-DM_SIZE=$M_SIZE -DN_SIZE=$N_SIZE -DCHECK_RESULT=$CHECK_RESULT"
