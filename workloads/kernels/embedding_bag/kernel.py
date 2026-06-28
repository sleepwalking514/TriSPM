"""
Triton AOT compilation: embedding_bag kernel -> LLVM IR for RISC-V.

First irregular-access workload in TriSPM.  Pooled embedding gather over a
fixed-length bag of indices:

    for k in 0..L_MAX:
        idx = indices[off_start + k]
        acc += table[idx, :]
    output[bag, :] = acc

C2 grouped bags: each program handles BAG_GROUP consecutive bags so the SPM
lowering has an outer scf.for to ping-pong row staging across.  When
BAG_GROUP == 1 the IR shape collapses to the v1 single-bag form, so the v1
matcher still applies.

W1 (this file) goes through the existing cache lowering only.  The kernel
intentionally does NOT call any SPM intrinsic; the SPM path is added later
by ConvertMemoryToSPM.cpp's indirect-tile recognizer (single-buffer in C1,
double-buffer in C2).
"""
import os

import torch
import triton
import triton.language as tl


def env_int(name: str) -> int:
    value = os.getenv(name)
    if value is None:
        raise RuntimeError(
            f"{name} must be exported from experiment.toml by run_experiment.py"
        )
    return int(value)


B = env_int("EMB_B")
L_MAX = env_int("EMB_L_MAX")
D = env_int("EMB_D")
NUM_ROWS = env_int("EMB_NUM_ROWS")
BAG_GROUP = env_int("EMB_BAG_GROUP")

if B % BAG_GROUP != 0:
    raise RuntimeError(
        f"EMB_B ({B}) must be divisible by EMB_BAG_GROUP ({BAG_GROUP}); "
        f"v1/C2 do not support tail masking"
    )

GRID_X = B // BAG_GROUP


@triton.jit
def embedding_bag(table_ptr, indices_ptr, offsets_ptr, out_ptr,
                  B: tl.constexpr, L_MAX: tl.constexpr,
                  D: tl.constexpr, NUM_ROWS: tl.constexpr,
                  BAG_GROUP: tl.constexpr):
    pid = tl.program_id(0)

    # Outer loop over the BAG_GROUP bags this program owns.  The compiler's
    # indirect-tile matcher recognizes this loop's body (the inner gather +
    # reduction over k) and ping-pongs row staging across iterations.
    for local_b in range(0, BAG_GROUP):
        bag = pid * BAG_GROUP + local_b
        off_start = tl.load(offsets_ptr + bag)

        acc = tl.zeros((D,), dtype=tl.float32)

        for k in range(0, L_MAX):
            idx = tl.load(indices_ptr + off_start + k)
            row_ptr = tl.make_block_ptr(
                base=table_ptr, shape=(NUM_ROWS * D,), strides=(1,),
                offsets=(idx * D,),
                block_shape=(D,), order=(0,))
            v = tl.load(row_ptr)
            acc += v

        out_block_ptr = tl.make_block_ptr(
            base=out_ptr, shape=(B * D,), strides=(1,),
            offsets=(bag * D,),
            block_shape=(D,), order=(0,))
        tl.store(out_block_ptr, acc)


# --- AOT cross-compilation ---
NNZ = B * L_MAX
table   = torch.empty(NUM_ROWS, D, dtype=torch.float32)
indices = torch.empty(NNZ, dtype=torch.int32)
offsets = torch.empty(B + 1, dtype=torch.int32)
out     = torch.empty(B, D, dtype=torch.float32)

embedding_bag[(GRID_X,)](table, indices, offsets, out,
                         B, L_MAX, D, NUM_ROWS, BAG_GROUP)
