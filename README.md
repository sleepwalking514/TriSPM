# TriSPM

TriSPM is a research compiler and simulator artifact for compiler-controlled
scratchpad promotion on cache-rich RISC-V CPUs.  Graph-visible tensors stay on
the normal cache-coherent memory path; the compiler promotes only bounded,
high-reuse kernel-local regions into an explicitly managed scratchpad memory
(SPM).

The public repository ties together three pieces:

- `compiler/`: Triton CPU fork with the TriSPM lowering pipeline.
- `simulator/`: gem5 fork with the RISC-V SPM, DMA engine, address regions, and
  simulation configuration.
- `workloads/`: kernels, graph manifests, build/run helpers, and reproducible
  experiment drivers.

Internal paper drafts, planning notes, archived logs, and generated evidence
are intentionally kept outside the public artifact and ignored at the top
level.

## Scope

TriSPM is not a general-purpose scratchpad allocator and it is not primarily a
collection of hand-written DMA kernels.  The compiler targets analyzable tensor
IR regions and leaves unsupported or unprofitable regions on the cache path.

Currently covered footprint classes include:

- GEMM-like contraction reuse.
- Multi-pass row residency for normalization and canonical softmax-style row
  reductions.
- Derived-value residency for canonical softmax `exp(x - max)` reuse.
- Conservative generic affine-tile promotion for static full-tile transfers.

Promotion records and rejection reasons are part of the artifact contract:
rejected candidates document the conservative fallback behavior.

## Checkout

This repository uses two large submodules:

```bash
git submodule update --init --recursive
```

The upstream build instructions for the forks remain in `compiler/README.md`
and `simulator/README.md`.  The top-level repository assumes the checked-out
layout shown above.

The recursive submodule checkout also initializes nested third-party
dependencies used by the compiler fork, including SLEEF.

## Prerequisites

The workload drivers cross-compile RISC-V binaries and then run them under the
TriSPM gem5 fork.  A working setup needs:

- A Python environment for the Triton CPU fork.
- A built Triton/LLVM tree, with `llc` available at
  `compiler/llvm-project/build/bin/llc`.
- A built RISCV gem5 binary at `simulator/build/RISCV/gem5.opt`.
- A RISC-V cross-compilation toolchain with Clang and a sysroot.

`RISCV_TOOLCHAIN_ROOT` must point at the RISC-V toolchain root.  The workload
scripts expect this layout:

```text
$RISCV_TOOLCHAIN_ROOT/bin/clang
$RISCV_TOOLCHAIN_ROOT/sysroot/
```

For example:

```bash
export RISCV_TOOLCHAIN_ROOT=/opt/riscv
```

## Running Experiments

The main reproducibility entry point is the paper-scoped campaign generator:

```bash
cd workloads
./scripts/paper_experiments.py --campaign paper-experiments
```

This writes the generated run plan under:

```text
workloads/m5out/campaigns/paper-experiments/
```

Add `--run` to execute selected phases:

```bash
cd workloads
./scripts/paper_experiments.py \
  --campaign paper-experiments \
  --phase kernel-headline \
  --run \
  --jobs 4
```

Useful phases include:

- `kernel-headline`
- `graph-headline`
- `graph-scale`
- `graph-hw-sensitivity`
- `graph-profile`
- `softmax-fairness`
- `attention-algorithm-fairness`
- `generic-affine-fallback`
- `generic-affine-fallback-perf`
- `gemm-tuning-mechanism`
- `split`
- `cache-capacity-fairness`
- `xspm-instruction`

For individual kernels:

```bash
cd workloads
./scripts/run_experiment.py matmul --mode cache --preset steady --tag example-cache
./scripts/run_experiment.py matmul --mode spm --preset steady --tag example-spm
```

For graph-level runs:

```bash
cd workloads
./scripts/graph_eval.py decoder_canonical_mh8 --preset large
./scripts/graph_eval.py decoder_canonical_mh8 --preset large_profile
```

Generated binaries, gem5 outputs, logs, campaign plans, and comparison tables
are written under `workloads/build/`, `workloads/m5out/`, and
`workloads/logs/`.  These paths are local artifacts and are ignored by Git.

## Script Layout

The stable workload commands live directly under `workloads/scripts/`.  One-off
paper sweeps, historical diagnostics, and hand-written comparison runners are
kept only when they preserve a useful reproduction path; see
`workloads/scripts/README.md` for the current classification.

## Claim Boundaries

When using or extending this artifact:

- Report graph rows as whole-graph ROI cycles, not as sums of per-node windows.
- Treat graph-visible tensors as cache-backed interfaces.
- Treat FlashAttention-style attention as a separate algorithmic track from
  canonical `QK^T -> softmax -> PV`.
- Treat irregular workloads as scoped evidence; small or low-reuse regions can
  lose when DMA setup and synchronization dominate.
