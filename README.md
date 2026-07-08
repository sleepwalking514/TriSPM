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

## Scope

TriSPM extends the Triton CPU compilation flow with compiler-managed scratchpad
promotion for RVV tensor programs.  The compiler analyzes bounded tensor
regions in lowered Triton CPU IR, promotes profitable tiles and temporaries to
SPM, inserts DMA/wait operations, and leaves unsupported accesses on the
ordinary cache path.

The artifact covers the access patterns evaluated in the paper:

- Contraction tile reuse for GEMM-like operators.
- Multi-pass row reuse for LayerNorm and canonical Softmax.
- Temporary-value reuse such as cached `exp(x - max)` in Softmax.
- Generic affine-tile streaming for static full-tile transfers.

Graph-visible tensors remain ordinary cache-coherent tensors between operators;
TriSPM applies SPM promotion inside admitted kernel scopes.

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
- A RISC-V cross-compilation toolchain with Clang and a target sysroot.

`RISCV_TOOLCHAIN_ROOT` must point at the RISC-V toolchain root.  The workload
scripts derive both `clang` and `--sysroot` from this root, so no separate
`SYSROOT` variable is required.  The expected layout is:

```text
$RISCV_TOOLCHAIN_ROOT/bin/clang
$RISCV_TOOLCHAIN_ROOT/sysroot/
```

For example:

```bash
export RISCV_TOOLCHAIN_ROOT=/opt/riscv
```

## Running Experiments

The main reproducibility entry point is the paper experiment campaign driver.
A campaign is a named set of experiment rows; the name is used only to group the
generated run plan, logs, status, and summaries under `workloads/m5out/`.

```bash
cd workloads
./scripts/paper_experiments.py --campaign paper-experiments
```

The `--campaign` value names the output directory for the generated run plan:

```text
workloads/m5out/campaigns/paper-experiments/
```

By default this only writes the plan.  Add `--run` to execute selected rows.
The common filters are:

- `--phase`: include one phase, repeatable.
- `--from-phase`: include a phase and every later phase in paper order.
- `--label`: include rows whose label contains this substring, repeatable.
- `--jobs`: run multiple independent rows concurrently.

For example, run only the single-kernel results phase:

```bash
cd workloads
./scripts/paper_experiments.py \
  --campaign paper-experiments \
  --phase single-kernel-results \
  --run \
  --jobs 4
```

Run a single labeled subset without overlapping existing outputs:

```bash
cd workloads
./scripts/paper_experiments.py \
  --campaign paper-experiments \
  --phase graph-scale \
  --label large \
  --artifact-suffix trial1 \
  --run
```

Useful phases include:

- `single-kernel-results`
- `main-graph-result`
- `graph-scale`
- `graph-hardware-sensitivity`
- `graph-attribution`
- `softmax-algorithm-controls`
- `attention-algorithm-controls`
- `generic-affine-build`
- `generic-affine-performance`
- `gemm-tuning`
- `graph-cache-capacity`
- `graph-xspm-variant`

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

To regenerate canonical decoder graph fixtures:

```bash
cd workloads
./scripts/generators/generate_decoder_canonical.py --case small
./scripts/generators/generate_decoder_canonical.py --case base
./scripts/generators/generate_decoder_canonical.py --case large
```

Generated binaries, gem5 outputs, logs, campaign plans, and comparison tables
are written under `workloads/build/`, `workloads/m5out/`, and
`workloads/logs/`.  These paths are local artifacts and are ignored by Git.

## Script Layout

The stable workload commands live directly under `workloads/scripts/`.  Helper
implementation, graph generators, and report post-processors are grouped under
subdirectories; see `workloads/scripts/README.md` for the current layout.

## Claim Boundaries

When using or extending this artifact:

- Report graph rows as whole-graph ROI cycles, not as sums of per-node windows.
- Treat graph-visible tensors as cache-backed interfaces.
- Treat FlashAttention-style attention as a separate algorithmic track from
  canonical `QK^T -> softmax -> PV`.
- Treat irregular workloads as scoped evidence; small or low-reuse regions can
  lose when DMA setup and synchronization dominate.
