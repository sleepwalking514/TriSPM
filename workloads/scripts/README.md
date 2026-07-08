# Workload Scripts

This directory contains the public workload control plane for TriSPM.  The top
level is reserved for stable commands that users run directly.

## User Entry Points

- `paper_experiments.py`: generates and optionally executes the reproducible
  experiment matrix.  It writes a run plan, summary, and status files under
  `workloads/m5out/campaigns/<name>/`.
- `run_experiment.py`: builds, runs, verifies, and compares one kernel manifest.
- `graph_eval.py`: runs graph-level SPM/cache comparisons and emits
  compact evaluation artifacts.

Typical commands:

```bash
./scripts/paper_experiments.py --campaign paper-experiments
./scripts/paper_experiments.py --campaign paper-experiments --phase kernel-headline --run --jobs 4

./scripts/run_experiment.py matmul --mode cache --preset steady --tag example-cache
./scripts/run_experiment.py matmul --mode spm --preset steady --tag example-spm

./scripts/graph_eval.py decoder_canonical_mh8 --preset large
```

## Supporting Code

- `internal/`: helper implementation used by the user entry points, including
  graph placement, low-level build/run scripts, stats comparison, and shared
  artifact paths.
- `generators/generate_decoder_canonical.py`: regenerates the canonical decoder
  graph fixtures.  Use `--case small`, `--case base`, or `--case large`.
- `reports/`: post-processes generated campaign or gem5 output.
- `../tools/run_rvv.sh`: runs a built artifact on the RVV laptop path.

## Extending Experiments

New kernel experiments should normally be expressed through a kernel
`experiment.toml` plus `run_experiment.py`.  New graph experiments should use a
graph manifest plus `graph_eval.py`, and paper-scale batches should be added as
rows in `paper_experiments.py`.
