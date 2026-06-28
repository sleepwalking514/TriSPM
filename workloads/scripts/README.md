# Workload Scripts

This directory contains the public workload control plane for TriSPM.  Keep the
top level reserved for stable entry points that a fresh artifact user can run
or inspect without knowing the paper development history.

## Stable Entry Points

- `paper_experiments.py`: generates and optionally executes the reproducible
  experiment matrix.
- `run_experiment.py`: builds, runs, and compares one kernel manifest.
- `graph_eval.py`: runs graph-level SPM/cache comparisons and emits
  compact evaluation artifacts.
- `graph_placement.py`: builds graph placement plans and graph run artifacts.
- `build_kernel.sh` and `run_gem5.sh`: low-level build/run helpers used by the
  Python drivers.
- `compare_stats.py`: extracts the compact SPM-vs-cache statistics tables.
- `trispm_paths.py`: shared repository-relative artifact path definitions.

## Maintained Helpers

- `generate_decoder_canonical_mh8.py`: regenerates canonical decoder graph
  manifests.
- `summarize_*.py`: post-processes already generated campaign or gem5 output.
- `../tools/run_rvv.sh`: runs a built artifact on the RVV laptop path.

## Historical Scripts

Older phase wrappers, local evidence archivers, tuning sweeps, and root-level
comparison runners were removed or folded into `paper_experiments.py`.  The
repository keeps the public reproduction path through generated campaign rows
instead of preserving one-off development commands.

New experiments should normally be expressed through `paper_experiments.py` or
`run_experiment.py` instead of adding another top-level ad hoc script.
