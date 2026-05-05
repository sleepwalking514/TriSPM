#!/usr/bin/env python3
"""Single source of truth for TriSPM workload artifact paths.

Layout:
  workloads/build/<kernel>/<spm|cache>-<flat-tag>/   build artifacts (.llir, .s, _test, _launcher.c)
  workloads/m5out/<kernel>/<shape>/<spm|cache>/<blocking>/
      gem5 outdir + stats.txt + roi-stats.txt + run.log
  workloads/m5out/<kernel>/<shape>/cache_best.json   best cache blocking for this shape
  workloads/m5out/<kernel>/<shape>/spm/<blocking>/compare_vs_cache_best.txt
  workloads/m5out/<kernel>/<shape>/spm/<blocking>/spm_stats.txt

Tags may contain '/' to create m5out subdirectories (e.g.
matmul uses "{M}x{N}x{K}/{bsM}x{bsN}x{bsK}"). The first tag component is the
shape; the remainder is the blocking/schedule. Build dirs flatten any slashes
since they are intermediate artifacts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

WORKLOADS_DIR = Path(__file__).resolve().parent.parent
BUILD_ROOT = WORKLOADS_DIR / "build"
M5OUT_ROOT = WORKLOADS_DIR / "m5out"

MODES = ("spm", "cache")


def _check_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def _flat_tag(tag: str) -> str:
    """Collapse '/' into '-' for filesystem-flat build dir names."""
    return tag.replace("/", "-")


def split_tag(tag: str) -> tuple[str, str]:
    """Return (shape, blocking) from a rendered tag."""
    parts = [part for part in tag.split("/") if part]
    if not parts:
        raise ValueError("tag must not be empty")
    shape = parts[0]
    blocking = "-".join(parts[1:]) if len(parts) > 1 else "default"
    return shape, blocking


def build_dir(kernel: str, mode: str, tag: str) -> Path:
    _check_mode(mode)
    return BUILD_ROOT / kernel / f"{mode}-{_flat_tag(tag)}"


def binary_path(kernel: str, mode: str, tag: str) -> Path:
    return build_dir(kernel, mode, tag) / f"{kernel}_test"


def m5out_dir(kernel: str, mode: str, tag: str) -> Path:
    _check_mode(mode)
    shape, blocking = split_tag(tag)
    return M5OUT_ROOT / kernel / shape / mode / blocking


def roi_stats_path(kernel: str, mode: str, tag: str) -> Path:
    """Filtered stats containing only the explicit ROI dump."""
    return m5out_dir(kernel, mode, tag) / "roi-stats.txt"


def run_log_path(kernel: str, mode: str, tag: str) -> Path:
    """Captured gem5 stdout/stderr, including the workload PASS/FAIL line."""
    return m5out_dir(kernel, mode, tag) / "run.log"


def shape_dir(kernel: str, tag: str) -> Path:
    shape, _ = split_tag(tag)
    return M5OUT_ROOT / kernel / shape


def cache_best_path(kernel: str, tag: str) -> Path:
    return shape_dir(kernel, tag) / "cache_best.json"


def compare_path(kernel: str, tag: str) -> Path:
    return m5out_dir(kernel, "spm", tag) / "compare_vs_cache_best.txt"


def spm_stats_path(kernel: str, tag: str) -> Path:
    """SPM-only signals (DMA traffic, bank conflicts, etc.) — no cache counterpart."""
    return m5out_dir(kernel, "spm", tag) / "spm_stats.txt"


def artifact_stats_path(kernel: str, tag: str) -> Path:
    return m5out_dir(kernel, "spm", tag) / "artifacts.txt"


def graph_build_dir(graph: str, mode: str) -> Path:
    _check_mode(mode)
    return BUILD_ROOT / "graphs" / graph / mode


def graph_binary_path(graph: str, mode: str) -> Path:
    return graph_build_dir(graph, mode) / f"{graph}_test"


def graph_m5out_dir(graph: str, mode: str) -> Path:
    _check_mode(mode)
    return M5OUT_ROOT / "graphs" / graph / mode / "default"


def graph_roi_stats_path(graph: str, mode: str) -> Path:
    return graph_m5out_dir(graph, mode) / "roi-stats.txt"


def graph_run_log_path(graph: str, mode: str) -> Path:
    return graph_m5out_dir(graph, mode) / "run.log"


def graph_compare_path(graph: str) -> Path:
    return graph_m5out_dir(graph, "spm") / "compare_vs_cache.txt"


def graph_spm_stats_path(graph: str) -> Path:
    return graph_m5out_dir(graph, "spm") / "spm_stats.txt"


def graph_report_path(graph: str) -> Path:
    return graph_m5out_dir(graph, "spm") / "graph_report.txt"


def graph_eval_json_path(graph: str) -> Path:
    return graph_m5out_dir(graph, "spm") / "phase6_eval.json"


def graph_eval_summary_path(graph: str) -> Path:
    return graph_m5out_dir(graph, "spm") / "phase6_summary.txt"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "what",
        choices=[
            "build_dir",
            "binary",
            "m5out_dir",
            "roi_stats",
            "run_log",
            "compare",
            "spm_stats",
            "artifact_stats",
            "cache_best",
            "graph_build_dir",
            "graph_binary",
            "graph_m5out_dir",
            "graph_roi_stats",
            "graph_run_log",
            "graph_compare",
            "graph_spm_stats",
            "graph_report",
            "graph_eval_json",
            "graph_eval_summary",
        ],
    )
    p.add_argument("kernel")
    p.add_argument("mode", nargs="?", choices=MODES)
    p.add_argument("--tag")
    args = p.parse_args()

    if args.what in (
        "compare",
        "spm_stats",
        "artifact_stats",
        "cache_best",
    ):
        if args.tag is None:
            p.error(f"{args.what} requires --tag")
        fns = {
            "compare": compare_path,
            "spm_stats": spm_stats_path,
            "artifact_stats": artifact_stats_path,
            "cache_best": cache_best_path,
        }
        print(fns[args.what](args.kernel, args.tag))
    elif args.what in (
        "graph_compare",
        "graph_spm_stats",
        "graph_report",
        "graph_eval_json",
        "graph_eval_summary",
    ):
        fns = {
            "graph_compare": graph_compare_path,
            "graph_spm_stats": graph_spm_stats_path,
            "graph_report": graph_report_path,
            "graph_eval_json": graph_eval_json_path,
            "graph_eval_summary": graph_eval_summary_path,
        }
        print(fns[args.what](args.kernel))
    elif args.what.startswith("graph_"):
        if args.mode is None:
            p.error(f"{args.what} requires mode to be one of {MODES}")
        fns = {
            "graph_build_dir": graph_build_dir,
            "graph_binary": graph_binary_path,
            "graph_m5out_dir": graph_m5out_dir,
            "graph_roi_stats": graph_roi_stats_path,
            "graph_run_log": graph_run_log_path,
        }
        print(fns[args.what](args.kernel, args.mode))
    else:
        if args.mode is None:
            p.error(f"{args.what} requires mode to be one of {MODES}")
        if args.tag is None:
            p.error(f"{args.what} requires --tag")
        fns = {
            "build_dir": build_dir,
            "binary": binary_path,
            "m5out_dir": m5out_dir,
            "roi_stats": roi_stats_path,
            "run_log": run_log_path,
        }
        print(fns[args.what](args.kernel, args.mode, args.tag))


if __name__ == "__main__":
    main()
