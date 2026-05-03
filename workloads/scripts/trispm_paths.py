#!/usr/bin/env python3
"""Single source of truth for TriSPM workload artifact paths.

Layout:
  workloads/build/<kernel>/<spm|cache>-<flat-tag>/   build artifacts (.llir, .s, _test, _launcher.c)
  workloads/m5out/<kernel>/<tag>/<spm|cache>/        gem5 outdir + stats.txt + roi-stats.txt
  workloads/m5out/<kernel>/<tag>/compare.txt         SPM-vs-cache delta table
  workloads/m5out/<kernel>/<tag>/compare.csv         CSV form of compare.txt
  workloads/m5out/<kernel>/<tag>/spm_stats.txt       SPM-only signals (DMA, banks, ...)
  workloads/m5out/<kernel>/<tag>/spm_stats.csv       CSV form of spm_stats.txt
  workloads/m5out/<kernel>/<tag>/artifacts.csv       build artifact line/marker counts

Tags may contain '/' to create m5out subdirectories (e.g.
matmul uses "{M}x{N}x{K}/{bsM}x{bsN}x{bsK}"). Build dirs flatten any
slashes to '-' since they are intermediate artifacts.
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


def build_dir(kernel: str, mode: str, tag: str) -> Path:
    _check_mode(mode)
    return BUILD_ROOT / kernel / f"{mode}-{_flat_tag(tag)}"


def binary_path(kernel: str, mode: str, tag: str) -> Path:
    return build_dir(kernel, mode, tag) / f"{kernel}_test"


def m5out_dir(kernel: str, mode: str, tag: str) -> Path:
    _check_mode(mode)
    return M5OUT_ROOT / kernel / tag / mode


def roi_stats_path(kernel: str, mode: str, tag: str) -> Path:
    """Filtered stats containing only the explicit ROI dump."""
    return m5out_dir(kernel, mode, tag) / "roi-stats.txt"


def compare_path(kernel: str, tag: str) -> Path:
    return M5OUT_ROOT / kernel / tag / "compare.txt"


def spm_stats_path(kernel: str, tag: str) -> Path:
    """SPM-only signals (DMA traffic, bank conflicts, etc.) — no cache counterpart."""
    return M5OUT_ROOT / kernel / tag / "spm_stats.txt"


def compare_csv_path(kernel: str, tag: str) -> Path:
    return M5OUT_ROOT / kernel / tag / "compare.csv"


def spm_stats_csv_path(kernel: str, tag: str) -> Path:
    return M5OUT_ROOT / kernel / tag / "spm_stats.csv"


def artifact_stats_path(kernel: str, tag: str) -> Path:
    return M5OUT_ROOT / kernel / tag / "artifacts.csv"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "what",
        choices=[
            "build_dir",
            "binary",
            "m5out_dir",
            "roi_stats",
            "compare",
            "spm_stats",
            "compare_csv",
            "spm_stats_csv",
            "artifact_stats",
        ],
    )
    p.add_argument("kernel")
    p.add_argument("mode", nargs="?", choices=MODES)
    p.add_argument("--tag", required=True)
    args = p.parse_args()

    if args.what in (
        "compare",
        "spm_stats",
        "compare_csv",
        "spm_stats_csv",
        "artifact_stats",
    ):
        fns = {
            "compare": compare_path,
            "spm_stats": spm_stats_path,
            "compare_csv": compare_csv_path,
            "spm_stats_csv": spm_stats_csv_path,
            "artifact_stats": artifact_stats_path,
        }
        print(fns[args.what](args.kernel, args.tag))
    else:
        if args.mode is None:
            p.error(f"{args.what} requires mode to be one of {MODES}")
        fns = {
            "build_dir": build_dir,
            "binary": binary_path,
            "m5out_dir": m5out_dir,
            "roi_stats": roi_stats_path,
        }
        print(fns[args.what](args.kernel, args.mode, args.tag))


if __name__ == "__main__":
    main()
