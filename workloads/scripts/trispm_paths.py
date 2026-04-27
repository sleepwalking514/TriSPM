#!/usr/bin/env python3
"""Single source of truth for TriSPM workload artifact paths.

Layout:
  workloads/build/<kernel>/<spm|cache>-<tag>/   build artifacts (.llir, .s, _test, _launcher.c)
  workloads/m5out/<kernel>/<tag>/<spm|cache>/   gem5 outdir + stats.txt + roi-stats.txt
  workloads/m5out/<kernel>/<tag>/compare.txt    SPM-vs-cache delta table

The shell scripts call this module with `--print` to avoid duplicating the
naming logic in bash.
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


def build_dir(kernel: str, mode: str, tag: str = "default") -> Path:
    _check_mode(mode)
    return BUILD_ROOT / kernel / f"{mode}-{tag}"


def binary_path(kernel: str, mode: str, tag: str = "default") -> Path:
    return build_dir(kernel, mode, tag) / f"{kernel}_test"


def m5out_dir(kernel: str, mode: str, tag: str = "default") -> Path:
    _check_mode(mode)
    return M5OUT_ROOT / kernel / tag / mode


def roi_stats_path(kernel: str, mode: str, tag: str = "default") -> Path:
    """Filtered stats containing only the explicit ROI dump."""
    return m5out_dir(kernel, mode, tag) / "roi-stats.txt"


def compare_path(kernel: str, tag: str = "default") -> Path:
    return M5OUT_ROOT / kernel / tag / "compare.txt"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("what", choices=["build_dir", "binary", "m5out_dir", "roi_stats", "compare"])
    p.add_argument("kernel")
    p.add_argument("mode", nargs="?", choices=MODES)
    p.add_argument("--tag", default="default")
    args = p.parse_args()

    if args.what == "compare":
        print(compare_path(args.kernel, args.tag))
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
