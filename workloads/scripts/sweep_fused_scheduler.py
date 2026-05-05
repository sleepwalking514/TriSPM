#!/usr/bin/env python3
"""Two-phase parallel sweep for matmul fused-scheduler evidence.

Phase A: For each matrix size, sweep blocking configs (fixed microM=8, windowK=4)
         to find the best blocking. Also runs cache baselines for speedup data.
Phase B: On each size's best blocking, sweep microM × windowK.

All runs use CHECK_RESULT=0. Results land in m5out/matmul/fused-sweep/.

Usage:
  ./scripts/sweep_fused_scheduler.py                     # both phases
  ./scripts/sweep_fused_scheduler.py --phase a           # blocking sweep only
  ./scripts/sweep_fused_scheduler.py --phase b           # microM×windowK only (needs phase A results)
  ./scripts/sweep_fused_scheduler.py --jobs 16           # limit parallelism
  ./scripts/sweep_fused_scheduler.py --dry-run           # print commands only
  ./scripts/sweep_fused_scheduler.py --sizes 64,256,512  # custom sizes
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
SWEEP_TAG_PREFIX = "fused-sweep"

# ─── Sweep space ──────────────────────────────────────────────────────────

# Each size paired with candidate blockings to try in Phase A.
SIZE_BLOCKING_CANDIDATES: list[tuple[int, list[tuple[int, int, int]]]] = [
    (64,   [(32, 32, 32)]),
    (256,  [(32, 32, 32), (64, 64, 32), (64, 64, 64)]),
    (512,  [(32, 32, 32), (64, 64, 32), (64, 64, 64), (128, 128, 32)]),
    (1024, [(64, 64, 32), (64, 64, 64), (128, 128, 32), (128, 128, 64)]),
]

PHASE_A_MICRO_M = 8
PHASE_A_WINDOW_K = 4

DEFAULT_MICRO_M = [4, 8, 16]
DEFAULT_WINDOW_K = [2, 4, 8]


# ─── Helpers ──────────────────────────────────────────────────────────────

def tag_for(size: int, bm: int, bn: int, bk: int, micro_m: int, window_k: int, mode: str) -> str:
    base = f"{SWEEP_TAG_PREFIX}/{size}x{size}x{size}/{bm}x{bn}x{bk}"
    if mode == "cache":
        return f"{base}/cache"
    return f"{base}/uM{micro_m}-wK{window_k}"


def roi_stats_path(size: int, bm: int, bn: int, bk: int, micro_m: int, window_k: int, mode: str) -> Path:
    tag = tag_for(size, bm, bn, bk, micro_m, window_k, mode)
    # Mirror trispm_paths layout
    cmd = [sys.executable, str(SCRIPTS_DIR / "trispm_paths.py"), "roi_stats", "matmul", mode, "--tag", tag]
    return Path(subprocess.check_output(cmd, text=True).strip())


def build_experiment_cmd(
    size: int, bm: int, bn: int, bk: int,
    micro_m: int, window_k: int,
    mode: str,
) -> tuple[str, list[str], dict[str, str]]:
    tag = tag_for(size, bm, bn, bk, micro_m, window_k, mode)
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "run_experiment.py"),
        "matmul",
        "--mode", mode if mode != "cache" else "cache",
        "--tag", tag,
        "--set", f"M={size}",
        "--set", f"N={size}",
        "--set", f"K={size}",
        "--set", f"BLOCK_SIZE_M={bm}",
        "--set", f"BLOCK_SIZE_N={bn}",
        "--set", f"BLOCK_SIZE_K={bk}",
        "--set", "CHECK_RESULT=0",
    ]
    env = os.environ.copy()
    env["TRITON_MICRO_M"] = str(micro_m)
    env["TRITON_SPM_WINDOW_K"] = str(window_k)
    return tag, cmd, env


def run_one(tag: str, cmd: list[str], env: dict[str, str]) -> tuple[str, int, float]:
    t0 = time.time()
    log_dir = SCRIPTS_DIR.parent / "m5out" / "matmul" / tag.replace("/", os.sep)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "sweep.log"
    with open(log_path, "w") as f:
        result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    return tag, result.returncode, elapsed


def run_parallel(jobs: list[tuple[str, list[str], dict[str, str]]], max_workers: int, dry_run: bool) -> dict[str, int]:
    if dry_run:
        for tag, cmd, env in jobs:
            um = env.get("TRITON_MICRO_M", "?")
            wk = env.get("TRITON_SPM_WINDOW_K", "?")
            print(f"  TRITON_MICRO_M={um} TRITON_SPM_WINDOW_K={wk} {' '.join(shlex.quote(c) for c in cmd)}")
        return {}

    print(f"  {len(jobs)} jobs, {max_workers} workers")
    t_start = time.time()
    results: dict[str, int] = {}
    done = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_one, tag, cmd, env): tag for tag, cmd, env in jobs}
        for future in as_completed(futures):
            tag, rc, elapsed = future.result()
            results[tag] = rc
            done += 1
            status = "OK" if rc == 0 else f"FAIL(rc={rc})"
            if rc != 0:
                failed += 1
            print(f"    [{done}/{len(jobs)}] {status}  {elapsed:6.1f}s  {tag}")

    total = time.time() - t_start
    print(f"  Phase done: {done - failed}/{len(jobs)} passed, {failed} failed, {total:.0f}s wall\n")
    return results


def read_cycles(stats_path: Path) -> float | None:
    if not stats_path.is_file():
        return None
    for line in stats_path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "system.cpu.numCycles":
            return float(parts[1])
    return None


# ─── Phase A ──────────────────────────────────────────────────────────────

def phase_a_jobs(sizes: list[tuple[int, list[tuple[int, int, int]]]]) -> list[tuple[str, list[str], dict[str, str]]]:
    jobs = []
    for size, blockings in sizes:
        for bm, bn, bk in blockings:
            # SPM run
            jobs.append(build_experiment_cmd(size, bm, bn, bk, PHASE_A_MICRO_M, PHASE_A_WINDOW_K, "spm"))
            # Cache baseline
            jobs.append(build_experiment_cmd(size, bm, bn, bk, PHASE_A_MICRO_M, PHASE_A_WINDOW_K, "cache"))
    return jobs


BEST_BLOCKING_FILE = "fused_sweep_best_blocking.json"


def find_best_blocking(sizes: list[tuple[int, list[tuple[int, int, int]]]]) -> dict[int, tuple[int, int, int]]:
    """After Phase A, pick the blocking with lowest SPM cycles per size."""
    best: dict[int, tuple[int, int, int]] = {}
    print("Phase A results:")
    print(f"  {'size':<12} {'blocking':<14} {'spm_cycles':>12} {'cache_cycles':>12} {'speedup':>8}")
    print(f"  {'-'*12} {'-'*14} {'-'*12} {'-'*12} {'-'*8}")

    for size, blockings in sizes:
        best_cycles = float("inf")
        best_blk = blockings[0]
        for bm, bn, bk in blockings:
            spm_path = roi_stats_path(size, bm, bn, bk, PHASE_A_MICRO_M, PHASE_A_WINDOW_K, "spm")
            cache_path = roi_stats_path(size, bm, bn, bk, PHASE_A_MICRO_M, PHASE_A_WINDOW_K, "cache")
            spm_cyc = read_cycles(spm_path)
            cache_cyc = read_cycles(cache_path)
            spm_s = f"{int(spm_cyc):,}" if spm_cyc else "MISSING"
            cache_s = f"{int(cache_cyc):,}" if cache_cyc else "MISSING"
            speedup_s = f"{(cache_cyc / spm_cyc - 1) * 100:.1f}%" if (spm_cyc and cache_cyc) else "-"
            marker = ""
            if spm_cyc and spm_cyc < best_cycles:
                best_cycles = spm_cyc
                best_blk = (bm, bn, bk)
                marker = " <-- best"
            print(f"  {size:<12} {bm}x{bn}x{bk:<10} {spm_s:>12} {cache_s:>12} {speedup_s:>8}{marker}")
        best[size] = best_blk

    # Persist so Phase B can be run independently
    out_path = SCRIPTS_DIR.parent / "m5out" / "matmul" / SWEEP_TAG_PREFIX / BEST_BLOCKING_FILE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {str(k): list(v) for k, v in best.items()}
    out_path.write_text(json.dumps(serializable, indent=2) + "\n")
    print(f"\n  Best blocking saved to {out_path.relative_to(SCRIPTS_DIR.parent)}")
    return best


def load_best_blocking() -> dict[int, tuple[int, int, int]]:
    path = SCRIPTS_DIR.parent / "m5out" / "matmul" / SWEEP_TAG_PREFIX / BEST_BLOCKING_FILE
    if not path.is_file():
        sys.exit(f"ERROR: {path} not found. Run --phase a first.")
    data = json.loads(path.read_text())
    return {int(k): tuple(v) for k, v in data.items()}


# ─── Phase B ──────────────────────────────────────────────────────────────

def phase_b_jobs(
    best: dict[int, tuple[int, int, int]],
    micro_ms: list[int],
    window_ks: list[int],
) -> list[tuple[str, list[str], dict[str, str]]]:
    jobs = []
    for size, (bm, bn, bk) in sorted(best.items()):
        for micro_m, window_k in itertools.product(micro_ms, window_ks):
            jobs.append(build_experiment_cmd(size, bm, bn, bk, micro_m, window_k, "spm"))
    return jobs


def summarize_phase_b(
    best: dict[int, tuple[int, int, int]],
    micro_ms: list[int],
    window_ks: list[int],
) -> None:
    """Print a summary table of Phase B results."""
    print("Phase B results (SPM cycles):")
    header_combos = [f"uM{m}-wK{w}" for m, w in itertools.product(micro_ms, window_ks)]
    print(f"  {'size':<12} {'blocking':<14} " + " ".join(f"{h:>14}" for h in header_combos))
    print(f"  {'-'*12} {'-'*14} " + " ".join("-" * 14 for _ in header_combos))

    for size, (bm, bn, bk) in sorted(best.items()):
        vals = []
        for micro_m, window_k in itertools.product(micro_ms, window_ks):
            path = roi_stats_path(size, bm, bn, bk, micro_m, window_k, "spm")
            cyc = read_cycles(path)
            vals.append(f"{int(cyc):>14,}" if cyc else f"{'MISSING':>14}")
        print(f"  {size:<12} {bm}x{bn}x{bk:<10} " + " ".join(vals))

    # Also show best combo per size
    print(f"\n  Best per size:")
    for size, (bm, bn, bk) in sorted(best.items()):
        best_cyc = float("inf")
        best_combo = ""
        for micro_m, window_k in itertools.product(micro_ms, window_ks):
            path = roi_stats_path(size, bm, bn, bk, micro_m, window_k, "spm")
            cyc = read_cycles(path)
            if cyc and cyc < best_cyc:
                best_cyc = cyc
                best_combo = f"uM{micro_m}-wK{window_k}"
        if best_combo:
            print(f"    {size}: {best_combo} ({int(best_cyc):,} cycles)")


# ─── Main ─────────────────────────────────────────────────────────────────

def parse_sizes_arg(s: str) -> list[tuple[int, list[tuple[int, int, int]]]]:
    """Map user-provided sizes to blocking candidates."""
    result = []
    size_to_blockings = {size: blks for size, blks in SIZE_BLOCKING_CANDIDATES}
    for token in s.split(","):
        n = int(token.strip())
        if n in size_to_blockings:
            result.append((n, size_to_blockings[n]))
        elif n <= 128:
            result.append((n, [(32, 32, 32)]))
        elif n <= 256:
            result.append((n, [(32, 32, 32), (64, 64, 32)]))
        else:
            result.append((n, [(64, 64, 32), (64, 64, 64), (128, 128, 32)]))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=("a", "b", "ab"), default="ab",
                   help="which phase to run (default: ab = both)")
    p.add_argument("--jobs", "-j", type=int, default=None,
                   help="max parallel gem5 instances (default: nproc/4)")
    p.add_argument("--dry-run", action="store_true",
                   help="print commands without executing")
    p.add_argument("--sizes", default=None,
                   help="comma-separated M=N=K values (e.g. '64,256,512')")
    p.add_argument("--micro-m", default=None,
                   help="Phase B: comma-separated microM values (default: 4,8,16)")
    p.add_argument("--window-k", default=None,
                   help="Phase B: comma-separated windowK values (default: 2,4,8)")
    args = p.parse_args()

    max_workers = args.jobs or max(1, os.cpu_count() // 4)
    sizes = parse_sizes_arg(args.sizes) if args.sizes else SIZE_BLOCKING_CANDIDATES
    micro_ms = [int(x) for x in args.micro_m.split(",")] if args.micro_m else DEFAULT_MICRO_M
    window_ks = [int(x) for x in args.window_k.split(",")] if args.window_k else DEFAULT_WINDOW_K

    run_a = "a" in args.phase
    run_b = "b" in args.phase

    if run_a:
        print("=" * 60)
        print("PHASE A: Blocking sweep (find best blocking per size)")
        print("=" * 60)
        jobs = phase_a_jobs(sizes)
        n_spm = sum(1 for _, cmd, _ in jobs if "cache" not in cmd)
        n_cache = len(jobs) - n_spm
        print(f"  {n_spm} SPM + {n_cache} cache = {len(jobs)} total runs")
        run_parallel(jobs, max_workers, args.dry_run)
        if not args.dry_run:
            best = find_best_blocking(sizes)
            print()

    if run_b:
        print("=" * 60)
        print("PHASE B: microM × windowK sweep (on best blocking)")
        print("=" * 60)
        if run_a and not args.dry_run:
            pass  # best already computed
        else:
            if args.dry_run and run_a:
                # For dry-run, use first blocking as placeholder
                best = {size: blks[0] for size, blks in sizes}
            else:
                best = load_best_blocking()
                # Filter to requested sizes
                requested = {size for size, _ in sizes}
                best = {k: v for k, v in best.items() if k in requested}

        jobs = phase_b_jobs(best, micro_ms, window_ks)
        print(f"  {len(best)} sizes × {len(micro_ms)} microM × {len(window_ks)} windowK = {len(jobs)} runs")
        run_parallel(jobs, max_workers, args.dry_run)
        if not args.dry_run:
            summarize_phase_b(best, micro_ms, window_ks)


if __name__ == "__main__":
    main()
