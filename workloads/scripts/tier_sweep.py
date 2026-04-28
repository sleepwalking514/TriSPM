#!/usr/bin/env python3
"""Tier x swizzle sweep for matmul (SPM build only).

Runs four configurations through run_experiment.py and prints a comparison
table:

  T2-GSM4 : A,B in Tier 2 (cacheable+DMA), GROUP_SIZE_M from manifest
  T2-GSM1 : A,B in Tier 2,                 GROUP_SIZE_M=1 (no swizzle)
  T3-GSM4 : A,B in Tier 3 (uncacheable DMA buf), GROUP_SIZE_M from manifest
  T3-GSM1 : A,B in Tier 3,                       GROUP_SIZE_M=1

Tier override flows through KERNEL_TIER_OVERRIDE, consumed by the Triton CPU
launcher generator (compiler/third_party/cpu/backend/driver.py).  C is left
at the SPMTensorPlacement default in every config (it is store-only and
falls through to Tier 2).

Usage:
  scripts/tier_sweep.py                      # 256x256x256 / 32x32x32 (manifest defaults)
  scripts/tier_sweep.py --size 64 --block 32 # smoke test
  scripts/tier_sweep.py --gsm 8              # override GSM 'big' value (default: from manifest)
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

import trispm_paths
from trispm_paths import WORKLOADS_DIR

SCRIPTS_DIR = Path(__file__).resolve().parent
KERNEL = "matmul"

# (label, gem5 stat name) — pulled into the comparison table.
METRICS = [
    ("numCycles",            "system.cpu.numCycles"),
    ("simInsts",             "simInsts"),
    ("ipc",                  "system.cpu.ipc"),
    ("l1d.demandMisses",     "system.l1d.demandMisses::total"),
    ("l1d.demandMissRate",   "system.l1d.demandMissRate::total"),
    ("l2.demandMisses",      "system.l2cache.demandMisses::total"),
    ("l2.demandMissRate",    "system.l2cache.demandMissRate::total"),
    ("spm_dma.bytes",        "system.spm_dma.bytesTransferred"),
    ("spm_dma.busyCycles",   "system.spm_dma.busyCycles"),
    ("spm_dma.waitStall",    "system.spm_dma.waitStallCycles"),
    ("spm.bankConflicts",    "system.spm.bankConflicts"),
]


def parse_stats(path: Path) -> dict[str, str]:
    """Parse the (single) ROI block emitted by run_gem5.sh into a flat dict."""
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        if "." in parts[0] or parts[0] in {"simInsts", "simOps"}:
            out[parts[0]] = parts[1]
    return out


def run(cmd: list[str], env: dict[str, str]) -> None:
    print(f"$ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env)


def run_one(label: str, tag: str, sets: list[str], tier_override: str) -> Path:
    env = os.environ.copy()
    env["KERNEL_TIER_OVERRIDE"] = tier_override
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "run_experiment.py"),
        KERNEL,
        "--mode", "spm",
        "--tag", tag,
    ]
    for kv in sets:
        cmd += ["--set", kv]
    print(f"\n========== {label}  (tier_override={tier_override!r}) ==========")
    run(cmd, env=env)
    return trispm_paths.roi_stats_path(KERNEL, "spm", tag)


def fmt_value(v: str | None) -> str:
    if v is None:
        return "-"
    try:
        f = float(v)
    except ValueError:
        return v
    if abs(f) >= 1 and f == int(f):
        return f"{int(f)}"
    return f"{f:.4f}"


def render_table(rows: list[tuple[str, ...]], headers: tuple[str, ...]) -> str:
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    fmt_row = lambda row: "  ".join(
        (cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i]))
        for i, cell in enumerate(row)
    )
    sep = "  ".join("-" * w for w in widths)
    return "\n".join([fmt_row(headers), sep] + [fmt_row(r) for r in rows])


def manifest_param(key: str) -> int:
    path = WORKLOADS_DIR / "kernels" / KERNEL / "experiment.toml"
    data = tomllib.loads(path.read_text())
    return int(data["params"][key])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", type=int, default=None,
                   help="set M=N=K (default: manifest)")
    p.add_argument("--block", type=int, default=None,
                   help="set BLOCK_SIZE_M=N=K (default: manifest)")
    p.add_argument("--gsm", type=int, default=None,
                   help="GROUP_SIZE_M for the 'big' arm (default: manifest)")
    p.add_argument("--prefix", default="tier-sweep",
                   help="m5out tag prefix (default: tier-sweep)")
    args = p.parse_args()

    size = args.size if args.size is not None else manifest_param("M")
    block = args.block if args.block is not None else manifest_param("BLOCK_SIZE_M")
    gsm_big = args.gsm if args.gsm is not None else manifest_param("GROUP_SIZE_M")
    if size % block != 0:
        sys.exit(f"ERROR: size {size} not divisible by block {block}")

    base_sets = [
        f"M={size}", f"N={size}", f"K={size}",
        f"BLOCK_SIZE_M={block}", f"BLOCK_SIZE_N={block}", f"BLOCK_SIZE_K={block}",
    ]
    tag_base = f"{args.prefix}/{size}x{size}x{size}/{block}x{block}x{block}"

    # (label, tier override, GSM value)
    configs = [
        ("T2-GSM4", "0=2,1=2", gsm_big),
        ("T2-GSM1", "0=2,1=2", 1),
        ("T3-GSM4", "0=3,1=3", gsm_big),
        ("T3-GSM1", "0=3,1=3", 1),
    ]

    label_to_tag = {label: f"{tag_base}/{label.lower()}" for label, _, _ in configs}
    label_to_stats: dict[str, dict[str, str]] = {}
    for label, override, gsm in configs:
        sets = base_sets + [f"GROUP_SIZE_M={gsm}"]
        roi = run_one(label, label_to_tag[label], sets, override)
        label_to_stats[label] = parse_stats(roi)

    # Sanity-dump: launcher allocator dispatch for every config, so the
    # summary self-attests which allocator each tier override actually
    # produced (T2 -> malloc, T3 -> dma_buf_malloc).
    dispatch_lines: list[str] = ["launcher dispatch (allocator per arg index):"]
    for label, _, _ in configs:
        path = trispm_paths.build_dir(KERNEL, "spm", label_to_tag[label]) / \
            f"{KERNEL}_launcher.c"
        dispatch_lines.append(f"  [{label}]")
        if path.is_file():
            for line in path.read_text().splitlines():
                if "case " in line and "return" in line:
                    dispatch_lines.append(f"    {line.strip()}")
        else:
            dispatch_lines.append(f"    <missing: {path}>")
    dispatch_block = "\n".join(dispatch_lines)
    print(f"\n{dispatch_block}")

    # Build comparison table.
    print(f"\n=========== tier x swizzle sweep ({size}^3, block {block}^3) ===========")
    headers = ("metric",) + tuple(label for label, _, _ in configs)
    rows: list[tuple[str, ...]] = []
    for label, stat in METRICS:
        row = (label,) + tuple(
            fmt_value(label_to_stats[cfg].get(stat))
            for cfg, _, _ in configs
        )
        rows.append(row)
    table_text = render_table(rows, headers)
    print(table_text)

    # Persist alongside the per-tag m5out artifacts.
    summary_path = trispm_paths.M5OUT_ROOT / KERNEL / args.prefix / \
        f"{size}x{size}x{size}_{block}x{block}x{block}_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(f"{dispatch_block}\n\n{table_text}\n")
    print(f"\nSummary written to {summary_path.relative_to(WORKLOADS_DIR)}")


if __name__ == "__main__":
    main()
