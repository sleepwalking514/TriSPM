#!/usr/bin/env python3
"""Print a compact SPM-vs-cache delta table from two gem5 stats dumps.

Inputs are explicit ROI stats files (one per mode); paths come from
trispm_paths.roi_stats_path.
"""

from __future__ import annotations

import argparse
from pathlib import Path


FIELDS = [
    ("numCycles", "system.cpu.numCycles"),
    ("simInsts", "simInsts"),
    ("simOps", "simOps"),
    ("spm_dma.transfers", "system.spm_dma.transfers"),
    ("spm_dma.bytes", "system.spm_dma.bytesTransferred"),
    ("spm.bytesRead", "system.spm.bytesRead::total"),
    ("spm.bankConflicts", "system.spm.bankConflicts"),
    ("l1d.demandMisses", "system.l1d.demandMisses::total"),
    ("l1d.overallMisses", "system.l1d.overallMisses::total"),
    ("l2.demandMisses", "system.l2cache.demandMisses::total"),
    ("l2.overallMisses", "system.l2cache.overallMisses::total"),
    ("issued.SimdMisc", "system.cpu.issuedInstType_0::SimdMisc"),
    ("issued.SimdFloatMultAcc", "system.cpu.issuedInstType_0::SimdFloatMultAcc"),
    ("issued.SimdWholeRegLoad", "system.cpu.issuedInstType_0::SimdWholeRegisterLoad"),
    ("issued.SimdWholeRegStore", "system.cpu.issuedInstType_0::SimdWholeRegisterStore"),
    ("issued.SimdUnitLoad", "system.cpu.issuedInstType_0::SimdUnitStrideLoad"),
    ("issued.SimdUnitStore", "system.cpu.issuedInstType_0::SimdUnitStrideStore"),
    ("issued.MemRead", "system.cpu.issuedInstType_0::MemRead"),
    ("issued.MemWrite", "system.cpu.issuedInstType_0::MemWrite"),
    ("issued.FloatMemRead", "system.cpu.issuedInstType_0::FloatMemRead"),
    ("issued.FloatMemWrite", "system.cpu.issuedInstType_0::FloatMemWrite"),
]


def stats_blocks(path: Path) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    saw_marker = False

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("---------- Begin Simulation Statistics"):
            saw_marker = True
            current = {}
            continue
        if line.startswith("---------- End Simulation Statistics"):
            if current is not None:
                blocks.append(current)
                current = None
            continue

        if not line or line.startswith("#") or line.startswith("-"):
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        if current is None:
            if saw_marker:
                continue
            current = {}

        if "." in parts[0] or parts[0] in {"simInsts", "simOps"}:
            current[parts[0]] = parts[1]

    if current is not None and not saw_marker:
        blocks.append(current)

    return blocks


def load_stats(path: Path, section: str) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    blocks = stats_blocks(path)
    if not blocks:
        return {}
    return blocks[0] if section == "first" else blocks[-1]


def as_number(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fmt(value: str | None) -> str:
    return "-" if value is None else value


def fmt_delta(spm_value: str | None, cache_value: str | None) -> str:
    spm_num = as_number(spm_value)
    cache_num = as_number(cache_value)
    if spm_num is None or cache_num is None:
        return "-"
    delta = spm_num - cache_num
    if cache_num == 0:
        return f"{delta:+.0f}"
    return f"{delta:+.0f} ({delta / cache_num:+.1%})"


def print_summary(spm: dict[str, str], cache: dict[str, str], measure_iters: int) -> None:
    print(render_summary(spm, cache, measure_iters))


def render_summary(spm: dict[str, str], cache: dict[str, str], measure_iters: int) -> str:
    rows = []
    if measure_iters > 1:
        spm_cycles = as_number(spm.get("system.cpu.numCycles"))
        cache_cycles = as_number(cache.get("system.cpu.numCycles"))
        rows.append((
            "avgCycles/iter",
            "-" if spm_cycles is None else f"{spm_cycles / measure_iters:.1f}",
            "-" if cache_cycles is None else f"{cache_cycles / measure_iters:.1f}",
            fmt_delta(
                None if spm_cycles is None else str(spm_cycles / measure_iters),
                None if cache_cycles is None else str(cache_cycles / measure_iters),
            ),
        ))
    for label, stat_name in FIELDS:
        spm_value = spm.get(stat_name)
        cache_value = cache.get(stat_name)
        rows.append((label, fmt(spm_value), fmt(cache_value), fmt_delta(spm_value, cache_value)))

    widths = [
        max(len("stat"), *(len(row[0]) for row in rows)),
        max(len("spm"), *(len(row[1]) for row in rows)),
        max(len("cache"), *(len(row[2]) for row in rows)),
        max(len("delta"), *(len(row[3]) for row in rows)),
    ]

    header = ("stat", "spm", "cache", "delta")
    lines = [
        f"{header[0]:<{widths[0]}}  {header[1]:>{widths[1]}}  {header[2]:>{widths[2]}}  {header[3]:>{widths[3]}}",
        f"{'-' * widths[0]}  {'-' * widths[1]}  {'-' * widths[2]}  {'-' * widths[3]}",
    ]
    for row in rows:
        lines.append(f"{row[0]:<{widths[0]}}  {row[1]:>{widths[1]}}  {row[2]:>{widths[2]}}  {row[3]:>{widths[3]}}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spm", type=Path, required=True, help="ROI stats from SPM run")
    parser.add_argument("--cache", type=Path, required=True, help="ROI stats from cache-baseline run")
    parser.add_argument(
        "--section",
        choices=("first", "last"),
        default="first",
        help="which stats section to read when a file contains multiple dumps",
    )
    parser.add_argument(
        "--measure-iters",
        type=int,
        default=1,
        help="kernel launches inside the measured ROI; prints avgCycles/iter when >1",
    )
    parser.add_argument("--output", type=Path, default=None, help="write table to file instead of stdout")
    parser.add_argument("--quiet", action="store_true", help="do not print when writing --output")
    args = parser.parse_args()

    spm = load_stats(args.spm, args.section)
    cache = load_stats(args.cache, args.section)
    summary = render_summary(spm, cache, args.measure_iters)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(summary + "\n")
        if not args.quiet:
            print(f"Compare table written to {args.output}")
    else:
        print(summary)


if __name__ == "__main__":
    main()
