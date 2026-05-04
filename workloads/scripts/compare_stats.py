#!/usr/bin/env python3
"""Print a compact SPM-vs-cache delta table from two gem5 stats dumps.

Inputs are explicit ROI stats files (one per mode); paths come from
trispm_paths.roi_stats_path. Two text outputs:
  - compare table:  symmetric SPM vs cache signals (cycles, IPC, miss rates,
                    snoop traffic, SIMD activity). One file per (kernel, tag).
  - spm_stats table: SPM-only counters (DMA traffic/latency/stalls, bank
                    conflict, per-port bandwidth). Has no cache counterpart,
                    so it lives as a sibling text report.
"""

from __future__ import annotations

import argparse
from pathlib import Path


# (label, gem5 stat name) — symmetric signals where SPM and cache both make sense.
COMPARE_FIELDS = [
    ("numCycles", "system.cpu.numCycles"),
    ("simInsts", "simInsts"),
    ("ipc", "system.cpu.ipc"),
    ("l1d.demandMisses", "system.l1d.demandMisses::total"),
    ("l1d.demandMissRate", "system.l1d.demandMissRate::total"),
    ("l2.demandMisses", "system.l2cache.demandMisses::total"),
    ("l2.demandMissRate", "system.l2cache.demandMissRate::total"),
    ("l2bus.snoops", "system.l2bus.snoops"),
    ("l2bus.snoopTraffic", "system.l2bus.snoopTraffic"),
    ("l2bus.snoopFilterRequests", "system.l2bus.snoop_filter.totRequests"),
    ("l2bus.snoopFilterSnoops", "system.l2bus.snoop_filter.totSnoops"),
    ("membus.snoops", "system.membus.snoops"),
    ("membus.snoopTraffic", "system.membus.snoopTraffic"),
    ("membus.snoopFilterRequests", "system.membus.snoop_filter.totRequests"),
    ("membus.snoopFilterSnoops", "system.membus.snoop_filter.totSnoops"),
    ("issued.SimdFloatMultAcc", "system.cpu.issuedInstType_0::SimdFloatMultAcc"),
    ("issued.MemRead", "system.cpu.issuedInstType_0::MemRead"),
    ("issued.MemWrite", "system.cpu.issuedInstType_0::MemWrite"),
    ("issued.FloatMemRead", "system.cpu.issuedInstType_0::FloatMemRead"),
    ("issued.FloatMemWrite", "system.cpu.issuedInstType_0::FloatMemWrite"),
]

# (label, gem5 stat name) — SPM-only. Reported in spm_stats.txt.
SPM_ONLY_FIELDS = [
    ("spm_dma.transfers", "system.spm_dma.transfers"),
    ("spm_dma.transfers2D", "system.spm_dma.transfers2D"),
    ("spm_dma.bytes", "system.spm_dma.bytesTransferred"),
    ("spm_dma.busyCycles", "system.spm_dma.busyCycles"),
    ("spm_dma.avgLatency", "system.spm_dma.avgLatency"),
    ("spm_dma.queueFullStalls", "system.spm_dma.queueFullStalls"),
    ("spm_dma.waitStallCycles", "system.spm_dma.waitStallCycles"),
    ("spm_dma.avgWaitStallCycles", "system.spm_dma.avgWaitStallCycles"),
    ("spm_dma.waitPollBusy", "system.spm_dma.waitPollBusy"),
    ("spm_dma.waitPollIdle", "system.spm_dma.waitPollIdle"),
    ("spm.bytesRead", "system.spm.bytesRead::total"),
    ("spm.bytesWritten", "system.spm.bytesWritten::total"),
    ("spm.numReads", "system.spm.numReads::total"),
    ("spm.numWrites", "system.spm.numWrites::total"),
    ("spm.bankConflicts", "system.spm.bankConflicts"),
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


def is_integer_text(value: str | None) -> bool:
    return value is not None and value.lstrip("+-").isdigit()


def fmt(value: str | None) -> str:
    return "-" if value is None else value


def fmt_delta(spm_value: str | None, cache_value: str | None) -> str:
    spm_num = as_number(spm_value)
    cache_num = as_number(cache_value)
    if spm_num is None or cache_num is None:
        return "-"
    delta = spm_num - cache_num
    # Use float formatting for fractional stats (rates, IPC), integer otherwise.
    delta_str = (
        f"{delta:+.0f}"
        if is_integer_text(spm_value) and is_integer_text(cache_value)
        else f"{delta:+.4f}" if abs(delta) < 1 else f"{delta:+.0f}"
    )
    if cache_num == 0:
        return delta_str
    return f"{delta_str} ({delta / cache_num:+.1%})"


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


COMPARE_HEADERS: tuple[str, ...] = ("stat", "spm", "cache", "delta")


def compare_rows(spm: dict[str, str], cache: dict[str, str], measure_iters: int) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
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
    for label, stat_name in COMPARE_FIELDS:
        spm_value = spm.get(stat_name)
        cache_value = cache.get(stat_name)
        rows.append((label, fmt(spm_value), fmt(cache_value), fmt_delta(spm_value, cache_value)))
    return rows


def render_compare(spm: dict[str, str], cache: dict[str, str], measure_iters: int) -> str:
    return render_table(compare_rows(spm, cache, measure_iters), COMPARE_HEADERS)


SPM_ONLY_HEADERS: tuple[str, ...] = ("stat", "value", "note")


def spm_only_rows(spm: dict[str, str], total_cycles: float | None) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    busy = as_number(spm.get("system.spm_dma.busyCycles"))
    if busy is not None and total_cycles and total_cycles > 0:
        rows.append((
            "spm_dma.busyFraction",
            f"{busy / total_cycles:.4f}",
            "fraction of CPU cycles DMA was busy",
        ))
    wait = as_number(spm.get("system.spm_dma.waitStallCycles"))
    if wait is not None and total_cycles and total_cycles > 0:
        rows.append((
            "spm_dma.waitFraction",
            f"{wait / total_cycles:.4f}",
            "fraction of CPU cycles spent in spm.dma.w polling",
        ))
    for label, stat_name in SPM_ONLY_FIELDS:
        rows.append((label, fmt(spm.get(stat_name)), ""))
    return rows


def render_spm_only(spm: dict[str, str], total_cycles: float | None) -> str:
    return render_table(spm_only_rows(spm, total_cycles), SPM_ONLY_HEADERS)


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
    parser.add_argument("--output", type=Path, default=None,
                        help="write compare table to file instead of stdout")
    parser.add_argument("--spm-only-output", type=Path, default=None,
                        help="also write SPM-only signals (DMA, banks, ...) to this path")
    parser.add_argument("--quiet", action="store_true", help="do not print when writing --output")
    args = parser.parse_args()

    spm = load_stats(args.spm, args.section)
    cache = load_stats(args.cache, args.section)
    compare_text = render_compare(spm, cache, args.measure_iters)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(compare_text + "\n")
        if not args.quiet:
            print(f"Compare table written to {args.output}")
    else:
        print(compare_text)

    if args.spm_only_output:
        spm_only_text = render_spm_only(spm, as_number(spm.get("system.cpu.numCycles")))
        args.spm_only_output.parent.mkdir(parents=True, exist_ok=True)
        args.spm_only_output.write_text(spm_only_text + "\n")
        if not args.quiet:
            print(f"SPM-only stats written to {args.spm_only_output}")


if __name__ == "__main__":
    main()
