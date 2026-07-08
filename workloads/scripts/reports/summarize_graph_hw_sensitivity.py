#!/usr/bin/env python3
"""Summarize decoder MH8 hardware-sensitivity rows against default SPM.

This is a postprocess-only helper.  It reads the ROI stats produced by
the `graph-hw-sensitivity` phase in paper_experiments.py and treats
paper-hw-default as the denominator.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


VARIANTS = [
    ("paper-hw-default", "Default", ""),
    ("paper-hw-spm-lat-6ns", "SPM latency 6ns", "--spm_lat 6ns"),
    ("paper-hw-spm-bw-16gib", "SPM bandwidth 16GiB/s", "--spm_bw 16GiB/s"),
    ("paper-hw-spm-banks-4", "SPM banks 4", "--spm_num_banks 4"),
    (
        "paper-hw-dma-ctrl-4x",
        "DMA control path 4x",
        "--dma_pio_lat 20ns --dma_desc_lat 40ns",
    ),
]


def load_stats(path: Path) -> dict[str, str]:
    stats: dict[str, str] = {}
    in_block = False
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("---------- Begin Simulation Statistics"):
            in_block = True
            continue
        if line.startswith("---------- End Simulation Statistics") and in_block:
            break
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0]
        if "." in name or name in {"simInsts", "simOps"}:
            stats[name] = parts[1]
    return stats


def number(stats: dict[str, str], name: str) -> float | None:
    try:
        return float(stats[name])
    except (KeyError, ValueError):
        return None


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(row))) + " |"

    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    return "\n".join([fmt(headers), sep] + [fmt(row) for row in rows])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-out",
        default="decoder_canonical_mh8/large",
        help="graph output directory under workloads/m5out/graphs",
    )
    parser.add_argument(
        "--workloads-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="default: workloads/m5out/graphs/<graph-out>/hw_sensitivity_summary",
    )
    args = parser.parse_args()

    workloads = args.workloads_dir
    graph_root = workloads / "m5out" / "graphs" / args.graph_out
    out_dir = args.output_dir or (graph_root / "hw_sensitivity_summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    default_roi = graph_root / "paper-hw-default" / "default" / "roi-stats.txt"
    if not default_roi.is_file():
        raise SystemExit(f"missing default SPM ROI: {default_roi}")
    default_stats = load_stats(default_roi)
    default_cycles = number(default_stats, "system.cpu.numCycles")
    if not default_cycles:
        raise SystemExit(f"default SPM cycles unavailable in {default_roi}")

    rows: list[dict[str, str]] = []
    for tag, label, flags in VARIANTS:
        roi = graph_root / tag / "default" / "roi-stats.txt"
        if not roi.is_file():
            raise SystemExit(f"missing ROI for {tag}: {roi}")
        stats = load_stats(roi)
        cycles = number(stats, "system.cpu.numCycles")
        wait = number(stats, "system.spm_dma.waitStallCycles")
        busy = number(stats, "system.spm_dma.busyCycles")
        slowdown = None if cycles is None else cycles / default_cycles
        cycle_delta_pct = None if cycles is None else (cycles - default_cycles) / default_cycles
        rows.append({
            "tag": tag,
            "label": label,
            "flags": flags,
            "cycles": "" if cycles is None else f"{cycles:.0f}",
            "slowdown_vs_default_spm": "" if slowdown is None else f"{slowdown:.3f}",
            "cycle_delta_vs_default_pct": "" if cycle_delta_pct is None else f"{cycle_delta_pct:+.1%}",
            "ipc": stats.get("system.cpu.ipc", ""),
            "l1d_misses": stats.get("system.l1d.demandMisses::total", ""),
            "l2_misses": stats.get("system.l2cache.demandMisses::total", ""),
            "dma_bytes": stats.get("system.spm_dma.bytesTransferred", ""),
            "dma_busy_frac": "" if busy is None or not cycles else f"{busy / cycles:.4f}",
            "dma_wait_frac": "" if wait is None or not cycles else f"{wait / cycles:.4f}",
            "bank_conflicts": stats.get("system.spm.bankConflicts", ""),
            "roi": str(roi.relative_to(workloads)),
        })

    csv_path = out_dir / "summary_vs_default_spm.csv"
    fields = [
        "tag", "label", "flags", "cycles", "slowdown_vs_default_spm",
        "cycle_delta_vs_default_pct", "ipc", "l1d_misses", "l2_misses",
        "dma_bytes", "dma_busy_frac", "dma_wait_frac", "bank_conflicts", "roi",
    ]
    with csv_path.open("w", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    table_rows = [
        [
            row["label"],
            f"`{row['flags']}`" if row["flags"] else "`<default>`",
            row["cycles"],
            f"{row['slowdown_vs_default_spm']}x",
            row["cycle_delta_vs_default_pct"],
            row["ipc"],
            row["dma_wait_frac"],
            row["bank_conflicts"],
        ]
        for row in rows
    ]
    md_path = out_dir / "summary_vs_default_spm.md"
    md_path.write_text(
        "# Decoder MH8 Hardware Sensitivity vs Default SPM\n\n"
        f"Graph: `{args.graph_out}`\n\n"
        f"Default SPM ROI: `{default_roi.relative_to(workloads)}`\n\n"
        + format_table(
            [
                "Variant", "Flags", "Cycles", "Slowdown vs default",
                "Cycle delta", "IPC", "DMA wait", "Bank conflicts",
            ],
            table_rows,
        )
        + "\n",
    )

    print(f"Summary CSV: {csv_path}")
    print(f"Summary MD:  {md_path}")


if __name__ == "__main__":
    main()
