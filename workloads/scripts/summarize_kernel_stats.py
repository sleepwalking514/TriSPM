#!/usr/bin/env python3
"""Summarize per-kernel gem5 stats from graph profiling runs.

Profiling graph harnesses print one `KERNEL_STATS <label>` line immediately
before `m5_dump_reset_stats(0, 0)`.  This script pairs those labels with the
corresponding stats blocks in gem5 `stats.txt` and emits a compact table.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import compare_stats


FIELDS = [
    ("cycles", "system.cpu.numCycles"),
    ("simInsts", "simInsts"),
    ("ipc", "system.cpu.ipc"),
    ("l1d_misses", "system.l1d.demandMisses::total"),
    ("l2_misses", "system.l2cache.demandMisses::total"),
    ("dram_read_bytes", compare_stats.DRAM_READ_STAT),
    ("dram_write_bytes", compare_stats.DRAM_WRITE_STAT),
    ("dma_transfers", "system.spm_dma.transfers"),
    ("dma_bytes", "system.spm_dma.bytesTransferred"),
    ("dma_wait", "system.spm_dma.waitStallCycles"),
]

FIELD_COLUMNS = [name for name, _stat in FIELDS] + ["dram_total_bytes"]


def kernel_labels(run_log: Path) -> list[str]:
    labels: list[str] = []
    for raw in run_log.read_text(errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("KERNEL_STATS "):
            labels.append(line.split(None, 1)[1])
    return labels


def pick(stats: dict[str, str], name: str) -> str:
    return stats.get(name, "")


def dram_total(stats: dict[str, str]) -> str:
    return compare_stats.sum_stats(
        stats,
        (compare_stats.DRAM_READ_STAT, compare_stats.DRAM_WRITE_STAT),
    ) or ""


def rows(labels: list[str], blocks: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for idx, label in enumerate(labels):
        block = blocks[idx] if idx < len(blocks) else {}
        row = {"index": str(idx), "kernel": label}
        for column, stat_name in FIELDS:
            row[column] = pick(block, stat_name)
        row["dram_total_bytes"] = dram_total(block)
        out.append(row)
    return out


def render_markdown(records: list[dict[str, str]]) -> str:
    headers = ["index", "kernel", *FIELD_COLUMNS]
    widths = [
        max(len(header), *(len(row.get(header, "")) for row in records))
        for header in headers
    ]

    def fmt(row: dict[str, str]) -> str:
        return "| " + " | ".join(
            row.get(header, "").ljust(widths[i]) for i, header in enumerate(headers)
        ) + " |"

    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [
        fmt({header: header for header in headers}),
        sep,
        *(fmt(row) for row in records),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", required=True, type=Path,
                        help="gem5 stats.txt from a GRAPH_DUMP_KERNEL_STATS run")
    parser.add_argument("--run-log", required=True, type=Path,
                        help="matching graph run.log containing KERNEL_STATS labels")
    parser.add_argument("--output", type=Path, default=None,
                        help="optional Markdown output path")
    parser.add_argument("--csv", type=Path, default=None,
                        help="optional CSV output path")
    args = parser.parse_args()

    labels = kernel_labels(args.run_log)
    blocks = compare_stats.stats_blocks(args.stats)
    records = rows(labels, blocks)
    if not labels:
        raise SystemExit(f"ERROR: no KERNEL_STATS labels found in {args.run_log}")
    if len(blocks) < len(labels):
        raise SystemExit(
            f"ERROR: only {len(blocks)} stats blocks for {len(labels)} kernel labels"
        )
    if len(blocks) > len(labels):
        extra = len(blocks) - len(labels)
        print(f"warning: ignoring {extra} extra stats block(s) after kernel labels")

    text = render_markdown(records)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["index", "kernel", *FIELD_COLUMNS]
            )
            writer.writeheader()
            writer.writerows(records)


if __name__ == "__main__":
    main()
