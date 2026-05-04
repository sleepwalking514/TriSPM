#!/usr/bin/env python3
"""Parse per-checkpoint L2 cache stats from dma_l2_warming gem5 output.

gem5 produces multiple stats blocks when m5_dump_stats() is called
repeatedly.  This script extracts the relevant l2cache counters from
each block and presents them as a text table.

Usage:
    python3 parse_l2_warming.py <stats.txt>
    python3 parse_l2_warming.py <m5out_dir>
"""

import sys
import re
import os
from pathlib import Path

PHASE_NAMES = ["A (DMA cacheable→SPM)", "B (scalar read, L2 warm)",
               "C (DMA UC→SPM)", "D (scalar read, L2 cold)"]

COUNTERS = [
    "system.cpu.numCycles",
    "system.l2cache.overallHits::total",
    "system.l2cache.overallMisses::total",
    "system.l2cache.overallAccesses::total",
    "system.l2cache.overallMissRate::total",
]


def parse_stats_blocks(path: Path) -> list[dict[str, str]]:
    """Split stats.txt into blocks and extract counters from each."""
    text = path.read_text()
    blocks = text.split("---------- Begin Simulation Statistics ----------")
    results = []
    for block in blocks[1:]:  # skip preamble before first block
        counters = {}
        for line in block.splitlines():
            for c in COUNTERS:
                if line.strip().startswith(c):
                    parts = line.split()
                    if len(parts) >= 2:
                        counters[c] = parts[1]
        if counters:
            results.append(counters)
    return results


def print_table(blocks: list[dict[str, str]]):
    """Pretty-print the per-phase stats."""
    # Header
    short_names = [c.split(".")[-1] for c in COUNTERS]
    header = f"{'Phase':<30s}" + "".join(f"{n:>20s}" for n in short_names)
    print(header)
    print("-" * len(header))

    for i, block in enumerate(blocks):
        name = PHASE_NAMES[i] if i < len(PHASE_NAMES) else f"Phase {i}"
        row = f"{name:<30s}"
        for c in COUNTERS:
            row += f"{block.get(c, 'N/A'):>20s}"
        print(row)

    # Summary comparison
    if len(blocks) >= 4:
        print("\n--- Key comparison: B vs D ---")
        b_hits = blocks[1].get("system.l2cache.overallHits::total", "?")
        d_hits = blocks[3].get("system.l2cache.overallHits::total", "?")
        b_misses = blocks[1].get("system.l2cache.overallMisses::total", "?")
        d_misses = blocks[3].get("system.l2cache.overallMisses::total", "?")
        print(f"  Phase B (L2-warm scalar read): hits={b_hits}, misses={b_misses}")
        print(f"  Phase D (cold scalar read):    hits={d_hits}, misses={d_misses}")
        if b_hits != "?" and d_hits != "?":
            bh, dh = int(b_hits), int(d_hits)
            if dh == 0 and bh > 0:
                print("  → L2-warming CONFIRMED: B has hits, D has none.")
            elif bh > dh:
                print(f"  → L2-warming CONFIRMED: B has {bh - dh} more L2 hits than D.")
            else:
                print("  → WARNING: L2-warming NOT observed. Check flush logic.")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    path = Path(sys.argv[1])

    if path.is_dir():
        path = path / "stats.txt"
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)

    blocks = parse_stats_blocks(path)
    if not blocks:
        print("ERROR: no stats blocks found", file=sys.stderr)
        sys.exit(1)

    print_table(blocks)


if __name__ == "__main__":
    main()
