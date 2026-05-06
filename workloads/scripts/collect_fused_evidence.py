#!/usr/bin/env python3
"""Collect fused-sweep results and generate docs/evidence/matmul-fused-scheduler.md."""
from __future__ import annotations

import re
from pathlib import Path

WORKLOADS = Path(__file__).resolve().parent.parent
M5OUT = WORKLOADS / "m5out" / "matmul" / "fused-sweep"
EVIDENCE_OUT = WORKLOADS.parent / "docs" / "evidence" / "matmul-fused-scheduler.md"

STATS_OF_INTEREST = [
    "system.cpu.numCycles",
    "system.cpu.ipc",
    "simInsts",
    "system.l1d.demandMissRate::total",
    "system.l1d.demandMisses::total",
    "system.l1d.demandAccesses::total",
    "system.l2cache.demandMissRate::total",
    "system.l2cache.demandMisses::total",
    "system.spm.bytesRead::cpu.data",
    "system.spm.bytesWritten::total",
    "system.spm_dma.transfers",
    "system.spm_dma.bytesTransferred",
    "system.spm_dma.busyCycles",
    "system.spm_dma.avgLatency",
    "system.spm_dma.waitStallCycles",
    "system.spm_dma.queueFullStalls",
]


def parse_stats(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            out[parts[0]] = parts[1]
    return out


def get_cycles(stats: dict[str, str]) -> int | None:
    v = stats.get("system.cpu.numCycles")
    if v:
        return int(v)
    return None


def parse_tag(dirname: str) -> dict[str, str]:
    """Parse directory name like '1024x1024x1024-64x64x64-uM8-wK4' or '...-cache'."""
    m = re.match(
        r"(\d+)x(\d+)(?:x(\d+))?-(\d+)x(\d+)x(\d+)-(?:uM(\d+)-wK(\d+)|cache)",
        dirname,
    )
    if not m:
        return {}
    size = m.group(1)
    k = m.group(3) or size
    return {
        "M": size, "N": m.group(2), "K": k,
        "BM": m.group(4), "BN": m.group(5), "BK": m.group(6),
        "microM": m.group(7) or "", "windowK": m.group(8) or "",
    }


def collect_spm() -> list[dict]:
    rows = []
    spm_dir = M5OUT / "spm"
    if not spm_dir.is_dir():
        return rows
    for d in sorted(spm_dir.iterdir()):
        if not d.is_dir():
            continue
        stats_file = d / "stats.txt"
        info = parse_tag(d.name)
        if not info or not info["microM"]:
            continue
        stats = parse_stats(stats_file)
        cyc = get_cycles(stats)
        if cyc is None:
            continue
        rows.append({**info, "cycles": cyc, "stats": stats})
    return rows


def collect_cache() -> list[dict]:
    rows = []
    cache_dir = M5OUT / "cache"
    if not cache_dir.is_dir():
        return rows
    for d in sorted(cache_dir.iterdir()):
        if not d.is_dir():
            continue
        stats_file = d / "stats.txt"
        info = parse_tag(d.name)
        if not info:
            continue
        stats = parse_stats(stats_file)
        cyc = get_cycles(stats)
        if cyc is None:
            continue
        rows.append({**info, "cycles": cyc, "stats": stats})
    return rows


def fmt_int(n: int) -> str:
    return f"{n:,}"


def generate_md(spm_rows: list[dict], cache_rows: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# Matmul Fused-Scheduler Evidence")
    lines.append("")
    lines.append("Collected from `workloads/m5out/matmul/fused-sweep/`.")
    lines.append("gem5 O3 CPU, SE mode, single-core, DMA queue depth = 32.")
    lines.append("")

    # --- Section 1: Blocking comparison (Phase A) ---
    lines.append("## 1. Blocking Comparison (Phase A)")
    lines.append("")
    lines.append("Fixed microM=8, windowK=4. SPM vs cache baseline per blocking config.")
    lines.append("")

    # Group cache by (M, BM, BN, BK)
    cache_by_key: dict[tuple, int] = {}
    for r in cache_rows:
        key = (r["M"], r["BM"], r["BN"], r["BK"])
        cache_by_key[key] = r["cycles"]

    # Group SPM Phase A points (uM8-wK4)
    phase_a = [r for r in spm_rows if r["microM"] == "8" and r["windowK"] == "4"]

    sizes_seen: list[str] = []
    for r in phase_a:
        if r["M"] not in sizes_seen:
            sizes_seen.append(r["M"])

    lines.append("| Size | Blocking | SPM cycles | Cache cycles | Speedup |")
    lines.append("|------|----------|-----------|-------------|---------|")
    for size in sizes_seen:
        size_rows = [r for r in phase_a if r["M"] == size]
        for r in sorted(size_rows, key=lambda x: x["cycles"]):
            key = (r["M"], r["BM"], r["BN"], r["BK"])
            cache_cyc = cache_by_key.get(key)
            speedup = f"{cache_cyc / r['cycles']:.2f}×" if cache_cyc else "-"
            cache_s = fmt_int(cache_cyc) if cache_cyc else "-"
            lines.append(
                f"| {size}³ | {r['BM']}×{r['BN']}×{r['BK']} | "
                f"{fmt_int(r['cycles'])} | {cache_s} | {speedup} |"
            )

    lines.append("")

    # --- Section 2: microM × windowK sweep (Phase B) ---
    lines.append("## 2. microM × windowK Sweep (Phase B)")
    lines.append("")
    lines.append("Best blocking per size from Phase A, sweep scheduler parameters.")
    lines.append("")

    # Determine best blocking per size
    best_blocking: dict[str, tuple[str, str, str]] = {}
    for size in sizes_seen:
        size_rows = [r for r in phase_a if r["M"] == size]
        if size_rows:
            best = min(size_rows, key=lambda x: x["cycles"])
            best_blocking[size] = (best["BM"], best["BN"], best["BK"])

    # For each size, build a table
    for size in sizes_seen:
        blk = best_blocking.get(size)
        if not blk:
            continue
        bm, bn, bk = blk
        size_rows = [
            r for r in spm_rows
            if r["M"] == size and r["BM"] == bm and r["BN"] == bn and r["BK"] == bk
        ]
        if not size_rows:
            continue

        # Get unique microM and windowK values
        micro_ms = sorted(set(r["microM"] for r in size_rows), key=int)
        window_ks = sorted(set(r["windowK"] for r in size_rows), key=int)

        cache_key = (size, bm, bn, bk)
        cache_cyc = cache_by_key.get(cache_key)

        lines.append(f"### {size}³ (blocking {bm}×{bn}×{bk}, cache={fmt_int(cache_cyc) if cache_cyc else '?'})")
        lines.append("")
        header = "| microM \\ windowK | " + " | ".join(f"wK={w}" for w in window_ks) + " |"
        sep = "|" + "---|" * (len(window_ks) + 1)
        lines.append(header)
        lines.append(sep)

        for um in micro_ms:
            cells = []
            for wk in window_ks:
                match = [r for r in size_rows if r["microM"] == um and r["windowK"] == wk]
                if match:
                    cells.append(fmt_int(match[0]["cycles"]))
                else:
                    cells.append("-")
            lines.append(f"| uM={um} | " + " | ".join(cells) + " |")

        lines.append("")

    # --- Section 3: Best configurations summary ---
    lines.append("## 3. Best Configurations")
    lines.append("")
    lines.append("| Size | Best Blocking | Best microM | Best windowK | SPM cycles | Cache cycles | Speedup |")
    lines.append("|------|--------------|-------------|-------------|-----------|-------------|---------|")

    for size in sizes_seen:
        size_rows = [r for r in spm_rows if r["M"] == size]
        if not size_rows:
            continue
        best = min(size_rows, key=lambda x: x["cycles"])
        cache_key = (size, best["BM"], best["BN"], best["BK"])
        cache_cyc = cache_by_key.get(cache_key)
        # Also find best cache across all blockings for this size
        all_cache = [c for c in cache_rows if c["M"] == size]
        best_cache = min(all_cache, key=lambda x: x["cycles"]) if all_cache else None
        best_cache_cyc = best_cache["cycles"] if best_cache else cache_cyc
        speedup = f"{best_cache_cyc / best['cycles']:.2f}×" if best_cache_cyc else "-"
        lines.append(
            f"| {size}³ | {best['BM']}×{best['BN']}×{best['BK']} | "
            f"{best['microM']} | {best['windowK']} | "
            f"{fmt_int(best['cycles'])} | {fmt_int(best_cache_cyc) if best_cache_cyc else '-'} | {speedup} |"
        )

    lines.append("")

    # --- Section 4: Key observations (scheduler parameter trends) ---
    lines.append("## 4. Scheduler Parameter Trends")
    lines.append("")
    lines.append("- **microM=8 is the sweet spot** for sizes ≥256³. At 64³ microM=4 wins")
    lines.append("  (fewer descriptors matter more when total work is small).")
    lines.append("- **Larger windowK is consistently better** (more B-tile residency in SPM")
    lines.append("  reduces DMA traffic). wK=8 > wK=4 > wK=2 across all sizes ≥256³.")
    lines.append("- **64³ is insensitive to windowK** because K/BK=2, so any wK≥2 covers")
    lines.append("  the entire B working set.")
    lines.append("- **32×32×32 blocking dominates for SPM** even at 1024³, despite 8× more")
    lines.append("  DMA descriptors than 64×64×32. The smaller tile size enables better")
    lines.append("  latency hiding and reduces SPM capacity pressure.")
    lines.append("- **microM=16/32 degrades significantly** at large sizes — gem5 O3 pipeline")
    lines.append("  stalls from the longer micro-loop body outweigh reduced descriptor count.")
    lines.append("- **DMA queue depth 32** eliminates all queueFullStalls (previously observed")
    lines.append("  with depth=4 at 1024³/32×32×32).")
    lines.append("")

    # --- Section 5: Detailed SPM vs Cache comparison ---
    lines.append("## 5. SPM vs Cache Microarchitecture Breakdown (Best Config per Size)")
    lines.append("")
    lines.append("Why is SPM faster? Per-size comparison of the best SPM config against")
    lines.append("the best cache config (lowest-cycle blocking for each).")
    lines.append("")

    detail_metrics = [
        ("Cycles", "system.cpu.numCycles", "int"),
        ("IPC", "system.cpu.ipc", "float"),
        ("Instructions", "simInsts", "int"),
        ("L1d miss rate", "system.l1d.demandMissRate::total", "pct"),
        ("L1d misses", "system.l1d.demandMisses::total", "int"),
        ("L1d accesses", "system.l1d.demandAccesses::total", "int"),
        ("L2 miss rate", "system.l2cache.demandMissRate::total", "pct"),
        ("L2 misses", "system.l2cache.demandMisses::total", "int"),
    ]

    spm_detail_metrics = [
        ("SPM reads (CPU)", "system.spm.bytesRead::cpu.data", "bytes"),
        ("SPM writes (total)", "system.spm.bytesWritten::total", "bytes"),
        ("DMA transfers", "system.spm_dma.transfers", "int"),
        ("DMA bytes", "system.spm_dma.bytesTransferred", "bytes"),
        ("DMA busy cycles", "system.spm_dma.busyCycles", "int"),
        ("DMA avg latency", "system.spm_dma.avgLatency", "float"),
        ("DMA wait stalls", "system.spm_dma.waitStallCycles", "int"),
        ("DMA queue full", "system.spm_dma.queueFullStalls", "int"),
    ]

    def fmt_stat(val: str | None, typ: str) -> str:
        if not val or val == "-":
            return "-"
        try:
            if typ == "int":
                return fmt_int(int(val))
            elif typ == "float":
                return f"{float(val):.3f}"
            elif typ == "pct":
                return f"{float(val)*100:.2f}%"
            elif typ == "bytes":
                b = int(val)
                if b >= 1024*1024:
                    return f"{b/(1024*1024):.1f} MiB"
                elif b >= 1024:
                    return f"{b/1024:.1f} KiB"
                return fmt_int(b)
        except (ValueError, TypeError):
            return val
        return val

    for size in sizes_seen:
        # Best SPM
        size_spm = [r for r in spm_rows if r["M"] == size]
        if not size_spm:
            continue
        best_spm = min(size_spm, key=lambda x: x["cycles"])
        # Best cache
        size_cache = [r for r in cache_rows if r["M"] == size]
        if not size_cache:
            continue
        best_cache_r = min(size_cache, key=lambda x: x["cycles"])

        lines.append(f"### {size}³")
        lines.append("")
        spm_label = f"SPM (uM{best_spm['microM']}-wK{best_spm['windowK']}, {best_spm['BM']}×{best_spm['BN']}×{best_spm['BK']})"
        cache_label = f"Cache ({best_cache_r['BM']}×{best_cache_r['BN']}×{best_cache_r['BK']})"
        lines.append(f"| Metric | {spm_label} | {cache_label} |")
        lines.append("|--------|---|---|")

        for label, stat, typ in detail_metrics:
            sv = fmt_stat(best_spm["stats"].get(stat), typ)
            cv = fmt_stat(best_cache_r["stats"].get(stat), typ)
            lines.append(f"| {label} | {sv} | {cv} |")

        lines.append(f"| | | |")
        lines.append(f"| **SPM-only metrics** | | |")
        for label, stat, typ in spm_detail_metrics:
            sv = fmt_stat(best_spm["stats"].get(stat), typ)
            lines.append(f"| {label} | {sv} | - |")

        lines.append("")

    # --- Section 6: Key observations ---
    # (renumber from 4 to 6 since we inserted section 5)

    lines.append("## 6. Interpretability Notes")
    lines.append("")
    lines.append("**Why SPM wins — the mechanism:**")
    lines.append("")
    lines.append("1. **Near-zero L1d miss rate under SPM.** The CPU loads/stores hit SPM at")
    lines.append("   L1 latency (1 cycle). The remaining L1d misses are stack/control only.")
    lines.append("2. **DMA hides memory latency.** While the CPU computes on tile N, DMA")
    lines.append("   prefetches tile N+1 from DRAM→SPM in the background. The `waitStallCycles`")
    lines.append("   metric shows how much time the CPU actually blocked waiting for DMA.")
    lines.append("3. **Higher IPC.** Fewer cache misses → fewer pipeline stalls → more useful")
    lines.append("   work per cycle.")
    lines.append("4. **Cache baseline suffers from capacity misses.** The working set exceeds")
    lines.append("   L1d (32 KiB) and often L2, causing high miss rates and long stalls.")
    lines.append("")
    lines.append("**Why smaller blocking helps SPM more than cache:**")
    lines.append("")
    lines.append("- Smaller tiles fit entirely in SPM with room for double-buffering.")
    lines.append("- Each DMA transfer is short → completes before the CPU finishes the")
    lines.append("  current tile → near-perfect overlap.")
    lines.append("- Cache with small tiles loses spatial locality (stride patterns don't")
    lines.append("  align with cache lines), increasing miss rate.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    spm_rows = collect_spm()
    cache_rows = collect_cache()
    print(f"Collected {len(spm_rows)} SPM runs, {len(cache_rows)} cache runs")

    md = generate_md(spm_rows, cache_rows)
    EVIDENCE_OUT.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_OUT.write_text(md)
    print(f"Written to {EVIDENCE_OUT}")


if __name__ == "__main__":
    main()
