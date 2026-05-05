#!/usr/bin/env python3
"""Run graph compare and emit Phase 6-friendly evaluation artifacts.

The output is intentionally compact and machine-readable enough for later
paper aggregation: it records the manifest path, placement decisions, run
artifact paths, result-gate status, and selected SPM/cache stats.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import compare_stats
import graph_placement
import trispm_paths
from trispm_paths import WORKLOADS_DIR


SUMMARY_STATS = [
    ("numCycles", "system.cpu.numCycles"),
    ("simInsts", "simInsts"),
    ("ipc", "system.cpu.ipc"),
    ("l1d.demandMisses", "system.l1d.demandMisses::total"),
    ("l2.demandMisses", "system.l2cache.demandMisses::total"),
    ("spm_dma.transfers", "system.spm_dma.transfers"),
    ("spm_dma.bytes", "system.spm_dma.bytesTransferred"),
    ("spm_dma.waitStallCycles", "system.spm_dma.waitStallCycles"),
    ("spm.bytesRead", "system.spm.bytesRead::total"),
    ("spm.bytesWritten", "system.spm.bytesWritten::total"),
]


def rel(path: Path) -> str:
    return str(path.relative_to(WORKLOADS_DIR))


def run_graph_compare(graph: str, skip_build: bool, gem5_flags: list[str]) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "graph_placement.py"),
        graph,
        "--mode",
        "compare",
    ]
    if skip_build:
        cmd.append("--skip-build")
    for flag in gem5_flags:
        cmd.append(f"--gem5-flag={flag}")
    subprocess.run(cmd, check=True)


def load_graph_plan(graph: str) -> tuple[dict, list[graph_placement.NodePlan]]:
    data = graph_placement.load_graph(graph)
    plans = graph_placement.build_plan(graph, data)
    return data, plans


def placement_records(plans: list[graph_placement.NodePlan]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for plan in plans:
        for idx, tensor in enumerate(plan.args):
            decision = plan.decisions[idx]
            records.append({
                "node": plan.name,
                "kernel": plan.kernel,
                "arg_index": idx,
                "tensor": tensor,
                "tier": decision.tier,
                "reason": decision.reason,
            })
    return records


def selected_stats(stats: dict[str, str]) -> dict[str, str | None]:
    return {label: stats.get(name) for label, name in SUMMARY_STATS}


def numeric(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def delta_record(spm: dict[str, str], cache: dict[str, str]) -> dict[str, object]:
    spm_cycles = numeric(spm.get("system.cpu.numCycles"))
    cache_cycles = numeric(cache.get("system.cpu.numCycles"))
    if spm_cycles is None or cache_cycles is None:
        return {}
    delta = spm_cycles - cache_cycles
    return {
        "spm_cycles": spm_cycles,
        "cache_cycles": cache_cycles,
        "delta_cycles": delta,
        "delta_pct": None if cache_cycles == 0 else delta / cache_cycles,
        "speedup_cache_over_spm": None if spm_cycles == 0 else cache_cycles / spm_cycles,
    }


def render_summary(payload: dict[str, object]) -> str:
    delta = payload.get("cycle_delta", {})
    if not isinstance(delta, dict):
        delta = {}
    lines = [
        f"graph: {payload['graph']}",
        f"manifest: {payload['manifest']}",
        f"compare: {payload['artifacts']['compare']}",
        f"report: {payload['artifacts']['report']}",
        "",
        "result gates:",
    ]
    result_gates = payload.get("result_gates", {})
    if isinstance(result_gates, dict):
        for mode in ("spm", "cache"):
            lines.append(f"  {mode}: {result_gates.get(mode, 'UNKNOWN')}")
    if delta:
        pct = delta.get("delta_pct")
        speedup = delta.get("speedup_cache_over_spm")
        lines += [
            "",
            "cycle summary:",
            f"  spm_cycles: {delta.get('spm_cycles')}",
            f"  cache_cycles: {delta.get('cache_cycles')}",
            f"  delta_cycles: {delta.get('delta_cycles')}",
            f"  delta_pct: {'-' if pct is None else f'{pct:+.1%}'}",
            f"  cache_over_spm_speedup: {'-' if speedup is None else f'{speedup:.3f}x'}",
        ]
    lines += [
        "",
        "placement decisions:",
    ]
    for record in payload.get("placement", []):
        if not isinstance(record, dict):
            continue
        lines.append(
            f"  {record['node']}.{record['arg_index']} "
            f"{record['tensor']}: Tier {record['tier']} ({record['reason']})"
        )
    return "\n".join(lines) + "\n"


def write_eval_artifacts(graph: str, graph_data: dict, plans: list[graph_placement.NodePlan]) -> tuple[Path, Path]:
    spm_roi = trispm_paths.graph_roi_stats_path(graph, "spm")
    cache_roi = trispm_paths.graph_roi_stats_path(graph, "cache")
    spm_stats = compare_stats.load_stats(spm_roi, "first")
    cache_stats = compare_stats.load_stats(cache_roi, "first")
    manifest = graph_data.get("_path", WORKLOADS_DIR / "graphs" / graph / "graph.toml")

    payload: dict[str, object] = {
        "graph": graph,
        "manifest": rel(Path(manifest)),
        "artifacts": {
            "spm_roi_stats": rel(spm_roi),
            "cache_roi_stats": rel(cache_roi),
            "compare": rel(trispm_paths.graph_compare_path(graph)),
            "spm_stats": rel(trispm_paths.graph_spm_stats_path(graph)),
            "report": rel(trispm_paths.graph_report_path(graph)),
        },
        "result_gates": {
            "spm": graph_placement.graph_log_status(graph, "spm"),
            "cache": graph_placement.graph_log_status(graph, "cache"),
        },
        "placement": placement_records(plans),
        "stats": {
            "spm": selected_stats(spm_stats),
            "cache": selected_stats(cache_stats),
        },
        "cycle_delta": delta_record(spm_stats, cache_stats),
    }

    out_json = trispm_paths.graph_eval_json_path(graph)
    out_summary = trispm_paths.graph_eval_summary_path(graph)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    out_summary.write_text(render_summary(payload))
    return out_json, out_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", help="graph name under workloads/graphs/<name>/graph.toml")
    parser.add_argument("--skip-build", action="store_true",
                        help="reuse existing graph ELFs before running gem5")
    parser.add_argument("--skip-run", action="store_true",
                        help="reuse existing graph ROI stats and only rewrite eval artifacts")
    parser.add_argument("--gem5-flag", action="append", default=[],
                        help="extra gem5 run_spm.py flag passed through graph compare")
    args = parser.parse_args()

    graph_data, plans = load_graph_plan(args.graph)
    if not args.skip_run:
        run_graph_compare(args.graph, args.skip_build, args.gem5_flag)
    out_json, out_summary = write_eval_artifacts(args.graph, graph_data, plans)
    print(f"Phase 6 graph eval JSON:    {rel(out_json)}")
    print(f"Phase 6 graph eval summary: {rel(out_summary)}")


if __name__ == "__main__":
    main()
