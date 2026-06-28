#!/usr/bin/env python3
"""Run graph compare and emit compact evaluation artifacts.

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


def cache_capacity_suffix(
    cache_l1d_size: str | None = None,
    cache_l2_size: str | None = None,
) -> str:
    return graph_placement.cache_capacity_suffix(cache_l1d_size, cache_l2_size)


def graph_cache_capacity_m5out_dir(
    graph: str,
    cache_l1d_size: str | None = None,
    cache_l2_size: str | None = None,
) -> Path:
    return graph_placement.graph_cache_capacity_m5out_dir(
        graph,
        cache_l1d_size,
        cache_l2_size,
    )


def graph_cache_capacity_roi_stats_path(
    graph: str,
    cache_l1d_size: str | None = None,
    cache_l2_size: str | None = None,
) -> Path:
    return graph_cache_capacity_m5out_dir(
        graph, cache_l1d_size, cache_l2_size) / "roi-stats.txt"


def graph_cache_capacity_run_log_path(
    graph: str,
    cache_l1d_size: str | None = None,
    cache_l2_size: str | None = None,
) -> Path:
    return graph_cache_capacity_m5out_dir(
        graph, cache_l1d_size, cache_l2_size) / "run.log"


def graph_capacity_eval_json_path(
    graph: str,
    cache_l1d_size: str | None = None,
    cache_l2_size: str | None = None,
    spm_tag: str | None = None,
) -> Path:
    return graph_placement.graph_m5out_dir(graph, "spm", spm_tag) / (
        f"graph_eval_cache_{cache_capacity_suffix(cache_l1d_size, cache_l2_size)}.json"
    )


def graph_capacity_eval_summary_path(
    graph: str,
    cache_l1d_size: str | None = None,
    cache_l2_size: str | None = None,
    spm_tag: str | None = None,
) -> Path:
    return graph_placement.graph_m5out_dir(graph, "spm", spm_tag) / (
        f"graph_summary_cache_{cache_capacity_suffix(cache_l1d_size, cache_l2_size)}.txt"
    )


def graph_capacity_compare_path(
    graph: str,
    cache_l1d_size: str | None = None,
    cache_l2_size: str | None = None,
    spm_tag: str | None = None,
) -> Path:
    return graph_placement.graph_capacity_compare_path(
        graph, cache_l1d_size, cache_l2_size, spm_tag)


def graph_capacity_report_path(
    graph: str,
    cache_l1d_size: str | None = None,
    cache_l2_size: str | None = None,
    spm_tag: str | None = None,
) -> Path:
    return graph_placement.graph_capacity_report_path(
        graph, cache_l1d_size, cache_l2_size, spm_tag)


def run_graph_compare(
    graph: str,
    preset: str | None,
    artifact_tag: str | None,
    skip_build: bool,
    full_compare: bool,
    gem5_flags: list[str],
    cache_l1d_size: str | None,
    cache_l2_size: str | None,
    spm_tag: str | None,
) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "graph_placement.py"),
        graph,
        "--mode",
        "compare",
    ]
    if preset:
        cmd += ["--preset", preset]
    if artifact_tag:
        cmd += ["--artifact-tag", artifact_tag]
    if skip_build:
        cmd.append("--skip-build")
    if cache_l1d_size:
        cmd += ["--cache-l1d-size", cache_l1d_size]
    if cache_l2_size:
        cmd += ["--cache-l2-size", cache_l2_size]
    if (cache_l1d_size or cache_l2_size) and not full_compare:
        cmd.append("--cache-only")
    if spm_tag:
        cmd += ["--spm-tag", spm_tag]
    for flag in gem5_flags:
        cmd.append(f"--gem5-flag={flag}")
    subprocess.run(cmd, check=True)


def load_graph_plan(
    graph: str,
    preset: str | None,
    artifact_tag: str | None = None,
) -> tuple[str, dict, list[graph_placement.NodePlan]]:
    data = graph_placement.apply_graph_preset(graph_placement.load_graph(graph), preset)
    graph_output = graph_placement.graph_artifact_name(graph, preset, artifact_tag)
    plans = graph_placement.build_plan(graph_output, data)
    return graph_output, data, plans


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
            f"{record['tensor']}: {record['reason']}"
        )
    return "\n".join(lines) + "\n"


def write_eval_artifacts(
    graph: str,
    graph_data: dict,
    plans: list[graph_placement.NodePlan],
    cache_l1d_size: str | None = None,
    cache_l2_size: str | None = None,
    spm_tag: str | None = None,
) -> tuple[Path, Path]:
    spm_roi = trispm_paths.graph_roi_stats_path(graph, "spm", spm_tag)
    cache_roi = (
        graph_cache_capacity_roi_stats_path(graph, cache_l1d_size, cache_l2_size)
        if cache_l1d_size or cache_l2_size
        else trispm_paths.graph_roi_stats_path(graph, "cache")
    )
    spm_stats = compare_stats.load_stats(spm_roi, "first")
    cache_stats = compare_stats.load_stats(cache_roi, "first")
    manifest = graph_data.get("_path", WORKLOADS_DIR / "graphs" / graph / "graph.toml")
    compare = (
        graph_capacity_compare_path(graph, cache_l1d_size, cache_l2_size, spm_tag)
        if cache_l1d_size or cache_l2_size
        else trispm_paths.graph_compare_path(graph, spm_tag)
    )
    spm_only = trispm_paths.graph_spm_stats_path(graph, spm_tag)
    report = (
        graph_capacity_report_path(graph, cache_l1d_size, cache_l2_size, spm_tag)
        if cache_l1d_size or cache_l2_size
        else trispm_paths.graph_report_path(graph, spm_tag)
    )
    cache_run_log = (
        graph_cache_capacity_run_log_path(graph, cache_l1d_size, cache_l2_size)
        if cache_l1d_size or cache_l2_size
        else trispm_paths.graph_run_log_path(graph, "cache")
    )

    harness_params = graph_placement.graph_harness_params(graph_data, "spm")
    measure_iters = int(str(harness_params.get("MEASURE_ITERS", "1")), 0)

    if cache_l1d_size or cache_l2_size or spm_tag:
        compare.parent.mkdir(parents=True, exist_ok=True)
        compare.write_text(
            compare_stats.render_compare(spm_stats, cache_stats, measure_iters) + "\n"
        )
        spm_only.parent.mkdir(parents=True, exist_ok=True)
        spm_only.write_text(
            compare_stats.render_spm_only(
                spm_stats,
                compare_stats.as_number(spm_stats.get("system.cpu.numCycles")),
            )
            + "\n"
        )
        graph_placement.write_graph_report(
            graph,
            graph_data,
            plans,
            compare,
            spm_only,
            cache_l1d_size,
            cache_l2_size,
            spm_tag,
        )

    payload: dict[str, object] = {
        "graph": graph,
        "manifest": rel(Path(manifest)),
        "graph_preset": graph_data.get("_preset"),
        "cache_l1d_size": cache_l1d_size or "32KiB",
        "cache_l2_size": cache_l2_size or "512KiB",
        "spm_tag": spm_tag or "spm",
        "artifacts": {
            "spm_roi_stats": rel(spm_roi),
            "cache_roi_stats": rel(cache_roi),
            "cache_run_log": rel(cache_run_log),
            "compare": rel(compare),
            "spm_stats": rel(spm_only),
            "report": rel(report),
        },
        "result_gates": {
            "spm": graph_placement.graph_log_status(graph, "spm", spm_tag=spm_tag),
            "cache": (
                graph_placement.graph_log_status(
                    graph, "cache", cache_l1d_size, cache_l2_size)
                if cache_l1d_size or cache_l2_size
                else graph_placement.graph_log_status(graph, "cache")
            ),
        },
        "placement": placement_records(plans),
        "stats": {
            "spm": selected_stats(spm_stats),
            "cache": selected_stats(cache_stats),
        },
        "cycle_delta": delta_record(spm_stats, cache_stats),
    }

    out_json = (
        graph_capacity_eval_json_path(graph, cache_l1d_size, cache_l2_size, spm_tag)
        if cache_l1d_size or cache_l2_size else trispm_paths.graph_eval_json_path(graph, spm_tag)
    )
    out_summary = (
        graph_capacity_eval_summary_path(graph, cache_l1d_size, cache_l2_size, spm_tag)
        if cache_l1d_size or cache_l2_size
        else trispm_paths.graph_eval_summary_path(graph, spm_tag)
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    out_summary.write_text(render_summary(payload))
    return out_json, out_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", help="graph name under workloads/graphs/<name>/graph.toml")
    parser.add_argument("--preset", default=None,
                        help="graph preset under [presets.<name>] in graph.toml")
    parser.add_argument("--skip-build", action="store_true",
                        help="reuse existing graph ELFs before running gem5")
    parser.add_argument("--skip-run", action="store_true",
                        help="reuse existing graph ROI stats and only rewrite eval artifacts")
    parser.add_argument("--full-compare", action="store_true",
                        help="with cache capacity flags, run SPM plus cache instead of cache-only")
    parser.add_argument("--cache-l1d-size", default=None,
                        help="cache-only L1D capacity for capacity/fairness baselines")
    parser.add_argument("--cache-l2-size", default=None,
                        help="cache-only L2 capacity for capacity/fairness baselines")
    parser.add_argument("--spm-tag", default=None,
                        help="optional SPM output directory tag, e.g. spm_lat_2ns")
    parser.add_argument("--artifact-tag", default=None,
                        help="optional suffix for graph build/m5out artifacts")
    parser.add_argument("--gem5-flag", action="append", default=[],
                        help="extra gem5 run_spm.py flag passed through graph compare")
    args = parser.parse_args()

    graph_output, graph_data, plans = load_graph_plan(
        args.graph,
        args.preset,
        args.artifact_tag,
    )
    if not args.skip_run:
        run_graph_compare(
            args.graph,
            args.preset,
            args.artifact_tag,
            args.skip_build,
            args.full_compare,
            args.gem5_flag,
            args.cache_l1d_size,
            args.cache_l2_size,
            args.spm_tag,
        )
    out_json, out_summary = write_eval_artifacts(
        graph_output,
        graph_data,
        plans,
        args.cache_l1d_size,
        args.cache_l2_size,
        args.spm_tag,
    )
    print(f"Graph eval JSON:    {rel(out_json)}")
    print(f"Graph eval summary: {rel(out_summary)}")


if __name__ == "__main__":
    main()
