#!/usr/bin/env python3
"""Parallel Phase 4 graph sweep for attention_smoke.

The Phase 4 ramp has an intentional dependency order:

  P4a: shape sweep with conservative/default node blocking
  P4b: graph-node matmul blocking sweep on those shapes
  P4c: aggregate the per-graph phase6_eval.json files into one table

This script keeps those as phases, but provides one entry point so a full run can
be launched and aggregated reproducibly.

Examples:
  ./scripts/sweep_p4_graph.py --phase a --jobs 8
  ./scripts/sweep_p4_graph.py --phase abc --jobs 8
  ./scripts/sweep_p4_graph.py --phase a --shape-preset smoke --dry-run
  ./scripts/sweep_p4_graph.py --phase b --shapes 32x64x32x128 --jobs 4
  ./scripts/sweep_p4_graph.py --phase c --sweep-name p4abc
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import tomllib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
WORKLOADS_DIR = SCRIPTS_DIR.parent
GRAPHS_DIR = WORKLOADS_DIR / "graphs"
M5OUT_ROOT = WORKLOADS_DIR / "m5out"
BASE_GRAPH = "attention_smoke"

MATMUL_GROUPS = ("qkv", "qk", "pv", "o_proj", "ffn_up", "ffn_down")
GROUP_NODES = {
    "qkv": ("q_proj", "k_proj", "v_proj"),
    "qk": ("qk",),
    "pv": ("pv",),
    "o_proj": ("o_proj",),
    "ffn_up": ("ffn_up",),
    "ffn_down": ("ffn_down",),
}

MATMUL_CANDIDATES = [
    (32, 32, 32),
    (32, 64, 32),
    (64, 32, 32),
    (64, 64, 32),
    (32, 32, 64),
    (32, 64, 64),
    (64, 32, 64),
    (64, 64, 64),
    (32, 128, 32),
    (64, 128, 32),
    (32, 128, 64),
    (64, 128, 64),
]


@dataclass(frozen=True)
class ShapeSpec:
    seq: int
    d_model: int
    head_dim: int
    ffn_dim: int

    @property
    def label(self) -> str:
        return f"s{self.seq}_d{self.d_model}_h{self.head_dim}_f{self.ffn_dim}"


@dataclass(frozen=True)
class SweepJob:
    graph: str
    phase: str
    variant: str
    shape: ShapeSpec
    blocking_group: str
    blocking: dict[str, tuple[int, int, int]]


def docs_shapes() -> list[ShapeSpec]:
    shapes: list[ShapeSpec] = []
    for seq, d_model, head_dim, ffn_mult in itertools.product(
        (32, 64),
        (64, 128),
        (32, 64),
        (2, 4),
    ):
        if head_dim <= d_model:
            shapes.append(ShapeSpec(seq, d_model, head_dim, ffn_mult * d_model))
    return shapes


def smoke_shapes() -> list[ShapeSpec]:
    return [
        ShapeSpec(32, 32, 16, 64),
        ShapeSpec(32, 64, 32, 128),
    ]


def parse_shapes(text: str) -> list[ShapeSpec]:
    shapes: list[ShapeSpec] = []
    for raw in text.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        parts = token.split("x")
        if len(parts) != 4:
            raise ValueError(
                f"shape {raw!r} must use SEQxD_MODELxHEAD_DIMxFFN_DIM")
        seq, d_model, head_dim, ffn_dim = (int(part, 0) for part in parts)
        if head_dim > d_model:
            raise ValueError(f"shape {raw!r} has HEAD_DIM > D_MODEL")
        shapes.append(ShapeSpec(seq, d_model, head_dim, ffn_dim))
    return shapes


def block_dim(dim: int, preferred: int) -> int:
    if dim <= 0:
        raise ValueError(f"invalid dimension {dim}")
    for candidate in (preferred, 128, 64, 32, 16, 8, 4, 2, 1):
        if candidate <= dim and dim % candidate == 0:
            return candidate
    return 1


def default_blocking(shape: ShapeSpec) -> dict[str, tuple[int, int, int]]:
    seq = shape.seq
    d = shape.d_model
    h = shape.head_dim
    f = shape.ffn_dim
    return {
        "qkv": (block_dim(seq, 32), block_dim(h, 16), block_dim(d, 32)),
        "qk": (block_dim(seq, 32), block_dim(seq, 16), block_dim(h, 16)),
        "pv": (block_dim(seq, 32), block_dim(h, 16), block_dim(seq, 32)),
        "o_proj": (block_dim(seq, 32), block_dim(d, 16), block_dim(h, 16)),
        "ffn_up": (block_dim(seq, 32), block_dim(f, 32), block_dim(d, 32)),
        "ffn_down": (block_dim(seq, 32), block_dim(d, 16), block_dim(f, 32)),
    }


def group_dims(shape: ShapeSpec, group: str) -> tuple[int, int, int]:
    if group == "qkv":
        return shape.seq, shape.head_dim, shape.d_model
    if group == "qk":
        return shape.seq, shape.seq, shape.head_dim
    if group == "pv":
        return shape.seq, shape.head_dim, shape.seq
    if group == "o_proj":
        return shape.seq, shape.d_model, shape.head_dim
    if group == "ffn_up":
        return shape.seq, shape.ffn_dim, shape.d_model
    if group == "ffn_down":
        return shape.seq, shape.d_model, shape.ffn_dim
    raise ValueError(f"unknown matmul group {group!r}")


def is_legal_blocking(dims: tuple[int, int, int], blk: tuple[int, int, int]) -> bool:
    return all(b <= d and d % b == 0 for d, b in zip(dims, blk))


def legal_group_candidates(shape: ShapeSpec, group: str) -> list[tuple[int, int, int]]:
    dims = group_dims(shape, group)
    seen: set[tuple[int, int, int]] = set()
    out: list[tuple[int, int, int]] = []
    for candidate in MATMUL_CANDIDATES:
        if candidate in seen or not is_legal_blocking(dims, candidate):
            continue
        seen.add(candidate)
        out.append(candidate)
    return out


def graph_name(sweep_name: str, shape: ShapeSpec, phase: str, variant: str) -> str:
    raw = f"p4_{sweep_name}_{phase}_{shape.label}_{variant}"
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in raw)
    return safe[:180]


def phase_a_jobs(sweep_name: str, shapes: list[ShapeSpec]) -> list[SweepJob]:
    jobs: list[SweepJob] = []
    for shape in shapes:
        blocking = default_blocking(shape)
        jobs.append(SweepJob(
            graph=graph_name(sweep_name, shape, "p4a", "default"),
            phase="p4a",
            variant="default",
            shape=shape,
            blocking_group="all",
            blocking=blocking,
        ))
    return jobs


def phase_b_jobs(sweep_name: str, shapes: list[ShapeSpec]) -> list[SweepJob]:
    jobs: list[SweepJob] = []
    for shape in shapes:
        base = default_blocking(shape)
        for group in MATMUL_GROUPS:
            for blk in legal_group_candidates(shape, group):
                if blk == base[group]:
                    continue
                blocking = dict(base)
                blocking[group] = blk
                bm, bn, bk = blk
                variant = f"{group}_{bm}x{bn}x{bk}"
                jobs.append(SweepJob(
                    graph=graph_name(sweep_name, shape, "p4b", variant),
                    phase="p4b",
                    variant=variant,
                    shape=shape,
                    blocking_group=group,
                    blocking=blocking,
                ))
    return jobs


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML value: {value!r}")


def write_kv_lines(lines: list[str], values: dict[str, Any]) -> None:
    for key, value in values.items():
        lines.append(f"{key} = {toml_value(value)}")


def render_graph_toml(data: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("[graph]")
    write_kv_lines(lines, data["graph"])
    lines += ["", "[harness]"]
    harness = data["harness"]
    write_kv_lines(lines, {k: v for k, v in harness.items() if k not in {"params", "build"}})
    lines += ["", "[harness.params]"]
    write_kv_lines(lines, harness["params"])
    lines += ["", "[harness.build]"]
    write_kv_lines(lines, harness["build"])

    for tensor, values in data["tensors"].items():
        lines += ["", f"[tensors.{tensor}]"]
        write_kv_lines(lines, values)

    for node, values in data["nodes"].items():
        lines += ["", f"[nodes.{node}]"]
        write_kv_lines(lines, {k: v for k, v in values.items() if k != "params"})
        lines += ["", f"[nodes.{node}.params]"]
        write_kv_lines(lines, values["params"])

    return "\n".join(lines) + "\n"


def load_base_graph() -> dict[str, Any]:
    path = GRAPHS_DIR / BASE_GRAPH / "graph.toml"
    return tomllib.loads(path.read_text())


def set_matmul_node(
    graph: dict[str, Any],
    node: str,
    m: int,
    n: int,
    k: int,
    block: tuple[int, int, int],
) -> None:
    params = graph["nodes"][node]["params"]
    bm, bn, bk = block
    params.update({
        "M": m,
        "N": n,
        "K": k,
        "BLOCK_SIZE_M": bm,
        "BLOCK_SIZE_N": bn,
        "BLOCK_SIZE_K": bk,
        "CHECK_RESULT": 0,
    })


def set_harness_matmul_grid(
    harness_params: dict[str, Any],
    prefix: str,
    block: tuple[int, int, int],
) -> None:
    bm, bn, _ = block
    harness_params[f"{prefix}_BLOCK_M"] = bm
    harness_params[f"{prefix}_BLOCK_N"] = bn


def assert_harness_node_grid_match(
    graph: dict[str, Any],
    harness_params: dict[str, Any],
) -> None:
    for group, prefix in (
        ("qkv", "QKV"),
        ("qk", "QK"),
        ("pv", "PV"),
        ("o_proj", "O_PROJ"),
        ("ffn_up", "FFN_UP"),
        ("ffn_down", "FFN_DOWN"),
    ):
        expected = (
            int(harness_params[f"{prefix}_BLOCK_M"]),
            int(harness_params[f"{prefix}_BLOCK_N"]),
        )
        for node in GROUP_NODES[group]:
            params = graph["nodes"][node]["params"]
            actual = (int(params["BLOCK_SIZE_M"]), int(params["BLOCK_SIZE_N"]))
            if actual != expected:
                raise ValueError(
                    f"{node} node blocking {actual} does not match "
                    f"harness {prefix} grid {expected}")


def apply_job_to_graph(base: dict[str, Any], job: SweepJob) -> dict[str, Any]:
    graph = json.loads(json.dumps(base))
    shape = job.shape
    graph["graph"]["name"] = job.graph
    graph["graph"]["description"] = (
        f"P4 graph sweep generated from {BASE_GRAPH}: "
        f"{shape.label}, {job.phase}, {job.variant}."
    )
    graph["harness"]["source"] = "harness.c"
    harness_params = graph["harness"]["params"]
    harness_params.update({
        "SEQ": shape.seq,
        "D_MODEL": shape.d_model,
        "HEAD_DIM": shape.head_dim,
        "FFN_DIM": shape.ffn_dim,
        "BLOCK": 16,
        "CHECK_RESULT": 1,
        "FLUSH_BEFORE_ROI": 1,
    })

    for group, prefix in (
        ("qkv", "QKV"),
        ("qk", "QK"),
        ("pv", "PV"),
        ("o_proj", "O_PROJ"),
        ("ffn_up", "FFN_UP"),
        ("ffn_down", "FFN_DOWN"),
    ):
        set_harness_matmul_grid(harness_params, prefix, job.blocking[group])

    harness_params["K_TRANSPOSE_BLOCK_M"] = block_dim(shape.seq, 16)
    harness_params["K_TRANSPOSE_BLOCK_N"] = block_dim(shape.head_dim, 16)

    nodes = graph["nodes"]
    nodes["layer_norm"]["params"].update({
        "M": shape.seq,
        "N": shape.d_model,
        "CHECK_RESULT": 0,
        "LAYERNORM_FLUSH_BEFORE_ROI": 1,
    })
    nodes["ln2"]["params"].update(nodes["layer_norm"]["params"])

    for node in GROUP_NODES["qkv"]:
        set_matmul_node(
            graph, node, shape.seq, shape.head_dim, shape.d_model, job.blocking["qkv"])
    set_matmul_node(
        graph, "qk", shape.seq, shape.seq, shape.head_dim, job.blocking["qk"])
    set_matmul_node(
        graph, "pv", shape.seq, shape.head_dim, shape.seq, job.blocking["pv"])
    set_matmul_node(
        graph, "o_proj", shape.seq, shape.d_model, shape.head_dim, job.blocking["o_proj"])
    set_matmul_node(
        graph, "ffn_up", shape.seq, shape.ffn_dim, shape.d_model, job.blocking["ffn_up"])
    set_matmul_node(
        graph, "ffn_down", shape.seq, shape.d_model, shape.ffn_dim, job.blocking["ffn_down"])

    nodes["k_transpose"]["params"].update({
        "M": shape.seq,
        "N": shape.head_dim,
        "BLOCK_M": harness_params["K_TRANSPOSE_BLOCK_M"],
        "BLOCK_N": harness_params["K_TRANSPOSE_BLOCK_N"],
        "CHECK_RESULT": 0,
    })
    nodes["softmax"]["params"].update({
        "M": shape.seq,
        "N": shape.seq,
        "BLOCK_N": block_dim(shape.seq, 64),
        "SPM_ROW_BLOCK": 1,
        "SPM_ROW_GROUP_BLOCKS": 1,
        "SPM_INTERNAL_ROW_BLOCK": 0,
        "CHECK_RESULT": 0,
    })
    for node in ("attn_residual_add", "final_residual_add"):
        nodes[node]["params"].update({
            "SIZE": shape.seq * shape.d_model,
            "BLOCK_SIZE": 16,
            "CHECK_RESULT": 0,
        })
    nodes["ffn_activation"]["params"].update({
        "SIZE": shape.seq * shape.ffn_dim,
        "BLOCK_SIZE": 16,
        "CHECK_RESULT": 0,
    })
    assert_harness_node_grid_match(graph, harness_params)

    for node, values in nodes.items():
        values["tag"] = f"graph/{job.graph}/{node}"

    return graph


def prepare_graph(job: SweepJob) -> Path:
    base = load_base_graph()
    graph = apply_job_to_graph(base, job)
    out_dir = GRAPHS_DIR / job.graph
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "graph.toml").write_text(render_graph_toml(graph))
    shutil.copy2(GRAPHS_DIR / BASE_GRAPH / "harness.c", out_dir / "harness.c")
    meta = {
        "graph": job.graph,
        "phase": job.phase,
        "variant": job.variant,
        "shape": {
            "seq": job.shape.seq,
            "d_model": job.shape.d_model,
            "head_dim": job.shape.head_dim,
            "ffn_dim": job.shape.ffn_dim,
        },
        "blocking_group": job.blocking_group,
        "blocking": {
            group: list(block)
            for group, block in sorted(job.blocking.items())
        },
    }
    (out_dir / "p4_sweep_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out_dir


def job_command(job: SweepJob, skip_build: bool, gem5_flags: list[str]) -> list[str]:
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "phase6_graph_eval.py"),
        job.graph,
    ]
    if skip_build:
        cmd.append("--skip-build")
    for flag in gem5_flags:
        cmd.append(f"--gem5-flag={flag}")
    return cmd


def run_one(
    job: SweepJob,
    sweep_name: str,
    skip_build: bool,
    gem5_flags: list[str],
    env_updates: dict[str, str],
) -> tuple[str, int, float]:
    prepare_graph(job)
    env = os.environ.copy()
    env.update(env_updates)
    log_dir = aggregate_dir(sweep_name) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job.graph}.log"
    t0 = time.time()
    with log_path.open("w") as log:
        proc = subprocess.run(
            job_command(job, skip_build, gem5_flags),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return job.graph, proc.returncode, time.time() - t0


def run_parallel(
    jobs: list[SweepJob],
    sweep_name: str,
    max_workers: int,
    dry_run: bool,
    skip_build: bool,
    gem5_flags: list[str],
    env_updates: dict[str, str],
) -> dict[str, int]:
    if dry_run:
        for job in jobs:
            cmd = job_command(job, skip_build, gem5_flags)
            env_text = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(env_updates.items()))
            print(
                f"  {job.graph}: {job.phase} {job.shape.label} {job.variant}\n"
                f"    {env_text} {' '.join(shlex.quote(c) for c in cmd)}")
        return {}

    print(f"  {len(jobs)} graph jobs, {max_workers} workers")
    results: dict[str, int] = {}
    failed = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(run_one, job, sweep_name, skip_build, gem5_flags, env_updates): job
            for job in jobs
        }
        for idx, future in enumerate(as_completed(futures), 1):
            graph, rc, elapsed = future.result()
            results[graph] = rc
            status = "OK" if rc == 0 else f"FAIL(rc={rc})"
            if rc != 0:
                failed += 1
            print(f"    [{idx}/{len(jobs)}] {status} {elapsed:7.1f}s {graph}")
    total = time.time() - t0
    print(f"  Done: {len(jobs) - failed}/{len(jobs)} passed, {failed} failed, {total:.0f}s wall\n")
    return results


def aggregate_dir(sweep_name: str) -> Path:
    return M5OUT_ROOT / "graphs" / "p4_sweep" / sweep_name


def graph_eval_path(graph: str) -> Path:
    return M5OUT_ROOT / "graphs" / graph / "spm" / "default" / "phase6_eval.json"


def read_meta(graph: str) -> dict[str, Any]:
    path = GRAPHS_DIR / graph / "p4_sweep_meta.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def discover_graphs(sweep_name: str) -> list[str]:
    prefix = f"p4_{sweep_name}_"
    graphs = []
    for path in sorted(GRAPHS_DIR.glob(f"{prefix}*/p4_sweep_meta.json")):
        graphs.append(path.parent.name)
    return graphs


def aggregate_results(sweep_name: str, graphs: list[str] | None = None) -> tuple[Path, Path, Path]:
    if graphs is None:
        graphs = discover_graphs(sweep_name)
    rows: list[dict[str, Any]] = []
    for graph in sorted(graphs):
        meta = read_meta(graph)
        eval_path = graph_eval_path(graph)
        payload: dict[str, Any] = {}
        if eval_path.is_file():
            payload = json.loads(eval_path.read_text())

        shape = meta.get("shape", {})
        delta = payload.get("cycle_delta", {})
        spm_stats = payload.get("stats", {}).get("spm", {})
        cache_stats = payload.get("stats", {}).get("cache", {})
        spm_cycles = as_float(delta.get("spm_cycles"))
        wait_cycles = as_float(spm_stats.get("spm_dma.waitStallCycles"))
        spm_l1 = as_float(spm_stats.get("l1d.demandMisses"))
        cache_l1 = as_float(cache_stats.get("l1d.demandMisses"))
        spm_l2 = as_float(spm_stats.get("l2.demandMisses"))
        cache_l2 = as_float(cache_stats.get("l2.demandMisses"))
        row = {
            "graph": graph,
            "phase": meta.get("phase", ""),
            "variant": meta.get("variant", ""),
            "blocking_group": meta.get("blocking_group", ""),
            "seq": shape.get("seq", ""),
            "d_model": shape.get("d_model", ""),
            "head_dim": shape.get("head_dim", ""),
            "ffn_dim": shape.get("ffn_dim", ""),
            "spm_result": payload.get("result_gates", {}).get("spm", "MISSING"),
            "cache_result": payload.get("result_gates", {}).get("cache", "MISSING"),
            "spm_cycles": delta.get("spm_cycles"),
            "cache_cycles": delta.get("cache_cycles"),
            "delta_pct": delta.get("delta_pct"),
            "speedup_cache_over_spm": delta.get("speedup_cache_over_spm"),
            "spm_dma_bytes": spm_stats.get("spm_dma.bytes"),
            "spm_dma_wait_cycles": spm_stats.get("spm_dma.waitStallCycles"),
            "spm_dma_wait_fraction": (
                None if spm_cycles in (None, 0) or wait_cycles is None
                else wait_cycles / spm_cycles
            ),
            "l1d_demand_misses_spm": spm_stats.get("l1d.demandMisses"),
            "l1d_demand_misses_cache": cache_stats.get("l1d.demandMisses"),
            "l1d_demand_misses_delta": (
                None if spm_l1 is None or cache_l1 is None else spm_l1 - cache_l1
            ),
            "l2_demand_misses_spm": spm_stats.get("l2.demandMisses"),
            "l2_demand_misses_cache": cache_stats.get("l2.demandMisses"),
            "l2_demand_misses_delta": (
                None if spm_l2 is None or cache_l2 is None else spm_l2 - cache_l2
            ),
            "eval_json": str(eval_path.relative_to(WORKLOADS_DIR)) if eval_path.exists() else "",
        }
        for group, block in sorted(meta.get("blocking", {}).items()):
            row[f"{group}_block"] = "x".join(str(x) for x in block)
        rows.append(row)

    out_dir = aggregate_dir(sweep_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "p4abc_summary.json"
    out_csv = out_dir / "p4abc_summary.csv"
    out_md = out_dir / "p4abc_summary.md"

    out_json.write_text(json.dumps(rows, indent=2) + "\n")
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "graph", "phase", "variant", "blocking_group",
        "seq", "d_model", "head_dim", "ffn_dim",
        "spm_result", "cache_result", "spm_cycles", "cache_cycles",
        "delta_pct", "speedup_cache_over_spm", "spm_dma_bytes",
        "spm_dma_wait_cycles", "spm_dma_wait_fraction",
        "l1d_demand_misses_spm", "l1d_demand_misses_cache", "l1d_demand_misses_delta",
        "l2_demand_misses_spm", "l2_demand_misses_cache", "l2_demand_misses_delta",
        "eval_json",
    ]
    fieldnames = preferred + [name for name in fieldnames if name not in preferred]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# P4 Graph Sweep Summary: {sweep_name}",
        "",
        "| phase | shape | variant | SPM cycles | cache cycles | speedup | wait frac | result |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        shape = f"{row['seq']}x{row['d_model']}x{row['head_dim']}x{row['ffn_dim']}"
        speedup = as_float(row.get("speedup_cache_over_spm"))
        wait = as_float(row.get("spm_dma_wait_fraction"))
        lines.append(
            f"| {row['phase']} | {shape} | {row['variant']} | "
            f"{fmt_int(row.get('spm_cycles'))} | {fmt_int(row.get('cache_cycles'))} | "
            f"{'-' if speedup is None else f'{speedup:.3f}x'} | "
            f"{'-' if wait is None else f'{wait:.3f}'} | "
            f"{row['spm_result']}/{row['cache_result']} |")
    out_md.write_text("\n".join(lines) + "\n")
    return out_json, out_csv, out_md


def fmt_int(value: Any) -> str:
    number = as_float(value)
    return "-" if number is None else f"{int(number):,}"


def limit_jobs(jobs: list[SweepJob], max_runs: int | None) -> list[SweepJob]:
    if max_runs is None:
        return jobs
    return jobs[:max_runs]


def split_run_limit(
    a_jobs: list[SweepJob],
    b_jobs: list[SweepJob],
    max_runs: int | None,
) -> tuple[list[SweepJob], list[SweepJob]]:
    if max_runs is None:
        return a_jobs, b_jobs
    a_limited = a_jobs[:max_runs]
    remaining = max(0, max_runs - len(a_limited))
    return a_limited, b_jobs[:remaining]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--phase", choices=("a", "b", "c", "ab", "abc"), default="a",
                        help="phase(s) to run; c aggregates existing eval JSONs")
    parser.add_argument("--sweep-name", default="p4abc",
                        help="name used in generated graph names and aggregate dir")
    parser.add_argument("--shape-preset", choices=("docs", "smoke"), default="docs",
                        help="shape set when --shapes is not provided")
    parser.add_argument("--shapes", default=None,
                        help="comma-separated SEQxD_MODELxHEAD_DIMxFFN_DIM values")
    parser.add_argument("--jobs", "-j", type=int, default=None,
                        help="max parallel graph compare jobs (default: nproc/4)")
    parser.add_argument("--max-runs", type=int, default=None,
                        help="debug helper: truncate the generated run list")
    parser.add_argument("--dry-run", action="store_true",
                        help="print graph jobs and commands without creating/running them")
    parser.add_argument("--skip-build", action="store_true",
                        help="pass --skip-build through to phase6_graph_eval.py")
    parser.add_argument("--gem5-flag", action="append", default=[],
                        help="extra gem5 run_spm.py flag passed through graph compare")
    parser.add_argument("--micro-m", type=int, default=8,
                        help="fixed TRITON_MICRO_M for this sweep")
    parser.add_argument("--window-k", type=int, default=4,
                        help="fixed TRITON_SPM_WINDOW_K for this sweep")
    args = parser.parse_args()

    shapes = parse_shapes(args.shapes) if args.shapes else (
        docs_shapes() if args.shape_preset == "docs" else smoke_shapes()
    )
    max_workers = args.jobs or max(1, (os.cpu_count() or 4) // 4)
    env_updates = {
        "TRITON_MICRO_M": str(args.micro_m),
        "TRITON_SPM_WINDOW_K": str(args.window_k),
    }

    a_jobs: list[SweepJob] = []
    b_jobs: list[SweepJob] = []
    if "a" in args.phase:
        a_jobs = phase_a_jobs(args.sweep_name, shapes)
    if "b" in args.phase:
        b_jobs = phase_b_jobs(args.sweep_name, shapes)
    a_jobs, b_jobs = split_run_limit(a_jobs, b_jobs, args.max_runs)
    all_jobs = a_jobs + b_jobs

    if all_jobs:
        print("=" * 72)
        print("P4 graph sweep runs")
        print("=" * 72)
        print(f"  shapes:  {len(shapes)}")
        print(f"  P4a runs: {len(a_jobs)}")
        print(f"  P4b runs: {len(b_jobs)}")
        print(f"  env:     TRITON_MICRO_M={args.micro_m} TRITON_SPM_WINDOW_K={args.window_k}")
    if a_jobs:
        print("\nP4a: graph shape sweep")
        run_parallel(
            a_jobs,
            args.sweep_name,
            max_workers,
            args.dry_run,
            args.skip_build,
            args.gem5_flag,
            env_updates,
        )
    if b_jobs:
        print("\nP4b: graph-node matmul blocking sweep")
        run_parallel(
            b_jobs,
            args.sweep_name,
            max_workers,
            args.dry_run,
            args.skip_build,
            args.gem5_flag,
            env_updates,
        )

    if "c" in args.phase:
        print("=" * 72)
        print("P4c aggregation")
        print("=" * 72)
        if args.dry_run:
            print(f"  would aggregate graphs matching p4_{args.sweep_name}_*")
        else:
            graph_list = [job.graph for job in all_jobs] if all_jobs else None
            out_json, out_csv, out_md = aggregate_results(args.sweep_name, graph_list)
            print(f"  JSON: {out_json.relative_to(WORKLOADS_DIR)}")
            print(f"  CSV:  {out_csv.relative_to(WORKLOADS_DIR)}")
            print(f"  MD:   {out_md.relative_to(WORKLOADS_DIR)}")


if __name__ == "__main__":
    main()
