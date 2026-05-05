#!/usr/bin/env python3
"""Build/verify graph-level conservative placement artifacts.

The graph planner sits above single-kernel SPM placement.  It reads tensor-edge
metadata, chooses backing tiers for each node argument, and builds SPM artifacts
with KERNEL_TIER_OVERRIDE so the generated launcher allocation dispatch can be
checked without changing ConvertMemoryToSPM.

Usage:
  scripts/graph_placement.py layer_norm_qkv --mode plan
  scripts/graph_placement.py layer_norm_qkv --mode verify
  scripts/graph_placement.py layer_norm_qkv --mode run
  scripts/graph_placement.py layer_norm_qkv --mode compare
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import run_experiment
import trispm_paths
from trispm_paths import WORKLOADS_DIR

SCRIPTS_DIR = Path(__file__).resolve().parent
GRAPHS_DIR = WORKLOADS_DIR / "graphs"

TIER_SPM_RESIDENT = 1
TIER_CACHEABLE = 2
TIER_UNCACHEABLE_DMA = 3


@dataclass(frozen=True)
class TensorDecision:
    tier: int
    reason: str


@dataclass(frozen=True)
class NodePlan:
    name: str
    kernel: str
    tag: str
    args: list[str]
    params: dict[str, str]
    arg_tiers: dict[int, int]
    decisions: dict[int, TensorDecision]

    @property
    def tier_override(self) -> str:
        return ",".join(f"{idx}={tier}" for idx, tier in sorted(self.arg_tiers.items()))

    @property
    def c_name(self) -> str:
        return sanitize_c_ident(self.name)


def load_graph(name: str) -> dict[str, Any]:
    path = GRAPHS_DIR / name / "graph.toml"
    if not path.is_file():
        sys.exit(f"ERROR: graph manifest not found: {path}")
    data = tomllib.loads(path.read_text())
    data["_path"] = path
    return data


def sanitize_c_ident(value: str) -> str:
    ident = re.sub(r"\W+", "_", value.strip())
    ident = ident.strip("_")
    if not ident:
        raise ValueError(f"cannot derive a C identifier from {value!r}")
    if ident[0].isdigit():
        ident = f"n_{ident}"
    return ident


def choose_tensor_tier(tensor_name: str, tensor: dict[str, Any]) -> TensorDecision:
    kind = str(tensor.get("kind", "")).strip()
    read_only = bool(tensor.get("read_only", False))
    dma_only = bool(tensor.get("dma_only", False))
    has_producer = bool(tensor.get("producer"))
    consumers = tensor.get("consumers", [])
    if consumers is None:
        consumers = []

    if int(tensor.get("tier", 0) or 0) == TIER_SPM_RESIDENT:
        raise ValueError(
            f"tensor {tensor_name!r} requests Tier 1, but graph Tier 1 is future work")

    if kind in {"intermediate", "producer_output", "graph_output"} or has_producer:
        return TensorDecision(
            TIER_CACHEABLE,
            "producer output / intermediate activation stays cacheable",
        )

    if read_only and dma_only and kind in {"external_input", "external_weight"}:
        return TensorDecision(
            TIER_UNCACHEABLE_DMA,
            "external read-only DMA-only tensor may use uncacheable backing",
        )

    if consumers:
        return TensorDecision(
            TIER_CACHEABLE,
            "downstream use is visible, so keep cacheable",
        )

    return TensorDecision(
        TIER_CACHEABLE,
        "conservative default",
    )


def build_plan(graph_name: str, graph: dict[str, Any]) -> list[NodePlan]:
    tensors = graph.get("tensors", {})
    nodes = graph.get("nodes", {})
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("graph manifest must define [tensors.*]")
    if not isinstance(nodes, dict) or not nodes:
        raise ValueError("graph manifest must define [nodes.*]")

    plans: list[NodePlan] = []
    for node_name, node in nodes.items():
        kernel = str(node.get("kernel", "")).strip()
        if not kernel:
            raise ValueError(f"node {node_name!r} missing kernel")

        args = list(node.get("args", []))
        if not args:
            raise ValueError(f"node {node_name!r} must list pointer args")

        params = {str(k): str(v) for k, v in dict(node.get("params", {})).items()}
        tag = str(node.get("tag", f"graph/{graph_name}/{node_name}"))

        arg_tiers: dict[int, int] = {}
        decisions: dict[int, TensorDecision] = {}
        for idx, tensor_name in enumerate(args):
            if tensor_name not in tensors:
                raise ValueError(
                    f"node {node_name!r} arg {idx} references unknown tensor "
                    f"{tensor_name!r}")
            decision = choose_tensor_tier(tensor_name, tensors[tensor_name])
            arg_tiers[idx] = decision.tier
            decisions[idx] = decision

        plans.append(NodePlan(
            name=node_name,
            kernel=kernel,
            tag=tag,
            args=args,
            params=params,
            arg_tiers=arg_tiers,
            decisions=decisions,
        ))

    return plans


def run(cmd: list[str], env: dict[str, str] | None = None, echo: bool = True) -> None:
    if echo:
        print(f"$ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env)


def build_node(plan: NodePlan) -> None:
    manifest = run_experiment.load_manifest(plan.kernel)
    params = run_experiment.merged_params(
        manifest,
        preset=None,
        overrides=plan.params,
        mode="spm",
    )
    env = run_experiment.export_env(manifest, params)
    env["KERNEL_TIER_OVERRIDE"] = plan.tier_override
    run([
        str(SCRIPTS_DIR / "build_kernel.sh"),
        plan.kernel,
        "--mode", "spm",
        "--tag", plan.tag,
    ], env=env)


def node_build_dir(plan: NodePlan, mode: str) -> Path:
    return trispm_paths.build_dir(plan.kernel, mode, plan.tag)


def graph_build_dir(graph_name: str, mode: str) -> Path:
    return trispm_paths.graph_build_dir(graph_name, mode)


def graph_binary_path(graph_name: str, mode: str) -> Path:
    return trispm_paths.graph_binary_path(graph_name, mode)


def graph_m5out_dir(graph_name: str, mode: str) -> Path:
    return trispm_paths.graph_m5out_dir(graph_name, mode)


def graph_run_log_path(graph_name: str, mode: str) -> Path:
    return trispm_paths.graph_run_log_path(graph_name, mode)


def graph_roi_stats_path(graph_name: str, mode: str) -> Path:
    return trispm_paths.graph_roi_stats_path(graph_name, mode)


def graph_compare_path(graph_name: str) -> Path:
    return trispm_paths.graph_compare_path(graph_name)


def graph_spm_stats_path(graph_name: str) -> Path:
    return trispm_paths.graph_spm_stats_path(graph_name)


def graph_report_path(graph_name: str) -> Path:
    return trispm_paths.graph_report_path(graph_name)


def render_graph_cflags(graph: dict[str, Any]) -> str:
    harness = dict(graph.get("harness", {}))
    params = {str(k): str(v) for k, v in dict(harness.get("params", {})).items()}
    macros = harness.get("build", {}).get("c_macros", [])
    try:
        return " ".join(f"-D{macro.format(**params)}" for macro in macros)
    except KeyError as exc:
        raise ValueError(
            f"[harness.build].c_macros references unknown param {exc.args[0]!r}"
        ) from exc


def replace_c_identifier(text: str, old: str, new: str) -> str:
    return re.sub(rf"\b{re.escape(old)}\b", new, text)


def namespace_asm(text: str, old: str, new: str) -> str:
    return replace_c_identifier(text, old, new)


def node_symbol(plan: NodePlan, suffix: str = "") -> str:
    return f"{plan.kernel}_{plan.c_name}{suffix}"


def namespace_graph_nodes(
    out_dir: Path,
    graph_name: str,
    plans: list[NodePlan],
    node_dirs: dict[str, Path],
) -> list[dict[str, Path]]:
    name_counts = Counter(plan.c_name for plan in plans)
    collisions = sorted(name for name, count in name_counts.items() if count > 1)
    if collisions:
        joined = ", ".join(collisions)
        sys.exit(f"ERROR: graph {graph_name} has duplicate C node names: {joined}")

    artifacts: list[dict[str, Path]] = []
    launcher_units: list[str] = [
        '#include "graph_nodes.h"',
        "",
    ]
    header_lines = [
        f"#ifndef {sanitize_c_ident(graph_name).upper()}_GRAPH_NODES_H",
        f"#define {sanitize_c_ident(graph_name).upper()}_GRAPH_NODES_H",
        "",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "",
    ]

    for plan in plans:
        build_dir = node_dirs[plan.name]
        source_symbol = plan.kernel
        asm_symbol = node_symbol(plan)
        launch_symbol = node_symbol(plan, "_launch")
        alloc_symbol = node_symbol(plan, "_alloc")
        free_symbol = node_symbol(plan, "_free_all")
        record_symbol = node_symbol(plan, "_record_malloc")
        malloc_ptrs_symbol = node_symbol(plan, "_malloc_ptrs")
        malloc_count_symbol = node_symbol(plan, "_malloc_count")

        asm_src = build_dir / f"{plan.kernel}.s"
        launcher_src = build_dir / f"{plan.kernel}_launcher.c"
        launcher_hdr = build_dir / f"{plan.kernel}_launcher.h"

        node_prefix = f"{plan.c_name}_{plan.kernel}"
        asm_dst = out_dir / f"{node_prefix}.s"
        launcher_dst = out_dir / f"{node_prefix}_launcher.c"
        header_dst = out_dir / f"{node_prefix}_launcher.h"

        asm_text = namespace_asm(asm_src.read_text(), source_symbol, asm_symbol)
        asm_dst.write_text(asm_text)

        header_text = launcher_hdr.read_text()
        for old, new in (
            (f"{plan.kernel}_launch", launch_symbol),
            (f"{plan.kernel}_alloc", alloc_symbol),
            (f"{plan.kernel}_free_all", free_symbol),
        ):
            header_text = replace_c_identifier(header_text, old, new)
        header_text = header_text.replace(
            f"{plan.kernel.upper()}_LAUNCHER_H",
            f"{node_prefix.upper()}_LAUNCHER_H",
        )
        header_dst.write_text(header_text)

        launcher_text = launcher_src.read_text()
        launcher_text = launcher_text.replace(
            f'#include "{plan.kernel}_launcher.h"',
            f'#include "{node_prefix}_launcher.h"',
        )
        replacements = (
            (f"{plan.kernel}_record_malloc", record_symbol),
            (f"{plan.kernel}_malloc_ptrs", malloc_ptrs_symbol),
            (f"{plan.kernel}_malloc_count", malloc_count_symbol),
            (f"{plan.kernel}_launch", launch_symbol),
            (f"{plan.kernel}_alloc", alloc_symbol),
            (f"{plan.kernel}_free_all", free_symbol),
            (plan.kernel, asm_symbol),
        )
        for old, new in replacements:
            launcher_text = replace_c_identifier(launcher_text, old, new)
        launcher_dst.write_text(launcher_text)
        launcher_units += [
            f"/* graph node: {plan.name} ({plan.kernel}) */",
            launcher_text,
            "",
        ]

        header_lines += [
            f'#include "{node_prefix}_launcher.h"',
            f"#define {plan.c_name}_launch {launch_symbol}",
            f"#define {plan.c_name}_alloc {alloc_symbol}",
            f"#define {plan.c_name}_free_all {free_symbol}",
            "",
        ]
        artifacts.append({"asm": asm_dst, "launcher_c": launcher_dst, "launcher_h": header_dst})

    header_lines += [f"#endif /* {sanitize_c_ident(graph_name).upper()}_GRAPH_NODES_H */", ""]
    (out_dir / "graph_nodes.h").write_text("\n".join(header_lines))
    (out_dir / "graph_node_launchers.c").write_text("\n".join(launcher_units))
    return artifacts


def source_env() -> dict[str, str]:
    cmd = (
        "set -euo pipefail; "
        f"source {shlex.quote(str(WORKLOADS_DIR / 'env.sh'))}; "
        "export TRISPM_ROOT LLC LLC_FLAGS CLANG CLANG_FLAGS GEM5 GEM5_RUN_SCRIPT; "
        "python3 - <<'PY'\n"
        "import json, os\n"
        "keys = ['TRISPM_ROOT', 'LLC', 'LLC_FLAGS', 'CLANG', 'CLANG_FLAGS', "
        "'GEM5', 'GEM5_RUN_SCRIPT']\n"
        "print(json.dumps({k: os.environ.get(k, '') for k in keys}))\n"
        "PY"
    )
    proc = subprocess.run(
        ["bash", "-lc", cmd],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    data = json.loads(proc.stdout)
    env = os.environ.copy()
    env.update({k: str(v) for k, v in data.items() if v})
    return env


def compile_graph(graph_name: str, graph: dict[str, Any], plans: list[NodePlan], mode: str) -> None:
    env = source_env()
    out_dir = graph_build_dir(graph_name, mode)
    out_dir.mkdir(parents=True, exist_ok=True)

    node_dirs = {plan.name: node_build_dir(plan, mode) for plan in plans}
    required: list[Path] = []
    for plan in plans:
        build_dir = node_dirs[plan.name]
        required += [
            build_dir / f"{plan.kernel}.s",
            build_dir / f"{plan.kernel}_launcher.c",
            build_dir / f"{plan.kernel}_launcher.h",
        ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        detail = "\n".join(f"  {path}" for path in missing)
        sys.exit(f"ERROR: missing node build artifacts:\n{detail}")

    harness_cfg = dict(graph.get("harness", {}))
    harness_source = GRAPHS_DIR / graph_name / str(harness_cfg.get("source", "harness.c"))
    if not harness_source.is_file():
        sys.exit(f"ERROR: graph harness source not found: {harness_source}")

    graph_cflags = render_graph_cflags(graph)
    node_artifacts = namespace_graph_nodes(out_dir, graph_name, plans, node_dirs)
    binary = graph_binary_path(graph_name, mode)
    include_flags = [f"-I{out_dir}"]
    asm_sources = [str(artifact["asm"]) for artifact in node_artifacts]
    launcher_sources = [str(out_dir / "graph_node_launchers.c")]
    cmd = [
        env["CLANG"],
        *shlex.split(env["CLANG_FLAGS"]),
        *shlex.split(graph_cflags),
        *include_flags,
        f"-I{env['TRISPM_ROOT']}/simulator/src/scratchpad_mem",
        *asm_sources,
        *launcher_sources,
        str(harness_source),
        "-lm",
        "-o",
        str(binary),
    ]
    print(f"\n========== link executable graph ({mode}) ==========")
    run(cmd, env=env)
    print(f"  -> {binary.relative_to(WORKLOADS_DIR)}")


def build_graph_executable(graph_name: str, graph: dict[str, Any], plans: list[NodePlan], mode: str) -> None:
    for plan in plans:
        print(
            f"\n========== build {plan.name} "
            f"({mode}, tier_override={plan.tier_override!r}) ==========")
        if mode == "spm":
            build_node(plan)
        else:
            manifest = run_experiment.load_manifest(plan.kernel)
            params = run_experiment.merged_params(
                manifest,
                preset=None,
                overrides=plan.params,
                mode="cache",
            )
            env = run_experiment.export_env(manifest, params)
            run([
                str(SCRIPTS_DIR / "build_kernel.sh"),
                plan.kernel,
                "--mode", "cache",
                "--tag", plan.tag,
            ], env=env)
    compile_graph(graph_name, graph, plans, mode)


def run_graph_executable(graph_name: str, mode: str, gem5_flags: list[str]) -> None:
    env = source_env()
    binary = graph_binary_path(graph_name, mode)
    if not binary.is_file():
        sys.exit(f"ERROR: graph binary not found: {binary}")
    m5out = graph_m5out_dir(graph_name, mode)
    run_log = graph_run_log_path(graph_name, mode)
    roi_stats = graph_roi_stats_path(graph_name, mode)
    m5out.mkdir(parents=True, exist_ok=True)

    cmd = [
        env["GEM5"],
        f"--outdir={m5out}",
        env["GEM5_RUN_SCRIPT"],
        "--binary",
        str(binary),
    ]
    if mode == "cache":
        cmd.append("--cache_baseline")
    cmd += gem5_flags

    print(f"\n===== Running graph {graph_name} ({mode}) on gem5 =====")
    print(f"  binary: {binary}")
    print(f"  outdir: {m5out}")
    print(f"  flags:  {' '.join(gem5_flags) if gem5_flags else '<none>'}\n")
    with run_log.open("w") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        status = proc.wait()
    if status != 0:
        sys.exit(status)

    stats = m5out / "stats.txt"
    if stats.is_file():
        in_block = False
        with stats.open(errors="replace") as src, roi_stats.open("w") as dst:
            for line in src:
                if "---------- Begin Simulation Statistics ----------" in line:
                    in_block = True
                if in_block:
                    dst.write(line)
                if "---------- End Simulation Statistics" in line and in_block:
                    break
        print(f"\nROI stats written to {roi_stats}")
    print(f"Run log written to {run_log}")


def validate_graph_run(graph_name: str, mode: str) -> None:
    run_log = graph_run_log_path(graph_name, mode)
    if not run_log.is_file():
        sys.exit(f"ERROR: graph run log is missing: {run_log}")
    text = run_log.read_text(errors="replace")
    bad_lines = [
        line for line in text.splitlines()
        if re.search(r"\b(FAIL|MISMATCH):", line)
    ]
    if "PASS: graph outputs correct" in text and not bad_lines:
        print(f"Result gate passed: graph {graph_name} {mode}")
        return
    detail = "\n".join(bad_lines[:12]) if bad_lines else "PASS line was not found"
    sys.exit(
        f"ERROR: graph {graph_name} {mode} failed result gate.\n"
        f"Log: {run_log.relative_to(WORKLOADS_DIR)}\n"
        f"{detail}"
    )


def ensure_graph_roi_stats(graph_name: str, mode: str) -> Path:
    roi_stats = graph_roi_stats_path(graph_name, mode)
    if not roi_stats.is_file():
        sys.exit(
            f"ERROR: graph {mode} ROI stats are missing: "
            f"{roi_stats.relative_to(WORKLOADS_DIR)}"
        )
    return roi_stats


def run_graph_compare_stats(graph_name: str) -> tuple[Path, Path]:
    spm_stats = ensure_graph_roi_stats(graph_name, "spm")
    cache_stats = ensure_graph_roi_stats(graph_name, "cache")
    compare = graph_compare_path(graph_name)
    spm_only = graph_spm_stats_path(graph_name)
    run([
        str(SCRIPTS_DIR / "compare_stats.py"),
        "--spm", str(spm_stats),
        "--cache", str(cache_stats),
        "--measure-iters", "1",
        "--output", str(compare),
        "--spm-only-output", str(spm_only),
    ])
    return compare, spm_only


def graph_log_status(graph_name: str, mode: str) -> str:
    run_log = graph_run_log_path(graph_name, mode)
    if not run_log.is_file():
        return "missing run.log"
    text = run_log.read_text(errors="replace")
    bad = re.search(r"\b(FAIL|MISMATCH):", text)
    if "PASS: graph outputs correct" in text and not bad:
        return "PASS"
    if bad:
        return "FAIL"
    if "SKIP: graph result check disabled" in text:
        return "SKIP"
    return "UNKNOWN"


def summarize_run_rows(graph_name: str) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for mode in ("spm", "cache"):
        rows.append((
            mode,
            str(graph_binary_path(graph_name, mode).relative_to(WORKLOADS_DIR)),
            str(graph_roi_stats_path(graph_name, mode).relative_to(WORKLOADS_DIR)),
            graph_log_status(graph_name, mode),
        ))
    return rows


def write_graph_report(
    graph_name: str,
    graph: dict[str, Any],
    plans: list[NodePlan],
    compare: Path,
    spm_only: Path,
) -> Path:
    out = graph_report_path(graph_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    graph_meta = dict(graph.get("graph", {}))
    harness_params = {
        str(k): str(v)
        for k, v in dict(dict(graph.get("harness", {})).get("params", {})).items()
    }
    lines = [
        f"# Graph Compare Report: {graph_name}",
        "",
        f"description: {graph_meta.get('description', '')}",
        "",
        "## Harness Params",
        "",
        render_table(
            [(key, value) for key, value in sorted(harness_params.items())],
            ("param", "value"),
        ),
        "",
        "## Placement Decisions",
        "",
        render_table(plan_as_rows(plans), ("node", "kernel", "arg", "tier", "reason")),
        "",
        "## Run Artifacts",
        "",
        render_table(summarize_run_rows(graph_name), ("mode", "binary", "roi_stats", "result")),
        "",
        "## SPM vs Cache",
        "",
        f"compare: {compare.relative_to(WORKLOADS_DIR)}",
        f"spm_only: {spm_only.relative_to(WORKLOADS_DIR)}",
        "",
        compare.read_text().rstrip(),
        "",
        "## SPM-only Stats",
        "",
        spm_only.read_text().rstrip(),
        "",
    ]
    out.write_text("\n".join(lines))
    return out


def compare_graph(
    graph_name: str,
    graph: dict[str, Any],
    plans: list[NodePlan],
    skip_build: bool,
    gem5_flags: list[str],
) -> None:
    for mode in ("spm", "cache"):
        if not skip_build:
            build_graph_executable(graph_name, graph, plans, mode)
        run_graph_executable(graph_name, mode, gem5_flags)
        validate_graph_run(graph_name, mode)

    compare, spm_only = run_graph_compare_stats(graph_name)
    report = write_graph_report(graph_name, graph, plans, compare, spm_only)
    print(f"Graph compare saved: {compare.relative_to(WORKLOADS_DIR)}")
    print(f"Graph SPM stats:    {spm_only.relative_to(WORKLOADS_DIR)}")
    print(f"Graph report:       {report.relative_to(WORKLOADS_DIR)}")


def expected_allocator(tier: int, kernel: str) -> str:
    if tier == TIER_SPM_RESIDENT:
        return "spm_malloc(nbytes)"
    if tier == TIER_UNCACHEABLE_DMA:
        return "dma_buf_malloc(nbytes)"
    return f"{kernel}_record_malloc(malloc(nbytes))"


def verify_node(plan: NodePlan) -> bool:
    build_dir = trispm_paths.build_dir(plan.kernel, "spm", plan.tag)
    launcher = build_dir / f"{plan.kernel}_launcher.c"
    if not launcher.is_file():
        print(f"  [FAIL] missing launcher: {launcher}")
        return False

    text = launcher.read_text()
    ok = True
    print(f"\n===== graph placement verify: {plan.name} ({plan.kernel}) =====")
    for idx, tier in sorted(plan.arg_tiers.items()):
        expect = expected_allocator(tier, plan.kernel)
        pattern = rf"case\s+{idx}:\s+return\s+{re.escape(expect)};"
        matched = re.search(pattern, text) is not None
        status = "PASS" if matched else "FAIL"
        tensor = plan.args[idx]
        reason = plan.decisions[idx].reason
        print(
            f"  [{status}] arg{idx} {tensor}: Tier {tier} -> {expect} "
            f"({reason})")
        ok = ok and matched
    return ok


def plan_as_rows(plans: list[NodePlan]) -> list[tuple[str, str, str, str, str]]:
    rows: list[tuple[str, str, str, str, str]] = []
    for plan in plans:
        for idx, tensor in enumerate(plan.args):
            decision = plan.decisions[idx]
            rows.append((
                plan.name,
                plan.kernel,
                f"arg{idx}:{tensor}",
                f"Tier {decision.tier}",
                decision.reason,
            ))
    return rows


def render_table(rows: list[tuple[str, ...]], headers: tuple[str, ...]) -> str:
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def fmt(row: tuple[str, ...]) -> str:
        return "  ".join(row[i].ljust(widths[i]) for i in range(len(row)))

    sep = tuple("-" * w for w in widths)
    return "\n".join([fmt(headers), fmt(sep)] + [fmt(row) for row in rows])


def write_plan_json(graph_name: str, plans: list[NodePlan]) -> Path:
    out = trispm_paths.BUILD_ROOT / "graphs" / graph_name / "placement_plan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "graph": graph_name,
        "nodes": [
            {
                "name": plan.name,
                "kernel": plan.kernel,
                "tag": plan.tag,
                "tier_override": plan.tier_override,
                "args": [
                    {
                        "index": idx,
                        "tensor": plan.args[idx],
                        "tier": plan.arg_tiers[idx],
                        "reason": plan.decisions[idx].reason,
                    }
                    for idx in sorted(plan.arg_tiers)
                ],
            }
            for plan in plans
        ],
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("graph", help="graph name under workloads/graphs/<name>/graph.toml")
    parser.add_argument(
        "--mode",
        choices=("plan", "build", "verify", "build-exec", "run", "compare"),
        default="plan",
    )
    parser.add_argument(
        "--exec-mode",
        choices=("spm", "cache"),
        default="spm",
        help="mode for executable graph build/run",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="run an existing executable graph binary",
    )
    parser.add_argument(
        "--gem5-flag",
        action="append",
        default=[],
        help="extra gem5 run_spm.py flag for executable graph runs",
    )
    args = parser.parse_args()

    graph = load_graph(args.graph)
    try:
        plans = build_plan(args.graph, graph)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")

    rows = plan_as_rows(plans)
    print(f"\n===== graph placement plan: {args.graph} =====")
    print(render_table(rows, ("node", "kernel", "arg", "tier", "reason")))
    plan_json = write_plan_json(args.graph, plans)
    print(f"\nPlan written to {plan_json.relative_to(WORKLOADS_DIR)}")

    if args.mode in {"build", "verify"}:
        for plan in plans:
            print(
                f"\n========== build {plan.name} "
                f"(tier_override={plan.tier_override!r}) ==========")
            build_node(plan)

    if args.mode == "verify":
        all_ok = True
        for plan in plans:
            all_ok = verify_node(plan) and all_ok
        if not all_ok:
            sys.exit(1)
        print(f"\n{args.graph}: graph placement verification passed")

    if args.mode == "build-exec":
        if args.skip_build:
            compile_graph(args.graph, graph, plans, args.exec_mode)
        else:
            build_graph_executable(args.graph, graph, plans, args.exec_mode)

    if args.mode == "run" and not args.skip_build:
        build_graph_executable(args.graph, graph, plans, args.exec_mode)

    if args.mode == "run":
        run_graph_executable(args.graph, args.exec_mode, args.gem5_flag)
        validate_graph_run(args.graph, args.exec_mode)

    if args.mode == "compare":
        compare_graph(args.graph, graph, plans, args.skip_build, args.gem5_flag)


if __name__ == "__main__":
    main()
