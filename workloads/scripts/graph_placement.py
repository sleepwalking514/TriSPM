#!/usr/bin/env python3
"""Build/verify graph-level conservative placement artifacts.

The graph planner sits above single-kernel SPM placement.  It reads tensor-edge
metadata, chooses backing tiers for each node argument, and builds SPM artifacts
with KERNEL_TIER_OVERRIDE so the generated launcher allocation dispatch can be
checked without changing ConvertMemoryToSPM.

Usage:
  scripts/graph_placement.py layer_norm_qkv --mode plan
  scripts/graph_placement.py layer_norm_qkv --mode verify
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import tomllib
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


def load_graph(name: str) -> dict[str, Any]:
    path = GRAPHS_DIR / name / "graph.toml"
    if not path.is_file():
        sys.exit(f"ERROR: graph manifest not found: {path}")
    data = tomllib.loads(path.read_text())
    data["_path"] = path
    return data


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
    parser.add_argument("--mode", choices=("plan", "build", "verify"), default="plan")
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


if __name__ == "__main__":
    main()
