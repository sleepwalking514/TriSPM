#!/usr/bin/env python3
"""Build/verify graph-level conservative placement artifacts.

The graph planner sits above single-kernel SPM placement.  It reads tensor-edge
metadata and keeps graph tensors on ordinary DRAM by default.  SPM kernels can
still stage tiles with DMA internally, but launcher-visible DRAM allocation no
longer has the legacy split backing model.

Usage:
  scripts/graph_placement.py layer_norm_qkv --mode plan
  scripts/graph_placement.py layer_norm_qkv --mode verify
  scripts/graph_placement.py layer_norm_qkv --mode run
  scripts/graph_placement.py layer_norm_qkv --mode compare
  scripts/graph_placement.py layer_norm_qkv --mode fusion
"""
from __future__ import annotations

import argparse
import copy
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

FUSION_ABLATION_VARIANTS = [
    {
        "id": "A2",
        "key": "fused_cache",
        "label": "fused cache",
        "variant_macro": 0,
        "cache_baseline": True,
        "scheduling_changed": True,
        "spm_resident": False,
        "materializes_ln_out": False,
        "consumer_a_source": "ordinary-cache fused tile",
    },
    {
        "id": "A3",
        "key": "fused_spm_resident",
        "label": "fused SPM resident",
        "variant_macro": 1,
        "cache_baseline": False,
        "scheduling_changed": True,
        "spm_resident": True,
        "materializes_ln_out": False,
        "consumer_a_source": "resident SPM tile",
    },
    {
        "id": "A4",
        "key": "fused_spm_forced_materialize",
        "label": "fused SPM forced materialize",
        "variant_macro": 2,
        "cache_baseline": False,
        "scheduling_changed": True,
        "spm_resident": False,
        "materializes_ln_out": True,
        "consumer_a_source": "ordinary DRAM ln_out reload through DMA",
    },
]


@dataclass(frozen=True)
class TensorDecision:
    reason: str


@dataclass(frozen=True)
class NodePlan:
    name: str
    kernel: str
    tag: str
    args: list[str]
    params: dict[str, str]
    env: dict[str, str]
    decisions: dict[int, TensorDecision]

    @property
    def c_name(self) -> str:
        return sanitize_c_ident(self.name)


def load_graph(name: str) -> dict[str, Any]:
    path = GRAPHS_DIR / name / "graph.toml"
    if not path.is_file():
        sys.exit(f"ERROR: graph manifest not found: {path}")
    data = tomllib.loads(path.read_text())
    data["_path"] = path
    data["_source_graph"] = name
    return data


def merge_dict_recursive(base: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            merge_dict_recursive(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def apply_graph_preset(graph: dict[str, Any], preset: str | None) -> dict[str, Any]:
    if not preset:
        return graph
    presets = graph.get("presets", {})
    if not isinstance(presets, dict) or preset not in presets:
        sys.exit(f"ERROR: preset {preset!r} not in graph manifest")
    merged = copy.deepcopy(graph)
    preset_cfg = merged.get("presets", {}).get(preset)
    if not isinstance(preset_cfg, dict):
        sys.exit(f"ERROR: graph preset {preset!r} must be a table")
    merge_dict_recursive(merged, preset_cfg)
    merged["_preset"] = preset
    return merged


def graph_output_name(graph_name: str, preset: str | None) -> str:
    return graph_name if not preset else f"{graph_name}/{preset}"


def graph_artifact_name(
    graph_name: str,
    preset: str | None,
    artifact_tag: str | None = None,
) -> str:
    output = graph_output_name(graph_name, preset)
    return output if not artifact_tag else f"{output}/{artifact_tag}"


def graph_node_tag(graph_output: str, node_name: str) -> str:
    return f"graph/{graph_output}/{node_name}"


def sanitize_c_ident(value: str) -> str:
    ident = re.sub(r"\W+", "_", value.strip())
    ident = ident.strip("_")
    if not ident:
        raise ValueError(f"cannot derive a C identifier from {value!r}")
    if ident[0].isdigit():
        ident = f"n_{ident}"
    return ident


def choose_tensor_placement(
    tensor_name: str,
    tensor: dict[str, Any],
) -> TensorDecision:
    kind = str(tensor.get("kind", "")).strip()
    read_only = bool(tensor.get("read_only", False))
    dma_only = bool(tensor.get("dma_only", False))
    has_producer = bool(tensor.get("producer"))
    consumers = tensor.get("consumers", [])
    if consumers is None:
        consumers = []

    if kind in {"intermediate", "producer_output", "graph_output"} or has_producer:
        return TensorDecision(
            "producer output / intermediate activation stays on ordinary DRAM path",
        )

    if read_only and dma_only and kind in {"external_input", "external_weight"}:
        return TensorDecision(
            "external read-only DMA-only tensor uses ordinary DRAM backing",
        )

    if consumers:
        return TensorDecision(
            "downstream use is visible, so keep ordinary DRAM backing",
        )

    return TensorDecision(
        "conservative default",
    )


def build_plan(
    graph_name: str,
    graph: dict[str, Any],
    mode: str | None = None,
) -> list[NodePlan]:
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
        env = {str(k): str(v) for k, v in dict(node.get("env", {})).items()}
        if mode:
            mode_cfg = node.get(mode, {})
            if not isinstance(mode_cfg, dict):
                raise ValueError(f"node {node_name!r} [{mode}] override must be a table")
            params.update({
                str(k): str(v)
                for k, v in dict(mode_cfg.get("params", {})).items()
            })
            env.update({
                str(k): str(v)
                for k, v in dict(mode_cfg.get("env", {})).items()
            })
        tag = str(node.get("tag", graph_node_tag(graph_name, node_name)))

        decisions: dict[int, TensorDecision] = {}
        for idx, tensor_name in enumerate(args):
            if tensor_name not in tensors:
                raise ValueError(
                    f"node {node_name!r} arg {idx} references unknown tensor "
                    f"{tensor_name!r}")
            decision = choose_tensor_placement(
                tensor_name,
                tensors[tensor_name],
            )
            decisions[idx] = decision

        plans.append(NodePlan(
            name=node_name,
            kernel=kernel,
            tag=tag,
            args=args,
            params=params,
            env=env,
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
    env.update({k: str(v) for k, v in manifest.get("env", {}).items()})
    env.update(plan.env)
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


def graph_m5out_dir(graph_name: str, mode: str, spm_tag: str | None = None) -> Path:
    return trispm_paths.graph_m5out_dir(graph_name, mode, spm_tag)


def cache_capacity_suffix(
    cache_l1d_size: str | None = None,
    cache_l2_size: str | None = None,
) -> str:
    parts = []
    if cache_l1d_size:
        parts.append(f"l1d_{cache_l1d_size.replace('/', '_')}")
    if cache_l2_size:
        parts.append(f"l2_{cache_l2_size.replace('/', '_')}")
    return "_".join(parts)


def graph_cache_capacity_m5out_dir(
    graph_name: str,
    cache_l1d_size: str | None = None,
    cache_l2_size: str | None = None,
) -> Path:
    return (
        WORKLOADS_DIR
        / "m5out"
        / "graphs"
        / graph_name
        / f"cache_{cache_capacity_suffix(cache_l1d_size, cache_l2_size)}"
        / "default"
    )


def graph_mode_m5out_dir(
    graph_name: str,
    mode: str,
    cache_l1d_size: str | None = None,
    cache_l2_size: str | None = None,
    spm_tag: str | None = None,
) -> Path:
    if mode == "cache" and (cache_l1d_size or cache_l2_size):
        return graph_cache_capacity_m5out_dir(
            graph_name,
            cache_l1d_size,
            cache_l2_size,
        )
    return graph_m5out_dir(graph_name, mode, spm_tag if mode == "spm" else None)


def graph_run_log_path(graph_name: str, mode: str) -> Path:
    return trispm_paths.graph_run_log_path(graph_name, mode)


def graph_mode_run_log_path(
    graph_name: str,
    mode: str,
    cache_l1d_size: str | None = None,
    cache_l2_size: str | None = None,
    spm_tag: str | None = None,
) -> Path:
    return graph_mode_m5out_dir(
        graph_name, mode, cache_l1d_size, cache_l2_size, spm_tag) / "run.log"


def graph_roi_stats_path(graph_name: str, mode: str) -> Path:
    return trispm_paths.graph_roi_stats_path(graph_name, mode)


def graph_mode_roi_stats_path(
    graph_name: str,
    mode: str,
    cache_l1d_size: str | None = None,
    cache_l2_size: str | None = None,
    spm_tag: str | None = None,
) -> Path:
    return graph_mode_m5out_dir(
        graph_name, mode, cache_l1d_size, cache_l2_size, spm_tag) / "roi-stats.txt"


def graph_compare_path(graph_name: str, spm_tag: str | None = None) -> Path:
    return trispm_paths.graph_compare_path(graph_name, spm_tag)


def graph_capacity_compare_path(
    graph_name: str,
    cache_l1d_size: str | None = None,
    cache_l2_size: str | None = None,
    spm_tag: str | None = None,
) -> Path:
    return graph_m5out_dir(graph_name, "spm", spm_tag) / (
        f"compare_vs_cache_{cache_capacity_suffix(cache_l1d_size, cache_l2_size)}.txt"
    )


def graph_spm_stats_path(graph_name: str, spm_tag: str | None = None) -> Path:
    return trispm_paths.graph_spm_stats_path(graph_name, spm_tag)


def graph_report_path(graph_name: str, spm_tag: str | None = None) -> Path:
    return trispm_paths.graph_report_path(graph_name, spm_tag)


def graph_capacity_report_path(
    graph_name: str,
    cache_l1d_size: str | None = None,
    cache_l2_size: str | None = None,
    spm_tag: str | None = None,
) -> Path:
    return graph_m5out_dir(graph_name, "spm", spm_tag) / (
        f"graph_report_cache_{cache_capacity_suffix(cache_l1d_size, cache_l2_size)}.txt"
    )


def graph_fused_build_dir(graph_name: str) -> Path:
    return trispm_paths.graph_fused_build_dir(graph_name)


def graph_fusion_ablation_build_dir(graph_name: str, variant_key: str) -> Path:
    return WORKLOADS_DIR / "build" / "graphs" / graph_name / "fusion_ablation" / variant_key


def graph_fused_binary_path(graph_name: str) -> Path:
    return trispm_paths.graph_fused_binary_path(graph_name)


def graph_fusion_ablation_binary_path(graph_name: str, variant_key: str) -> Path:
    return (
        graph_fusion_ablation_build_dir(graph_name, variant_key)
        / f"{graph_name}_{variant_key}_test"
    )


def graph_fused_m5out_dir(graph_name: str) -> Path:
    return trispm_paths.graph_fused_m5out_dir(graph_name)


def graph_fusion_ablation_m5out_dir(graph_name: str, variant_key: str) -> Path:
    return WORKLOADS_DIR / "m5out" / "graphs" / graph_name / "fusion_ablation" / variant_key


def graph_fused_roi_stats_path(graph_name: str) -> Path:
    return trispm_paths.graph_fused_roi_stats_path(graph_name)


def graph_fusion_ablation_roi_stats_path(graph_name: str, variant_key: str) -> Path:
    return graph_fusion_ablation_m5out_dir(graph_name, variant_key) / "roi-stats.txt"


def graph_fused_run_log_path(graph_name: str) -> Path:
    return trispm_paths.graph_fused_run_log_path(graph_name)


def graph_fusion_ablation_run_log_path(graph_name: str, variant_key: str) -> Path:
    return graph_fusion_ablation_m5out_dir(graph_name, variant_key) / "run.log"


def graph_fused_compare_cache_path(graph_name: str) -> Path:
    return trispm_paths.graph_fused_compare_cache_path(graph_name)


def graph_fused_compare_unfused_path(graph_name: str) -> Path:
    return trispm_paths.graph_fused_compare_unfused_path(graph_name)


def graph_fusion_report_path(graph_name: str) -> Path:
    return trispm_paths.graph_fusion_report_path(graph_name)


def graph_fusion_json_path(graph_name: str) -> Path:
    return trispm_paths.graph_fusion_json_path(graph_name)


def graph_fusion_ablation_report_path(graph_name: str) -> Path:
    return (
        WORKLOADS_DIR
        / "m5out"
        / "graphs"
        / graph_name
        / "fusion_ablation"
        / "ablation_report.md"
    )


def graph_fusion_ablation_json_path(graph_name: str) -> Path:
    return (
        WORKLOADS_DIR
        / "m5out"
        / "graphs"
        / graph_name
        / "fusion_ablation"
        / "ablation_report.json"
    )


def graph_harness_params(graph: dict[str, Any], mode: str | None = None) -> dict[str, str]:
    harness = dict(graph.get("harness", {}))
    params = {str(k): str(v) for k, v in dict(harness.get("params", {})).items()}
    if mode:
        mode_cfg = harness.get(mode, {})
        if not isinstance(mode_cfg, dict):
            raise ValueError(f"[harness.{mode}] override must be a table")
        params.update({
            str(k): str(v)
            for k, v in dict(mode_cfg.get("params", {})).items()
        })
    return params


def render_graph_cflags(graph: dict[str, Any], mode: str | None = None) -> str:
    harness = dict(graph.get("harness", {}))
    params = graph_harness_params(graph, mode)
    macros = harness.get("build", {}).get("c_macros", [])
    try:
        return " ".join(f"-D{macro.format(**params)}" for macro in macros)
    except KeyError as exc:
        raise ValueError(
            f"[harness.build].c_macros references unknown param {exc.args[0]!r}"
        ) from exc


def render_fused_graph_cflags(graph: dict[str, Any], fusion: dict[str, Any]) -> str:
    base = render_graph_cflags(graph)
    micros = [
        f"-DGRAPH_FUSION_MICRO_M={int(fusion.get('micro_m', 8))}",
        f"-DGRAPH_FUSION_WINDOW_K={int(fusion.get('window_k', 4))}",
    ]
    return " ".join([base, *micros]).strip()


def render_fused_ablation_cflags(
    graph: dict[str, Any],
    fusion: dict[str, Any],
    variant_macro: int,
) -> str:
    base = render_fused_graph_cflags(graph, fusion)
    return f"{base} -DGRAPH_FUSION_VARIANT={variant_macro}".strip()


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
            (f"{plan.kernel}_arg_bytes", node_symbol(plan, "_arg_bytes")),
            (f"{plan.kernel}_arg_uses_dma_buf", node_symbol(plan, "_arg_uses_dma_buf")),
            (f"{plan.kernel}_arg_malloc", node_symbol(plan, "_arg_malloc")),
            (f"{plan.kernel}_align_up_size", node_symbol(plan, "_align_up_size")),
            (f"{plan.kernel}_head_dma_enqueue_2d", node_symbol(plan, "_head_dma_enqueue_2d")),
            (f"{plan.kernel}_head_dma_wait", node_symbol(plan, "_head_dma_wait")),
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
    source_graph = str(graph.get("_source_graph", graph_name))
    harness_source = GRAPHS_DIR / source_graph / str(harness_cfg.get("source", "harness.c"))
    if not harness_source.is_file():
        sys.exit(f"ERROR: graph harness source not found: {harness_source}")

    graph_cflags = render_graph_cflags(graph, mode)
    node_artifacts = namespace_graph_nodes(out_dir, graph_name, plans, node_dirs)
    binary = graph_binary_path(graph_name, mode)
    include_flags = [f"-I{out_dir}"]
    asm_sources = [str(artifact["asm"]) for artifact in node_artifacts]
    launcher_sources = [str(out_dir / "graph_node_launchers.c")]
    real_hw_cflags: list[str] = []
    real_hw_sources: list[str] = []
    if env.get("TRISPM_REAL_HW") == "1":
        real_hw_cflags = [
            "-DTRISPM_REAL_HW",
            f"-DSPM_BASE={env.get('TRITON_SPM_BASE', '0x40000000')}",
        ]
        real_hw_sources = [
            f"{env['TRISPM_ROOT']}/simulator/src/scratchpad_mem/libspm_real_hw.c"
        ]
    cmd = [
        env["CLANG"],
        *shlex.split(env["CLANG_FLAGS"]),
        *shlex.split(graph_cflags),
        *real_hw_cflags,
        *include_flags,
        f"-I{env['TRISPM_ROOT']}/simulator/src/scratchpad_mem",
        *asm_sources,
        *launcher_sources,
        str(harness_source),
        *real_hw_sources,
        "-lm",
        "-o",
        str(binary),
    ]
    print(f"\n========== link executable graph ({mode}) ==========")
    run(cmd, env=env)
    print(f"  -> {binary.relative_to(WORKLOADS_DIR)}")


def build_graph_executable(graph_name: str, graph: dict[str, Any], plans: list[NodePlan], mode: str) -> None:
    for plan in plans:
        print(f"\n========== build {plan.name} ({mode}) ==========")
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
            env.update({k: str(v) for k, v in manifest.get("env", {}).items()})
            env.update(plan.env)
            run([
                str(SCRIPTS_DIR / "build_kernel.sh"),
                plan.kernel,
                "--mode", "cache",
                "--tag", plan.tag,
            ], env=env)
    compile_graph(graph_name, graph, plans, mode)


def run_graph_executable(
    graph_name: str,
    mode: str,
    gem5_flags: list[str],
    cache_l1d_size: str | None = None,
    cache_l2_size: str | None = None,
    spm_tag: str | None = None,
) -> None:
    env = source_env()
    binary = graph_binary_path(graph_name, mode)
    if not binary.is_file():
        sys.exit(f"ERROR: graph binary not found: {binary}")
    m5out = graph_mode_m5out_dir(
        graph_name, mode, cache_l1d_size, cache_l2_size, spm_tag)
    run_log = graph_mode_run_log_path(
        graph_name, mode, cache_l1d_size, cache_l2_size, spm_tag)
    roi_stats = graph_mode_roi_stats_path(
        graph_name, mode, cache_l1d_size, cache_l2_size, spm_tag)
    m5out.mkdir(parents=True, exist_ok=True)

    cmd = [
        env["GEM5"],
        f"--outdir={m5out}",
        env["GEM5_RUN_SCRIPT"],
        "--binary",
        str(binary),
    ]
    display_flags = list(gem5_flags)
    if mode == "cache":
        cmd.append("--cache_baseline")
        display_flags.insert(0, "--cache_baseline")
        if cache_l1d_size:
            cmd += ["--l1d_size", cache_l1d_size]
            display_flags += ["--l1d_size", cache_l1d_size]
        if cache_l2_size:
            cmd += ["--l2_size", cache_l2_size]
            display_flags += ["--l2_size", cache_l2_size]
    cmd += gem5_flags

    print(f"\n===== Running graph {graph_name} ({mode}) on gem5 =====")
    print(f"  binary: {binary}")
    print(f"  outdir: {m5out}")
    print(f"  flags:  {' '.join(display_flags) if display_flags else '<none>'}\n")
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


def validate_graph_run(
    graph: dict[str, Any],
    graph_name: str,
    mode: str,
    cache_l1d_size: str | None = None,
    cache_l2_size: str | None = None,
    spm_tag: str | None = None,
) -> None:
    run_log = graph_mode_run_log_path(
        graph_name, mode, cache_l1d_size, cache_l2_size, spm_tag)
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
    harness_params = graph_harness_params(graph, mode)
    check_result = str(harness_params.get("CHECK_RESULT", "1")).strip().lower()
    if check_result in {"0", "false", "no", "off"} and not bad_lines:
        print(f"Result gate skipped: graph {graph_name} {mode} has CHECK_RESULT=0")
        return
    detail = "\n".join(bad_lines[:12]) if bad_lines else "PASS line was not found"
    sys.exit(
        f"ERROR: graph {graph_name} {mode} failed result gate.\n"
        f"Log: {run_log.relative_to(WORKLOADS_DIR)}\n"
        f"{detail}"
    )


def ensure_graph_roi_stats(
    graph_name: str,
    mode: str,
    cache_l1d_size: str | None = None,
    cache_l2_size: str | None = None,
    spm_tag: str | None = None,
) -> Path:
    roi_stats = graph_mode_roi_stats_path(
        graph_name, mode, cache_l1d_size, cache_l2_size, spm_tag)
    if not roi_stats.is_file():
        sys.exit(
            f"ERROR: graph {mode} ROI stats are missing: "
            f"{roi_stats.relative_to(WORKLOADS_DIR)}"
        )
    return roi_stats


def run_graph_compare_stats(
    graph: dict[str, Any],
    graph_name: str,
    cache_l1d_size: str | None = None,
    cache_l2_size: str | None = None,
    spm_tag: str | None = None,
) -> tuple[Path, Path]:
    spm_stats = ensure_graph_roi_stats(graph_name, "spm", spm_tag=spm_tag)
    cache_stats = ensure_graph_roi_stats(
        graph_name, "cache", cache_l1d_size, cache_l2_size)
    compare = (
        graph_capacity_compare_path(
            graph_name, cache_l1d_size, cache_l2_size, spm_tag)
        if cache_l1d_size or cache_l2_size else trispm_paths.graph_compare_path(graph_name, spm_tag)
    )
    spm_only = graph_spm_stats_path(graph_name, spm_tag)
    harness_params = graph_harness_params(graph, "spm")
    measure_iters = str(harness_params.get("MEASURE_ITERS", "1"))
    run([
        str(SCRIPTS_DIR / "compare_stats.py"),
        "--spm", str(spm_stats),
        "--cache", str(cache_stats),
        "--measure-iters", measure_iters,
        "--output", str(compare),
        "--spm-only-output", str(spm_only),
    ])
    return compare, spm_only


def graph_log_status(
    graph_name: str,
    mode: str,
    cache_l1d_size: str | None = None,
    cache_l2_size: str | None = None,
    spm_tag: str | None = None,
) -> str:
    run_log = graph_mode_run_log_path(
        graph_name, mode, cache_l1d_size, cache_l2_size, spm_tag)
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


def fused_graph_log_status(graph_name: str) -> str:
    run_log = graph_fused_run_log_path(graph_name)
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


def fusion_ablation_log_status(graph_name: str, variant_key: str) -> str:
    run_log = graph_fusion_ablation_run_log_path(graph_name, variant_key)
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


def summarize_run_rows(
    graph_name: str,
    cache_l1d_size: str | None = None,
    cache_l2_size: str | None = None,
    spm_tag: str | None = None,
) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for mode in ("spm", "cache"):
        label = mode
        tag_for_mode = spm_tag if mode == "spm" else None
        if mode == "spm" and spm_tag:
            label = spm_tag
        if mode == "cache" and (cache_l1d_size or cache_l2_size):
            label = f"cache_{cache_capacity_suffix(cache_l1d_size, cache_l2_size)}"
        rows.append((
            label,
            str(graph_binary_path(graph_name, mode).relative_to(WORKLOADS_DIR)),
            str(graph_mode_roi_stats_path(
                graph_name, mode, cache_l1d_size, cache_l2_size, tag_for_mode
            ).relative_to(WORKLOADS_DIR)),
            graph_log_status(
                graph_name, mode, cache_l1d_size, cache_l2_size, tag_for_mode),
        ))
    return rows


def write_graph_report(
    graph_name: str,
    graph: dict[str, Any],
    plans: list[NodePlan],
    compare: Path,
    spm_only: Path,
    cache_l1d_size: str | None = None,
    cache_l2_size: str | None = None,
    spm_tag: str | None = None,
) -> Path:
    out = (
        graph_capacity_report_path(graph_name, cache_l1d_size, cache_l2_size, spm_tag)
        if cache_l1d_size or cache_l2_size else graph_report_path(graph_name, spm_tag)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    graph_meta = dict(graph.get("graph", {}))
    harness_params = graph_harness_params(graph, "spm")
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
        render_table(plan_as_rows(plans), ("node", "kernel", "arg", "reason")),
        "",
        "## Run Artifacts",
        "",
        render_table(
            summarize_run_rows(graph_name, cache_l1d_size, cache_l2_size, spm_tag),
            ("mode", "binary", "roi_stats", "result"),
        ),
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


def parse_int_param(graph: dict[str, Any], name: str) -> int:
    harness = dict(graph.get("harness", {}))
    params = dict(harness.get("params", {}))
    if name not in params:
        raise ValueError(f"[harness.params] missing {name}")
    return int(str(params[name]), 0)


def get_layer_norm_qkv_fusion(graph: dict[str, Any]) -> dict[str, Any] | None:
    fusion_root = graph.get("fusion", {})
    if not isinstance(fusion_root, dict):
        return None
    fusion = fusion_root.get("layer_norm_qkv")
    if not isinstance(fusion, dict) or not bool(fusion.get("enabled", False)):
        return None
    return fusion


def vregs_per_row(block_n: int) -> int | None:
    if block_n == 8:
        return 1
    if block_n == 16:
        return 2
    if block_n == 32:
        return 4
    return None


def estimate_fusion_layout_bytes(
    block_m: int,
    d_model: int,
    block_n: int,
    block_k: int,
    window_k: int,
) -> int:
    def align_up(value: int, alignment: int) -> int:
        return (value + alignment - 1) & ~(alignment - 1)

    x_tile = block_m * d_model * 4
    b_window = block_k * block_n * 4 * window_k
    acc = block_m * block_n * 4
    return align_up(x_tile, 64) + align_up(b_window, 64) + align_up(acc, 64)


def validate_layer_norm_qkv_fusion(
    graph_name: str,
    graph: dict[str, Any],
    plans: list[NodePlan],
) -> dict[str, Any]:
    fusion = get_layer_norm_qkv_fusion(graph)
    if fusion is None:
        raise ValueError(
            f"graph {graph_name!r} does not enable [fusion.layer_norm_qkv]")

    if graph_name != "layer_norm_qkv":
        raise ValueError(
            "the first P1 fused path is intentionally limited to "
            "workloads/graphs/layer_norm_qkv")

    plan_by_name = {plan.name: plan for plan in plans}
    required_nodes = ("layer_norm", "q_proj", "k_proj", "v_proj")
    missing = [name for name in required_nodes if name not in plan_by_name]
    if missing:
        raise ValueError(
            "layer_norm_qkv fusion requires graph nodes: "
            + ", ".join(required_nodes)
            + f" (missing {', '.join(missing)})"
        )

    ln = plan_by_name["layer_norm"]
    q = plan_by_name["q_proj"]
    k = plan_by_name["k_proj"]
    v = plan_by_name["v_proj"]
    if ln.kernel != "layer_norm" or any(plan.kernel != "matmul" for plan in (q, k, v)):
        raise ValueError("layer_norm_qkv fusion requires layer_norm + matmul consumers")
    if ln.args[3] != "ln_out" or any(plan.args[0] != "ln_out" for plan in (q, k, v)):
        raise ValueError("layer_norm_qkv fusion requires ln_out producer-consumer edge")
    if [q.args[1], k.args[1], v.args[1]] != ["wq", "wk", "wv"]:
        raise ValueError("layer_norm_qkv fusion expects q/k/v projection weights")
    tensors = graph.get("tensors", {})
    ln_out_tensor = tensors.get("ln_out", {}) if isinstance(tensors, dict) else {}
    ln_out_consumers = list(ln_out_tensor.get("consumers", []))
    expected_consumers = list(fusion.get("consumers", ["q_proj", "k_proj", "v_proj"]))
    if sorted(ln_out_consumers) != sorted(expected_consumers):
        raise ValueError(
            "layer_norm_qkv fusion only supports closed ln_out regions; "
            f"manifest consumers are {ln_out_consumers}, expected {expected_consumers}")

    m = parse_int_param(graph, "M")
    d_model = parse_int_param(graph, "D_MODEL")
    proj_n = parse_int_param(graph, "PROJ_N")
    block_m = parse_int_param(graph, "BLOCK_SIZE_M")
    block_n = parse_int_param(graph, "BLOCK_SIZE_N")
    block_k = parse_int_param(graph, "BLOCK_SIZE_K")
    micro_m = int(fusion.get("micro_m", 8))
    window_k = int(fusion.get("window_k", 4))
    spm_size = int(os.environ.get("SPM_SIZE_BYTES") or os.environ.get("TRITON_SPM_SIZE", "32768"), 0)

    reasons: list[str] = []
    if m % block_m != 0:
        reasons.append("GRAPH_M is not divisible by BLOCK_SIZE_M")
    if proj_n % block_n != 0:
        reasons.append("GRAPH_PROJ_N is not divisible by BLOCK_SIZE_N")
    if d_model % block_k != 0:
        reasons.append("GRAPH_D_MODEL is not divisible by BLOCK_SIZE_K")
    if block_m % micro_m != 0:
        reasons.append("BLOCK_SIZE_M is not divisible by fusion micro_m")
    if micro_m not in {2, 4, 8, 16}:
        reasons.append("fusion micro_m must be one of 2, 4, 8, 16")
    vregs = vregs_per_row(block_n)
    if vregs is None:
        reasons.append("BLOCK_SIZE_N must be 8, 16, or 32 for the RVV fused path")
    elif micro_m * vregs > 32:
        reasons.append("fusion micro_m exceeds vector register budget")
    if window_k <= 0:
        reasons.append("fusion window_k must be positive")
    if window_k > 32:
        reasons.append("fusion window_k exceeds DMA queue depth 32")

    layout_bytes = estimate_fusion_layout_bytes(
        block_m, d_model, block_n, block_k, window_k)
    if layout_bytes > spm_size:
        reasons.append(
            f"fusion SPM layout needs {layout_bytes}B > SPM size {spm_size}B")

    for node in (q, k, v):
        n = int(node.params.get("N", "0"), 0)
        k_dim = int(node.params.get("K", "0"), 0)
        if n != proj_n or k_dim != d_model:
            reasons.append(f"{node.name} shape does not match graph harness dims")
        if int(node.params.get("BLOCK_SIZE_M", "0"), 0) != block_m:
            reasons.append(f"{node.name} BLOCK_SIZE_M does not match harness")
        if int(node.params.get("BLOCK_SIZE_N", "0"), 0) != block_n:
            reasons.append(f"{node.name} BLOCK_SIZE_N does not match harness")
        if int(node.params.get("BLOCK_SIZE_K", "0"), 0) != block_k:
            reasons.append(f"{node.name} BLOCK_SIZE_K does not match harness")

    if reasons:
        raise ValueError("; ".join(reasons))

    x_tile_bytes = block_m * d_model * 4
    ln_out_bytes = m * d_model * 4
    m_tiles = m // block_m
    n_tiles = proj_n // block_n
    consumers = expected_consumers
    per_consumer_a_dma_removed = x_tile_bytes * m_tiles * n_tiles
    total_a_dma_removed = per_consumer_a_dma_removed * len(consumers)
    return {
        "graph": graph_name,
        "source": str(fusion.get("source", "fused_harness.c")),
        "producer": str(fusion.get("producer", "layer_norm")),
        "consumers": consumers,
        "resident_tensor": str(fusion.get("resident_tensor", "ln_out")),
        "micro_m": micro_m,
        "window_k": window_k,
        "spm_layout_bytes": layout_bytes,
        "spm_size_bytes": spm_size,
        "tile_bytes": x_tile_bytes,
        "full_materialization_bytes": ln_out_bytes,
        "materialization_removed_bytes": ln_out_bytes,
        "consumer_edges_using_resident_tensor": [
            {
                "edge": f"layer_norm->{consumer}",
                "resident_tensor": str(fusion.get("resident_tensor", "ln_out")),
                "materialization_boundary_removed": True,
            }
            for consumer in consumers
        ],
        "per_edge_dma_removed": {
            f"layer_norm->{consumer}": per_consumer_a_dma_removed
            for consumer in consumers
        },
        "spm_bytes_reused": total_a_dma_removed,
        "dma_bytes_removed_estimate": total_a_dma_removed,
    }


def compile_fused_layer_norm_qkv(
    graph_name: str,
    graph: dict[str, Any],
    fusion_info: dict[str, Any],
) -> None:
    env = source_env()
    out_dir = graph_fused_build_dir(graph_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_name = fusion_info["source"]
    source_graph = str(graph.get("_source_graph", graph_name))
    source = GRAPHS_DIR / source_graph / source_name
    if not source.is_file():
        sys.exit(f"ERROR: fused graph source not found: {source}")

    fusion = get_layer_norm_qkv_fusion(graph)
    assert fusion is not None
    cflags = render_fused_graph_cflags(graph, fusion)
    binary = graph_fused_binary_path(graph_name)
    cmd = [
        env["CLANG"],
        *shlex.split(env["CLANG_FLAGS"]),
        "-O3",
        *shlex.split(cflags),
        f"-I{env['TRISPM_ROOT']}/simulator/src/scratchpad_mem",
        str(source),
        "-lm",
        "-o",
        str(binary),
    ]
    print(f"\n========== build fused graph {graph_name} ==========")
    run(cmd, env=env)
    print(f"  -> {binary.relative_to(WORKLOADS_DIR)}")


def compile_fusion_ablation_variant(
    graph_name: str,
    graph: dict[str, Any],
    fusion_info: dict[str, Any],
    variant: dict[str, Any],
) -> None:
    env = source_env()
    out_dir = graph_fusion_ablation_build_dir(graph_name, str(variant["key"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    source_name = fusion_info["source"]
    source_graph = str(graph.get("_source_graph", graph_name))
    source = GRAPHS_DIR / source_graph / source_name
    if not source.is_file():
        sys.exit(f"ERROR: fused graph source not found: {source}")

    fusion = get_layer_norm_qkv_fusion(graph)
    assert fusion is not None
    cflags = render_fused_ablation_cflags(
        graph,
        fusion,
        int(variant["variant_macro"]),
    )
    binary = graph_fusion_ablation_binary_path(graph_name, str(variant["key"]))
    cmd = [
        env["CLANG"],
        *shlex.split(env["CLANG_FLAGS"]),
        "-O3",
        *shlex.split(cflags),
        f"-I{env['TRISPM_ROOT']}/simulator/src/scratchpad_mem",
        str(source),
        "-lm",
        "-o",
        str(binary),
    ]
    print(
        f"\n========== build fusion ablation {variant['id']} "
        f"({variant['key']}) ==========")
    run(cmd, env=env)
    print(f"  -> {binary.relative_to(WORKLOADS_DIR)}")


def run_fused_graph_executable(graph_name: str, gem5_flags: list[str]) -> None:
    env = source_env()
    binary = graph_fused_binary_path(graph_name)
    if not binary.is_file():
        sys.exit(f"ERROR: fused graph binary not found: {binary}")

    m5out = graph_fused_m5out_dir(graph_name)
    run_log = graph_fused_run_log_path(graph_name)
    roi_stats = graph_fused_roi_stats_path(graph_name)
    m5out.mkdir(parents=True, exist_ok=True)

    cmd = [
        env["GEM5"],
        f"--outdir={m5out}",
        env["GEM5_RUN_SCRIPT"],
        "--binary",
        str(binary),
        *gem5_flags,
    ]

    print(f"\n===== Running graph {graph_name} (fused) on gem5 =====")
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


def run_fusion_ablation_variant(
    graph_name: str,
    variant: dict[str, Any],
    gem5_flags: list[str],
) -> None:
    env = source_env()
    variant_key = str(variant["key"])
    binary = graph_fusion_ablation_binary_path(graph_name, variant_key)
    if not binary.is_file():
        sys.exit(f"ERROR: fusion ablation binary not found: {binary}")

    m5out = graph_fusion_ablation_m5out_dir(graph_name, variant_key)
    run_log = graph_fusion_ablation_run_log_path(graph_name, variant_key)
    roi_stats = graph_fusion_ablation_roi_stats_path(graph_name, variant_key)
    m5out.mkdir(parents=True, exist_ok=True)

    cmd = [
        env["GEM5"],
        f"--outdir={m5out}",
        env["GEM5_RUN_SCRIPT"],
        "--binary",
        str(binary),
    ]
    display_flags = list(gem5_flags)
    if bool(variant["cache_baseline"]):
        cmd.append("--cache_baseline")
        display_flags.insert(0, "--cache_baseline")
    cmd += gem5_flags

    print(
        f"\n===== Running graph {graph_name} fusion ablation "
        f"{variant['id']} ({variant_key}) on gem5 =====")
    print(f"  binary: {binary}")
    print(f"  outdir: {m5out}")
    print(f"  flags:  {' '.join(display_flags) if display_flags else '<none>'}\n")
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


def validate_fused_graph_run(graph_name: str) -> None:
    run_log = graph_fused_run_log_path(graph_name)
    if not run_log.is_file():
        sys.exit(f"ERROR: fused graph run log is missing: {run_log}")
    text = run_log.read_text(errors="replace")
    bad_lines = [
        line for line in text.splitlines()
        if re.search(r"\b(FAIL|MISMATCH):", line)
    ]
    if "PASS: graph outputs correct" in text and not bad_lines:
        print(f"Result gate passed: graph {graph_name} fused")
        return
    detail = "\n".join(bad_lines[:12]) if bad_lines else "PASS line was not found"
    sys.exit(
        f"ERROR: graph {graph_name} fused failed result gate.\n"
        f"Log: {run_log.relative_to(WORKLOADS_DIR)}\n"
        f"{detail}"
    )


def validate_fusion_ablation_run(graph_name: str, variant: dict[str, Any]) -> None:
    variant_key = str(variant["key"])
    run_log = graph_fusion_ablation_run_log_path(graph_name, variant_key)
    if not run_log.is_file():
        sys.exit(f"ERROR: fusion ablation run log is missing: {run_log}")
    text = run_log.read_text(errors="replace")
    bad_lines = [
        line for line in text.splitlines()
        if re.search(r"\b(FAIL|MISMATCH):", line)
    ]
    if "PASS: graph outputs correct" in text and not bad_lines:
        print(
            f"Result gate passed: graph {graph_name} "
            f"{variant['id']} ({variant_key})")
        return
    detail = "\n".join(bad_lines[:12]) if bad_lines else "PASS line was not found"
    sys.exit(
        f"ERROR: graph {graph_name} fusion ablation {variant['id']} "
        f"failed result gate.\n"
        f"Log: {run_log.relative_to(WORKLOADS_DIR)}\n"
        f"{detail}"
    )


def ensure_fused_roi_stats(graph_name: str) -> Path:
    roi_stats = graph_fused_roi_stats_path(graph_name)
    if not roi_stats.is_file():
        sys.exit(
            f"ERROR: graph fused ROI stats are missing: "
            f"{roi_stats.relative_to(WORKLOADS_DIR)}"
        )
    return roi_stats


def ensure_fusion_ablation_roi_stats(graph_name: str, variant_key: str) -> Path:
    roi_stats = graph_fusion_ablation_roi_stats_path(graph_name, variant_key)
    if not roi_stats.is_file():
        sys.exit(
            f"ERROR: graph fusion ablation ROI stats are missing: "
            f"{roi_stats.relative_to(WORKLOADS_DIR)}"
        )
    return roi_stats


def run_fused_compare_stats(graph_name: str) -> tuple[Path, Path]:
    fused_stats = ensure_fused_roi_stats(graph_name)
    cache_stats = ensure_graph_roi_stats(graph_name, "cache")
    unfused_stats = ensure_graph_roi_stats(graph_name, "spm")
    compare_cache = graph_fused_compare_cache_path(graph_name)
    compare_unfused = graph_fused_compare_unfused_path(graph_name)
    run([
        str(SCRIPTS_DIR / "compare_stats.py"),
        "--spm", str(fused_stats),
        "--cache", str(cache_stats),
        "--measure-iters", "1",
        "--output", str(compare_cache),
        "--spm-label", "fused",
        "--cache-label", "cache",
    ])
    run([
        str(SCRIPTS_DIR / "compare_stats.py"),
        "--spm", str(fused_stats),
        "--cache", str(unfused_stats),
        "--measure-iters", "1",
        "--output", str(compare_unfused),
        "--spm-only-output", str(graph_fused_m5out_dir(graph_name) / "spm_stats.txt"),
        "--spm-label", "fused",
        "--cache-label", "unfused_spm",
    ])
    return compare_cache, compare_unfused


def read_stat(path: Path, name: str) -> float | None:
    if not path.is_file():
        return None
    for raw in path.read_text(errors="replace").splitlines():
        parts = raw.split()
        if len(parts) >= 2 and parts[0] == name:
            try:
                return float(parts[1])
            except ValueError:
                return None
    return None


SUMMARY_STAT_NAMES = {
    "cycles": "system.cpu.numCycles",
    "simInsts": "simInsts",
    "ipc": "system.cpu.ipc",
    "l1d_misses": "system.l1d.demandMisses::total",
    "l2_misses": "system.l2cache.demandMisses::total",
    "spm_dma_bytes": "system.spm_dma.bytesTransferred",
    "spm_dma_transfers": "system.spm_dma.transfers",
    "spm_dma_wait_cycles": "system.spm_dma.waitStallCycles",
    "spm_bytes_read": "system.spm.bytesRead::total",
    "spm_bytes_written": "system.spm.bytesWritten::total",
    "spm_bank_conflicts": "system.spm.bankConflicts",
}


def stat_value(path: Path, stat_name: str) -> float | None:
    return read_stat(path, stat_name)


def fmt_stat(value: float | None) -> str:
    if value is None:
        return "-"
    if float(value).is_integer():
        return f"{int(value)}"
    return f"{value:.6g}"


def fmt_ratio(numer: float | None, denom: float | None) -> str:
    if numer is None or denom in (None, 0):
        return "-"
    return f"{denom / numer:.3f}x"


def fusion_ablation_record(
    graph_name: str,
    variant: dict[str, Any],
    fusion_info: dict[str, Any],
) -> dict[str, Any]:
    variant_key = str(variant["key"])
    roi = graph_fusion_ablation_roi_stats_path(graph_name, variant_key)
    stats = {
        label: stat_value(roi, stat_name)
        for label, stat_name in SUMMARY_STAT_NAMES.items()
    }
    return {
        "id": variant["id"],
        "key": variant_key,
        "label": variant["label"],
        "result": fusion_ablation_log_status(graph_name, variant_key),
        "scheduling_changed": variant["scheduling_changed"],
        "spm_resident": variant["spm_resident"],
        "materializes_ln_out": variant["materializes_ln_out"],
        "consumer_a_source": variant["consumer_a_source"],
        "cache_baseline": variant["cache_baseline"],
        "materialization_removed_bytes": (
            0 if variant["materializes_ln_out"]
            else fusion_info["materialization_removed_bytes"]
        ),
        "estimated_consumer_a_dma_bytes": (
            0 if variant["cache_baseline"] or variant["spm_resident"]
            else fusion_info["dma_bytes_removed_estimate"]
        ),
        "stats": stats,
        "artifacts": {
            "binary": str(
                graph_fusion_ablation_binary_path(
                    graph_name, variant_key).relative_to(WORKLOADS_DIR)
            ),
            "roi_stats": str(roi.relative_to(WORKLOADS_DIR)),
            "run_log": str(
                graph_fusion_ablation_run_log_path(
                    graph_name, variant_key).relative_to(WORKLOADS_DIR)
            ),
        },
    }


def write_fusion_ablation_report(
    graph_name: str,
    graph: dict[str, Any],
    fusion_info: dict[str, Any],
) -> tuple[Path, Path]:
    records = [
        fusion_ablation_record(graph_name, variant, fusion_info)
        for variant in FUSION_ABLATION_VARIANTS
    ]
    by_id = {str(record["id"]): record for record in records}
    a2_cycles = by_id["A2"]["stats"]["cycles"]
    a3_cycles = by_id["A3"]["stats"]["cycles"]
    a4_cycles = by_id["A4"]["stats"]["cycles"]

    summary_rows = [
        (
            str(record["id"]),
            str(record["label"]),
            str(record["result"]),
            fmt_stat(record["stats"]["cycles"]),
            fmt_stat(record["stats"]["spm_dma_bytes"]),
            fmt_stat(record["stats"]["spm_dma_transfers"]),
            "yes" if bool(record["spm_resident"]) else "no",
            "yes" if bool(record["materializes_ln_out"]) else "no",
            str(record["consumer_a_source"]),
        )
        for record in records
    ]
    comparison_rows = [
        (
            "A3 / A2",
            "SPM residency value over ordinary fused cache",
            fmt_ratio(a3_cycles, a2_cycles),
        ),
        (
            "A3 / A4",
            "resident tile reuse value over forced materialization",
            fmt_ratio(a3_cycles, a4_cycles),
        ),
        (
            "A4 / A2",
            "fused SPM harness without resident reuse vs fused cache",
            fmt_ratio(a4_cycles, a2_cycles),
        ),
    ]
    stat_rows: list[tuple[str, str, str, str, str, str, str]] = []
    for record in records:
        stats = record["stats"]
        stat_rows.append((
            str(record["id"]),
            fmt_stat(stats["simInsts"]),
            fmt_stat(stats["ipc"]),
            fmt_stat(stats["l1d_misses"]),
            fmt_stat(stats["l2_misses"]),
            fmt_stat(stats["spm_dma_wait_cycles"]),
            fmt_stat(stats["spm_bank_conflicts"]),
        ))

    report = graph_fusion_ablation_report_path(graph_name)
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# P1.1 Fusion Ablation Report: {graph_name}",
        "",
        "This P1.1 ablation run keeps the same graph inputs and result checker",
        "across fused variants, then changes only the fused cache/SPM",
        "residency/materialization policy.  The fused harness is hand-tuned",
        "C/RVV evidence for attribution, not the main graph-executor baseline.",
        "",
        "## Variants",
        "",
        render_table(
            summary_rows,
            (
                "id",
                "variant",
                "result",
                "cycles",
                "dma_bytes",
                "dma_desc",
                "spm_resident",
                "materializes_ln_out",
                "consumer_a_source",
            ),
        ),
        "",
        "## Key Ratios",
        "",
        render_table(comparison_rows, ("comparison", "meaning", "speedup")),
        "",
        "## Secondary Stats",
        "",
        render_table(
            stat_rows,
            (
                "id",
                "simInsts",
                "ipc",
                "l1d_misses",
                "l2_misses",
                "dma_wait_cycles",
                "spm_bank_conflicts",
            ),
        ),
        "",
        "## Artifacts",
        "",
    ]
    artifact_rows = [
        (
            str(record["id"]),
            str(record["artifacts"]["binary"]),
            str(record["artifacts"]["roi_stats"]),
            str(record["artifacts"]["run_log"]),
        )
        for record in records
    ]
    lines += [
        render_table(artifact_rows, ("id", "binary", "roi_stats", "run_log")),
        "",
    ]
    report.write_text("\n".join(line.rstrip() for line in lines) + "\n")

    payload = {
        "graph": graph_name,
        "manifest": str(Path(graph.get("_path", "")).relative_to(WORKLOADS_DIR)),
        "fusion": fusion_info,
        "records": records,
        "comparisons": {
            "A3_over_A2": None if not a3_cycles or not a2_cycles else a2_cycles / a3_cycles,
            "A3_over_A4": None if not a3_cycles or not a4_cycles else a4_cycles / a3_cycles,
            "A4_over_A2": None if not a4_cycles or not a2_cycles else a2_cycles / a4_cycles,
        },
        "artifacts": {
            "report": str(report.relative_to(WORKLOADS_DIR)),
        },
    }
    out_json = graph_fusion_ablation_json_path(graph_name)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    return report, out_json


def write_fusion_report(
    graph_name: str,
    graph: dict[str, Any],
    fusion_info: dict[str, Any],
    compare_cache: Path,
    compare_unfused: Path,
) -> Path:
    report = graph_fusion_report_path(graph_name)
    report.parent.mkdir(parents=True, exist_ok=True)

    fused_stats = graph_fused_roi_stats_path(graph_name)
    unfused_stats = graph_roi_stats_path(graph_name, "spm")
    cache_stats = graph_roi_stats_path(graph_name, "cache")
    fused_cycles = read_stat(fused_stats, "system.cpu.numCycles")
    unfused_cycles = read_stat(unfused_stats, "system.cpu.numCycles")
    cache_cycles = read_stat(cache_stats, "system.cpu.numCycles")
    fused_dma_bytes = read_stat(fused_stats, "system.spm_dma.bytesTransferred")
    unfused_dma_bytes = read_stat(unfused_stats, "system.spm_dma.bytesTransferred")

    summary_rows = [
        ("result", fused_graph_log_status(graph_name)),
        ("fused_cycles", "-" if fused_cycles is None else f"{fused_cycles:.0f}"),
        ("unfused_spm_cycles", "-" if unfused_cycles is None else f"{unfused_cycles:.0f}"),
        ("cache_cycles", "-" if cache_cycles is None else f"{cache_cycles:.0f}"),
        (
            "speedup_vs_unfused_spm",
            "-" if not fused_cycles or not unfused_cycles else f"{unfused_cycles / fused_cycles:.3f}x",
        ),
        (
            "speedup_vs_cache",
            "-" if not fused_cycles or not cache_cycles else f"{cache_cycles / fused_cycles:.3f}x",
        ),
        (
            "measured_dma_bytes_removed_vs_unfused",
            "-" if fused_dma_bytes is None or unfused_dma_bytes is None else f"{unfused_dma_bytes - fused_dma_bytes:.0f}",
        ),
        ("estimated_ln_out_materialization_removed_bytes", str(fusion_info["materialization_removed_bytes"])),
        ("spm_layout_bytes", str(fusion_info["spm_layout_bytes"])),
        ("spm_size_bytes", str(fusion_info["spm_size_bytes"])),
    ]
    resident_edge_rows = [
        (
            str(record["edge"]),
            str(record["resident_tensor"]),
            "yes" if bool(record["materialization_boundary_removed"]) else "no",
        )
        for record in fusion_info["consumer_edges_using_resident_tensor"]
    ]
    dma_edge_rows = [
        (edge, str(bytes_removed))
        for edge, bytes_removed in fusion_info["per_edge_dma_removed"].items()
    ]
    lines = [
        f"# Fused Producer-Consumer Report: {graph_name}",
        "",
        "## Summary",
        "",
        render_table(summary_rows, ("metric", "value")),
        "",
        "## Fusion Record",
        "",
        render_table([
            ("producer", str(fusion_info["producer"])),
            ("consumers", ", ".join(fusion_info["consumers"])),
            ("resident_tensor", str(fusion_info["resident_tensor"])),
            ("micro_m", str(fusion_info["micro_m"])),
            ("window_k", str(fusion_info["window_k"])),
            ("spm_bytes_reused", str(fusion_info["spm_bytes_reused"])),
            ("dma_bytes_removed_estimate", str(fusion_info["dma_bytes_removed_estimate"])),
        ], ("field", "value")),
        "",
        "## Consumer Edges Using Resident Tensor",
        "",
        render_table(
            resident_edge_rows,
            ("edge", "resident_tensor", "materialization_boundary_removed"),
        ),
        "",
        "## Per-Edge Consumer DMA Removed",
        "",
        render_table(dma_edge_rows, ("edge", "bytes")),
        "",
        "## Run Artifacts",
        "",
        render_table([
            (
                "fused",
                str(graph_fused_binary_path(graph_name).relative_to(WORKLOADS_DIR)),
                str(graph_fused_roi_stats_path(graph_name).relative_to(WORKLOADS_DIR)),
                fused_graph_log_status(graph_name),
            ),
            (
                "unfused_spm",
                str(graph_binary_path(graph_name, "spm").relative_to(WORKLOADS_DIR)),
                str(graph_roi_stats_path(graph_name, "spm").relative_to(WORKLOADS_DIR)),
                graph_log_status(graph_name, "spm"),
            ),
            (
                "cache",
                str(graph_binary_path(graph_name, "cache").relative_to(WORKLOADS_DIR)),
                str(graph_roi_stats_path(graph_name, "cache").relative_to(WORKLOADS_DIR)),
                graph_log_status(graph_name, "cache"),
            ),
        ], ("mode", "binary", "roi_stats", "result")),
        "",
        "## Fused vs Cache",
        "",
        f"compare: {compare_cache.relative_to(WORKLOADS_DIR)}",
        "",
        compare_cache.read_text().rstrip(),
        "",
        "## Fused vs Unfused SPM",
        "",
        f"compare: {compare_unfused.relative_to(WORKLOADS_DIR)}",
        "",
        compare_unfused.read_text().rstrip(),
        "",
    ]
    report.write_text("\n".join(lines))

    payload = dict(fusion_info)
    payload.update({
        "result": fused_graph_log_status(graph_name),
        "fused_cycles": fused_cycles,
        "unfused_spm_cycles": unfused_cycles,
        "cache_cycles": cache_cycles,
        "speedup_vs_unfused_spm": (
            None if not fused_cycles or not unfused_cycles else unfused_cycles / fused_cycles
        ),
        "speedup_vs_cache": (
            None if not fused_cycles or not cache_cycles else cache_cycles / fused_cycles
        ),
        "fused_dma_bytes": fused_dma_bytes,
        "unfused_spm_dma_bytes": unfused_dma_bytes,
        "measured_dma_bytes_removed_vs_unfused": (
            None if fused_dma_bytes is None or unfused_dma_bytes is None
            else unfused_dma_bytes - fused_dma_bytes
        ),
        "artifacts": {
            "report": str(report.relative_to(WORKLOADS_DIR)),
            "compare_vs_cache": str(compare_cache.relative_to(WORKLOADS_DIR)),
            "compare_vs_unfused_spm": str(compare_unfused.relative_to(WORKLOADS_DIR)),
            "roi_stats": str(fused_stats.relative_to(WORKLOADS_DIR)),
            "run_log": str(graph_fused_run_log_path(graph_name).relative_to(WORKLOADS_DIR)),
        },
    })
    graph_fusion_json_path(graph_name).write_text(json.dumps(payload, indent=2) + "\n")
    return report


def run_layer_norm_qkv_fusion(
    graph_name: str,
    graph: dict[str, Any],
    plans: list[NodePlan],
    skip_build: bool,
    gem5_flags: list[str],
) -> None:
    try:
        fusion_info = validate_layer_norm_qkv_fusion(graph_name, graph, plans)
    except ValueError as exc:
        sys.exit(f"ERROR: fusion rejected: {exc}")

    print("\n===== graph fusion plan: layer_norm_qkv =====")
    print(render_table([
        ("producer", str(fusion_info["producer"])),
        ("consumers", ", ".join(fusion_info["consumers"])),
        ("resident_tensor", str(fusion_info["resident_tensor"])),
        ("micro_m", str(fusion_info["micro_m"])),
        ("window_k", str(fusion_info["window_k"])),
        ("spm_layout_bytes", str(fusion_info["spm_layout_bytes"])),
        ("materialization_removed_bytes", str(fusion_info["materialization_removed_bytes"])),
    ], ("field", "value")))

    if not skip_build:
        for mode in ("spm", "cache"):
            build_graph_executable(graph_name, graph, plans, mode)
        compile_fused_layer_norm_qkv(graph_name, graph, fusion_info)

    for mode in ("spm", "cache"):
        run_graph_executable(
            graph_name,
            mode,
            gem5_flags,
        )
        validate_graph_run(graph, graph_name, mode)

    run_fused_graph_executable(graph_name, gem5_flags)
    validate_fused_graph_run(graph_name)

    compare_cache, compare_unfused = run_fused_compare_stats(graph_name)
    report = write_fusion_report(
        graph_name,
        graph,
        fusion_info,
        compare_cache,
        compare_unfused,
    )
    print(f"Fused vs cache:       {compare_cache.relative_to(WORKLOADS_DIR)}")
    print(f"Fused vs unfused SPM: {compare_unfused.relative_to(WORKLOADS_DIR)}")
    print(f"Fusion report:        {report.relative_to(WORKLOADS_DIR)}")


def run_layer_norm_qkv_fusion_ablation(
    graph_name: str,
    graph: dict[str, Any],
    plans: list[NodePlan],
    skip_build: bool,
    gem5_flags: list[str],
) -> None:
    try:
        fusion_info = validate_layer_norm_qkv_fusion(graph_name, graph, plans)
    except ValueError as exc:
        sys.exit(f"ERROR: fusion ablation rejected: {exc}")

    print("\n===== graph fusion ablation plan: layer_norm_qkv =====")
    rows = [
        (
            str(variant["id"]),
            str(variant["key"]),
            str(variant["consumer_a_source"]),
            "yes" if bool(variant["cache_baseline"]) else "no",
            "yes" if bool(variant["spm_resident"]) else "no",
            "yes" if bool(variant["materializes_ln_out"]) else "no",
        )
        for variant in FUSION_ABLATION_VARIANTS
    ]
    print(render_table(
        rows,
        ("id", "key", "consumer_a_source", "cache_run", "spm_resident", "mat_ln_out"),
    ))

    if not skip_build:
        for variant in FUSION_ABLATION_VARIANTS:
            compile_fusion_ablation_variant(
                graph_name,
                graph,
                fusion_info,
                variant,
            )

    for variant in FUSION_ABLATION_VARIANTS:
        run_fusion_ablation_variant(graph_name, variant, gem5_flags)
        validate_fusion_ablation_run(graph_name, variant)

    report, out_json = write_fusion_ablation_report(graph_name, graph, fusion_info)
    print(f"Fusion ablation report: {report.relative_to(WORKLOADS_DIR)}")
    print(f"Fusion ablation JSON:   {out_json.relative_to(WORKLOADS_DIR)}")


def compare_graph(
    graph_name: str,
    graph: dict[str, Any],
    plans: list[NodePlan],
    skip_build: bool,
    gem5_flags: list[str],
    cache_l1d_size: str | None = None,
    cache_l2_size: str | None = None,
    cache_only: bool = False,
    spm_tag: str | None = None,
) -> None:
    modes = ("cache",) if cache_only else ("spm", "cache")
    compare_plans: dict[str, list[NodePlan]] = {}
    for mode in modes:
        mode_plans = build_plan(graph_name, graph, mode=mode)
        compare_plans[mode] = mode_plans
        if not skip_build:
            build_graph_executable(graph_name, graph, mode_plans, mode)
        run_graph_executable(
            graph_name,
            mode,
            gem5_flags,
            cache_l1d_size,
            cache_l2_size,
            spm_tag if mode == "spm" else None,
        )
        validate_graph_run(
            graph,
            graph_name,
            mode,
            cache_l1d_size,
            cache_l2_size,
            spm_tag if mode == "spm" else None,
        )

    if cache_only:
        return

    compare, spm_only = run_graph_compare_stats(
        graph,
        graph_name,
        cache_l1d_size,
        cache_l2_size,
        spm_tag,
    )
    report = write_graph_report(
        graph_name,
        graph,
        compare_plans.get("spm", plans),
        compare,
        spm_only,
        cache_l1d_size,
        cache_l2_size,
        spm_tag,
    )
    print(f"Graph compare saved: {compare.relative_to(WORKLOADS_DIR)}")
    print(f"Graph SPM stats:    {spm_only.relative_to(WORKLOADS_DIR)}")
    print(f"Graph report:       {report.relative_to(WORKLOADS_DIR)}")


def verify_node(plan: NodePlan) -> bool:
    build_dir = trispm_paths.build_dir(plan.kernel, "spm", plan.tag)
    launcher = build_dir / f"{plan.kernel}_launcher.c"
    if not launcher.is_file():
        print(f"  [FAIL] missing launcher: {launcher}")
        return False

    text = launcher.read_text()
    print(f"\n===== graph placement verify: {plan.name} ({plan.kernel}) =====")
    default_pattern = rf"default:\s+return\s+{re.escape(plan.kernel)}_arg_malloc\(arg_index,\s*nbytes\);"
    has_default_allocator = re.search(default_pattern, text) is not None
    has_legacy_cases = re.search(
        r"case\s+\d+:\s+return\s+(?:spm_malloc|dma_buf_malloc)\(nbytes\);",
        text,
    ) is not None
    print(
        f"  [{'PASS' if has_default_allocator else 'FAIL'}] ordinary DRAM default allocator")
    print(
        f"  [{'FAIL' if has_legacy_cases else 'PASS'}] no legacy allocator cases")
    return has_default_allocator and not has_legacy_cases


def plan_as_rows(plans: list[NodePlan]) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for plan in plans:
        for idx, tensor in enumerate(plan.args):
            decision = plan.decisions[idx]
            rows.append((
                plan.name,
                plan.kernel,
                f"arg{idx}:{tensor}",
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
                "args": [
                    {
                        "index": idx,
                        "tensor": plan.args[idx],
                        "reason": plan.decisions[idx].reason,
                    }
                    for idx in range(len(plan.args))
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
        "--preset",
        default=None,
        help="graph preset under [presets.<name>] in graph.toml",
    )
    parser.add_argument(
        "--mode",
        choices=(
            "plan",
            "build",
            "verify",
            "build-exec",
            "run",
            "compare",
            "fusion",
            "fusion-ablation",
        ),
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
    parser.add_argument(
        "--cache-l1d-size",
        default=None,
        help="cache-only L1D capacity for capacity/fairness baselines, e.g. 256KiB",
    )
    parser.add_argument(
        "--cache-l2-size",
        default=None,
        help="cache-only L2 capacity for capacity/fairness baselines, e.g. 1MiB",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="with --mode compare, run only the cache executable and skip report compare",
    )
    parser.add_argument(
        "--spm-tag",
        default=None,
        help="optional output directory tag for SPM graph variants, e.g. spm_lat_2ns",
    )
    parser.add_argument(
        "--artifact-tag",
        default=None,
        help="optional suffix for graph build/m5out artifacts; isolates concurrent campaigns",
    )
    args = parser.parse_args()

    graph = apply_graph_preset(load_graph(args.graph), args.preset)
    graph_output = graph_artifact_name(args.graph, args.preset, args.artifact_tag)
    try:
        plans = build_plan(graph_output, graph)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")

    rows = plan_as_rows(plans)
    print(f"\n===== graph placement plan: {graph_output} =====")
    print(render_table(rows, ("node", "kernel", "arg", "reason")))
    plan_json = write_plan_json(graph_output, plans)
    print(f"\nPlan written to {plan_json.relative_to(WORKLOADS_DIR)}")

    if args.mode in {"build", "verify"}:
        build_plans = build_plan(
            graph_output,
            graph,
            mode="spm",
        )
        for plan in build_plans:
            print(f"\n========== build {plan.name} ==========")
            build_node(plan)

    if args.mode == "verify":
        all_ok = True
        for plan in build_plans:
            all_ok = verify_node(plan) and all_ok
        if not all_ok:
            sys.exit(1)
        print(f"\n{graph_output}: graph placement verification passed")

    if args.mode == "build-exec":
        mode_plans = build_plan(
            graph_output,
            graph,
            mode=args.exec_mode,
        )
        if args.skip_build:
            compile_graph(graph_output, graph, mode_plans, args.exec_mode)
        else:
            build_graph_executable(graph_output, graph, mode_plans, args.exec_mode)

    if args.mode == "run" and not args.skip_build:
        mode_plans = build_plan(
            graph_output,
            graph,
            mode=args.exec_mode,
        )
        build_graph_executable(graph_output, graph, mode_plans, args.exec_mode)

    if args.mode == "run":
        mode_plans = build_plan(
            graph_output,
            graph,
            mode=args.exec_mode,
        )
        run_graph_executable(
            graph_output,
            args.exec_mode,
            args.gem5_flag,
            args.cache_l1d_size,
            args.cache_l2_size,
            args.spm_tag if args.exec_mode == "spm" else None,
        )
        validate_graph_run(
            graph,
            graph_output,
            args.exec_mode,
            args.cache_l1d_size,
            args.cache_l2_size,
            args.spm_tag if args.exec_mode == "spm" else None,
        )

    if args.mode == "compare":
        compare_graph(
            graph_output,
            graph,
            plans,
            args.skip_build,
            args.gem5_flag,
            args.cache_l1d_size,
            args.cache_l2_size,
            args.cache_only,
            args.spm_tag,
        )

    if args.mode == "fusion":
        run_layer_norm_qkv_fusion(
            graph_output,
            graph,
            plans,
            args.skip_build,
            args.gem5_flag,
        )

    if args.mode == "fusion-ablation":
        run_layer_norm_qkv_fusion_ablation(
            graph_output,
            graph,
            plans,
            args.skip_build,
            args.gem5_flag,
        )


if __name__ == "__main__":
    main()
