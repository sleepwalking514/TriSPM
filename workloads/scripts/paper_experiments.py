#!/usr/bin/env python3
"""Generate and optionally run the paper experiment matrix.

This runner is intentionally paper-scoped: it encodes the reproducible table
rows on top of the tuned defaults, without pulling in the older
ablation-heavy fresh-eval profiles.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trispm_paths import WORKLOADS_DIR


SCRIPTS_DIR = Path(__file__).resolve().parent
CAMPAIGN_ROOT = WORKLOADS_DIR / "m5out" / "campaigns"
PHASE_ORDER = [
    "kernel-headline",
    "graph-headline",
    "graph-scale",
    "graph-hw-sensitivity",
    "graph-profile",
    "softmax-fairness",
    "attention-algorithm-fairness",
    "generic-affine-fallback",
    "generic-affine-fallback-perf",
    "gemm-tuning-mechanism",
    "split",
    "cache-capacity-fairness",
    "xspm-instruction",
]
SERIAL_PHASES = {
    "graph-headline",
    "graph-hw-sensitivity",
    "graph-profile",
}

MACHINES = {
    "Cache-base": {"mode": "cache", "l1d": "32KiB", "l2": "512KiB"},
    "Cache-capacity": {"mode": "cache", "l1d": None, "l2": "1MiB"},
    "Cache-stress": {"mode": "cache", "l1d": "256KiB", "l2": "4MiB"},
    "TriSPM": {"mode": "spm", "l1d": "32KiB", "l2": "512KiB", "spm": "32KiB"},
}

GRAPH_SEQ = 512
GRAPH_D_MODEL = 512
GRAPH_HEAD_DIM = 64
GRAPH_FFN_DIM = 2048

MATMUL_BEST = {
    "qkv-proj": {
        "shape": {"M": GRAPH_SEQ, "N": GRAPH_HEAD_DIM, "K": GRAPH_D_MODEL},
        "cache": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1},
        "spm": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 2},
    },
    "attn-qk": {
        "shape": {"M": GRAPH_SEQ, "N": GRAPH_SEQ, "K": GRAPH_HEAD_DIM},
        "cache": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1},
        "spm": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1},
    },
    "attn-pv": {
        "shape": {"M": GRAPH_SEQ, "N": GRAPH_HEAD_DIM, "K": GRAPH_SEQ},
        "cache": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1},
        "spm": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 2},
    },
    "attn-o-proj": {
        "shape": {"M": GRAPH_SEQ, "N": GRAPH_D_MODEL, "K": GRAPH_HEAD_DIM},
        "cache": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1},
        "spm": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1},
    },
    "ffn-up": {
        "shape": {"M": GRAPH_SEQ, "N": GRAPH_FFN_DIM, "K": GRAPH_D_MODEL},
        "cache": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1},
        "spm": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1},
    },
    "ffn-down": {
        "shape": {"M": GRAPH_SEQ, "N": GRAPH_D_MODEL, "K": GRAPH_FFN_DIM},
        "cache": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1},
        "spm": {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1},
    },
}

SPM_MATMUL_ENV = {
    "TRITON_MICRO_M": 8,
    "TRITON_SPM_WINDOW_K": 4,
    "TRITON_SPM_PROMOTION_REPORT": 1,
}

GENERIC_AFFINE_FALLBACK_ENV = {
    "TRITON_SPM_PROMOTION_REPORT": 1,
    "TRITON_SPM_GENERIC_AFFINE_TILE_MIN_BYTES": 0,
}

NO_SPLIT_MATMUL_ENV = {
    "TRITON_MICRO_M": 32,
    "TRITON_SPM_WINDOW_K": 1,
    "TRITON_SPM_PROMOTION_REPORT": 1,
}


@dataclass(frozen=True)
class Row:
    phase: str
    workload: str
    label: str
    command: list[str]
    mode: str
    machine: str
    row_kind: str
    paper_table: bool = False
    comparison: str = ""
    estimated_minutes: float = 0.0
    env: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def row_id(self) -> str:
        stable = {
            "phase": self.phase,
            "workload": self.workload,
            "label": self.label,
            "command": self.command,
            "env": self.env,
            "machine": self.machine,
        }
        digest = hashlib.sha1(
            json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:10]
        return f"{self.phase}-{self.workload}-{digest}"

    def as_record(self) -> dict[str, object]:
        return {
            "row_id": self.row_id,
            "phase": self.phase,
            "workload": self.workload,
            "label": self.label,
            "mode": self.mode,
            "machine": self.machine,
            "row_kind": self.row_kind,
            "paper_table": self.paper_table,
            "comparison": self.comparison,
            "estimated_minutes": self.estimated_minutes,
            "env": self.env,
            "command": self.command,
            "command_text": shlex.join(self.command),
            "metadata": self.metadata,
        }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def add_set(cmd: list[str], params: dict[str, object]) -> None:
    for key in sorted(params):
        cmd += ["--set", f"{key}={params[key]}"]


def add_env(cmd: list[str], env: dict[str, object]) -> None:
    for key in sorted(env):
        cmd += ["--env", f"{key}={env[key]}"]


def suffix_tag(tag: str, suffix: str) -> str:
    if not suffix:
        return tag
    return f"{tag}-{suffix}"


def suffix_graph_tag(tag: str, suffix: str) -> str:
    if not suffix:
        return tag
    return f"{tag}/{suffix}"


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


def capacity_tag(
    tag: str,
    cache_l1d_size: str | None = None,
    cache_l2_size: str | None = None,
) -> str:
    if not cache_l1d_size and not cache_l2_size:
        return tag
    suffix = cache_capacity_suffix(cache_l1d_size, cache_l2_size)
    parts = [part for part in tag.split("/") if part]
    if parts and parts[-1] == suffix:
        return tag
    return f"{tag}/{suffix}"


def kernel_cmd(
    kernel: str,
    mode: str,
    params: dict[str, object],
    *,
    tag: str,
    machine: str,
    preset: str | None = None,
    env: dict[str, object] | None = None,
) -> list[str]:
    cmd = [sys.executable, str(SCRIPTS_DIR / "run_experiment.py"), kernel, "--mode", mode]
    if preset:
        cmd += ["--preset", preset]
    cmd += ["--tag", tag]
    add_set(cmd, params)
    if env:
        add_env(cmd, env)
    machine_cfg = MACHINES[machine]
    if mode == "cache":
        if machine_cfg.get("l1d"):
            cmd += ["--cache-l1d-size", str(machine_cfg["l1d"])]
        if machine_cfg.get("l2"):
            cmd += ["--cache-l2-size", str(machine_cfg["l2"])]
    return cmd


def graph_cmd(
    graph: str,
    *,
    preset: str | None,
    machine: str,
    spm_tag: str | None = None,
    artifact_tag: str | None = None,
    skip_build: bool = False,
    gem5_flags: list[str] | None = None,
) -> list[str]:
    cmd = [sys.executable, str(SCRIPTS_DIR / "graph_eval.py"), graph]
    if preset:
        cmd += ["--preset", preset]
    if artifact_tag:
        cmd += ["--artifact-tag", artifact_tag]
    if skip_build:
        cmd.append("--skip-build")
    machine_cfg = MACHINES[machine]
    if machine == "TriSPM":
        if spm_tag:
            cmd += ["--spm-tag", spm_tag]
    else:
        if machine_cfg.get("l1d"):
            cmd += ["--cache-l1d-size", str(machine_cfg["l1d"])]
        if machine_cfg.get("l2"):
            cmd += ["--cache-l2-size", str(machine_cfg["l2"])]
        if spm_tag:
            cmd += ["--spm-tag", spm_tag]
    for flag in gem5_flags or []:
        cmd.append(f"--gem5-flag={flag}")
    return cmd


def graph_spm_run_cmd(
    graph: str,
    *,
    preset: str | None,
    spm_tag: str | None = None,
    artifact_tag: str | None = None,
    skip_build: bool = False,
    gem5_flags: list[str] | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "graph_placement.py"),
        graph,
        "--mode",
        "run",
        "--exec-mode",
        "spm",
    ]
    if preset:
        cmd += ["--preset", preset]
    if artifact_tag:
        cmd += ["--artifact-tag", artifact_tag]
    if skip_build:
        cmd.append("--skip-build")
    if spm_tag:
        cmd += ["--spm-tag", spm_tag]
    for flag in gem5_flags or []:
        cmd.append(f"--gem5-flag={flag}")
    return cmd


def matmul_tag(role: str, mode: str, cfg: dict[str, int], suffix: str) -> str:
    shape = MATMUL_BEST[role]["shape"]
    block = f"{cfg['BLOCK_SIZE_M']}x{cfg['BLOCK_SIZE_N']}x{cfg['BLOCK_SIZE_K']}"
    return (
        f"{shape['M']}x{shape['N']}x{shape['K']}/paper-{role}-"
        f"{mode}-{block}-gsm{cfg['GROUP_SIZE_M']}{suffix}"
    )


def kernel_headline_rows() -> list[Row]:
    rows: list[Row] = []
    for role, record in MATMUL_BEST.items():
        for mode, machine in (("cache", "Cache-base"), ("spm", "TriSPM")):
            cfg = dict(record[mode])
            params = {**record["shape"], **cfg, "CHECK_RESULT": 0}
            env = SPM_MATMUL_ENV if mode == "spm" else None
            tag = matmul_tag(role, mode, cfg, "")
            rows.append(Row(
                "kernel-headline",
                "matmul",
                f"{role}-{mode}-best",
                kernel_cmd("matmul", mode, params, tag=tag, machine=machine, env=env),
                mode,
                machine,
                "kernel-run",
                paper_table=True,
                comparison="best-cache-vs-best-spm",
                estimated_minutes=20,
                metadata={"role": role, "shape": record["shape"], "blocking": cfg},
            ))

    for mode, machine, params, env, tag in (
        (
            "cache",
            "Cache-base",
            {
                "M": 512, "N": 512, "CHECK_RESULT": 0,
                "SPM_ROW_BLOCK": 1, "SPM_ROW_GROUP_BLOCKS": 1,
                "SPM_INTERNAL_ROW_BLOCK": 0,
            },
            None,
            "512x512/paper-decoder-ln-cache",
        ),
        (
            "spm",
            "TriSPM",
            {
                "M": 512, "N": 512, "CHECK_RESULT": 0,
                "SPM_ROW_BLOCK": 4, "SPM_ROW_GROUP_BLOCKS": 8,
                "SPM_INTERNAL_ROW_BLOCK": 1,
            },
            {"TRITON_SPM_PROMOTION_REPORT": 1},
            "512x512/paper-decoder-ln-spm-rb4-rg8",
        ),
    ):
        rows.append(Row(
            "kernel-headline",
            "layer_norm",
            f"decoder-ln-{mode}-best",
            kernel_cmd("layer_norm", mode, params, tag=tag, machine=machine, env=env),
            mode,
            machine,
            "kernel-run",
            paper_table=True,
            comparison="best-cache-vs-best-spm",
            estimated_minutes=8,
            metadata={"shape": {"M": 512, "N": 512}, "params": params},
        ))

    for mode, machine, params, env, tag in (
        (
            "cache",
            "Cache-base",
            {
                "M": 512, "N": 512, "BLOCK_N": 512, "CHECK_RESULT": 0,
                "SPM_ROW_BLOCK": 1, "SPM_ROW_GROUP_BLOCKS": 1,
                "SPM_INTERNAL_ROW_BLOCK": 0,
            },
            None,
            "512x512/paper-decoder-attn-cache-bn512",
        ),
        (
            "spm",
            "TriSPM",
            {
                "M": 512, "N": 512, "BLOCK_N": 32, "CHECK_RESULT": 0,
                "SPM_ROW_BLOCK": 2, "SPM_ROW_GROUP_BLOCKS": 8,
                "SPM_INTERNAL_ROW_BLOCK": 1,
            },
            {"TRITON_SPM_SOFTMAX_CACHE_EXP": 1, "TRITON_SPM_PROMOTION_REPORT": 1},
            "512x512/paper-decoder-attn-spm-bn32-rb2-rg8-exp1",
        ),
    ):
        rows.append(Row(
            "kernel-headline",
            "softmax",
            f"decoder-attn-{mode}-best",
            kernel_cmd("softmax", mode, params, tag=tag, machine=machine, env=env),
            mode,
            machine,
            "kernel-run",
            paper_table=True,
            comparison="best-cache-vs-best-spm",
            estimated_minutes=8,
            metadata={"shape": {"M": 512, "N": 512}, "params": params},
        ))
    return rows


def graph_headline_rows() -> list[Row]:
    spm_tag = "paper-decoder-best"
    rows = [
        Row(
            "graph-headline",
            "decoder_canonical_mh8",
            "large-TriSPM-vs-Cache-base",
            graph_cmd("decoder_canonical_mh8", preset="large", machine="TriSPM", spm_tag=spm_tag),
            "compare",
            "TriSPM",
            "graph-compare",
            paper_table=True,
            comparison="TriSPM-vs-Cache-base",
            estimated_minutes=80,
            metadata={"graph": "decoder_canonical_mh8", "preset": "large", "spm_tag": spm_tag},
        ),
        Row(
            "cache-capacity-fairness",
            "decoder_canonical_mh8",
            "large-TriSPM-vs-Cache-capacity",
            graph_cmd("decoder_canonical_mh8", preset="large", machine="Cache-capacity", spm_tag=spm_tag, skip_build=True),
            "compare",
            "Cache-capacity",
            "graph-cache-capacity",
            paper_table=True,
            comparison="TriSPM-vs-Cache-capacity",
            estimated_minutes=80,
            metadata={"graph": "decoder_canonical_mh8", "preset": "large", "spm_tag": spm_tag},
        ),
        Row(
            "cache-capacity-fairness",
            "decoder_canonical_mh8",
            "large-TriSPM-vs-Cache-stress",
            graph_cmd("decoder_canonical_mh8", preset="large", machine="Cache-stress", spm_tag=spm_tag, skip_build=True),
            "compare",
            "Cache-stress",
            "graph-cache-capacity",
            paper_table=True,
            comparison="TriSPM-vs-Cache-stress",
            estimated_minutes=80,
            metadata={"graph": "decoder_canonical_mh8", "preset": "large", "spm_tag": spm_tag},
        ),
    ]
    return rows


def graph_scale_rows() -> list[Row]:
    configs = (
        (
            "decoder_canonical_small_mh4",
            "small-TriSPM-vs-Cache-base",
            "paper-decoder-small",
            45,
            {
                "scale": "small",
                "SEQ": 256,
                "HEADS": 4,
                "D_MODEL": 256,
                "HEAD_DIM": 64,
                "FFN_DIM": 1024,
            },
        ),
        (
            "decoder_canonical_large_mh16",
            "large-TriSPM-vs-Cache-base",
            "paper-decoder-large",
            240,
            {
                "scale": "large",
                "SEQ": 1024,
                "HEADS": 16,
                "D_MODEL": 1024,
                "HEAD_DIM": 64,
                "FFN_DIM": 4096,
            },
        ),
    )
    return [
        Row(
            "graph-scale",
            graph,
            label,
            graph_cmd(graph, preset="large", machine="TriSPM", spm_tag=spm_tag),
            "compare",
            "TriSPM",
            "graph-compare",
            paper_table=True,
            comparison="TriSPM-vs-Cache-base",
            estimated_minutes=minutes,
            metadata={"graph": graph, "preset": "large", "spm_tag": spm_tag, **metadata},
        )
        for graph, label, spm_tag, minutes, metadata in configs
    ]


def graph_hw_sensitivity_rows() -> list[Row]:
    graph = "decoder_canonical_mh8"
    preset = "large"
    rows: list[Row] = []
    variants = (
        ("default", "Default", "paper-hw-default", [], False),
        ("spm-lat-6ns", "SPM latency 6ns", "paper-hw-spm-lat-6ns", ["--spm_lat", "6ns"], True),
        ("spm-bw-16gib", "SPM bandwidth 16GiB/s", "paper-hw-spm-bw-16gib", ["--spm_bw", "16GiB/s"], True),
        ("spm-banks-4", "SPM banks 4", "paper-hw-spm-banks-4", ["--spm_num_banks", "4"], True),
        (
            "dma-ctrl-4x",
            "DMA control path 4x",
            "paper-hw-dma-ctrl-4x",
            ["--dma_pio_lat", "20ns", "--dma_desc_lat", "40ns"],
            True,
        ),
    )
    for variant, label, spm_tag, gem5_flags, skip_build in variants:
        rows.append(Row(
            "graph-hw-sensitivity",
            graph,
            label,
            graph_spm_run_cmd(
                graph,
                preset=preset,
                spm_tag=spm_tag,
                skip_build=skip_build,
                gem5_flags=gem5_flags,
            ),
            "spm",
            "TriSPM",
            "graph-spm-run",
            paper_table=True,
            comparison="graph-hw-sensitivity",
            estimated_minutes=80,
            metadata={
                "graph": graph,
                "preset": preset,
                "variant": variant,
                "spm_tag": spm_tag,
                "baseline_spm_tag": "paper-hw-default",
                "gem5_flags": gem5_flags,
            },
        ))
    return rows


def graph_profile_rows() -> list[Row]:
    graph = "decoder_canonical_mh8"
    return [
        Row(
            "graph-profile",
            graph,
            "large-profile-TriSPM-vs-Cache-base",
            graph_cmd(
                graph,
                preset="large_profile",
                machine="TriSPM",
                spm_tag="paper-profile",
                artifact_tag="paper-profile",
            ),
            "compare",
            "TriSPM",
            "graph-profile",
            paper_table=False,
            comparison="per-kernel-attribution",
            estimated_minutes=120,
            metadata={
                "graph": graph,
                "preset": "large_profile",
                "spm_tag": "paper-profile",
                "artifact_tag": "paper-profile",
                "postprocess": "summarize_kernel_stats.py",
            },
        )
    ]


def softmax_fairness_rows() -> list[Row]:
    common = {"M": 512, "N": 512, "CHECK_RESULT": 0}
    rows: list[Row] = []
    rows.append(Row(
        "softmax-fairness",
        "softmax",
        "canonical-cache-full-row",
        kernel_cmd(
            "softmax",
            "cache",
            {**common, "BLOCK_N": 512, "SPM_ROW_BLOCK": 1, "SPM_ROW_GROUP_BLOCKS": 1, "SPM_INTERNAL_ROW_BLOCK": 0},
            tag="512x512/paper-softmax-canonical-cache-bn512",
            machine="Cache-base",
        ),
        "cache",
        "Cache-base",
        "algorithm-fairness",
        paper_table=True,
        metadata={"algorithm": "canonical", "exp_cache": None},
    ))
    for exp_cache in (0, 1):
        rows.append(Row(
            "softmax-fairness",
            "softmax",
            f"canonical-spm-exp-cache-{exp_cache}",
            kernel_cmd(
                "softmax",
                "spm",
                {**common, "BLOCK_N": 32, "SPM_ROW_BLOCK": 2, "SPM_ROW_GROUP_BLOCKS": 8, "SPM_INTERNAL_ROW_BLOCK": 1},
                env={"TRITON_SPM_SOFTMAX_CACHE_EXP": exp_cache, "TRITON_SPM_PROMOTION_REPORT": 1},
                tag=f"512x512/paper-softmax-canonical-spm-bn32-rb2-rg8-exp{exp_cache}",
                machine="TriSPM",
            ),
            "spm",
            "TriSPM",
            "algorithm-fairness",
            paper_table=True,
            metadata={"algorithm": "canonical", "exp_cache": exp_cache},
        ))
    rows.append(Row(
        "softmax-fairness",
        "softmax_online",
        "online-cache-bn32",
        kernel_cmd(
            "softmax_online",
            "cache",
            {**common, "BLOCK_N": 32, "CAUSAL": 0},
            tag="512x512/paper-softmax-online-cache-bn32",
            machine="Cache-base",
        ),
        "cache",
        "Cache-base",
        "algorithm-fairness",
        paper_table=True,
        metadata={"algorithm": "online"},
    ))
    rows.append(Row(
        "softmax-fairness",
        "softmax_online",
        "online-spm-bn32",
        kernel_cmd(
            "softmax_online",
            "spm",
            {**common, "BLOCK_N": 32, "CAUSAL": 0},
            env={"TRITON_SPM_PROMOTION_REPORT": 1},
            tag="512x512/paper-softmax-online-spm-bn32",
            machine="TriSPM",
        ),
        "spm",
        "TriSPM",
        "algorithm-fairness",
        paper_table=True,
        metadata={
            "algorithm": "online",
            "spm_reductions": 1,
            "affine_tile_candidates": "reported_in_promotion_sidecar",
        },
    ))
    return rows


def attention_fairness_rows() -> list[Row]:
    rows: list[Row] = []
    canonical_tag = "paper-canonical-attention-s512h64-c1"
    rows.append(Row(
        "attention-algorithm-fairness",
        "canonical_attention",
        "canonical-graph-TriSPM-vs-Cache-base",
        graph_cmd("canonical_attention", preset="s512h64-c1", machine="TriSPM", spm_tag=canonical_tag),
        "compare",
        "TriSPM",
        "graph-compare",
        paper_table=True,
        comparison="canonical-cache-vs-canonical-spm",
        estimated_minutes=60,
        metadata={"graph": "canonical_attention", "preset": "s512h64-c1"},
    ))
    flash_common = {
        "BATCH": 1,
        "HEADS": 1,
        "SEQ": 512,
        "HEAD_DIM": 64,
        "BLOCK_M": 16,
        "BLOCK_N": 16,
        "CAUSAL": 1,
        "CHECK_RESULT": 0,
        "WARMUP_ITERS": 2,
        "MEASURE_ITERS": 5,
        "FLUSH_BEFORE_ROI": 0,
    }
    rows.append(Row(
        "attention-algorithm-fairness",
        "flash_attention",
        "flash-kernel-cache-s512h64",
        kernel_cmd(
            "flash_attention",
            "cache",
            flash_common,
            tag="s512_h64/paper-flash-cache-bm16-bn16",
            machine="Cache-base",
        ),
        "cache",
        "Cache-base",
        "kernel-run",
        paper_table=True,
        comparison="flash-cache",
        estimated_minutes=60,
        metadata={"algorithm": "flash_attention", "shape": {"SEQ": 512, "HEAD_DIM": 64}},
    ))
    rows.append(Row(
        "attention-algorithm-fairness",
        "flash_attention",
        "flash-kernel-spm-headkv",
        kernel_cmd(
            "flash_attention",
            "spm",
            flash_common,
            preset="s512h64-c1",
            tag="s512_h64/paper-flash-spm-headkv-bm16-bn16",
            machine="TriSPM",
        ),
        "spm",
        "TriSPM",
        "kernel-run",
        paper_table=True,
        comparison="flash-spm-headkv",
        estimated_minutes=60,
        metadata={
            "algorithm": "flash_attention",
            "shape": {"SEQ": 512, "HEAD_DIM": 64},
            "spm_policy": "s512h64-c1",
            "affine_tile_candidates": "reported_in_promotion_sidecar",
        },
    ))
    return rows


def generic_affine_fallback_rows() -> list[Row]:
    rows: list[Row] = []
    for role, record in MATMUL_BEST.items():
        cfg = dict(record["spm"])
        params = {**record["shape"], **cfg, "CHECK_RESULT": 0}
        tag = matmul_tag(role, "generic-fallback", cfg, "")
        rows.append(Row(
            "generic-affine-fallback",
            "matmul",
            f"{role}-generic-fallback-build",
            kernel_cmd(
                "matmul",
                "build",
                params,
                tag=tag,
                machine="TriSPM",
                env={**SPM_MATMUL_ENV, **GENERIC_AFFINE_FALLBACK_ENV},
            ),
            "build",
            "TriSPM",
            "generic-fallback-build",
            paper_table=False,
            estimated_minutes=1,
            metadata={
                "source_table": "Table 3",
                "role": role,
                "shape": record["shape"],
                "blocking": cfg,
                "spm_policy": "generic-affine-fallback",
            },
        ))

    for params, tag, label, source_table in (
        (
            {
                "M": 512, "N": 512, "CHECK_RESULT": 0,
                "SPM_ROW_BLOCK": 1, "SPM_ROW_GROUP_BLOCKS": 1,
                "SPM_INTERNAL_ROW_BLOCK": 0,
            },
            "512x512/paper-decoder-ln-generic-fallback-rb1-rg1",
            "decoder-ln-generic-fallback-build",
            "Table 3",
        ),
        (
            {
                "M": 512, "N": 512, "BLOCK_N": 32, "CHECK_RESULT": 0,
                "SPM_ROW_BLOCK": 1, "SPM_ROW_GROUP_BLOCKS": 1,
                "SPM_INTERNAL_ROW_BLOCK": 0,
            },
            "512x512/paper-decoder-attn-softmax-generic-fallback-bn32-rb1-rg1",
            "decoder-attn-softmax-generic-fallback-build",
            "Table 3/Table 5",
        ),
    ):
        rows.append(Row(
            "generic-affine-fallback",
            "layer_norm" if "ln" in label else "softmax",
            label,
            kernel_cmd(
                "layer_norm" if "ln" in label else "softmax",
                "build",
                params,
                tag=tag,
                machine="TriSPM",
                env=GENERIC_AFFINE_FALLBACK_ENV,
            ),
            "build",
            "TriSPM",
            "generic-fallback-build",
            paper_table=False,
            estimated_minutes=1,
            metadata={
                "source_table": source_table,
                "shape": {"M": 512, "N": 512},
                "params": params,
                "spm_policy": "generic-affine-fallback",
            },
        ))

    common = {"M": 512, "N": 512, "CHECK_RESULT": 0}
    rows.append(Row(
        "generic-affine-fallback",
        "softmax",
        "canonical-softmax-generic-fallback-build",
        kernel_cmd(
            "softmax",
            "build",
            {**common, "BLOCK_N": 32, "SPM_ROW_BLOCK": 1,
             "SPM_ROW_GROUP_BLOCKS": 1, "SPM_INTERNAL_ROW_BLOCK": 0},
            tag="512x512/paper-softmax-canonical-generic-fallback-bn32-rb1-rg1",
            machine="TriSPM",
            env=GENERIC_AFFINE_FALLBACK_ENV,
        ),
        "build",
        "TriSPM",
        "generic-fallback-build",
        paper_table=False,
        estimated_minutes=1,
        metadata={
            "source_table": "Table 5",
            "algorithm": "canonical",
            "spm_policy": "generic-affine-fallback",
        },
    ))
    rows.append(Row(
        "generic-affine-fallback",
        "softmax_online",
        "online-softmax-generic-fallback-build",
        kernel_cmd(
            "softmax_online",
            "build",
            {**common, "BLOCK_N": 32, "CAUSAL": 0},
            tag="512x512/paper-softmax-online-generic-fallback-bn32",
            machine="TriSPM",
            env=GENERIC_AFFINE_FALLBACK_ENV,
        ),
        "build",
        "TriSPM",
        "generic-fallback-build",
        paper_table=False,
        estimated_minutes=1,
        metadata={
            "source_table": "Table 5",
            "algorithm": "online",
            "spm_policy": "generic-affine-fallback",
        },
    ))

    flash_common = {
        "BATCH": 1,
        "HEADS": 1,
        "SEQ": 256,
        "HEAD_DIM": 64,
        "BLOCK_M": 16,
        "BLOCK_N": 16,
        "CAUSAL": 1,
        "CHECK_RESULT": 0,
        "WARMUP_ITERS": 2,
        "MEASURE_ITERS": 5,
        "FLUSH_BEFORE_ROI": 0,
    }
    rows.append(Row(
        "generic-affine-fallback",
        "flash_attention",
        "flash-attention-generic-fallback-build",
        kernel_cmd(
            "flash_attention",
            "build",
            flash_common,
            tag="s256_h64/paper-flash-generic-fallback-bm16-bn16",
            machine="TriSPM",
            env=GENERIC_AFFINE_FALLBACK_ENV,
        ),
        "build",
        "TriSPM",
        "generic-fallback-build",
        paper_table=False,
        estimated_minutes=1,
        metadata={
            "source_table": "Table 6",
            "algorithm": "flash_attention",
            "shape": {"SEQ": 256, "HEAD_DIM": 64},
            "spm_policy": "generic-affine-fallback",
        },
    ))
    return rows


def generic_affine_fallback_perf_rows() -> list[Row]:
    rows: list[Row] = []
    for role, record in MATMUL_BEST.items():
        cfg = dict(record["spm"])
        params = {**record["shape"], **cfg, "CHECK_RESULT": 0}
        generic_tag = matmul_tag(role, "generic-fallback-perf", cfg, "")
        cache_tag = capacity_tag(
            matmul_tag(role, "cache", dict(record["cache"]), ""),
            "32KiB",
            "512KiB",
        )
        tuned_tag = matmul_tag(role, "spm", cfg, "")
        rows.append(Row(
            "generic-affine-fallback-perf",
            "matmul",
            f"{role}-generic-fallback-perf",
            kernel_cmd(
                "matmul",
                "spm",
                params,
                tag=generic_tag,
                machine="TriSPM",
                env={**SPM_MATMUL_ENV, **GENERIC_AFFINE_FALLBACK_ENV},
            ),
            "spm",
            "TriSPM",
            "generic-fallback-perf",
            paper_table=False,
            comparison="cache-vs-tuned-vs-generic-fallback",
            estimated_minutes=20,
            metadata={
                "source_table": "Table 3",
                "role": role,
                "shape": record["shape"],
                "blocking": cfg,
                "cache_tag": cache_tag,
                "tuned_spm_tag": tuned_tag,
                "generic_spm_tag": generic_tag,
                "measure_iters": 1,
                "spm_policy": "generic-affine-fallback",
            },
        ))

    for params, generic_tag, label, source_table, cache_tag, tuned_tag in (
        (
            {
                "M": 512, "N": 512, "CHECK_RESULT": 0,
                "SPM_ROW_BLOCK": 1, "SPM_ROW_GROUP_BLOCKS": 1,
                "SPM_INTERNAL_ROW_BLOCK": 0,
            },
            "512x512/paper-decoder-ln-generic-fallback-perf-rb1-rg1",
            "decoder-ln-generic-fallback-perf",
            "Table 3",
            capacity_tag("512x512/paper-decoder-ln-cache", "32KiB", "512KiB"),
            "512x512/paper-decoder-ln-spm-rb4-rg8",
        ),
        (
            {
                "M": 512, "N": 512, "BLOCK_N": 32, "CHECK_RESULT": 0,
                "SPM_ROW_BLOCK": 1, "SPM_ROW_GROUP_BLOCKS": 1,
                "SPM_INTERNAL_ROW_BLOCK": 0,
            },
            "512x512/paper-decoder-attn-softmax-generic-fallback-perf-bn32-rb1-rg1",
            "decoder-attn-softmax-generic-fallback-perf",
            "Table 3/Table 5",
            capacity_tag(
                "512x512/paper-decoder-attn-cache-bn512",
                "32KiB",
                "512KiB",
            ),
            "512x512/paper-decoder-attn-spm-bn32-rb2-rg8-exp1",
        ),
    ):
        workload = "layer_norm" if "ln" in label else "softmax"
        rows.append(Row(
            "generic-affine-fallback-perf",
            workload,
            label,
            kernel_cmd(
                workload,
                "spm",
                params,
                tag=generic_tag,
                machine="TriSPM",
                env=GENERIC_AFFINE_FALLBACK_ENV,
            ),
            "spm",
            "TriSPM",
            "generic-fallback-perf",
            paper_table=False,
            comparison="cache-vs-tuned-vs-generic-fallback",
            estimated_minutes=8,
            metadata={
                "source_table": source_table,
                "shape": {"M": 512, "N": 512},
                "params": params,
                "cache_tag": cache_tag,
                "tuned_spm_tag": tuned_tag,
                "generic_spm_tag": generic_tag,
                "measure_iters": 1,
                "spm_policy": "generic-affine-fallback",
            },
        ))

    common = {"M": 512, "N": 512, "CHECK_RESULT": 0}
    rows.append(Row(
        "generic-affine-fallback-perf",
        "softmax",
        "canonical-softmax-generic-fallback-perf",
        kernel_cmd(
            "softmax",
            "spm",
            {**common, "BLOCK_N": 32, "SPM_ROW_BLOCK": 1,
             "SPM_ROW_GROUP_BLOCKS": 1, "SPM_INTERNAL_ROW_BLOCK": 0},
            tag="512x512/paper-softmax-canonical-generic-fallback-perf-bn32-rb1-rg1",
            machine="TriSPM",
            env=GENERIC_AFFINE_FALLBACK_ENV,
        ),
        "spm",
        "TriSPM",
        "generic-fallback-perf",
        paper_table=False,
        comparison="cache-vs-tuned-vs-generic-fallback",
        estimated_minutes=8,
        metadata={
            "source_table": "Table 5",
            "algorithm": "canonical",
            "cache_tag": capacity_tag(
                "512x512/paper-softmax-canonical-cache-bn512",
                "32KiB",
                "512KiB",
            ),
            "tuned_spm_tag": "512x512/paper-softmax-canonical-spm-bn32-rb2-rg8-exp1",
            "generic_spm_tag": (
                "512x512/paper-softmax-canonical-generic-fallback-perf-bn32-rb1-rg1"
            ),
            "measure_iters": 1,
            "spm_policy": "generic-affine-fallback",
        },
    ))
    rows.append(Row(
        "generic-affine-fallback-perf",
        "softmax_online",
        "online-softmax-generic-fallback-perf",
        kernel_cmd(
            "softmax_online",
            "spm",
            {**common, "BLOCK_N": 32, "CAUSAL": 0},
            tag="512x512/paper-softmax-online-generic-fallback-perf-bn32",
            machine="TriSPM",
            env=GENERIC_AFFINE_FALLBACK_ENV,
        ),
        "spm",
        "TriSPM",
        "generic-fallback-perf",
        paper_table=False,
        comparison="cache-vs-tuned-vs-generic-fallback",
        estimated_minutes=8,
        metadata={
            "source_table": "Table 5",
            "algorithm": "online",
            "cache_tag": capacity_tag(
                "512x512/paper-softmax-online-cache-bn32",
                "32KiB",
                "512KiB",
            ),
            "tuned_spm_tag": "512x512/paper-softmax-online-spm-bn32",
            "generic_spm_tag": "512x512/paper-softmax-online-generic-fallback-perf-bn32",
            "measure_iters": 1,
            "spm_policy": "generic-affine-fallback",
        },
    ))

    flash_common = {
        "BATCH": 1,
        "HEADS": 1,
        "SEQ": 256,
        "HEAD_DIM": 64,
        "BLOCK_M": 16,
        "BLOCK_N": 16,
        "CAUSAL": 1,
        "CHECK_RESULT": 0,
        "WARMUP_ITERS": 2,
        "MEASURE_ITERS": 5,
        "FLUSH_BEFORE_ROI": 0,
    }
    rows.append(Row(
        "generic-affine-fallback-perf",
        "flash_attention",
        "flash-attention-generic-fallback-perf",
        kernel_cmd(
            "flash_attention",
            "spm",
            flash_common,
            tag="s256_h64/paper-flash-generic-fallback-perf-bm16-bn16",
            machine="TriSPM",
            env=GENERIC_AFFINE_FALLBACK_ENV,
        ),
        "spm",
        "TriSPM",
        "generic-fallback-perf",
        paper_table=False,
        comparison="cache-vs-tuned-vs-generic-fallback",
        estimated_minutes=60,
        metadata={
            "source_table": "Table 6",
            "algorithm": "flash_attention",
            "shape": {"SEQ": 256, "HEAD_DIM": 64},
            "cache_tag": capacity_tag(
                "s256_h64/paper-flash-cache-bm16-bn16-tail-parallel",
                "32KiB",
                "512KiB",
            ),
            "tuned_spm_tag": "s256_h64/paper-flash-spm-headkv-bm16-bn16-tail-parallel",
            "generic_spm_tag": "s256_h64/paper-flash-generic-fallback-perf-bm16-bn16",
            "measure_iters": 5,
            "spm_policy": "generic-affine-fallback",
        },
    ))
    return rows


def gemm_mechanism_rows() -> list[Row]:
    role = "ffn-down"
    shape = MATMUL_BEST[role]["shape"]
    rows: list[Row] = []
    for micro_m in (4, 8, 16):
        for window_k in (2, 4, 8):
            cfg = MATMUL_BEST[role]["spm"]
            tag = (
                f"{shape['M']}x{shape['N']}x{shape['K']}/"
                f"paper-mech-{role}-32x32x32-gsm1-uM{micro_m}-wK{window_k}"
            )
            rows.append(Row(
                "gemm-tuning-mechanism",
                "matmul",
                f"{role}-uM{micro_m}-wK{window_k}",
                kernel_cmd(
                    "matmul",
                    "spm",
                    {**shape, **cfg, "CHECK_RESULT": 0},
                    env={
                        "TRITON_MICRO_M": micro_m,
                        "TRITON_SPM_WINDOW_K": window_k,
                        "TRITON_SPM_PROMOTION_REPORT": 1,
                    },
                    tag=tag,
                    machine="TriSPM",
                ),
                "spm",
                "TriSPM",
                "mechanism-sweep",
                paper_table=True,
                estimated_minutes=20,
                metadata={"role": role, "micro_m": micro_m, "window_k": window_k},
            ))
    return rows


def split_rows() -> list[Row]:
    role = "ffn-down"
    shape = MATMUL_BEST[role]["shape"]
    cfg = MATMUL_BEST[role]["spm"]
    return [
        Row(
            "split",
            "matmul",
            f"{role}-no-split",
            kernel_cmd(
                "matmul",
                "spm",
                {**shape, **cfg, "CHECK_RESULT": 0},
                env=NO_SPLIT_MATMUL_ENV,
                tag=(
                    f"{shape['M']}x{shape['N']}x{shape['K']}/"
                    f"paper-split-{role}-no-split-32x32x32-gsm1"
                ),
                machine="TriSPM",
            ),
            "spm",
            "TriSPM",
            "mechanism-ablation",
            paper_table=True,
            estimated_minutes=20,
            metadata={
                "role": role,
                "split": "off",
                "micro_m": 32,
                "window_k": 1,
                "baseline_split_row": "gemm-tuning-mechanism ffn-down-uM8-wK4",
            },
        )
    ]


def xspm_instruction_rows() -> list[Row]:
    tag = "paper-decoder-xspm1"
    return [
        Row(
            "xspm-instruction",
            "decoder_canonical_mh8",
            "large-graph-xspm1-spm-only",
            graph_spm_run_cmd("decoder_canonical_mh8", preset="large", spm_tag=tag),
            "spm",
            "TriSPM",
            "graph-spm-run",
            paper_table=True,
            comparison="xspm1-vs-graph-headline-mmio",
            estimated_minutes=80,
            env={"TRITON_USE_XSPM_INSN": "1"},
            metadata={
                "graph": "decoder_canonical_mh8",
                "preset": "large",
                "spm_tag": tag,
                "baseline_spm_tag": "paper-decoder-best",
                "TRITON_USE_XSPM_INSN": 1,
                "cache_baseline": "reuse graph-headline cache/default",
            },
        )
    ]


def build_rows() -> list[Row]:
    rows: list[Row] = []
    rows.extend(kernel_headline_rows())
    rows.extend(graph_headline_rows())
    rows.extend(graph_scale_rows())
    rows.extend(graph_hw_sensitivity_rows())
    rows.extend(graph_profile_rows())
    rows.extend(softmax_fairness_rows())
    rows.extend(attention_fairness_rows())
    rows.extend(generic_affine_fallback_rows())
    rows.extend(generic_affine_fallback_perf_rows())
    rows.extend(gemm_mechanism_rows())
    rows.extend(split_rows())
    rows.extend(xspm_instruction_rows())
    return rows


def apply_artifact_suffix(rows: list[Row], suffix: str) -> list[Row]:
    if not suffix:
        return rows

    updated: list[Row] = []
    for row in rows:
        command = list(row.command)
        metadata = dict(row.metadata)

        if command and command[1].endswith("run_experiment.py"):
            if "--tag" in command:
                index = command.index("--tag") + 1
                old_tag = command[index]
                command[index] = suffix_tag(command[index], suffix)
                if metadata.get("generic_spm_tag") == old_tag:
                    metadata["generic_spm_tag"] = command[index]
        elif command and command[1].endswith("graph_eval.py"):
            row_suffix = f"{suffix}-{row.phase}-{row.workload}-{row.label}"
            row_suffix = row_suffix.replace("/", "-")
            if "--skip-build" in command:
                command.remove("--skip-build")
            if row.phase == "cache-capacity-fairness":
                command.append("--full-compare")
            command += ["--artifact-tag", row_suffix]
            if "--spm-tag" in command:
                index = command.index("--spm-tag") + 1
                command[index] = suffix_graph_tag(command[index], suffix)
                metadata["spm_tag"] = command[index]
            metadata["artifact_tag"] = row_suffix
        elif command and command[1].endswith("graph_placement.py"):
            if row.phase == "graph-hw-sensitivity":
                row_suffix = f"{suffix}-{row.phase}-{row.workload}"
            else:
                row_suffix = f"{suffix}-{row.phase}-{row.workload}-{row.label}"
            row_suffix = row_suffix.replace("/", "-")
            command += ["--artifact-tag", row_suffix]
            if "--spm-tag" in command:
                index = command.index("--spm-tag") + 1
                command[index] = suffix_graph_tag(command[index], suffix)
                metadata["spm_tag"] = command[index]
            if "baseline_spm_tag" in metadata:
                metadata["baseline_spm_tag"] = suffix_graph_tag(
                    str(metadata["baseline_spm_tag"]), suffix)
            metadata["artifact_tag"] = row_suffix

        updated.append(Row(
            row.phase,
            row.workload,
            suffix_tag(row.label, suffix),
            command,
            row.mode,
            row.machine,
            row.row_kind,
            row.paper_table,
            row.comparison,
            row.estimated_minutes,
            dict(row.env),
            metadata,
        ))
    return updated


def filter_rows(
    rows: list[Row],
    phases: list[str],
    labels: list[str],
    from_phase: str | None = None,
) -> list[Row]:
    if from_phase:
        if from_phase not in PHASE_ORDER:
            sys.exit(f"ERROR: unknown phase for --from-phase: {from_phase}")
        phase_rank = {phase: index for index, phase in enumerate(PHASE_ORDER)}
        start = phase_rank[from_phase]
        rows = [
            row for row in rows
            if phase_rank.get(row.phase, len(PHASE_ORDER)) >= start
        ]
    if phases:
        allowed = set(phases)
        rows = [row for row in rows if row.phase in allowed]
    if labels:
        needles = tuple(labels)
        rows = [row for row in rows if any(needle in row.label for needle in needles)]
    return rows


def write_plan(rows: list[Row], campaign_dir: Path) -> None:
    campaign_dir.mkdir(parents=True, exist_ok=True)
    records = [row.as_record() for row in rows]
    phases: dict[str, int] = {}
    for row in rows:
        phases[row.phase] = phases.get(row.phase, 0) + 1
    payload = {
        "metadata": {
            "generated_at": now_iso(),
            "row_count": len(rows),
            "phases": phases,
            "source": "workloads/scripts/paper_experiments.py",
        },
        "rows": records,
    }
    (campaign_dir / "run_list.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (campaign_dir / "run_list.csv").open("w", newline="") as f:
        fieldnames = [
            "row_id", "phase", "workload", "label", "mode", "machine",
            "row_kind", "paper_table", "comparison", "estimated_minutes",
            "command_text",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})

    lines = [
        f"# Paper Experiment Campaign: {campaign_dir.name}",
        "",
        f"- total rows: {len(rows)}",
        "- default action: plan only; pass `--run` to execute",
        "",
        "## Phase Summary",
        "",
        "| Phase | Rows |",
        "|---|---:|",
    ]
    for phase, count in sorted(phases.items()):
        lines.append(f"| {phase} | {count} |")
    lines += ["", "## Commands", ""]
    for record in records:
        lines += [
            f"### {record['row_id']}",
            "",
            f"- phase: `{record['phase']}`",
            f"- workload: `{record['workload']}`",
            f"- label: `{record['label']}`",
            "",
            "```bash",
            record["command_text"],
            "```",
            "",
        ]
        if record.get("env"):
            env_text = " ".join(f"{k}={v}" for k, v in sorted(record["env"].items()))
            lines += [f"env: `{env_text}`", ""]
    (campaign_dir / "summary.md").write_text("\n".join(lines))


def load_status(path: Path) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text())
    return {}


def save_status(path: Path, status: dict[str, Any]) -> None:
    path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")


def run_row(row: Row, campaign_dir: Path) -> dict[str, Any]:
    logs_dir = campaign_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{row.row_id}.log"
    started = time.time()
    env = os.environ.copy()
    env.update(row.env)
    with log_path.open("w") as log:
        log.write(f"$ {shlex.join(row.command)}\n")
        if row.env:
            log.write(f"# env {json.dumps(row.env, sort_keys=True)}\n")
        log.flush()
        proc = subprocess.run(row.command, stdout=log, stderr=subprocess.STDOUT, env=env)
    finished = time.time()
    return {
        **row.as_record(),
        "status": "pass" if proc.returncode == 0 else "fail",
        "returncode": proc.returncode,
        "started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(timespec="seconds"),
        "finished_at": datetime.fromtimestamp(finished, timezone.utc).isoformat(timespec="seconds"),
        "elapsed_seconds": finished - started,
        "log_path": str(log_path.relative_to(WORKLOADS_DIR)),
    }


def run_batch(
    rows: list[Row],
    campaign_dir: Path,
    status_path: Path,
    status: dict[str, Any],
    jobs: int,
    force: bool,
) -> None:
    pending = [
        row for row in rows
        if force or status.get(row.row_id, {}).get("status") != "pass"
    ]
    if not pending:
        return

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures: dict[Future[dict[str, Any]], Row] = {}
        pending_iter = iter(pending)

        def submit_next() -> None:
            try:
                row = next(pending_iter)
            except StopIteration:
                return
            print(f"START {row.row_id} {row.label}", flush=True)
            futures[pool.submit(run_row, row, campaign_dir)] = row

        for _ in range(min(jobs, len(pending))):
            submit_next()

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                row = futures.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 - preserve row status
                    result = {
                        **row.as_record(),
                        "status": "fail",
                        "error": repr(exc),
                        "finished_at": now_iso(),
                    }
                status[row.row_id] = result
                save_status(status_path, status)
                print(f"{result['status'].upper()} {row.row_id} {row.label}", flush=True)
                submit_next()


def run_rows(
    rows: list[Row],
    campaign_dir: Path,
    jobs: int,
    force: bool,
    parallel_phases: bool,
) -> None:
    status_path = campaign_dir / "status.json"
    status = load_status(status_path)
    pending = [row for row in rows if force or status.get(row.row_id, {}).get("status") != "pass"]
    if not pending:
        print("No pending rows.")
        return

    phase_rank = {phase: index for index, phase in enumerate(PHASE_ORDER)}
    if parallel_phases:
        ordered = sorted(
            rows,
            key=lambda row: phase_rank.get(row.phase, len(PHASE_ORDER)),
        )
        print(f"PHASES parallel jobs={jobs}", flush=True)
        run_batch(ordered, campaign_dir, status_path, status, jobs, force)
        return

    phases = sorted({row.phase for row in rows}, key=lambda phase: phase_rank.get(phase, len(PHASE_ORDER)))
    for phase in phases:
        batch = [row for row in rows if row.phase == phase]
        phase_jobs = 1 if phase in SERIAL_PHASES else jobs
        print(f"PHASE {phase} jobs={phase_jobs}", flush=True)
        run_batch(batch, campaign_dir, status_path, status, phase_jobs, force)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="paper-experiments")
    parser.add_argument("--phase", action="append", default=[], help="phase to include; repeatable")
    parser.add_argument(
        "--from-phase",
        choices=PHASE_ORDER,
        default=None,
        help="include this phase and every later phase in paper order",
    )
    parser.add_argument("--label", action="append", default=[], help="substring filter on labels; repeatable")
    parser.add_argument(
        "--artifact-suffix",
        default="",
        help="suffix kernel tags and graph artifact directories to avoid overlapping another campaign",
    )
    parser.add_argument("--run", action="store_true", help="execute pending rows after writing the plan")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--parallel-phases",
        action="store_true",
        help="run all selected pending rows in one parallel batch instead of phase order",
    )
    parser.add_argument("--force", action="store_true", help="rerun rows even if status is pass")
    args = parser.parse_args()

    rows = filter_rows(build_rows(), args.phase, args.label, args.from_phase)
    rows = apply_artifact_suffix(rows, args.artifact_suffix)
    campaign_dir = CAMPAIGN_ROOT / args.campaign
    write_plan(rows, campaign_dir)
    print(f"Plan: {campaign_dir.relative_to(WORKLOADS_DIR)}/run_list.json")
    print(f"Summary: {campaign_dir.relative_to(WORKLOADS_DIR)}/summary.md")
    if args.run:
        run_rows(rows, campaign_dir, max(1, args.jobs), args.force, args.parallel_phases)


if __name__ == "__main__":
    main()
