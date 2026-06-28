#!/usr/bin/env python3
"""TriSPM experiment driver: build → run gem5 → compare.

Reads a kernel's experiment.toml manifest, exports its params as env vars
for the build, and orchestrates spm/cache/compare runs in a single command.

Usage:
  run_experiment.py <kernel> --mode spm
  run_experiment.py <kernel> --mode cache-search --sweep blocking
  run_experiment.py <kernel> --mode spm-compare [--preset spm-candidate]
  run_experiment.py <kernel> --mode verify
  run_experiment.py <kernel> --sweep blocking [--preset steady]

Modes:
  spm       build + run with SPM
  cache     build + run cache-baseline
  cache-search  run cache candidates and write per-shape cache_best.json
  spm-compare   compare one SPM candidate against cache_best.json
  verify    build both and check LLIR for SPM markers
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
from pathlib import Path

import trispm_paths
from trispm_paths import WORKLOADS_DIR

SCRIPTS_DIR = Path(__file__).resolve().parent


def load_manifest(kernel: str) -> dict:
    path = WORKLOADS_DIR / "kernels" / kernel / "experiment.toml"
    if not path.is_file():
        sys.exit(f"ERROR: manifest not found: {path}")
    return tomllib.loads(path.read_text())


def merged_params(
    manifest: dict,
    preset: str | None,
    overrides: dict[str, str],
    mode: str | None = None,
) -> dict[str, str]:
    params: dict = dict(manifest.get("params", {}))
    if preset:
        preset_params = manifest.get("presets", {}).get(preset)
        if preset_params is None:
            sys.exit(f"ERROR: preset {preset!r} not in manifest")
        params.update(preset_params)
    if mode:
        params.update(manifest.get("mode_params", {}).get(mode, {}))
    params.update(overrides)
    return {k: str(v) for k, v in params.items()}


def render_cflags(manifest: dict, params: dict[str, str]) -> str:
    """Render C preprocessor flags declared by the kernel manifest."""
    macros = manifest.get("build", {}).get("c_macros", [])
    try:
        return " ".join(f"-D{macro.format(**params)}" for macro in macros)
    except KeyError as e:
        sys.exit(f"ERROR: [build].c_macros references unknown param {e.args[0]!r}")


def export_env(manifest: dict, params: dict[str, str]) -> dict[str, str]:
    """Return env with manifest params and rendered C build flags exported."""
    prefix = manifest["kernel"].get("env_prefix", "")
    env = os.environ.copy()
    for k, v in params.items():
        env[f"{prefix}{k}"] = v
    env["KERNEL_CFLAGS"] = render_cflags(manifest, params)
    return env


def preset_env(manifest: dict, preset: str | None) -> dict[str, str]:
    if not preset:
        return {}
    return {k: str(v) for k, v in manifest.get("preset_env", {}).get(preset, {}).items()}


def run(cmd: list[str], env: dict[str, str] | None = None, echo: bool = True) -> None:
    if echo:
        print(f"$ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env)


def rel_workloads_path(path: Path) -> str:
    return str(path.relative_to(WORKLOADS_DIR))


def parse_bool_param(params: dict[str, str], name: str, default: bool) -> bool:
    value = params.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def do_build(kernel: str, mode: str, tag: str, env: dict[str, str]) -> None:
    run([str(SCRIPTS_DIR / "build_kernel.sh"), kernel, "--mode", mode, "--tag", tag], env=env)


def do_run(kernel: str, mode: str, tag: str, gem5_flags: list[str], env: dict[str, str]) -> None:
    cmd = [str(SCRIPTS_DIR / "run_gem5.sh"), kernel, "--mode", mode, "--tag", tag]
    if gem5_flags:
        cmd += ["--"] + gem5_flags
    run(cmd, env=env)


def cache_gem5_flags(
    cache_l1d_size: str | None,
    cache_l2_size: str | None,
) -> list[str]:
    flags = ["--cache_baseline"]
    if cache_l1d_size:
        flags += ["--l1d_size", cache_l1d_size]
    if cache_l2_size:
        flags += ["--l2_size", cache_l2_size]
    return flags


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


def validate_run_result(kernel: str, mode: str, tag: str, params: dict[str, str]) -> None:
    """Fail loudly if a checked workload did not report a clean PASS."""
    if not parse_bool_param(params, "CHECK_RESULT", default=True):
        return

    run_log = trispm_paths.run_log_path(kernel, mode, tag)
    if not run_log.is_file():
        sys.exit(
            f"ERROR: {kernel} {mode} result check was enabled, but run log is "
            f"missing: {rel_workloads_path(run_log)}"
        )

    text = run_log.read_text(errors="replace")
    has_pass = re.search(r"\bPASS:", text) is not None
    bad_lines = [
        line for line in text.splitlines()
        if re.search(r"\b(FAIL|MISMATCH|SKIP):", line)
    ]
    if has_pass and not bad_lines:
        print(f"Result gate passed: {kernel} {mode} ({rel_workloads_path(run_log)})")
        return

    detail = "\n".join(bad_lines[:12]) if bad_lines else "PASS line was not found"
    sys.exit(
        f"ERROR: {kernel} {mode} failed result gate for tag={tag!r}.\n"
        f"Log: {rel_workloads_path(run_log)}\n"
        f"{detail}\n"
        "No compare/artifact tables were generated for this run."
    )


def remove_compare_outputs(kernel: str, tag: str) -> None:
    for path in (
        trispm_paths.compare_path(kernel, tag),
        trispm_paths.spm_stats_path(kernel, tag),
        trispm_paths.artifact_stats_path(kernel, tag),
    ):
        if path.exists():
            path.unlink()


def cache_best_output_path(
    kernel: str,
    tag: str,
    cache_l1d_size: str | None = None,
    cache_l2_size: str | None = None,
) -> Path:
    if not cache_l1d_size and not cache_l2_size:
        return trispm_paths.cache_best_path(kernel, tag)
    shape_dir = trispm_paths.shape_dir(kernel, tag)
    return shape_dir / (
        f"cache_best_{cache_capacity_suffix(cache_l1d_size, cache_l2_size)}.json"
    )


def remove_cache_best_outputs(
    kernel: str,
    tags: list[str],
    cache_l1d_size: str | None = None,
    cache_l2_size: str | None = None,
) -> None:
    seen: set[Path] = set()
    for tag in tags:
        path = cache_best_output_path(
            kernel, tag, cache_l1d_size, cache_l2_size)
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            path.unlink()


def do_compare_stats(
    kernel: str,
    spm_tag: str,
    cache_tag: str,
    measure_iters: int,
    compare_out: Path | None = None,
    spm_only_out: Path | None = None,
) -> None:
    spm_stats = trispm_paths.roi_stats_path(kernel, "spm", spm_tag)
    cache_stats = trispm_paths.roi_stats_path(kernel, "cache", cache_tag)
    compare = compare_out or trispm_paths.compare_path(kernel, spm_tag)
    spm_only = spm_only_out or trispm_paths.spm_stats_path(kernel, spm_tag)
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "compare_stats.py"),
        "--spm", str(spm_stats),
        "--cache", str(cache_stats),
        "--measure-iters", str(measure_iters),
        "--output", str(compare),
        "--spm-only-output", str(spm_only),
        "--quiet",
    ]
    run(cmd, echo=False)
    print(f"Compare saved:  {rel_workloads_path(compare)}")
    print(f"SPM-only saved: {rel_workloads_path(spm_only)}")


def capacity_compare_path(
    kernel: str,
    tag: str,
    cache_l1d_size: str | None = None,
    cache_l2_size: str | None = None,
) -> Path:
    return trispm_paths.m5out_dir(kernel, "spm", tag) / (
        f"compare_vs_cache_{cache_capacity_suffix(cache_l1d_size, cache_l2_size)}.txt"
    )


ARTIFACT_PATTERNS = {
    "llir": "{kernel}.llir",
    "asm": "{kernel}.s",
    "launcher_c": "{kernel}_launcher.c",
    "promotion_json": "{kernel}_promotions.json",
}


def count_pattern(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text))


def artifact_rows(kernel: str, spm_tag: str, cache_tag: str | None = None) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    tag_by_mode = {
        "spm": spm_tag,
        "cache": cache_tag or spm_tag,
    }
    for mode in ("spm", "cache"):
        tag = tag_by_mode[mode]
        build_dir = trispm_paths.build_dir(kernel, mode, tag)
        for kind, template in ARTIFACT_PATTERNS.items():
            path = build_dir / template.format(kernel=kernel)
            if not path.is_file():
                continue
            text = path.read_text(errors="replace")
            line_count = 0 if not text else text.count("\n") + (0 if text.endswith("\n") else 1)
            rows.append({
                "mode": mode,
                "artifact": kind,
                "path": rel_workloads_path(path),
                "bytes": path.stat().st_size,
                "lines": line_count,
                "addrspace3": count_pattern(text, r"addrspace\(3\)"),
                "fence_iorw": count_pattern(text, r"fence iorw"),
                "spm_dma_wait": count_pattern(text, r"spm\.dma\.w|spm_dma_wait"),
                "spm_dma_enqueue": count_pattern(text, r"spm\.dma|spm_dma_"),
            })
    return rows


def write_artifact_stats(kernel: str, spm_tag: str, cache_tag: str | None = None) -> None:
    rows = artifact_rows(kernel, spm_tag, cache_tag)
    out = trispm_paths.artifact_stats_path(kernel, spm_tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "mode",
        "artifact",
        "path",
        "bytes",
        "lines",
        "addrspace3",
        "fence_iorw",
        "spm_dma_wait",
        "spm_dma_enqueue",
    ]
    widths = {
        key: max([len(key), *(len(str(row.get(key, ""))) for row in rows)])
        for key in headers
    }
    def fmt_row(row: dict[str, str | int] | dict[str, str]) -> str:
        return "  ".join(str(row.get(key, "")).ljust(widths[key]) for key in headers)
    sep = "  ".join("-" * widths[key] for key in headers)
    text_rows = [fmt_row({key: key for key in headers}), sep]
    text_rows += [fmt_row(row) for row in rows]
    out.write_text("\n".join(text_rows) + "\n")
    print(f"Artifact stats: {rel_workloads_path(out)}")


def do_verify(kernel: str, tag: str, manifest: dict) -> None:
    """Check SPM/cache LLIR markers against the manifest's expected policy."""
    spm_dir = trispm_paths.build_dir(kernel, "spm", tag)
    cache_dir = trispm_paths.build_dir(kernel, "cache", tag)
    expect_spm = bool(manifest["kernel"].get("expect_spm", True))
    verify_cfg = manifest["kernel"].get("verify", {})
    expect_dma = bool(verify_cfg.get(
        "expect_dma",
        expect_spm,
    ))
    expect_promotion_source = verify_cfg.get("expect_promotion_source")
    expect_promotion_reason = verify_cfg.get("expect_promotion_reason")
    expect_rejection_reason = verify_cfg.get("expect_rejection_reason")
    expect_rejection_source = verify_cfg.get("expect_rejection_source")
    expect_residency_plan = verify_cfg.get("expect_residency_plan")
    expect_affine_tile_reason = verify_cfg.get("expect_affine_tile_reason")
    expect_affine_tile_class = verify_cfg.get("expect_affine_tile_class")

    checks: list[tuple[str, bool]] = []
    all_ok = True

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal all_ok
        status = "PASS" if ok else "FAIL"
        suffix = f"  ({detail})" if detail else ""
        print(f"  [{status}] {name}{suffix}")
        checks.append((name, ok))
        if not ok:
            all_ok = False

    print(
        f"\n===== verify-spm-policy: {kernel} "
        f"(tag={tag}, expect_spm={expect_spm}) ====="
    )

    # 1. SPM LLIR should match this kernel's expected SPM policy.
    spm_llir = spm_dir / f"{kernel}.llir"
    if not spm_llir.is_file():
        check("spm llir exists", False, str(spm_llir))
    else:
        text = spm_llir.read_text()
        n_addrspace = len(re.findall(r"addrspace\(3\)", text))
        n_fence = len(re.findall(r"fence iorw", text))
        if expect_spm:
            check("spm llir has addrspace(3)", n_addrspace > 0, f"count={n_addrspace}")
            if expect_dma:
                check("spm llir has fence iorw", n_fence > 0, f"count={n_fence}")
            else:
                check("spm llir clean of fence iorw", n_fence == 0, f"count={n_fence}")
        else:
            check("spm llir clean of addrspace(3)", n_addrspace == 0, f"count={n_addrspace}")
            check("spm llir clean of fence iorw", n_fence == 0, f"count={n_fence}")

    # 2. Cache LLIR should NOT contain these markers
    cache_llir = cache_dir / f"{kernel}.llir"
    if not cache_llir.is_file():
        check("cache llir exists", False, str(cache_llir))
    else:
        text = cache_llir.read_text()
        n_addrspace = len(re.findall(r"addrspace\(3\)", text))
        n_fence = len(re.findall(r"fence iorw", text))
        check("cache llir clean of addrspace(3)", n_addrspace == 0, f"count={n_addrspace}")
        check("cache llir clean of fence iorw", n_fence == 0, f"count={n_fence}")

    # 3. Optional promotion evidence sidecar check. This is debug/evidence
    # only; verify never treats it as placement or scheduling policy.
    if expect_promotion_source:
        promotion_json = spm_dir / f"{kernel}_promotions.json"
        if not promotion_json.is_file():
            check("promotion json exists", False, str(promotion_json))
        else:
            report = json.loads(promotion_json.read_text())
            sources = [
                record.get("source")
                for record in report.get("promotions", [])
                if record.get("status") == "accepted"
            ]
            check(
                f"promotion source {expect_promotion_source!r}",
                expect_promotion_source in sources,
                json.dumps(sources, separators=(",", ":")),
            )
    if expect_promotion_reason:
        promotion_json = spm_dir / f"{kernel}_promotions.json"
        if not promotion_json.is_file():
            check("promotion json exists", False, str(promotion_json))
        else:
            report = json.loads(promotion_json.read_text())
            reasons = [
                record.get("reason_code")
                for record in report.get("promotions", [])
                if record.get("status") == "accepted"
            ]
            check(
                f"promotion reason {expect_promotion_reason!r}",
                expect_promotion_reason in reasons,
                json.dumps(reasons, separators=(",", ":")),
            )
    if expect_rejection_reason:
        promotion_json = spm_dir / f"{kernel}_promotions.json"
        if not promotion_json.is_file():
            check("promotion json exists", False, str(promotion_json))
        else:
            report = json.loads(promotion_json.read_text())
            reasons = [
                record.get("reason_code")
                for record in report.get("rejections", [])
                if record.get("status") == "rejected"
            ]
            check(
                f"rejection reason {expect_rejection_reason!r}",
                expect_rejection_reason in reasons,
                json.dumps(reasons, separators=(",", ":")),
            )
    if expect_rejection_source:
        promotion_json = spm_dir / f"{kernel}_promotions.json"
        if not promotion_json.is_file():
            check("promotion json exists", False, str(promotion_json))
        else:
            report = json.loads(promotion_json.read_text())
            sources = [
                record.get("source")
                for record in report.get("rejections", [])
                if record.get("status") == "rejected"
            ]
            check(
                f"rejection source {expect_rejection_source!r}",
                expect_rejection_source in sources,
                json.dumps(sources, separators=(",", ":")),
            )
    if expect_residency_plan:
        promotion_json = spm_dir / f"{kernel}_promotions.json"
        if not promotion_json.is_file():
            check("promotion json exists", False, str(promotion_json))
        else:
            report = json.loads(promotion_json.read_text())
            records = report.get("promotions", []) + report.get("rejections", [])
            matching = [
                record.get("residency_plan", {})
                for record in records
                if record.get("source") == expect_residency_plan
                and isinstance(record.get("residency_plan"), dict)
            ]
            check(
                f"residency plan for {expect_residency_plan!r}",
                len(matching) > 0,
                json.dumps(matching, separators=(",", ":")),
            )
    if expect_affine_tile_reason:
        promotion_json = spm_dir / f"{kernel}_promotions.json"
        if not promotion_json.is_file():
            check("promotion json exists", False, str(promotion_json))
        else:
            report = json.loads(promotion_json.read_text())
            reasons = [
                record.get("reason_code")
                for record in report.get("affine_tile_candidates", [])
            ]
            check(
                f"affine tile reason {expect_affine_tile_reason!r}",
                expect_affine_tile_reason in reasons,
                json.dumps(reasons, separators=(",", ":")),
            )
    if expect_affine_tile_class:
        promotion_json = spm_dir / f"{kernel}_promotions.json"
        if not promotion_json.is_file():
            check("promotion json exists", False, str(promotion_json))
        else:
            report = json.loads(promotion_json.read_text())
            classes = [
                record.get("schedule_class")
                for record in report.get("affine_tile_candidates", [])
            ]
            check(
                f"affine tile class {expect_affine_tile_class!r}",
                expect_affine_tile_class in classes,
                json.dumps(classes, separators=(",", ":")),
            )

    # 4. Launcher has alloc/free_all
    launcher_c = spm_dir / f"{kernel}_launcher.c"
    if not launcher_c.is_file():
        check("launcher.c exists", False, str(launcher_c))
    else:
        text = launcher_c.read_text()
        has_alloc = f"{kernel}_alloc" in text
        has_free = f"{kernel}_free_all" in text
        check(f"launcher has {kernel}_alloc", has_alloc)
        check(f"launcher has {kernel}_free_all", has_free)

    print()
    if all_ok:
        print(f"  {kernel}: ALL CHECKS PASSED")
    else:
        failed = [name for name, ok in checks if not ok]
        print(f"  {kernel}: {len(failed)} CHECK(S) FAILED: {', '.join(failed)}")

    return all_ok


def render_tag(template: str | None, params: dict[str, str], default: str | None) -> str:
    if not template:
        if default is None:
            sys.exit(
                "ERROR: kernel manifest has no tag_template and no fallback was supplied. "
                "Add `tag_template = \"...\"` under [kernel] in experiment.toml."
            )
        return default
    return template.format(**params)


def apply_preset_to_tag(tag: str, preset: str | None) -> str:
    if not preset:
        return tag
    if "/" in tag:
        shape, blocking = tag.split("/", 1)
        return f"{shape}/{preset}-{blocking}"
    return f"{preset}-{tag}"


def default_tag(manifest: dict, params: dict[str, str], preset: str | None) -> str:
    base_tag = render_tag(manifest["kernel"].get("tag_template"), params, default=None)
    return apply_preset_to_tag(base_tag, preset)


def stat_value(path: Path, stat_name: str) -> float:
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == stat_name:
            return float(parts[1])
    raise ValueError(f"{stat_name} not found in {path}")


def truthy_param(params: dict[str, str], name: str) -> bool:
    value = params.get(name, "0")
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def require_reduction_row_block_lowered(kernel: str, mode: str, tag: str,
                                        params: dict[str, str]) -> None:
    expected_sources = {
        "softmax": "Softmax x row block",
        "layer_norm": "LayerNorm x row block",
        "rms_norm": "RMSNorm x row block",
    }
    if kernel not in expected_sources or mode != "spm":
        return
    if not truthy_param(params, "SPM_INTERNAL_ROW_BLOCK"):
        return
    if kernel == "layer_norm":
        block_n = int(params.get("BLOCK_N", "0"))
        n = int(params.get("N", "0"))
        if block_n >= n:
            return

    promotion_json = (
        trispm_paths.build_dir(kernel, mode, tag) / f"{kernel}_promotions.json"
    )
    if not promotion_json.is_file():
        sys.exit(
            f"ERROR: {kernel} requested SPM_INTERNAL_ROW_BLOCK=1, but the "
            f"promotion sidecar is missing: {rel_workloads_path(promotion_json)}\n"
            "The harness must not shrink gridX unless the compiler accepted "
            f"the {expected_sources[kernel]} lowering."
        )

    report = json.loads(promotion_json.read_text())
    expected_source = expected_sources[kernel]
    accepted_sources = {
        record.get("source")
        for record in report.get("promotions", [])
        if record.get("status") == "accepted"
    }
    if expected_source in accepted_sources:
        return

    rejections = [
        {
            "source": record.get("source"),
            "reason_code": record.get("reason_code"),
            "reason": record.get("reason"),
        }
        for record in report.get("rejections", [])
        if record.get("status") == "rejected"
    ]
    sys.exit(
        f"ERROR: {kernel} requested SPM_INTERNAL_ROW_BLOCK=1, but the compiler "
        f"did not accept `{expected_source}` lowering.\n"
        f"Sidecar: {rel_workloads_path(promotion_json)}\n"
        f"Rejections: {json.dumps(rejections, separators=(',', ':'))}\n"
        "This configuration is invalid because the canonical kernel still "
        "expects gridX=M; shrinking gridX would leave rows uncomputed."
    )


def run_one_mode(
    kernel: str,
    manifest: dict,
    tag: str,
    run_mode: str,
    preset: str | None,
    overrides: dict[str, str],
    skip_build: bool,
    should_run: bool = True,
    cache_l1d_size: str | None = None,
    cache_l2_size: str | None = None,
) -> dict[str, str]:
    gem5_flags = (
        cache_gem5_flags(cache_l1d_size, cache_l2_size)
        if run_mode == "cache" else []
    )
    target_params = merged_params(manifest, preset, overrides, mode=run_mode)
    env = export_env(manifest, target_params)
    env.update({k: str(v) for k, v in manifest.get("env", {}).items()})
    if run_mode == "spm":
        env.update(preset_env(manifest, preset))
    env.update({k: str(v) for k, v in manifest.get("_cli_env", {}).items()})
    if not skip_build:
        do_build(kernel, run_mode, tag, env)
    if run_mode == "spm":
        require_reduction_row_block_lowered(kernel, run_mode, tag, target_params)
    if should_run:
        do_run(kernel, run_mode, tag, gem5_flags, env)
        validate_run_result(kernel, run_mode, tag, target_params)
    return target_params


def cache_best_record(
    kernel: str,
    tag: str,
    params: dict[str, str],
    cycles: float,
    cache_l1d_size: str | None,
    cache_l2_size: str | None,
) -> dict[str, object]:
    shape, blocking = trispm_paths.split_tag(tag)
    return {
        "kernel": kernel,
        "shape": shape,
        "blocking": blocking,
        "tag": tag,
        "mode": "cache",
        "numCycles": int(cycles) if cycles.is_integer() else cycles,
        "roi_stats": rel_workloads_path(trispm_paths.roi_stats_path(kernel, "cache", tag)),
        "run_log": rel_workloads_path(trispm_paths.run_log_path(kernel, "cache", tag)),
        "params": params,
        "cache_l1d_size": cache_l1d_size or "32KiB",
        "cache_l2_size": cache_l2_size or "512KiB",
    }


def write_cache_best(
    kernel: str,
    tag: str,
    record: dict[str, object],
    cache_l1d_size: str | None = None,
    cache_l2_size: str | None = None,
) -> None:
    path = cache_best_output_path(kernel, tag, cache_l1d_size, cache_l2_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"Cache best:     {rel_workloads_path(path)}")


def load_cache_best(
    kernel: str,
    tag: str,
    cache_l1d_size: str | None = None,
    cache_l2_size: str | None = None,
) -> dict[str, object]:
    path = cache_best_output_path(kernel, tag, cache_l1d_size, cache_l2_size)
    if not path.is_file():
        sys.exit(
            f"ERROR: cache best is missing for this shape: {rel_workloads_path(path)}\n"
            "Run cache-search for the shape first."
        )
    return json.loads(path.read_text())


def do_cache_search(
    kernel: str,
    manifest: dict,
    candidates: list[tuple[str, dict[str, str], dict[str, str], str | None]],
    skip_build: bool,
    cache_l1d_size: str | None,
    cache_l2_size: str | None,
) -> None:
    best_by_shape: dict[str, tuple[float, str, dict[str, str]]] = {}
    remove_cache_best_outputs(
        kernel,
        [tag for tag, _params, _overrides, _preset in candidates],
        cache_l1d_size,
        cache_l2_size,
    )
    for tag, params, overrides, preset in candidates:
        print(f"\n========== cache candidate: {tag} ==========")
        cache_params = run_one_mode(
            kernel, manifest, tag, "cache", preset, overrides, skip_build,
            should_run=True,
            cache_l1d_size=cache_l1d_size,
            cache_l2_size=cache_l2_size,
        )
        cycles = stat_value(
            trispm_paths.roi_stats_path(kernel, "cache", tag),
            "system.cpu.numCycles",
        )
        print(f"cache numCycles: {cycles:.0f}")
        shape, _ = trispm_paths.split_tag(tag)
        best = best_by_shape.get(shape)
        if best is None or cycles < best[0]:
            best_by_shape[shape] = (cycles, tag, cache_params)

    if not best_by_shape:
        sys.exit("ERROR: cache-search had no candidates")
    for _shape, (cycles, tag, params) in sorted(best_by_shape.items()):
        write_cache_best(
            kernel,
            tag,
            cache_best_record(
                kernel, tag, params, cycles, cache_l1d_size, cache_l2_size),
            cache_l1d_size,
            cache_l2_size,
        )


def do_spm_compare(
    kernel: str,
    manifest: dict,
    tag: str,
    preset: str | None,
    overrides: dict[str, str],
    skip_build: bool,
    params: dict[str, str],
    cache_l1d_size: str | None,
    cache_l2_size: str | None,
) -> None:
    if not cache_l1d_size and not cache_l2_size:
        remove_compare_outputs(kernel, tag)
    best = load_cache_best(kernel, tag, cache_l1d_size, cache_l2_size)
    cache_tag = str(best["tag"])
    cache_stats = trispm_paths.roi_stats_path(kernel, "cache", cache_tag)
    if not cache_stats.is_file():
        sys.exit(
            f"ERROR: cache_best.json points to missing stats: "
            f"{rel_workloads_path(cache_stats)}"
        )

    run_one_mode(kernel, manifest, tag, "spm", preset, overrides, skip_build,
                 should_run=True)
    compare_out = (
        capacity_compare_path(kernel, tag, cache_l1d_size, cache_l2_size)
        if cache_l1d_size or cache_l2_size else None
    )
    do_compare_stats(
        kernel,
        tag,
        cache_tag,
        int(params.get("MEASURE_ITERS", "1")),
        compare_out=compare_out,
        spm_only_out=None,
    )
    if not cache_l1d_size and not cache_l2_size:
        write_artifact_stats(kernel, tag, cache_tag)


def execute_one(
    kernel: str,
    manifest: dict,
    params: dict[str, str],
    tag: str,
    mode: str,
    preset: str | None,
    overrides: dict[str, str],
    skip_build: bool,
    cache_l1d_size: str | None,
    cache_l2_size: str | None,
) -> None:
    if mode == "cache-search":
        do_cache_search(
            kernel,
            manifest,
            [(tag, params, overrides, preset)],
            skip_build,
            cache_l1d_size,
            cache_l2_size,
        )
        return

    if mode == "spm-compare":
        do_spm_compare(
            kernel,
            manifest,
            tag,
            preset,
            overrides,
            skip_build,
            params,
            cache_l1d_size,
            cache_l2_size,
        )
        return

    # (run_mode, do_run?)
    targets = {
        "spm":     [("spm",   True)],
        "cache":   [("cache", True)],
        "build":   [("spm",   False), ("cache", False)],
        "verify":  [("spm",   False), ("cache", False)],
    }[mode]

    for run_mode, should_run in targets:
        run_one_mode(
            kernel,
            manifest,
            tag,
            run_mode,
            preset,
            overrides,
            skip_build,
            should_run=should_run,
            cache_l1d_size=cache_l1d_size if run_mode == "cache" else None,
            cache_l2_size=cache_l2_size if run_mode == "cache" else None,
        )

    if mode == "verify":
        ok = do_verify(kernel, tag, manifest)
        if not ok:
            sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kernel")
    p.add_argument(
        "--mode",
        choices=("spm", "cache", "cache-search", "spm-compare", "build", "verify"),
        default="spm-compare",
    )
    p.add_argument("--tag", default=None, help="override artifact tag")
    p.add_argument("--preset", default=None, help="apply [presets.<name>] from manifest")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                   help="override a single param (repeatable)")
    p.add_argument("--sweep", default=None, help="run [sweeps.<name>] from manifest")
    p.add_argument("--skip-build", action="store_true", help="reuse existing build artifacts")
    p.add_argument(
        "--cache-l1d-size",
        default=None,
        help="cache-only L1D capacity for capacity/fairness baselines, e.g. 256KiB",
    )
    p.add_argument(
        "--cache-l2-size",
        default=None,
        help="cache-only L2 capacity for capacity/fairness baselines, e.g. 1MiB",
    )
    p.add_argument("--env", action="append", default=[], metavar="KEY=VAL",
                   help="export one build/run environment variable (repeatable)")
    p.add_argument("--expect-spm", choices=("true", "false"), default=None,
                   help="override [kernel].expect_spm for verify mode")
    p.add_argument("--expect-dma", choices=("true", "false"), default=None,
                   help="override verify DMA/fence marker expectation")
    p.add_argument("--expect-promotion-source", default=None,
                   help="require an accepted promotion source in the debug sidecar")
    p.add_argument("--expect-promotion-reason", default=None,
                   help="require an accepted promotion reason_code in the debug sidecar")
    p.add_argument("--expect-rejection-reason", default=None,
                   help="require a rejected promotion reason_code in the debug sidecar")
    p.add_argument("--expect-rejection-source", default=None,
                   help="require a rejected promotion source in the debug sidecar")
    p.add_argument("--expect-residency-plan", default=None,
                   help="require a residency_plan entry for this promotion/rejection source")
    p.add_argument("--expect-affine-tile-reason", default=None,
                   help="require a generic affine tile candidate reason_code in the debug sidecar")
    p.add_argument("--expect-affine-tile-class", default=None,
                   help="require a generic affine tile candidate schedule_class in the debug sidecar")
    args = p.parse_args()

    manifest = load_manifest(args.kernel)
    overrides: dict[str, str] = {}
    for kv in args.set:
        if "=" not in kv:
            sys.exit(f"--set expects KEY=VAL, got {kv!r}")
        k, v = kv.split("=", 1)
        overrides[k] = v

    cli_env: dict[str, str] = {}
    for kv in args.env:
        if "=" not in kv:
            sys.exit(f"--env expects KEY=VAL, got {kv!r}")
        k, v = kv.split("=", 1)
        cli_env[k] = v

    if cli_env:
        manifest = dict(manifest)
        manifest["_cli_env"] = cli_env

    if (
        args.expect_spm is not None
        or args.expect_dma is not None
        or args.expect_promotion_source is not None
        or args.expect_promotion_reason is not None
        or args.expect_rejection_reason is not None
        or args.expect_rejection_source is not None
        or args.expect_residency_plan is not None
        or args.expect_affine_tile_reason is not None
        or args.expect_affine_tile_class is not None
    ):
        manifest = dict(manifest)
        kernel_cfg = dict(manifest["kernel"])
        verify_cfg = dict(kernel_cfg.get("verify", {}))
        if args.expect_spm is not None:
            kernel_cfg["expect_spm"] = args.expect_spm == "true"
        if args.expect_dma is not None:
            verify_cfg["expect_dma"] = args.expect_dma == "true"
        if args.expect_promotion_source is not None:
            verify_cfg["expect_promotion_source"] = args.expect_promotion_source
        if args.expect_promotion_reason is not None:
            verify_cfg["expect_promotion_reason"] = args.expect_promotion_reason
        if args.expect_rejection_reason is not None:
            verify_cfg["expect_rejection_reason"] = args.expect_rejection_reason
        if args.expect_rejection_source is not None:
            verify_cfg["expect_rejection_source"] = args.expect_rejection_source
        if args.expect_residency_plan is not None:
            verify_cfg["expect_residency_plan"] = args.expect_residency_plan
        if args.expect_affine_tile_reason is not None:
            verify_cfg["expect_affine_tile_reason"] = args.expect_affine_tile_reason
        if args.expect_affine_tile_class is not None:
            verify_cfg["expect_affine_tile_class"] = args.expect_affine_tile_class
        kernel_cfg["verify"] = verify_cfg
        manifest["kernel"] = kernel_cfg

    if args.sweep:
        sweep = manifest.get("sweeps", {}).get(args.sweep)
        if sweep is None:
            sys.exit(f"ERROR: sweep {args.sweep!r} not in manifest")
        candidates: list[tuple[str, dict[str, str], dict[str, str], str | None]] = []
        for index, value in enumerate(sweep["values"]):
            sweep_overrides = dict(overrides)
            if isinstance(value, dict):
                for k, v in value.items():
                    if k == "name":
                        continue
                    sweep_overrides[k] = str(v)
                label = str(value.get("name", f"candidate{index}"))
            else:
                axis = sweep.get("axis")
                if axis is None:
                    sys.exit(
                        f"ERROR: sweep {args.sweep!r} scalar values require axis"
                    )
                sweep_overrides[axis] = str(value)
                for mirrored in sweep.get("mirror", []):
                    sweep_overrides[mirrored] = str(value)
                label = f"{axis}={value}"
            params = merged_params(manifest, args.preset, sweep_overrides)
            base_tag = args.tag or render_tag(
                sweep.get("tag_template"),
                params,
                default=label,
            )
            tag = base_tag if args.tag else apply_preset_to_tag(base_tag, args.preset)
            if args.mode in {"cache", "cache-search"}:
                tag = capacity_tag(
                    tag,
                    args.cache_l1d_size,
                    args.cache_l2_size,
                )
            if args.mode == "cache-search":
                candidates.append((tag, params, sweep_overrides, args.preset))
            else:
                print(f"\n========== sweep {args.sweep}: {label} (tag={tag}) ==========")
                execute_one(args.kernel, manifest, params, tag, args.mode,
                            args.preset, sweep_overrides, args.skip_build,
                            args.cache_l1d_size,
                            args.cache_l2_size)
        if args.mode == "cache-search":
            do_cache_search(
                args.kernel,
                manifest,
                candidates,
                args.skip_build,
                args.cache_l1d_size,
                args.cache_l2_size,
            )
        return

    params = merged_params(manifest, args.preset, overrides)
    tag = args.tag or default_tag(manifest, params, args.preset)
    if args.mode in {"cache", "cache-search"}:
        tag = capacity_tag(tag, args.cache_l1d_size, args.cache_l2_size)
    execute_one(args.kernel, manifest, params, tag, args.mode,
                args.preset, overrides, args.skip_build,
                args.cache_l1d_size, args.cache_l2_size)


if __name__ == "__main__":
    main()
