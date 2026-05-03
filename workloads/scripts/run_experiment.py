#!/usr/bin/env python3
"""TriSPM experiment driver: build → run gem5 → compare.

Reads a kernel's experiment.toml manifest, exports its params as env vars
for the build, and orchestrates spm/cache/compare runs in a single command.

Usage:
  run_experiment.py <kernel> --mode spm
  run_experiment.py <kernel> --mode compare [--preset steady]
  run_experiment.py <kernel> --mode verify
  run_experiment.py <kernel> --sweep size [--preset steady]

Modes:
  spm       build + run with SPM
  cache     build + run cache-baseline
  compare   build + run both, then save delta table
  verify    build both, check LLIR for SPM markers + tier JSON
"""
from __future__ import annotations

import argparse
import csv
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


def do_build(kernel: str, mode: str, tag: str, env: dict[str, str]) -> None:
    run([str(SCRIPTS_DIR / "build_kernel.sh"), kernel, "--mode", mode, "--tag", tag], env=env)


def do_run(kernel: str, mode: str, tag: str, gem5_flags: list[str], env: dict[str, str]) -> None:
    cmd = [str(SCRIPTS_DIR / "run_gem5.sh"), kernel, "--mode", mode, "--tag", tag]
    if gem5_flags:
        cmd += ["--"] + gem5_flags
    run(cmd, env=env)


def do_compare(kernel: str, tag: str, measure_iters: int) -> None:
    spm_stats = trispm_paths.roi_stats_path(kernel, "spm", tag)
    cache_stats = trispm_paths.roi_stats_path(kernel, "cache", tag)
    compare = trispm_paths.compare_path(kernel, tag)
    spm_only = trispm_paths.spm_stats_path(kernel, tag)
    compare_csv = trispm_paths.compare_csv_path(kernel, tag)
    spm_only_csv = trispm_paths.spm_stats_csv_path(kernel, tag)
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "compare_stats.py"),
        "--spm", str(spm_stats),
        "--cache", str(cache_stats),
        "--measure-iters", str(measure_iters),
        "--output", str(compare),
        "--spm-only-output", str(spm_only),
        "--csv", str(compare_csv),
        "--spm-only-csv", str(spm_only_csv),
        "--quiet",
    ]
    run(cmd, echo=False)
    print(f"Compare saved:  {rel_workloads_path(compare)}")
    print(f"Compare CSV:    {rel_workloads_path(compare_csv)}")
    print(f"SPM-only saved: {rel_workloads_path(spm_only)}")
    print(f"SPM-only CSV:   {rel_workloads_path(spm_only_csv)}")


ARTIFACT_PATTERNS = {
    "llir": "{kernel}.llir",
    "asm": "{kernel}.s",
    "launcher_c": "{kernel}_launcher.c",
    "promotion_json": "{kernel}_promotions.json",
    "tier_json": "{kernel}_tiers.json",
}


def count_pattern(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text))


def artifact_rows(kernel: str, tag: str) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for mode in ("spm", "cache"):
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


def write_artifact_stats(kernel: str, tag: str) -> None:
    rows = artifact_rows(kernel, tag)
    out = trispm_paths.artifact_stats_path(kernel, tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    headers = (
        "mode",
        "artifact",
        "path",
        "bytes",
        "lines",
        "addrspace3",
        "fence_iorw",
        "spm_dma_wait",
        "spm_dma_enqueue",
    )
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Artifact stats: {rel_workloads_path(out)}")


def do_verify(kernel: str, tag: str, manifest: dict) -> None:
    """Check SPM/cache LLIR markers against the manifest's expected policy."""
    spm_dir = trispm_paths.build_dir(kernel, "spm", tag)
    cache_dir = trispm_paths.build_dir(kernel, "cache", tag)
    expect_spm = bool(manifest["kernel"].get("expect_spm", True))
    verify_cfg = manifest["kernel"].get("verify", {})
    expect_tier_json = verify_cfg.get(
        "expect_tier_json",
        "non_empty" if expect_spm else "empty",
    )
    expect_dma = bool(verify_cfg.get(
        "expect_dma",
        expect_spm and expect_tier_json == "non_empty",
    ))
    expect_promotion_source = verify_cfg.get("expect_promotion_source")
    expect_rejection_reason = verify_cfg.get("expect_rejection_reason")

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
        f"(tag={tag}, expect_spm={expect_spm}, "
        f"expect_tier_json={expect_tier_json}) ====="
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

    # 3. Tier JSON
    tier_json = spm_dir / f"{kernel}_tiers.json"
    if not tier_json.is_file():
        check("tier json exists", False, str(tier_json))
    else:
        tiers = json.loads(tier_json.read_text())
        if expect_tier_json == "non_empty":
            non_empty = len(tiers) > 0
            check("tier json non-empty", non_empty, json.dumps(tiers, separators=(",", ":")))
        elif expect_tier_json == "empty":
            empty = len(tiers) == 0
            check("tier json empty", empty, json.dumps(tiers, separators=(",", ":")))
        else:
            check("tier json expectation valid", False, str(expect_tier_json))

    # 3b. Optional promotion evidence sidecar check. This is debug/evidence
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
    return f"{preset}-{tag}"


def default_tag(manifest: dict, params: dict[str, str], preset: str | None) -> str:
    base_tag = render_tag(manifest["kernel"].get("tag_template"), params, default=None)
    return apply_preset_to_tag(base_tag, preset)


def execute_one(
    kernel: str,
    manifest: dict,
    params: dict[str, str],
    tag: str,
    mode: str,
    preset: str | None,
    overrides: dict[str, str],
    skip_build: bool,
) -> None:
    cache_gem5_flags = ["--cache_baseline"]

    # (run_mode, gem5_flags, do_run?)
    targets = {
        "spm":     [("spm",   [],                True)],
        "cache":   [("cache", cache_gem5_flags,  True)],
        "compare": [("spm",   [],                True),
                    ("cache", cache_gem5_flags,  True)],
        "build":   [("spm",   [],                False),
                    ("cache", cache_gem5_flags,  False)],
        "verify":  [("spm",   [],                False),
                    ("cache", cache_gem5_flags,  False)],
    }[mode]

    for run_mode, gem5_flags, should_run in targets:
        target_params = merged_params(manifest, preset, overrides, mode=run_mode)
        env = export_env(manifest, target_params)
        env.update({k: str(v) for k, v in manifest.get("env", {}).items()})
        env.update(preset_env(manifest, preset))
        env.update({k: str(v) for k, v in manifest.get("_cli_env", {}).items()})
        if not skip_build:
            do_build(kernel, run_mode, tag, env)
        if should_run:
            do_run(kernel, run_mode, tag, gem5_flags, env)

    if mode == "compare":
        do_compare(kernel, tag, int(params.get("MEASURE_ITERS", "1")))
        write_artifact_stats(kernel, tag)

    if mode == "verify":
        ok = do_verify(kernel, tag, manifest)
        if not ok:
            sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kernel")
    p.add_argument("--mode", choices=("spm", "cache", "compare", "build", "verify"), default="compare")
    p.add_argument("--tag", default=None, help="override artifact tag")
    p.add_argument("--preset", default=None, help="apply [presets.<name>] from manifest")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                   help="override a single param (repeatable)")
    p.add_argument("--sweep", default=None, help="run [sweeps.<name>] from manifest")
    p.add_argument("--skip-build", action="store_true", help="reuse existing build artifacts")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VAL",
                   help="export one build/run environment variable (repeatable)")
    p.add_argument("--expect-spm", choices=("true", "false"), default=None,
                   help="override [kernel].expect_spm for verify mode")
    p.add_argument("--expect-tier-json", choices=("empty", "non_empty"), default=None,
                   help="override verify tier sidecar expectation")
    p.add_argument("--expect-promotion-source", default=None,
                   help="require an accepted promotion source in the debug sidecar")
    p.add_argument("--expect-rejection-reason", default=None,
                   help="require a rejected promotion reason_code in the debug sidecar")
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
        or args.expect_tier_json is not None
        or args.expect_promotion_source is not None
        or args.expect_rejection_reason is not None
    ):
        manifest = dict(manifest)
        kernel_cfg = dict(manifest["kernel"])
        verify_cfg = dict(kernel_cfg.get("verify", {}))
        if args.expect_spm is not None:
            kernel_cfg["expect_spm"] = args.expect_spm == "true"
        if args.expect_tier_json is not None:
            verify_cfg["expect_tier_json"] = args.expect_tier_json
        if args.expect_promotion_source is not None:
            verify_cfg["expect_promotion_source"] = args.expect_promotion_source
        if args.expect_rejection_reason is not None:
            verify_cfg["expect_rejection_reason"] = args.expect_rejection_reason
        kernel_cfg["verify"] = verify_cfg
        manifest["kernel"] = kernel_cfg

    if args.sweep:
        sweep = manifest.get("sweeps", {}).get(args.sweep)
        if sweep is None:
            sys.exit(f"ERROR: sweep {args.sweep!r} not in manifest")
        axis = sweep["axis"]
        for value in sweep["values"]:
            sweep_overrides = dict(overrides)
            sweep_overrides[axis] = str(value)
            for mirrored in sweep.get("mirror", []):
                sweep_overrides[mirrored] = str(value)
            params = merged_params(manifest, args.preset, sweep_overrides)
            base_tag = args.tag or render_tag(sweep.get("tag_template"), params, default=f"{axis.lower()}{value}")
            tag = base_tag if args.tag else apply_preset_to_tag(base_tag, args.preset)
            print(f"\n========== sweep {args.sweep}: {axis}={value} (tag={tag}) ==========")
            execute_one(args.kernel, manifest, params, tag, args.mode,
                        args.preset, sweep_overrides, args.skip_build)
        return

    params = merged_params(manifest, args.preset, overrides)
    tag = args.tag or default_tag(manifest, params, args.preset)
    execute_one(args.kernel, manifest, params, tag, args.mode,
                args.preset, overrides, args.skip_build)


if __name__ == "__main__":
    main()
