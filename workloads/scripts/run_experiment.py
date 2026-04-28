#!/usr/bin/env python3
"""TriSPM experiment driver: build → run gem5 → compare.

Reads a kernel's experiment.toml manifest, exports its params as env vars
for the build, and orchestrates spm/cache/compare runs in a single command.

Usage:
  run_experiment.py <kernel> --mode spm
  run_experiment.py <kernel> --mode compare [--preset steady]
  run_experiment.py <kernel> --sweep size [--preset steady]

Modes:
  spm       build + run with SPM
  cache     build + run cache-baseline
  compare   build + run both, then save delta table
"""
from __future__ import annotations

import argparse
import os
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


def merged_params(manifest: dict, preset: str | None, overrides: dict[str, str]) -> dict[str, str]:
    params: dict = dict(manifest.get("params", {}))
    if preset:
        preset_params = manifest.get("presets", {}).get(preset)
        if preset_params is None:
            sys.exit(f"ERROR: preset {preset!r} not in manifest")
        params.update(preset_params)
    params.update(overrides)
    return {k: str(v) for k, v in params.items()}


def export_env(manifest: dict, params: dict[str, str]) -> dict[str, str]:
    """Return env with manifest params exported as <PREFIX><KEY>."""
    prefix = manifest["kernel"].get("env_prefix", "")
    env = os.environ.copy()
    for k, v in params.items():
        env[f"{prefix}{k}"] = v
    return env


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
    skip_build: bool,
) -> None:
    env = export_env(manifest, params)
    cache_gem5_flags = ["--cache_baseline"]

    # (run_mode, gem5_flags, do_run?)
    targets = {
        "spm":     [("spm",   [],                True)],
        "cache":   [("cache", cache_gem5_flags,  True)],
        "compare": [("spm",   [],                True),
                    ("cache", cache_gem5_flags,  True)],
        "build":   [("spm",   [],                False),
                    ("cache", cache_gem5_flags,  False)],
    }[mode]

    for run_mode, gem5_flags, should_run in targets:
        if not skip_build:
            do_build(kernel, run_mode, tag, env)
        if should_run:
            do_run(kernel, run_mode, tag, gem5_flags, env)

    if mode == "compare":
        do_compare(kernel, tag, int(params.get("MEASURE_ITERS", "1")))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kernel")
    p.add_argument("--mode", choices=("spm", "cache", "compare", "build"), default="compare")
    p.add_argument("--tag", default=None, help="override artifact tag")
    p.add_argument("--preset", default=None, help="apply [presets.<name>] from manifest")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                   help="override a single param (repeatable)")
    p.add_argument("--sweep", default=None, help="run [sweeps.<name>] from manifest")
    p.add_argument("--skip-build", action="store_true", help="reuse existing build artifacts")
    args = p.parse_args()

    manifest = load_manifest(args.kernel)
    overrides: dict[str, str] = {}
    for kv in args.set:
        if "=" not in kv:
            sys.exit(f"--set expects KEY=VAL, got {kv!r}")
        k, v = kv.split("=", 1)
        overrides[k] = v

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
            execute_one(args.kernel, manifest, params, tag, args.mode, args.skip_build)
        return

    params = merged_params(manifest, args.preset, overrides)
    tag = args.tag or default_tag(manifest, params, args.preset)
    execute_one(args.kernel, manifest, params, tag, args.mode, args.skip_build)


if __name__ == "__main__":
    main()
