#!/usr/bin/env python3
"""Summarize generic affine fallback gem5 performance rows."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import compare_stats
import trispm_paths


def rows_from_campaign(campaign: Path) -> list[dict[str, object]]:
    run_list = campaign / "run_list.json"
    if not run_list.is_file():
        sys.exit(f"ERROR: campaign run_list not found: {run_list}")
    payload = json.loads(run_list.read_text())
    return [
        dict(row)
        for row in payload.get("rows", [])
        if row.get("phase") == "generic-affine-fallback-perf"
    ]


def roi_path(kernel: str, mode: str, tag: str) -> Path:
    return trispm_paths.roi_stats_path(kernel, mode, tag)


def read_cycles(path: Path) -> int | None:
    if not path.is_file():
        return None
    stats = compare_stats.load_stats(path, "first")
    value = stats.get("system.cpu.numCycles")
    if value is None:
        return None
    return int(float(value))


def ratio(numerator: int | None, denominator: int | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def fmt_path(path: Path) -> str:
    try:
        return str(path.relative_to(trispm_paths.WORKLOADS_DIR))
    except ValueError:
        return str(path)


def summarize_row(row: dict[str, object]) -> dict[str, object]:
    kernel = str(row.get("workload", ""))
    metadata = dict(row.get("metadata", {}) or {})
    generic_tag = str(metadata.get("generic_spm_tag", ""))
    cache_tag = str(metadata.get("cache_tag", ""))
    tuned_tag = str(metadata.get("tuned_spm_tag", ""))

    generic_path = roi_path(kernel, "spm", generic_tag)
    cache_path = roi_path(kernel, "cache", cache_tag)
    tuned_path = roi_path(kernel, "spm", tuned_tag)
    generic_cycles = read_cycles(generic_path)
    cache_cycles = read_cycles(cache_path)
    tuned_cycles = read_cycles(tuned_path)

    return {
        "kernel": kernel,
        "label": row.get("label", ""),
        "source_table": metadata.get("source_table", ""),
        "role": metadata.get("role", metadata.get("algorithm", "")),
        "cache_cycles": cache_cycles,
        "tuned_spm_cycles": tuned_cycles,
        "generic_spm_cycles": generic_cycles,
        "cache_over_generic_speedup": ratio(cache_cycles, generic_cycles),
        "cache_over_tuned_speedup": ratio(cache_cycles, tuned_cycles),
        "tuned_over_generic_speedup": ratio(tuned_cycles, generic_cycles),
        "generic_over_tuned_slowdown": ratio(generic_cycles, tuned_cycles),
        "cache_roi": fmt_path(cache_path),
        "tuned_spm_roi": fmt_path(tuned_path),
        "generic_spm_roi": fmt_path(generic_path),
        "missing": [
            name
            for name, cycles in (
                ("cache", cache_cycles),
                ("tuned_spm", tuned_cycles),
                ("generic_spm", generic_cycles),
            )
            if cycles is None
        ],
    }


def write_csv(records: list[dict[str, object]], path: Path) -> None:
    fields = [
        "kernel",
        "label",
        "source_table",
        "role",
        "cache_cycles",
        "tuned_spm_cycles",
        "generic_spm_cycles",
        "cache_over_generic_speedup",
        "cache_over_tuned_speedup",
        "tuned_over_generic_speedup",
        "generic_over_tuned_slowdown",
        "missing",
        "cache_roi",
        "tuned_spm_roi",
        "generic_spm_roi",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["missing"] = ",".join(row.get("missing", []))
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--output-csv")
    args = parser.parse_args()

    campaign = trispm_paths.WORKLOADS_DIR / "m5out" / "campaigns" / args.campaign
    records = [summarize_row(row) for row in rows_from_campaign(campaign)]
    text = json.dumps(records, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
    if args.output_csv:
        write_csv(records, Path(args.output_csv))


if __name__ == "__main__":
    main()
