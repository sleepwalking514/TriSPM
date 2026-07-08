#!/usr/bin/env python3
"""Summarize generic affine-tile fallback evidence from promotion sidecars."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from internal import trispm_paths


def summarize_sidecar(path: Path, row: dict[str, object] | None = None) -> dict[str, object]:
    report = json.loads(path.read_text())
    candidates = report.get("affine_tile_candidates", [])
    promotions = report.get("promotions", [])
    generic_promotions = [
        record for record in promotions
        if str(record.get("reason_code", "")).startswith(
            "accepted_generic_affine_tile")
    ]
    classes = Counter(str(record.get("schedule_class", "missing"))
                      for record in candidates)
    statuses = Counter(str(record.get("status", "missing"))
                       for record in candidates)
    reasons = Counter(str(record.get("reason_code", "missing"))
                      for record in candidates)
    rejection_reasons = Counter(
        str(record.get("reason_code", "missing"))
        for record in report.get("rejections", [])
        if record.get("pattern") == "generic_affine_tile"
    )
    metadata = {}
    if row:
        metadata = dict(row.get("metadata", {}) or {})
    return {
        "kernel": report.get("kernel", path.stem.removesuffix("_promotions")),
        "workload": row.get("workload", "") if row else "",
        "label": row.get("label", "") if row else "",
        "source_table": metadata.get("source_table", ""),
        "role": metadata.get("role", metadata.get("algorithm", "")),
        "sidecar": str(path.relative_to(trispm_paths.WORKLOADS_DIR)),
        "candidate_count": len(candidates),
        "candidate_statuses": dict(sorted(statuses.items())),
        "schedule_classes": dict(sorted(classes.items())),
        "candidate_reasons": dict(reasons.most_common()),
        "generic_accepted": len(generic_promotions),
        "generic_accept_reasons": dict(Counter(
            str(record.get("reason_code", "missing"))
            for record in generic_promotions
        )),
        "generic_rejection_reasons": dict(rejection_reasons.most_common()),
    }


def sidecar_path(kernel: str, tag: str) -> Path:
    return trispm_paths.build_dir(kernel, "spm", tag) / f"{kernel}_promotions.json"


def rows_from_campaign(campaign: Path) -> list[dict[str, object]]:
    run_list = campaign / "run_list.json"
    if not run_list.is_file():
        sys.exit(f"ERROR: campaign run_list not found: {run_list}")
    payload = json.loads(run_list.read_text())
    rows: list[dict[str, object]] = []
    for row in payload.get("rows", []):
        command = row.get("command", [])
        if row.get("phase") != "generic-affine-fallback":
            continue
        if "--tag" not in command:
            continue
        row = dict(row)
        row["tag"] = str(command[command.index("--tag") + 1])
        rows.append(row)
    return rows


def write_csv(records: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "kernel",
        "workload",
        "label",
        "source_table",
        "role",
        "sidecar",
        "candidate_count",
        "generic_accepted",
        "candidate_statuses",
        "schedule_classes",
        "generic_accept_reasons",
        "generic_rejection_reasons",
        "candidate_reasons",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = dict(record)
            for key in fields:
                if isinstance(row.get(key), dict):
                    row[key] = json.dumps(row[key], sort_keys=True)
            writer.writerow({key: row.get(key, "") for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", help="campaign name under workloads/m5out/campaigns")
    parser.add_argument("--kernel", action="append", default=[])
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--output-json")
    parser.add_argument("--output-csv")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    if args.campaign:
        rows.extend(rows_from_campaign(
            trispm_paths.WORKLOADS_DIR / "m5out" / "campaigns" / args.campaign))
    if args.kernel or args.tag:
        if len(args.kernel) != len(args.tag):
            sys.exit("ERROR: --kernel and --tag must be repeated equally")
        rows.extend({"workload": kernel, "tag": tag, "label": "", "metadata": {}}
                    for kernel, tag in zip(args.kernel, args.tag))
    if not rows:
        sys.exit("ERROR: provide --campaign or paired --kernel/--tag")

    records = []
    for row in rows:
        kernel = str(row.get("workload", ""))
        tag = str(row.get("tag", ""))
        path = sidecar_path(kernel, tag)
        if not path.is_file():
            metadata = dict(row.get("metadata", {}) or {})
            records.append({
                "kernel": kernel,
                "workload": kernel,
                "label": row.get("label", ""),
                "source_table": metadata.get("source_table", ""),
                "role": metadata.get("role", metadata.get("algorithm", "")),
                "sidecar": str(path.relative_to(trispm_paths.WORKLOADS_DIR)),
                "candidate_count": 0,
                "generic_accepted": 0,
                "candidate_statuses": {"missing_sidecar": 1},
                "schedule_classes": {},
                "candidate_reasons": {},
                "generic_accept_reasons": {},
                "generic_rejection_reasons": {},
            })
            continue
        records.append(summarize_sidecar(path, row))

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
