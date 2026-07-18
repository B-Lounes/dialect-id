#!/usr/bin/env python3
"""Verify BIG-RUN final label export artifacts."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {
    "sample_id",
    "tar_path",
    "member",
    "final_dialect",
    "final_country",
    "final_region",
    "confidence_tier",
    "strong_model_only",
    "sure_model_only",
    "ultra_sure_model_only",
    "sure_with_weak_region",
    "sure_with_weak_top1",
    "top1_score",
    "margin",
    "top2_cc",
    "top3_cc",
    "weak_cc",
    "weak_top1_match",
    "weak_region_match",
}

POLICY_COLUMNS = (
    "strong_model_only",
    "sure_model_only",
    "ultra_sure_model_only",
    "sure_with_weak_region",
    "sure_with_weak_top1",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_int(value: object, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Expected integer for {name}, got {value!r}") from exc


def verify_policy_summary(summary: dict[str, Any], exported_rows: int) -> dict[str, int]:
    policy_counts = summary.get("sure_policy_counts")
    if not isinstance(policy_counts, dict):
        raise SystemExit("Final label summary is missing sure_policy_counts")
    missing = [name for name in POLICY_COLUMNS if name not in policy_counts]
    if missing:
        raise SystemExit(f"Final label summary is missing sure policy counts: {missing}")
    expected: dict[str, int] = {}
    for name in POLICY_COLUMNS:
        payload = policy_counts.get(name)
        if not isinstance(payload, dict):
            raise SystemExit(f"Final label summary policy count is malformed for {name}: {payload!r}")
        rows = as_int(payload.get("rows"), f"sure_policy_counts.{name}.rows")
        if rows < 0 or rows > exported_rows:
            raise SystemExit(f"Final label summary policy count out of range for {name}: {rows}")
        expected[name] = rows
    return expected


def count_policy_columns(manifest_rows: list[dict[str, str]], columns: list[str]) -> tuple[dict[str, int], int]:
    missing = [name for name in POLICY_COLUMNS if name not in columns]
    if missing:
        raise SystemExit(f"Cannot count missing policy columns: {missing}")
    counts = {name: 0 for name in POLICY_COLUMNS}
    rows_seen = 0
    for row in manifest_rows:
        file_path = Path(row["file"])
        with gzip.open(file_path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != columns:
                raise SystemExit(f"Final label shard header mismatch while counting policies: {file_path}")
            for sample in reader:
                rows_seen += 1
                for name in POLICY_COLUMNS:
                    value = str(sample.get(name, ""))
                    if value not in {"0", "1"}:
                        raise SystemExit(f"Invalid policy flag value in {file_path}: {name}={value!r}")
                    counts[name] += int(value)
    return counts, rows_seen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completeness-json", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--skip-header-check", action="store_true")
    parser.add_argument(
        "--check-policy-counts",
        action="store_true",
        help="Scan all final label shards and verify sure-policy flag sums against summary.json.",
    )
    args = parser.parse_args()

    completeness = read_json(args.completeness_json)
    summary_path = args.label_dir / "summary.json"
    manifest_path = args.label_dir / "manifest.csv"
    readme_path = args.label_dir / "README.md"
    success_path = args.label_dir / "_SUCCESS"
    in_progress_path = args.label_dir / "_EXPORT_IN_PROGRESS"
    lock_path = args.label_dir / "_EXPORT_LOCK"

    for path in (summary_path, manifest_path, readme_path, success_path):
        if not path.is_file():
            raise SystemExit(f"Missing final label artifact: {path}")
    if in_progress_path.exists():
        raise SystemExit(f"Final label export still marked in-progress: {in_progress_path}")
    if lock_path.exists():
        raise SystemExit(f"Final label export lock still exists: {lock_path}")

    summary = read_json(summary_path)
    success = read_json(success_path)
    expected_rows = int(completeness["sample_counts"]["decoded"])
    exported_rows = int(summary.get("exported_rows", -1))
    if exported_rows != expected_rows:
        raise SystemExit(f"Final label row mismatch: exported={exported_rows}, expected_decoded={expected_rows}")
    if int(success.get("exported_rows", -1)) != exported_rows:
        raise SystemExit(f"Final label success marker row mismatch: success={success}, summary_rows={exported_rows}")

    source_counts = summary.get("source_sample_counts", {})
    if int(source_counts.get("decoded", -1)) != expected_rows:
        raise SystemExit(f"Final label source decoded mismatch: {source_counts}")
    if int(source_counts.get("nondecoded", 0)) != 0:
        raise SystemExit(f"Final labels were exported from a tracker with nondecoded rows: {source_counts}")

    columns = list(summary.get("columns", []))
    missing_columns = sorted(REQUIRED_COLUMNS - set(columns))
    if missing_columns:
        raise SystemExit(f"Final label summary is missing required columns: {missing_columns}")
    expected_policy_counts = verify_policy_summary(summary, exported_rows)

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    if len(manifest_rows) != len(summary.get("files", [])):
        raise SystemExit(
            f"Final label manifest/file count mismatch: manifest={len(manifest_rows)}, "
            f"summary={len(summary.get('files', []))}"
        )

    row_sum = 0
    header_checked = 0
    for row in manifest_rows:
        file_path = Path(row["file"])
        if not file_path.is_file():
            raise SystemExit(f"Missing final label shard: {file_path}")
        expected_bytes = int(row["bytes"])
        actual_bytes = file_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise SystemExit(f"Final label shard size mismatch: {file_path}: actual={actual_bytes}, expected={expected_bytes}")
        row_sum += int(row["rows"])
        if not args.skip_header_check:
            with gzip.open(file_path, "rt", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, None)
            if header != columns:
                raise SystemExit(f"Final label shard header mismatch: {file_path}")
            header_checked += 1

    if row_sum != exported_rows:
        raise SystemExit(f"Final label manifest row sum mismatch: manifest_sum={row_sum}, exported={exported_rows}")

    policy_counts_checked = False
    observed_policy_counts = None
    if args.check_policy_counts:
        observed_policy_counts, policy_rows_seen = count_policy_columns(manifest_rows, columns)
        if policy_rows_seen != exported_rows:
            raise SystemExit(
                f"Final label policy scan row mismatch: scanned={policy_rows_seen}, exported={exported_rows}"
            )
        if observed_policy_counts != expected_policy_counts:
            raise SystemExit(
                "Final label policy count mismatch: "
                f"observed={observed_policy_counts}, summary={expected_policy_counts}"
            )
        policy_counts_checked = True

    payload = {
        "label_dir": str(args.label_dir),
        "final_label_rows": exported_rows,
        "final_label_files": len(manifest_rows),
        "manifest_row_sum": row_sum,
        "header_checked_files": header_checked,
        "sure_policy_counts": expected_policy_counts,
        "sure_policy_counts_checked": policy_counts_checked,
        "sure_policy_counts_observed": observed_policy_counts,
        "success_marker": str(success_path),
        "lock_absent": True,
    }
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
