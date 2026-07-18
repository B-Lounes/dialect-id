#!/usr/bin/env python3
"""Verify optional BIG-RUN for-sure manifest outputs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {
    "policy",
    "sample_id",
    "tar_path",
    "member",
    "duration",
    "final_dialect",
    "confidence_tier",
    "top1_score",
    "margin",
    "entropy",
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def as_int(value: object, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Expected integer for {name}, got {value!r}") from exc


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"Missing sure-manifest file list: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"Sure-manifest file list is empty: {path}")
    return rows


def expected_policy_counts(label_summary: dict[str, Any], policies: list[str]) -> dict[str, int] | None:
    counts = label_summary.get("sure_policy_counts")
    if not isinstance(counts, dict):
        return None
    expected: dict[str, int] = {}
    for policy in policies:
        payload = counts.get(policy)
        if not isinstance(payload, dict):
            return None
        expected[policy] = as_int(payload.get("rows"), f"final_labels.summary.sure_policy_counts.{policy}.rows")
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sure-dir", type=Path, required=True)
    parser.add_argument("--label-summary-json", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument(
        "--expect-uncapped",
        action="store_true",
        help="Require policy counts to match final label summary exactly. Use only when no cap/filter was used.",
    )
    args = parser.parse_args()

    summary = read_json(args.sure_dir / "summary.json")
    manifest_rows = read_manifest(args.sure_dir / "manifest.csv")
    policies = [str(policy) for policy in summary.get("policies", [])]
    max_rows_per_policy_dialect = as_int(summary.get("max_rows_per_policy_dialect", 0), "summary.max_rows_per_policy_dialect")
    skipped_by_cap = as_int(summary.get("skipped_by_cap", 0), "summary.skipped_by_cap")
    emitted_rows = as_int(summary.get("emitted_rows"), "summary.emitted_rows")
    scanned_rows = as_int(summary.get("scanned_rows"), "summary.scanned_rows")
    if scanned_rows < 0 or emitted_rows <= 0:
        raise SystemExit(f"Unexpected sure-manifest row counts: scanned={scanned_rows}, emitted={emitted_rows}")

    manifest_sum = 0
    observed_policy_counts: dict[str, int] = {policy: 0 for policy in policies}
    observed_policy_dialect_counts: dict[tuple[str, str], int] = {}
    header_checked_files = 0
    for row in manifest_rows:
        policy = str(row.get("policy", ""))
        dialect = str(row.get("final_dialect", ""))
        if policy not in policies:
            raise SystemExit(f"Manifest row has unexpected policy {policy!r}: {row}")
        rows_expected = as_int(row.get("rows"), "manifest.rows")
        file_path = Path(str(row.get("file", "")))
        if not file_path.is_file():
            raise SystemExit(f"Missing sure-manifest shard: {file_path}")
        expected_bytes = as_int(row.get("bytes"), "manifest.bytes")
        actual_bytes = file_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise SystemExit(f"Sure-manifest shard size mismatch: {file_path}: actual={actual_bytes}, expected={expected_bytes}")
        file_rows = 0
        with gzip.open(file_path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or []))
            if missing:
                raise SystemExit(f"Sure-manifest shard missing required columns {missing}: {file_path}")
            header_checked_files += 1
            for sample in reader:
                file_rows += 1
                if str(sample.get("policy", "")) != policy:
                    raise SystemExit(f"Policy mismatch in {file_path}: expected={policy}, row={sample.get('policy')}")
                if str(sample.get("final_dialect", "")) != dialect:
                    raise SystemExit(
                        f"Final dialect mismatch in {file_path}: expected={dialect}, row={sample.get('final_dialect')}"
                    )
                if not sample.get("tar_path") or not sample.get("member") or not sample.get("sample_id"):
                    raise SystemExit(f"Missing locator/sample id in {file_path}: {sample}")
        if file_rows != rows_expected:
            raise SystemExit(f"Sure-manifest shard row mismatch: {file_path}: rows={file_rows}, expected={rows_expected}")
        manifest_sum += file_rows
        observed_policy_counts[policy] = observed_policy_counts.get(policy, 0) + file_rows
        observed_policy_dialect_counts[(policy, dialect)] = file_rows

    if manifest_sum != emitted_rows:
        raise SystemExit(f"Sure-manifest emitted row mismatch: manifest_sum={manifest_sum}, summary={emitted_rows}")
    summary_policy_counts = {str(k): as_int(v, f"summary.policy_counts.{k}") for k, v in summary.get("policy_counts", {}).items()}
    if observed_policy_counts != summary_policy_counts:
        raise SystemExit(
            f"Sure-manifest policy counts mismatch: observed={observed_policy_counts}, summary={summary_policy_counts}"
        )

    label_expected_counts = None
    if args.label_summary_json:
        label_summary = read_json(args.label_summary_json)
        label_expected_counts = expected_policy_counts(label_summary, policies)
        if args.expect_uncapped:
            if max_rows_per_policy_dialect != 0 or skipped_by_cap != 0 or summary.get("dialect_filter") is not None:
                raise SystemExit("Cannot require uncapped counts: sure manifest was capped or dialect-filtered")
            if label_expected_counts is None:
                raise SystemExit("Final label summary does not contain comparable sure_policy_counts")
            if observed_policy_counts != label_expected_counts:
                raise SystemExit(
                    "Sure-manifest counts do not match final label summary: "
                    f"observed={observed_policy_counts}, expected={label_expected_counts}"
                )

    payload = {
        "sure_dir": str(args.sure_dir),
        "scanned_rows": scanned_rows,
        "emitted_rows": emitted_rows,
        "manifest_row_sum": manifest_sum,
        "files": len(manifest_rows),
        "header_checked_files": header_checked_files,
        "policies": policies,
        "policy_counts": observed_policy_counts,
        "label_summary_policy_counts": label_expected_counts,
        "expect_uncapped": bool(args.expect_uncapped),
    }
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
