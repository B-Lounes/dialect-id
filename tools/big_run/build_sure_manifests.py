#!/usr/bin/env python3
"""Build review/training manifests from BIG-RUN final label policy flags."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_POLICIES = (
    "sure_model_only",
    "ultra_sure_model_only",
    "sure_with_weak_region",
    "sure_with_weak_top1",
)

OUTPUT_FIELDS = [
    "policy",
    "sample_id",
    "manifest_row",
    "tar_path",
    "member",
    "youtube_id",
    "duration",
    "final_dialect",
    "final_country",
    "final_region",
    "confidence_tier",
    "top1_score",
    "margin",
    "entropy",
    "weak_cc",
    "weak_region",
    "weak_top1_match",
    "weak_region_match",
    "source_prediction_csv",
]


class GzipCsvWriter:
    def __init__(self, path: Path, fields: list[str], *, gzip_level: int) -> None:
        self.path = path
        self.rows = 0
        self._handle = gzip.open(path, "wt", encoding="utf-8", newline="", compresslevel=gzip_level)
        self._writer = csv.DictWriter(self._handle, fieldnames=fields)
        self._writer.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        self._writer.writerow(row)
        self.rows += 1

    def close(self) -> None:
        self._handle.close()


def parse_csv_list(value: str | None) -> set[str] | None:
    if value is None or not value.strip():
        return None
    return {item.strip().upper() for item in value.split(",") if item.strip()}


def parse_policy_list(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return list(DEFAULT_POLICIES)
    return [item.strip() for item in value.split(",") if item.strip()]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_clean_out_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise SystemExit(f"Output dir is not empty: {out_dir}; pass --overwrite to replace it")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def read_label_manifest(label_dir: Path) -> list[dict[str, str]]:
    manifest_path = label_dir / "manifest.csv"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing final label manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"Final label manifest is empty: {manifest_path}")
    return rows


def output_name(policy: str, dialect: str) -> str:
    safe_policy = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in policy)
    safe_dialect = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in dialect)
    return f"{safe_policy}__{safe_dialect}.csv.gz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--dialects", default=None, help="Optional comma-separated final dialect filter.")
    parser.add_argument(
        "--max-rows-per-policy-dialect",
        type=int,
        default=0,
        help="Optional cap per policy+dialect; 0 means no cap.",
    )
    parser.add_argument("--gzip-level", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    start = time.time()
    policies = parse_policy_list(args.policies)
    dialect_filter = parse_csv_list(args.dialects)
    require_clean_out_dir(args.out_dir, args.overwrite)
    manifest_rows = read_label_manifest(args.label_dir)
    label_summary = read_json(args.label_dir / "summary.json")
    columns = set(label_summary.get("columns", []))
    missing = [policy for policy in policies if policy not in columns]
    if missing:
        raise SystemExit(f"Final label summary is missing requested policy columns: {missing}")

    manifest_dir = args.out_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[tuple[str, str], GzipCsvWriter] = {}
    writer_rows: Counter[tuple[str, str]] = Counter()
    seen_rows = 0
    emitted_rows = 0
    skipped_by_cap = 0
    skipped_by_dialect = 0
    policy_counts: Counter[str] = Counter()
    dialect_counts: Counter[str] = Counter()
    policy_dialect_counts: Counter[tuple[str, str]] = Counter()

    try:
        for manifest_row in manifest_rows:
            shard_path = Path(manifest_row["file"])
            with gzip.open(shard_path, "rt", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                missing_fields = [field for field in OUTPUT_FIELDS if field != "policy" and field not in (reader.fieldnames or [])]
                if missing_fields:
                    raise SystemExit(f"Final label shard is missing fields {missing_fields}: {shard_path}")
                for row in reader:
                    seen_rows += 1
                    dialect = str(row.get("final_dialect") or "MISSING").upper()
                    if dialect_filter is not None and dialect not in dialect_filter:
                        skipped_by_dialect += 1
                        continue
                    for policy in policies:
                        if str(row.get(policy, "")) != "1":
                            continue
                        key = (policy, dialect)
                        if args.max_rows_per_policy_dialect > 0 and writer_rows[key] >= args.max_rows_per_policy_dialect:
                            skipped_by_cap += 1
                            continue
                        writer = writers.get(key)
                        if writer is None:
                            path = manifest_dir / output_name(policy, dialect)
                            writer = GzipCsvWriter(path, OUTPUT_FIELDS, gzip_level=args.gzip_level)
                            writers[key] = writer
                        out_row = {field: row.get(field, "") for field in OUTPUT_FIELDS if field != "policy"}
                        out_row["policy"] = policy
                        writer.write(out_row)
                        writer_rows[key] += 1
                        emitted_rows += 1
                        policy_counts[policy] += 1
                        dialect_counts[dialect] += 1
                        policy_dialect_counts[key] += 1
    finally:
        for writer in writers.values():
            writer.close()

    output_manifest_rows = []
    for (policy, dialect), rows in sorted(writer_rows.items()):
        path = manifest_dir / output_name(policy, dialect)
        output_manifest_rows.append(
            {
                "policy": policy,
                "final_dialect": dialect,
                "rows": rows,
                "file": str(path),
                "bytes": path.stat().st_size,
            }
        )

    with (args.out_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["policy", "final_dialect", "rows", "file", "bytes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_manifest_rows)

    summary = {
        "label_dir": str(args.label_dir),
        "out_dir": str(args.out_dir),
        "generated_at_epoch": time.time(),
        "wall_sec": time.time() - start,
        "source_label_rows": int(label_summary.get("exported_rows", 0)),
        "scanned_rows": seen_rows,
        "emitted_rows": emitted_rows,
        "policies": policies,
        "dialect_filter": sorted(dialect_filter) if dialect_filter else None,
        "max_rows_per_policy_dialect": args.max_rows_per_policy_dialect,
        "skipped_by_cap": skipped_by_cap,
        "skipped_by_dialect": skipped_by_dialect,
        "policy_counts": dict(policy_counts),
        "dialect_counts": dict(dialect_counts),
        "policy_dialect_counts": [
            {"policy": policy, "final_dialect": dialect, "rows": rows}
            for (policy, dialect), rows in sorted(policy_dialect_counts.items())
        ],
        "files": output_manifest_rows,
        "columns": OUTPUT_FIELDS,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# BIG-RUN For-Sure Manifests",
        "",
        f"- Source label dir: `{args.label_dir}`",
        f"- Scanned rows: `{seen_rows:,}`",
        f"- Emitted rows: `{emitted_rows:,}`",
        f"- Policies: `{', '.join(policies)}`",
        f"- Max rows per policy+dialect: `{args.max_rows_per_policy_dialect}`",
        "",
        "Each manifest row keeps the audio locator and the single-model final dialect label.",
        "Metadata-backed policies are audit subsets, not alternate labels.",
        "",
        "## Output Files",
        "",
        "| Policy | Dialect | Rows | File |",
        "| --- | --- | ---: | --- |",
    ]
    for row in output_manifest_rows:
        lines.append(f"| {row['policy']} | {row['final_dialect']} | {int(row['rows']):,} | `{row['file']}` |")
    lines.append("")
    (args.out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "scanned_rows": seen_rows,
                "emitted_rows": emitted_rows,
                "files": len(output_manifest_rows),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
