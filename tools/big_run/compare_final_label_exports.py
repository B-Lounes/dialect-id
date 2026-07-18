#!/usr/bin/env python3
"""Parallel row-exact comparison for BIG-RUN final-label exports."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable


POLICIES = (
    "strong_model_only",
    "sure_model_only",
    "ultra_sure_model_only",
    "sure_with_weak_region",
    "sure_with_weak_top1",
)
KEY_FIELDS = ("sample_id", "tar_path", "member")
REQUIRED_FIELDS = (*KEY_FIELDS, "final_dialect", "top1_score", "margin", "entropy", *POLICIES)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Missing required JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def as_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def bit(row: dict[str, str], field: str) -> int:
    return 1 if str(row.get(field, "")) == "1" else 0


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return tuple(str(row.get(field, "")) for field in KEY_FIELDS)  # type: ignore[return-value]


def read_manifest(label_dir: Path) -> list[dict[str, str]]:
    path = label_dir / "manifest.csv"
    if not path.is_file():
        raise SystemExit(f"Missing final-label manifest: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"Empty final-label manifest: {path}")
    return rows


def iter_rows(path: Path) -> Iterable[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(REQUIRED_FIELDS) - set(reader.fieldnames or []))
        if missing:
            raise RuntimeError(f"Final-label shard is missing fields {missing}: {path}")
        yield from reader


def compare_shard(task: tuple[int, str, str, int]) -> dict[str, Any]:
    file_idx, base_file_s, candidate_file_s, expected_rows = task
    base_file = Path(base_file_s)
    candidate_file = Path(candidate_file_s)
    if not base_file.is_file():
        raise RuntimeError(f"Missing base final-label shard: {base_file}")
    if not candidate_file.is_file():
        raise RuntimeError(f"Missing candidate final-label shard: {candidate_file}")

    total = changed = same = 0
    base_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    change_counts: Counter[tuple[str, str]] = Counter()
    changed_by_base: Counter[str] = Counter()
    changed_by_candidate: Counter[str] = Counter()
    policy_transitions: dict[str, Counter[tuple[int, int]]] = {policy: Counter() for policy in POLICIES}
    changed_policy_counts: dict[str, Counter[tuple[int, int]]] = {policy: Counter() for policy in POLICIES}
    score_delta_sum = margin_delta_sum = entropy_delta_sum = 0.0
    changed_score_delta_sum = changed_margin_delta_sum = changed_entropy_delta_sum = 0.0

    for row_idx, (base_row, candidate_row) in enumerate(
        zip_longest(iter_rows(base_file), iter_rows(candidate_file)), start=1
    ):
        if base_row is None or candidate_row is None:
            raise RuntimeError(
                "Shard row count mismatch while streaming: "
                f"file_index={file_idx}, row={row_idx}, "
                f"base_exhausted={base_row is None}, candidate_exhausted={candidate_row is None}"
            )
        total += 1
        if row_key(base_row) != row_key(candidate_row):
            raise RuntimeError(
                "Row key mismatch: "
                f"file_index={file_idx}, row={row_idx}, "
                f"base_key={row_key(base_row)}, candidate_key={row_key(candidate_row)}"
            )
        base_label = str(base_row.get("final_dialect", "")).upper()
        candidate_label = str(candidate_row.get("final_dialect", "")).upper()
        base_counts[base_label] += 1
        candidate_counts[candidate_label] += 1

        score_delta = as_float(candidate_row.get("top1_score")) - as_float(base_row.get("top1_score"))
        margin_delta = as_float(candidate_row.get("margin")) - as_float(base_row.get("margin"))
        entropy_delta = as_float(candidate_row.get("entropy")) - as_float(base_row.get("entropy"))
        score_delta_sum += score_delta
        margin_delta_sum += margin_delta
        entropy_delta_sum += entropy_delta

        for policy in POLICIES:
            transition = (bit(base_row, policy), bit(candidate_row, policy))
            policy_transitions[policy][transition] += 1

        if base_label == candidate_label:
            same += 1
        else:
            changed += 1
            change_counts[(base_label, candidate_label)] += 1
            changed_by_base[base_label] += 1
            changed_by_candidate[candidate_label] += 1
            changed_score_delta_sum += score_delta
            changed_margin_delta_sum += margin_delta
            changed_entropy_delta_sum += entropy_delta
            for policy in POLICIES:
                transition = (bit(base_row, policy), bit(candidate_row, policy))
                changed_policy_counts[policy][transition] += 1

    if total != expected_rows:
        raise RuntimeError(f"Shard row count mismatch in {base_file}: observed={total}, manifest={expected_rows}")

    return {
        "file_idx": file_idx,
        "total": total,
        "same": same,
        "changed": changed,
        "base_counts": base_counts,
        "candidate_counts": candidate_counts,
        "change_counts": change_counts,
        "changed_by_base": changed_by_base,
        "changed_by_candidate": changed_by_candidate,
        "policy_transitions": policy_transitions,
        "changed_policy_counts": changed_policy_counts,
        "score_delta_sum": score_delta_sum,
        "margin_delta_sum": margin_delta_sum,
        "entropy_delta_sum": entropy_delta_sum,
        "changed_score_delta_sum": changed_score_delta_sum,
        "changed_margin_delta_sum": changed_margin_delta_sum,
        "changed_entropy_delta_sum": changed_entropy_delta_sum,
    }


def compact_counter(counter: Counter[Any], *, keys: tuple[str, ...], limit: int) -> list[dict[str, Any]]:
    rows = []
    for item, count in counter.most_common(limit):
        if not isinstance(item, tuple):
            item = (item,)
        row = {key: value for key, value in zip(keys, item)}
        row["rows"] = count
        rows.append(row)
    return rows


def fmt_pct(num: int, den: int) -> str:
    return "NA" if den == 0 else f"{100.0 * num / den:.4f}%"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        tmp = Path(handle.name)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-label-dir", type=Path, required=True)
    parser.add_argument("--candidate-label-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-changes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=max(1, min(32, os.cpu_count() or 1)))
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    start = time.time()
    base_summary = read_json(args.base_label_dir / "summary.json")
    candidate_summary = read_json(args.candidate_label_dir / "summary.json")
    base_rows_expected = as_int(base_summary.get("exported_rows"))
    candidate_rows_expected = as_int(candidate_summary.get("exported_rows"))
    if base_rows_expected <= 0 or candidate_rows_expected <= 0:
        raise SystemExit(f"Invalid exported row counts: base={base_rows_expected}, candidate={candidate_rows_expected}")
    if base_rows_expected != candidate_rows_expected:
        raise SystemExit(f"Final-label row count mismatch: base={base_rows_expected}, candidate={candidate_rows_expected}")

    base_manifest = read_manifest(args.base_label_dir)
    candidate_manifest = read_manifest(args.candidate_label_dir)
    if len(base_manifest) != len(candidate_manifest):
        raise SystemExit(f"Manifest file count mismatch: base={len(base_manifest)}, candidate={len(candidate_manifest)}")

    tasks: list[tuple[int, str, str, int]] = []
    for file_idx, (base_file_row, candidate_file_row) in enumerate(zip(base_manifest, candidate_manifest)):
        base_file = Path(str(base_file_row.get("file", "")))
        candidate_file = Path(str(candidate_file_row.get("file", "")))
        base_file_rows = as_int(base_file_row.get("rows"))
        candidate_file_rows = as_int(candidate_file_row.get("rows"))
        if base_file_rows != candidate_file_rows:
            raise SystemExit(
                f"Manifest shard row mismatch at index {file_idx}: base={base_file_rows}, candidate={candidate_file_rows}"
            )
        tasks.append((file_idx, str(base_file), str(candidate_file), base_file_rows))

    total = changed = same = 0
    base_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    change_counts: Counter[tuple[str, str]] = Counter()
    changed_by_base: Counter[str] = Counter()
    changed_by_candidate: Counter[str] = Counter()
    policy_transitions: dict[str, Counter[tuple[int, int]]] = {policy: Counter() for policy in POLICIES}
    changed_policy_counts: dict[str, Counter[tuple[int, int]]] = {policy: Counter() for policy in POLICIES}
    score_delta_sum = margin_delta_sum = entropy_delta_sum = 0.0
    changed_score_delta_sum = changed_margin_delta_sum = changed_entropy_delta_sum = 0.0

    workers = max(1, min(args.workers, len(tasks)))
    print(
        json.dumps(
            {
                "event": "start_parallel_compare",
                "base_label_dir": str(args.base_label_dir),
                "candidate_label_dir": str(args.candidate_label_dir),
                "shards": len(tasks),
                "workers": workers,
                "rows_expected": base_rows_expected,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(compare_shard, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            total += int(result["total"])
            same += int(result["same"])
            changed += int(result["changed"])
            base_counts.update(result["base_counts"])
            candidate_counts.update(result["candidate_counts"])
            change_counts.update(result["change_counts"])
            changed_by_base.update(result["changed_by_base"])
            changed_by_candidate.update(result["changed_by_candidate"])
            for policy in POLICIES:
                policy_transitions[policy].update(result["policy_transitions"][policy])
                changed_policy_counts[policy].update(result["changed_policy_counts"][policy])
            score_delta_sum += float(result["score_delta_sum"])
            margin_delta_sum += float(result["margin_delta_sum"])
            entropy_delta_sum += float(result["entropy_delta_sum"])
            changed_score_delta_sum += float(result["changed_score_delta_sum"])
            changed_margin_delta_sum += float(result["changed_margin_delta_sum"])
            changed_entropy_delta_sum += float(result["changed_entropy_delta_sum"])
            if completed % args.progress_every == 0 or completed == len(tasks):
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "completed_shards": completed,
                            "total_shards": len(tasks),
                            "rows_compared": total,
                            "elapsed_sec": round(time.time() - start, 1),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    if total != base_rows_expected:
        raise SystemExit(f"Compared row count mismatch: compared={total}, expected={base_rows_expected}")

    policy_payload = {}
    for policy in POLICIES:
        transitions = policy_transitions[policy]
        policy_payload[policy] = {
            "base_false_candidate_false": transitions[(0, 0)],
            "base_false_candidate_true": transitions[(0, 1)],
            "base_true_candidate_false": transitions[(1, 0)],
            "base_true_candidate_true": transitions[(1, 1)],
            "net_gain": transitions[(0, 1)] - transitions[(1, 0)],
            "changed_rows": {
                "base_false_candidate_false": changed_policy_counts[policy][(0, 0)],
                "base_false_candidate_true": changed_policy_counts[policy][(0, 1)],
                "base_true_candidate_false": changed_policy_counts[policy][(1, 0)],
                "base_true_candidate_true": changed_policy_counts[policy][(1, 1)],
            },
        }

    top_change_rows = compact_counter(change_counts, keys=("base_final_dialect", "candidate_final_dialect"), limit=args.top_changes)
    for row in top_change_rows:
        row["pct_of_all"] = 0.0 if total == 0 else 100.0 * int(row["rows"]) / total
        row["pct_of_changed"] = 0.0 if changed == 0 else 100.0 * int(row["rows"]) / changed

    payload = {
        "base_label_dir": str(args.base_label_dir),
        "candidate_label_dir": str(args.candidate_label_dir),
        "rows_compared": total,
        "unchanged_rows": same,
        "changed_rows": changed,
        "changed_pct": 0.0 if total == 0 else 100.0 * changed / total,
        "key_mismatches": 0,
        "avg_score_delta": 0.0 if total == 0 else score_delta_sum / total,
        "avg_margin_delta": 0.0 if total == 0 else margin_delta_sum / total,
        "avg_entropy_delta": 0.0 if total == 0 else entropy_delta_sum / total,
        "changed_avg_score_delta": 0.0 if changed == 0 else changed_score_delta_sum / changed,
        "changed_avg_margin_delta": 0.0 if changed == 0 else changed_margin_delta_sum / changed,
        "changed_avg_entropy_delta": 0.0 if changed == 0 else changed_entropy_delta_sum / changed,
        "base_counts": dict(sorted(base_counts.items())),
        "candidate_counts": dict(sorted(candidate_counts.items())),
        "policy_transitions": policy_payload,
        "top_label_changes": top_change_rows,
        "changed_by_base": compact_counter(changed_by_base, keys=("base_final_dialect",), limit=args.top_changes),
        "changed_by_candidate": compact_counter(changed_by_candidate, keys=("candidate_final_dialect",), limit=args.top_changes),
        "parallel_compare": {
            "workers": workers,
            "elapsed_sec": time.time() - start,
            "script": str(Path(__file__).resolve()),
        },
        "notes": [
            "Rows are compared in final-label manifest order and every sample_id/tar_path/member key is validated.",
            "candidate is expected to be the refined single-model export.",
            "This report compares model outputs; it does not treat Qwen/path metadata as ground truth.",
        ],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(args.out_dir / "final_label_export_comparison.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_csv(
        args.out_dir / "top_label_changes.csv",
        top_change_rows,
        ["base_final_dialect", "candidate_final_dialect", "rows", "pct_of_all", "pct_of_changed"],
    )
    lines = [
        "# BIG-RUN Final Label Export Comparison",
        "",
        f"- Base labels: `{args.base_label_dir}`",
        f"- Candidate labels: `{args.candidate_label_dir}`",
        f"- Compared rows: `{total:,}`",
        f"- Changed labels: `{changed:,}` ({fmt_pct(changed, total)})",
        f"- Average score delta: `{payload['avg_score_delta']:.6f}`",
        f"- Average margin delta: `{payload['avg_margin_delta']:.6f}`",
        f"- Average entropy delta: `{payload['avg_entropy_delta']:.6f}`",
        f"- Parallel workers: `{workers}`",
        "",
        "## Policy Net Gains",
        "",
        "| Policy | Gained | Lost | Net | Stayed true |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for policy in POLICIES:
        row = policy_payload[policy]
        lines.append(
            f"| {policy} | {row['base_false_candidate_true']:,} | {row['base_true_candidate_false']:,} | "
            f"{row['net_gain']:,} | {row['base_true_candidate_true']:,} |"
        )
    lines.extend(
        [
            "",
            "## Top Label Changes",
            "",
            "| Base | Candidate | Rows | % all | % changed |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in top_change_rows[:25]:
        lines.append(
            f"| {row['base_final_dialect']} | {row['candidate_final_dialect']} | {int(row['rows']):,} | "
            f"{float(row['pct_of_all']):.4f}% | {float(row['pct_of_changed']):.4f}% |"
        )
    lines.append("")
    atomic_write_text(args.out_dir / "final_label_export_comparison.md", "\n".join(lines))
    print(
        json.dumps(
            {
                "event": "complete",
                "out_dir": str(args.out_dir),
                "rows_compared": total,
                "changed_rows": changed,
                "changed_pct": payload["changed_pct"],
                "elapsed_sec": round(time.time() - start, 1),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
