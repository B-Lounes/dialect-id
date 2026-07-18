#!/usr/bin/env python3
"""Verify the completed BIG-RUN final single-model artifact set.

The Qwen/path report is a subset sanity check by default.  Pass
``--require-qwen-full-coverage`` only when the Qwen input is expected to contain
one row for every final decoded sample.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def require_file(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"Missing required artifact: {path}")


def as_int(value: object, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Expected integer for {name}, got {value!r}") from exc


def ensure_clean_completeness(payload: dict[str, Any]) -> tuple[int, int]:
    bad_shards = {k: v for k, v in payload.get("shard_status", {}).items() if k != "done"}
    bad_tars = {k: v for k, v in payload.get("tar_status", {}).items() if k != "done"}
    sample_counts = payload.get("sample_counts", {})
    decoded = as_int(sample_counts.get("decoded"), "sample_counts.decoded")
    nondecoded = as_int(sample_counts.get("nondecoded", 0), "sample_counts.nondecoded")
    total = as_int(sample_counts.get("all"), "sample_counts.all")
    expected_shards = payload.get("expected_shards")
    done_shards = payload.get("done_shards")
    failed_shards = as_int(payload.get("failed_shards", 0), "completeness.failed_shards")
    missing_shards = as_int(payload.get("missing_shards", 0), "completeness.missing_shards")
    bad_done_markers = as_int(payload.get("bad_done_markers", 0), "completeness.bad_done_markers")
    csv_without_done = as_int(payload.get("csv_without_done_count", 0), "completeness.csv_without_done_count")
    done_without_csv = as_int(payload.get("done_without_csv_count", 0), "completeness.done_without_csv_count")
    if expected_shards is not None and done_shards is not None:
        expected_i = as_int(expected_shards, "completeness.expected_shards")
        done_i = as_int(done_shards, "completeness.done_shards")
        if expected_i <= 0 or done_i != expected_i:
            raise SystemExit(f"Shard completeness mismatch: done={done_i}, expected={expected_i}")
    if bad_shards or bad_tars or nondecoded or failed_shards or missing_shards or bad_done_markers or csv_without_done or done_without_csv:
        raise SystemExit(
            "Final run is not clean: "
            f"bad_shards={bad_shards}, bad_tars={bad_tars}, nondecoded_samples={nondecoded}, "
            f"failed_shards={failed_shards}, missing_shards={missing_shards}, "
            f"bad_done_markers={bad_done_markers}, csv_without_done={csv_without_done}, "
            f"done_without_csv={done_without_csv}"
        )
    if total != decoded:
        raise SystemExit(f"Final sample count mismatch: all={total}, decoded={decoded}")
    return total, decoded


def verify_sync_summary(sync_summary: dict[str, Any], decoded_rows: int) -> None:
    csv_rows = as_int(sync_summary.get("csv_sample_rows_seen"), "sync_summary.csv_sample_rows_seen")
    csv_decoded = as_int(sync_summary.get("csv_decoded_rows_seen"), "sync_summary.csv_decoded_rows_seen")
    csv_failed = as_int(sync_summary.get("csv_failed_rows_seen"), "sync_summary.csv_failed_rows_seen")
    failed_markers = as_int(sync_summary.get("failed_markers"), "sync_summary.failed_markers")
    missing_shards = as_int(sync_summary.get("missing_shards", 0), "sync_summary.missing_shards")
    bad_done_markers = as_int(sync_summary.get("bad_done_markers", 0), "sync_summary.bad_done_markers")
    csv_without_done = as_int(sync_summary.get("csv_without_done_count", 0), "sync_summary.csv_without_done_count")
    done_without_csv = as_int(sync_summary.get("done_without_csv_count", 0), "sync_summary.done_without_csv_count")
    expected_shards = sync_summary.get("expected_shards")
    done_markers = sync_summary.get("done_markers")
    if expected_shards is not None and done_markers is not None:
        expected_i = as_int(expected_shards, "sync_summary.expected_shards")
        done_i = as_int(done_markers, "sync_summary.done_markers")
        if expected_i <= 0 or done_i != expected_i:
            raise SystemExit(f"Sync summary shard mismatch: done={done_i}, expected={expected_i}")
    if csv_rows != decoded_rows or csv_decoded != decoded_rows:
        raise SystemExit(
            f"Sync summary row mismatch: csv_rows={csv_rows}, csv_decoded={csv_decoded}, decoded={decoded_rows}"
        )
    if csv_failed or failed_markers or missing_shards or bad_done_markers or csv_without_done or done_without_csv:
        raise SystemExit(
            "Sync summary reports incomplete/failed export: "
            f"csv_failed={csv_failed}, failed_markers={failed_markers}, missing_shards={missing_shards}, "
            f"bad_done_markers={bad_done_markers}, csv_without_done={csv_without_done}, done_without_csv={done_without_csv}"
        )


def verify_decision_summary(decision_summary: dict[str, Any], decoded_rows: int) -> None:
    decoded = as_int(decision_summary.get("decoded_rows"), "decision_summary.decoded_rows")
    nondecoded = as_int(decision_summary.get("nondecoded_rows", 0), "decision_summary.nondecoded_rows")
    overall = decision_summary.get("overall", {})
    overall_n = as_int(overall.get("n"), "decision_summary.overall.n")
    by_pred = decision_summary.get("by_predicted_dialect", [])
    by_pred_sum = sum(as_int(row.get("n"), "by_predicted_dialect.n") for row in by_pred)
    policies = decision_summary.get("for_sure_policies", [])
    policy_names = {str(row.get("policy", "")) for row in policies}
    required_policies = {
        "strong_model_only",
        "sure_model_only",
        "ultra_sure_model_only",
        "sure_with_weak_region",
        "sure_with_weak_top1",
    }
    if decoded != decoded_rows or overall_n != decoded_rows or by_pred_sum != decoded_rows:
        raise SystemExit(
            "Decision summary row mismatch: "
            f"decoded={decoded}, overall={overall_n}, by_pred_sum={by_pred_sum}, expected={decoded_rows}"
        )
    if nondecoded:
        raise SystemExit(f"Decision summary reports nondecoded rows: {nondecoded}")
    if not required_policies.issubset(policy_names):
        raise SystemExit(f"Decision summary missing for-sure policies: {sorted(required_policies - policy_names)}")


def verify_decision_card(decision_card: dict[str, Any], decoded_rows: int) -> None:
    card_rows = as_int(decision_card.get("decoded_rows"), "decision_card.decoded_rows")
    if card_rows != decoded_rows:
        raise SystemExit(f"Decision card row mismatch: card={card_rows}, expected={decoded_rows}")
    if decision_card.get("recommended_primary_policy") != "sure_model_only":
        raise SystemExit(f"Unexpected decision card primary policy: {decision_card.get('recommended_primary_policy')!r}")
    if decision_card.get("recommended_strict_policy") != "ultra_sure_model_only":
        raise SystemExit(f"Unexpected decision card strict policy: {decision_card.get('recommended_strict_policy')!r}")
    policy_counts = decision_card.get("policy_counts", {})
    required = ("sure_model_only", "ultra_sure_model_only", "sure_with_weak_region", "sure_with_weak_top1")
    missing = [name for name in required if name not in policy_counts]
    if missing:
        raise SystemExit(f"Decision card missing policy counts: {missing}")
    dialects = decision_card.get("dialects", [])
    if not dialects:
        raise SystemExit("Decision card has no dialect rows")
    dialect_total = sum(as_int(row.get("total_rows"), "decision_card.dialects.total_rows") for row in dialects)
    if dialect_total != decoded_rows:
        raise SystemExit(f"Decision card dialect total mismatch: total={dialect_total}, expected={decoded_rows}")


def verify_label_verification(label_verification: dict[str, Any], decoded_rows: int) -> None:
    rows = as_int(label_verification.get("final_label_rows"), "final_label_verification.final_label_rows")
    row_sum = as_int(label_verification.get("manifest_row_sum"), "final_label_verification.manifest_row_sum")
    files = as_int(label_verification.get("final_label_files"), "final_label_verification.final_label_files")
    headers = as_int(label_verification.get("header_checked_files"), "final_label_verification.header_checked_files")
    if rows != decoded_rows or row_sum != decoded_rows:
        raise SystemExit(
            f"Final label verification row mismatch: rows={rows}, manifest_sum={row_sum}, expected={decoded_rows}"
        )
    if files <= 0 or headers != files:
        raise SystemExit(f"Final label verification file/header mismatch: files={files}, headers={headers}")
    if label_verification.get("lock_absent") is not True:
        raise SystemExit("Final label verification did not prove lock_absent=true")


def verify_label_summary(label_summary: dict[str, Any], decoded_rows: int) -> None:
    exported = as_int(label_summary.get("exported_rows"), "final_labels.summary.exported_rows")
    source = label_summary.get("source_sample_counts", {})
    source_decoded = as_int(source.get("decoded"), "final_labels.summary.source_sample_counts.decoded")
    source_nondecoded = as_int(source.get("nondecoded", 0), "final_labels.summary.source_sample_counts.nondecoded")
    by_dialect = label_summary.get("by_final_dialect", [])
    by_dialect_sum = sum(as_int(row.get("rows"), "final_labels.summary.by_final_dialect.rows") for row in by_dialect)
    if exported != decoded_rows or source_decoded != decoded_rows or by_dialect_sum != decoded_rows:
        raise SystemExit(
            "Final label summary row mismatch: "
            f"exported={exported}, source_decoded={source_decoded}, by_dialect_sum={by_dialect_sum}, expected={decoded_rows}"
        )
    if source_nondecoded:
        raise SystemExit(f"Final label summary reports nondecoded source rows: {source_nondecoded}")


def verify_qwen_report(qwen_report: dict[str, Any], decoded_rows: int, *, require_full_coverage: bool) -> int:
    merged = qwen_report.get("merged", {})
    streamed = as_int(merged.get("streamed_rows"), "qwen_report.merged.streamed_rows")
    rows_with_keys = as_int(merged.get("rows_with_keys"), "qwen_report.merged.rows_with_keys")
    matched = as_int(merged.get("matched_rows"), "qwen_report.merged.matched_rows")
    decoded = as_int(merged.get("decoded_rows"), "qwen_report.merged.decoded_rows")
    missing = as_int(merged.get("missing_rows", 0), "qwen_report.merged.missing_rows")
    nondecoded = as_int(merged.get("nondecoded_rows", 0), "qwen_report.merged.nondecoded_rows")
    bad_json = as_int(merged.get("bad_json_rows", 0), "qwen_report.merged.bad_json_rows")
    missing_key = as_int(merged.get("missing_key_rows", 0), "qwen_report.merged.missing_key_rows")
    if min(streamed, rows_with_keys, matched, decoded) <= 0:
        raise SystemExit(f"Qwen sanity report has empty counts: {merged}")
    if streamed != rows_with_keys or matched != rows_with_keys or decoded != matched:
        raise SystemExit(
            "Qwen sanity report row mismatch: "
            f"streamed={streamed}, keys={rows_with_keys}, matched={matched}, decoded={decoded}"
        )
    if missing or nondecoded or bad_json or missing_key:
        raise SystemExit(
            "Qwen sanity report has bad rows: "
            f"missing={missing}, nondecoded={nondecoded}, bad_json={bad_json}, missing_key={missing_key}"
        )
    if matched > decoded_rows:
        raise SystemExit(f"Qwen sanity report exceeds final decoded rows: matched={matched}, final_decoded={decoded_rows}")
    if require_full_coverage and matched != decoded_rows:
        raise SystemExit(f"Qwen sanity report coverage mismatch: matched={matched}, final_decoded={decoded_rows}")
    return matched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument(
        "--require-qwen-report",
        action="store_true",
        help="Require Qwen/path sanity artifacts to exist and pass row checks.",
    )
    parser.add_argument(
        "--require-qwen-full-coverage",
        action="store_true",
        help="Require the Qwen/path sanity report to cover every final decoded row.",
    )
    args = parser.parse_args()

    reports = args.out_root / "reports"
    final_labels = args.out_root / "final_labels"
    qwen_dir = reports / "qwenft_sanity_ranked"
    final_decisions = reports / "final_decisions"

    completeness_path = reports / "completeness_after_sync.json"
    sync_summary_path = reports / "sync_summary.json"
    label_verification_path = reports / "final_labels_verification.json"
    label_summary_path = final_labels / "summary.json"
    decision_summary_path = final_decisions / "final_dialect_decision_summary.json"
    decision_md_path = final_decisions / "final_dialect_decision_summary.md"
    decision_card_dir = final_decisions / "decision_card"
    decision_card_json_path = decision_card_dir / "decision_card.json"
    decision_card_md_path = decision_card_dir / "decision_card.md"
    decision_card_by_dialect_path = decision_card_dir / "decision_card_by_dialect.csv"
    decision_card_review_path = decision_card_dir / "decision_card_review_priority.csv"
    qwen_json_path = qwen_dir / "qwenft_vs_final_single_model.json"
    qwen_md_path = qwen_dir / "qwenft_vs_final_single_model.md"

    require_file(decision_md_path)
    require_file(decision_card_md_path)
    require_file(decision_card_by_dialect_path)
    require_file(decision_card_review_path)

    completeness = read_json(completeness_path)
    sync_summary = read_json(sync_summary_path)
    label_verification = read_json(label_verification_path)
    label_summary = read_json(label_summary_path)
    decision_summary = read_json(decision_summary_path)
    decision_card = read_json(decision_card_json_path)

    total_rows, decoded_rows = ensure_clean_completeness(completeness)
    verify_sync_summary(sync_summary, decoded_rows)
    verify_label_verification(label_verification, decoded_rows)
    verify_label_summary(label_summary, decoded_rows)
    verify_decision_summary(decision_summary, decoded_rows)
    verify_decision_card(decision_card, decoded_rows)
    qwen_matched_rows = None
    if qwen_json_path.is_file() or qwen_md_path.is_file() or args.require_qwen_report:
        require_file(qwen_json_path)
        require_file(qwen_md_path)
        qwen_report = read_json(qwen_json_path)
        qwen_matched_rows = verify_qwen_report(
            qwen_report,
            decoded_rows,
            require_full_coverage=args.require_qwen_full_coverage,
        )

    payload = {
        "out_root": str(args.out_root),
        "sample_rows": total_rows,
        "decoded_rows": decoded_rows,
        "final_label_files": as_int(label_verification.get("final_label_files"), "final_label_files"),
        "qwen_matched_rows": qwen_matched_rows,
        "qwen_coverage_pct": None if qwen_matched_rows is None else (100.0 * qwen_matched_rows / decoded_rows if decoded_rows else 0.0),
        "qwen_report_required": bool(args.require_qwen_report),
        "qwen_full_coverage_required": bool(args.require_qwen_full_coverage),
        "artifacts": {
            "completeness": str(completeness_path),
            "sync_summary": str(sync_summary_path),
            "final_labels_verification": str(label_verification_path),
            "final_labels_summary": str(label_summary_path),
            "final_decision_summary": str(decision_summary_path),
            "final_decision_card": str(decision_card_json_path),
            "qwen_sanity_json": str(qwen_json_path) if qwen_json_path.is_file() else None,
            "qwen_sanity_md": str(qwen_md_path) if qwen_md_path.is_file() else None,
        },
    }
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
