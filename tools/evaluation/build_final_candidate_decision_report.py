#!/usr/bin/env python3
"""Build a concise final-candidate decision report from the 23-way scorecard."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", required=True)
    parser.add_argument("--acceptance-audit", required=True)
    parser.add_argument("--review-gate-status", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args()


def load_json(path: str | Path, *, optional: bool = False) -> Any:
    path = Path(path)
    if optional and not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if value is None:
        return "-"
    return str(value)


def status_label(row: dict[str, Any]) -> str:
    checks = [
        (row.get("old19_check") or {}).get("status"),
        (row.get("td_check") or {}).get("status"),
        (row.get("verified_east_check") or {}).get("status"),
    ]
    if row.get("ready"):
        return "ready"
    return f"{sum(check == 'pass' for check in checks)}/3 checks"


def next_action(scorecard: dict[str, Any], gate: dict[str, Any], top: dict[str, Any] | None) -> str:
    review_gate = scorecard.get("review_gate") or {}
    if scorecard.get("ready_for_final_23way_acceptance"):
        return "accept_or_promote_top_final_candidate_after_manual_review_of_report"
    if not review_gate.get("recommended_ready"):
        return "export_or_finish_reviewed_dj_km_so_csv_until_recommended_gate_is_ready"
    if not top:
        return "run_evaluations_for_candidate_checkpoints"
    checks = gate.get("top_checks") or {}
    missing = set(gate.get("top_missing_splits") or [])
    if "verified_dj_km_so_heldout_23way" in missing:
        return "train_or_eval_candidate_on_verified_dj_km_so_heldout_then_refresh_acceptance"
    if (checks.get("verified_east") or {}).get("status") != "pass":
        return "inspect_verified_dj_km_so_errors_or_train_a_better_candidate"
    if (checks.get("old19") or {}).get("status") != "pass":
        return "reject_candidate_or_retrain_without_old19_regression"
    if (checks.get("td") or {}).get("status") != "pass":
        return "reject_candidate_or_retrain_without_td_regression"
    return "refresh_acceptance_artifacts_and_inspect_remaining_audit_blockers"


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    scorecard = load_json(args.scorecard)
    audit = load_json(args.acceptance_audit, optional=True)
    review_status = load_json(args.review_gate_status, optional=True)
    gate = scorecard.get("final_candidate_gate") or {}
    leaderboard = scorecard.get("final_candidate_leaderboard") or []
    top = leaderboard[0] if leaderboard else None
    ready = bool(scorecard.get("ready_for_final_23way_acceptance"))
    decision = "accept" if ready else "wait"
    review_gate = scorecard.get("review_gate") or {}
    review_gate_status = review_status or {}
    verified_gate = scorecard.get("verified_east_gate") or {}
    selected = scorecard.get("selected_current_best") or {}
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "decision": decision,
        "ready_for_final_23way_acceptance": ready,
        "next_action": next_action(scorecard, gate, top),
        "scorecard": args.scorecard,
        "acceptance_audit": args.acceptance_audit if audit else None,
        "review_gate_status": args.review_gate_status if review_status else None,
        "review_gate": {
            "status": review_gate_status.get("status", review_gate.get("status")),
            "scorecard_status": review_gate.get("status"),
            "recommended_ready": review_gate_status.get(
                "recommended_ready",
                review_gate.get("recommended_ready"),
            ),
            "existing_export_count": review_gate_status.get(
                "existing_export_count",
                review_gate.get("existing_export_count"),
            ),
            "note": review_gate.get("note"),
            "next_action": review_gate_status.get("next_action", review_gate.get("next_action")),
        },
        "verified_east_gate": {
            "status": verified_gate.get("status"),
            "ready": verified_gate.get("ready"),
            "top_report": verified_gate.get("top_report"),
            "note": verified_gate.get("note"),
        },
        "final_candidate_gate": gate,
        "top_candidate": top,
        "top_candidates": leaderboard[: args.top_n],
        "selected_current_best": {
            "checkpoint": selected.get("checkpoint"),
            "report": selected.get("report"),
            "td_report": selected.get("td_report"),
        },
        "acceptance_audit_ready": (audit or {}).get("ready_for_final_acceptance") if audit else None,
        "remaining_blockers": [
            item.get("name")
            for item in (audit or {}).get("remaining_blockers", [])
            if isinstance(item, dict)
        ],
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = report.get("final_candidate_gate") or {}
    checks = gate.get("top_checks") or {}
    old19 = checks.get("old19") or {}
    td = checks.get("td") or {}
    verified = checks.get("verified_east") or {}
    lines = [
        "# Final Candidate Decision",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Decision: `{report['decision']}`",
        f"Ready for final 23-way acceptance: `{report['ready_for_final_23way_acceptance']}`",
        f"Next action: `{report['next_action']}`",
        "",
        "## Current Gates",
        "",
        f"Review gate: `{(report.get('review_gate') or {}).get('status')}`",
        f"Review recommended ready: `{(report.get('review_gate') or {}).get('recommended_ready')}`",
        f"Verified-east gate: `{(report.get('verified_east_gate') or {}).get('status')}`",
        f"Final-candidate gate: `{gate.get('status')}`",
        "",
        "## Closest Candidate",
        "",
        f"Candidate: `{gate.get('top_candidate_id')}`",
        f"Reports: `{gate.get('top_reports')}`",
        f"Missing splits: `{gate.get('top_missing_splits')}`",
        f"Old19: `{old19.get('status')}` accuracy_delta=`{fmt(old19.get('accuracy_delta'))}` macro_f1_delta=`{fmt(old19.get('macro_f1_delta'))}`",
        f"TD: `{td.get('status')}` validation_acc=`{fmt(td.get('validation_accuracy'))}` test_acc=`{fmt(td.get('test_accuracy'))}`",
        f"Verified-east: `{verified.get('status')}` note=`{verified.get('note')}`",
        "",
        "## Top Candidates",
        "",
        "| Rank | Candidate | Status | Missing | Old19 dF1 | TD val/test | Verified F1 | Verified support |",
        "|---:|---|---|---|---:|---|---:|---|",
    ]
    for idx, row in enumerate(report.get("top_candidates") or [], start=1):
        row_old19 = row.get("old19_check") or {}
        row_td = row.get("td_check") or {}
        row_verified = row.get("verified_east") or {}
        lines.append(
            "| {} | `{}` | `{}` | `{}` | {} | {}/{} | {} | `{}` |".format(
                idx,
                row.get("candidate_id"),
                status_label(row),
                row.get("missing_splits"),
                fmt(row_old19.get("macro_f1_delta")),
                fmt(row_td.get("validation_accuracy")),
                fmt(row_td.get("test_accuracy")),
                fmt(row_verified.get("macro_f1")),
                row.get("verified_east_per_label_support"),
            )
        )
    blockers = report.get("remaining_blockers") or []
    lines.extend(
        [
            "",
            "## Source Artifacts",
            "",
            f"Scorecard: `{report['scorecard']}`",
            f"Acceptance audit: `{report.get('acceptance_audit')}`",
            f"Review gate status: `{report.get('review_gate_status')}`",
            f"Remaining blockers: `{blockers}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    report = build_report(args)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(Path(args.out_md), report)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "ready_for_final_23way_acceptance": report["ready_for_final_23way_acceptance"],
                "top_candidate": (report.get("final_candidate_gate") or {}).get("top_candidate_id"),
                "next_action": report["next_action"],
                "out_json": str(out_json),
                "out_md": args.out_md,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
