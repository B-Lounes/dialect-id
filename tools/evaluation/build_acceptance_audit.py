#!/usr/bin/env python3
"""Build a rerunnable acceptance audit for the 23-way full-audio DID candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dialect_id.labels import DIALECTS


EXPECTED_CODES = [label.code for label in DIALECTS]
OLD19_CODES = [code for code in EXPECTED_CODES if code not in {"DJ", "KM", "SO", "TD"}]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", required=True)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--coverage-audit", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--min-old19-per-dialect-accuracy", type=float, default=0.90)
    parser.add_argument("--load-checkpoint", action="store_true")
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def status(pass_condition: bool, *, blocked: bool = False) -> str:
    if blocked:
        return "blocked"
    return "pass" if pass_condition else "fail"


def checkpoint_audit(checkpoint: Path | None, *, load_checkpoint: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "checkpoint": str(checkpoint) if checkpoint else None,
        "exists": bool(checkpoint and checkpoint.is_file()),
        "loaded": False,
        "can_load": None,
        "error": None,
    }
    if not checkpoint or not load_checkpoint:
        return result
    try:
        import torch

        payload = torch.load(checkpoint, map_location="cpu")
        model = payload.get("model", {}) if isinstance(payload, dict) else {}
        labels = payload.get("labels", {}) if isinstance(payload, dict) else {}
        args = payload.get("args", {}) if isinstance(payload, dict) else {}
        dialects = labels.get("dialects", []) if isinstance(labels, dict) else []
        label_codes = [str(item.get("code")) for item in dialects if isinstance(item, dict)]
        tensor_shapes = {}
        for key in (
            "heads.dialect.weight",
            "heads.dialect.bias",
            "heads.lang_type.weight",
            "heads.lang_type.bias",
            "heads.region.weight",
            "heads.region.bias",
            "heads.accent.weight",
            "heads.accent.bias",
        ):
            tensor = model.get(key)
            tensor_shapes[key] = list(tensor.shape) if tensor is not None and hasattr(tensor, "shape") else None
        result.update(
            {
                "loaded": True,
                "can_load": True,
                "step": payload.get("step"),
                "epoch": payload.get("epoch"),
                "state_type": payload.get("state_type"),
                "label_codes": label_codes,
                "tensor_shapes": tensor_shapes,
                "args": {
                    key: args.get(key)
                    for key in (
                        "model_type",
                        "full_audio_chunking",
                        "chunk_seconds",
                        "max_seconds",
                        "eval_max_seconds",
                        "region_logit_fusion_weight",
                    )
                },
            }
        )
    except Exception as exc:  # pragma: no cover - depends on optional torch/checkpoint env
        result.update({"can_load": False, "error": repr(exc)})
    return result


def selected_checkpoint(scorecard: dict[str, Any], cli_checkpoint: str | None) -> str | None:
    if cli_checkpoint:
        return cli_checkpoint
    selected = scorecard.get("selected_current_best") or {}
    return selected.get("checkpoint")


def metrics_for_split(scorecard: dict[str, Any], split: str) -> dict[str, float]:
    selected = scorecard.get("selected_current_best") or {}
    payload = (selected.get("splits") or {}).get(split) or {}
    metrics = payload.get("metrics") or payload
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}


def diagnostics_split(diagnostics: dict[str, Any], split: str) -> dict[str, Any]:
    return (diagnostics.get("splits") or {}).get(split) or {}


def old19_per_dialect_floor(diagnostics: dict[str, Any], min_accuracy: float) -> dict[str, Any]:
    split = diagnostics_split(diagnostics, "acgc_old19_test")
    rows = split.get("per_dialect") or []
    old_rows = [
        row for row in rows
        if row.get("dialect") in OLD19_CODES and int(row.get("support") or 0) > 0
    ]
    lows = [
        {
            "dialect": row.get("dialect"),
            "support": int(row.get("support") or 0),
            "accuracy": float(row.get("accuracy") or 0.0),
            "top_confusions": row.get("top_confusions") or "",
        }
        for row in old_rows
        if float(row.get("accuracy") or 0.0) < min_accuracy
    ]
    min_row = min(old_rows, key=lambda row: float(row.get("accuracy") or 0.0), default=None)
    return {
        "min_accuracy_threshold": min_accuracy,
        "old19_dialects_with_support": len(old_rows),
        "below_threshold": lows,
        "minimum": {
            "dialect": min_row.get("dialect"),
            "support": int(min_row.get("support") or 0),
            "accuracy": float(min_row.get("accuracy") or 0.0),
            "top_confusions": min_row.get("top_confusions") or "",
        } if min_row else None,
    }


def full_audio_evidence(checkpoint: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, Any]:
    args = checkpoint.get("args") or {}
    diag_splits = diagnostics.get("splits") or {}
    duration_ready = True
    chunk_ready = True
    split_evidence = {}
    for split_name, split in diag_splits.items():
        q = split.get("score_quantiles") or {}
        duration_quantiles = q.get("duration_seconds") or {}
        chunk_quantiles = q.get("chunk_count") or {}
        duration_ready = duration_ready and bool(duration_quantiles)
        chunk_ready = chunk_ready and bool(chunk_quantiles)
        split_evidence[split_name] = {
            "duration_p50": duration_quantiles.get("p50"),
            "duration_p99": duration_quantiles.get("p99"),
            "chunk_count_p90": chunk_quantiles.get("p90"),
            "chunk_count_p99": chunk_quantiles.get("p99"),
        }
    return {
        "checkpoint_full_audio_chunking": bool(args.get("full_audio_chunking")),
        "checkpoint_max_seconds": args.get("max_seconds"),
        "checkpoint_eval_max_seconds": args.get("eval_max_seconds"),
        "checkpoint_chunk_seconds": args.get("chunk_seconds"),
        "diagnostics_have_duration_buckets": duration_ready,
        "diagnostics_have_chunk_counts": chunk_ready,
        "splits": split_evidence,
    }


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    scorecard = load_json(args.scorecard)
    diagnostics = load_json(args.diagnostics)
    coverage = load_json(args.coverage_audit)
    checkpoint_path = selected_checkpoint(scorecard, args.checkpoint)
    checkpoint = checkpoint_audit(Path(checkpoint_path) if checkpoint_path else None, load_checkpoint=args.load_checkpoint)

    checkpoint_codes = checkpoint.get("label_codes") or EXPECTED_CODES
    label_set_pass = checkpoint_codes == EXPECTED_CODES
    dialect_shape = (checkpoint.get("tensor_shapes") or {}).get("heads.dialect.weight")
    dialect_head_pass = dialect_shape is None or int(dialect_shape[0]) == len(EXPECTED_CODES)
    lang_shape = (checkpoint.get("tensor_shapes") or {}).get("heads.lang_type.weight")
    region_shape = (checkpoint.get("tensor_shapes") or {}).get("heads.region.weight")
    accent_shape = (checkpoint.get("tensor_shapes") or {}).get("heads.accent.weight")
    heads_pass = dialect_head_pass and (lang_shape is None or int(lang_shape[0]) == 2) and region_shape is None and accent_shape is None

    old19_test = metrics_for_split(scorecard, "acgc_old19_test")
    old19_val = metrics_for_split(scorecard, "acgc_old19_validation")
    old19_rank_pass = int(scorecard.get("selected_eligible_rank") or 999) == 1
    regression = old19_per_dialect_floor(diagnostics, args.min_old19_per_dialect_accuracy)
    td_val = metrics_for_split(scorecard, "acgc_td_validation_23way")
    td_test = metrics_for_split(scorecard, "acgc_td_test_23way")
    td_pass = (td_val.get("accuracy", 0.0) >= 0.5) and (td_test.get("accuracy", 0.0) >= 0.5)
    review_gate = scorecard.get("review_gate") or {}
    verified_east_gate = scorecard.get("verified_east_gate") or {}
    final_candidate_gate = scorecard.get("final_candidate_gate") or {}
    dj_km_so_ready = bool(review_gate.get("recommended_ready")) and bool(final_candidate_gate.get("ready"))
    full_audio = full_audio_evidence(checkpoint, diagnostics)
    full_audio_pass = (
        bool(full_audio["checkpoint_full_audio_chunking"])
        and float(full_audio.get("checkpoint_chunk_seconds") or 0.0) == 30.0
        and float(full_audio.get("checkpoint_max_seconds") or 0.0) == 0.0
        and bool(full_audio["diagnostics_have_duration_buckets"])
        and bool(full_audio["diagnostics_have_chunk_counts"])
    )

    criteria = [
        {
            "id": 1,
            "name": "supports_all_23_dialects",
            "status": status(label_set_pass and heads_pass),
            "evidence": {
                "expected_codes": EXPECTED_CODES,
                "checkpoint_label_codes": checkpoint_codes,
                "dialect_head_shape": dialect_shape,
                "lang_type_head_shape": lang_shape,
                "learned_region_head_shape": region_shape,
                "learned_accent_head_shape": accent_shape,
            },
        },
        {
            "id": 2,
            "name": "old_dialect_acgc_match_or_improve",
            "status": status(old19_rank_pass),
            "evidence": {
                "selected_eligible_rank": scorecard.get("selected_eligible_rank"),
                "top_eligible_report": scorecard.get("top_eligible_report"),
                "old19_validation": old19_val,
                "old19_test": old19_test,
            },
        },
        {
            "id": 3,
            "name": "no_severe_old_dialect_regressions",
            "status": status(not regression["below_threshold"] and regression["old19_dialects_with_support"] == len(OLD19_CODES)),
            "evidence": regression,
        },
        {
            "id": 4,
            "name": "useful_td_behavior",
            "status": status(td_pass),
            "evidence": {
                "td_validation": td_val,
                "td_test": td_test,
            },
        },
        {
            "id": 5,
            "name": "reasonable_dj_km_so_verified_behavior",
            "status": status(False, blocked=not dj_km_so_ready),
            "evidence": {
                "review_gate": review_gate,
                "verified_east_gate": verified_east_gate,
                "final_candidate_gate": final_candidate_gate,
            },
        },
        {
            "id": 6,
            "name": "full_audio_30s_chunk_aggregation",
            "status": status(full_audio_pass),
            "evidence": full_audio,
        },
    ]
    ready = all(item["status"] == "pass" for item in criteria)
    return {
        "ready_for_final_acceptance": ready,
        "checkpoint": checkpoint,
        "expected_codes": EXPECTED_CODES,
        "old19_codes": OLD19_CODES,
        "criteria": criteria,
        "scorecard": {
            "path": args.scorecard,
            "ready_for_final_23way_acceptance": scorecard.get("ready_for_final_23way_acceptance"),
            "selected_eligible_rank": scorecard.get("selected_eligible_rank"),
            "top_eligible_report": scorecard.get("top_eligible_report"),
            "record_count": scorecard.get("record_count"),
            "final_candidate_gate_status": final_candidate_gate.get("status"),
            "final_candidate_top": final_candidate_gate.get("top_candidate_id"),
        },
        "diagnostics": {
            "path": args.diagnostics,
            "report_dirs": diagnostics.get("report_dirs"),
        },
        "coverage_audit": {
            "path": args.coverage_audit,
            "recommended_next_evals": len(coverage.get("recommended_next_evals") or []),
            "status_counts": coverage.get("status_counts"),
        },
        "remaining_blockers": [
            item for item in criteria if item["status"] in {"fail", "blocked"}
        ],
    }


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if value is None:
        return ""
    return str(value)


def write_markdown(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# Current 23-Way DID Acceptance Audit",
        "",
        f"Ready for final acceptance: `{audit['ready_for_final_acceptance']}`",
        "",
        "## Criteria",
        "",
        "| ID | Criterion | Status | Evidence |",
        "|---:|---|---|---|",
    ]
    for item in audit["criteria"]:
        ev = item["evidence"]
        if item["name"] == "supports_all_23_dialects":
            evidence = (
                f"dialect_head={ev.get('dialect_head_shape')}; "
                f"lang_type_head={ev.get('lang_type_head_shape')}; "
                f"labels={len(ev.get('checkpoint_label_codes') or [])}"
            )
        elif item["name"] == "old_dialect_acgc_match_or_improve":
            test = ev.get("old19_test") or {}
            evidence = (
                f"eligible_rank={ev.get('selected_eligible_rank')}; "
                f"old19_test_acc={fmt(test.get('accuracy'))}; "
                f"old19_test_macro_f1={fmt(test.get('macro_f1'))}"
            )
        elif item["name"] == "no_severe_old_dialect_regressions":
            min_row = ev.get("minimum") or {}
            evidence = (
                f"threshold={fmt(ev.get('min_accuracy_threshold'))}; "
                f"minimum={min_row.get('dialect')} {fmt(min_row.get('accuracy'))}; "
                f"below_threshold={len(ev.get('below_threshold') or [])}"
            )
        elif item["name"] == "useful_td_behavior":
            evidence = (
                f"TD val acc={fmt((ev.get('td_validation') or {}).get('accuracy'))}; "
                f"TD test acc={fmt((ev.get('td_test') or {}).get('accuracy'))}"
            )
        elif item["name"] == "reasonable_dj_km_so_verified_behavior":
            review = ev.get("review_gate") or {}
            verified = ev.get("verified_east_gate") or {}
            final_candidate = ev.get("final_candidate_gate") or {}
            evidence = (
                f"review_ready={review.get('recommended_ready')}; "
                f"verified_eval_ready={verified.get('ready')}; "
                f"verified_eval_status={verified.get('status')}; "
                f"verified_eval_top={verified.get('top_report')}; "
                f"final_candidate_ready={final_candidate.get('ready')}; "
                f"final_candidate_top={final_candidate.get('top_candidate_id')}; "
                f"note={final_candidate.get('note') or verified.get('note') or review.get('note')}"
            )
        elif item["name"] == "full_audio_30s_chunk_aggregation":
            evidence = (
                f"full_audio={ev.get('checkpoint_full_audio_chunking')}; "
                f"max_seconds={fmt(ev.get('checkpoint_max_seconds'))}; "
                f"chunk_seconds={fmt(ev.get('checkpoint_chunk_seconds'))}; "
                f"duration_buckets={ev.get('diagnostics_have_duration_buckets')}"
            )
        else:
            evidence = json.dumps(ev, sort_keys=True)
        lines.append(f"| {item['id']} | `{item['name']}` | `{item['status']}` | {evidence} |")

    lines.extend(["", "## Remaining Blockers", ""])
    blockers = audit.get("remaining_blockers") or []
    if not blockers:
        lines.append("None.")
    else:
        for item in blockers:
            lines.append(f"- `{item['name']}`: `{item['status']}`")

    lines.extend(
        [
            "",
            "## Source Artifacts",
            "",
            f"- Scorecard: `{audit['scorecard']['path']}`",
            f"- Diagnostics: `{audit['diagnostics']['path']}`",
            f"- Coverage audit: `{audit['coverage_audit']['path']}`",
            f"- Checkpoint: `{audit['checkpoint']['checkpoint']}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    audit = build_audit(args)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(Path(args.out_md), audit)
    print(
        json.dumps(
            {
                "ready_for_final_acceptance": audit["ready_for_final_acceptance"],
                "out_json": str(out_json),
                "out_md": args.out_md,
                "remaining_blockers": [item["name"] for item in audit["remaining_blockers"]],
            },
            sort_keys=True,
        )
    )
    if args.load_checkpoint and not audit["checkpoint"].get("loaded"):
        print(
            json.dumps(
                {
                    "error": "checkpoint_load_required_but_failed",
                    "checkpoint": audit["checkpoint"].get("checkpoint"),
                    "checkpoint_error": audit["checkpoint"].get("error"),
                    "out_json": str(out_json),
                    "out_md": args.out_md,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
