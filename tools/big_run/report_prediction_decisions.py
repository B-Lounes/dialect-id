#!/usr/bin/env python3
"""Parallel read-only decision summary from BIG-RUN prediction CSV shards.

This is safe to run while the export is still active: it reads only shards with
completed ``state/shard_*.done`` markers and writes reports outside the
prediction/state directories.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dialect_id.labels import CODE_TO_COUNTRY, CODE_TO_REGION
try:
    from .report_big_run_final_dialect_decisions import (
        add_metric,
        compact,
        format_num,
        format_pct,
        metric_dict,
        write_csv,
    )
except ImportError:  # Direct script execution.
    from report_big_run_final_dialect_decisions import (
        add_metric,
        compact,
        format_num,
        format_pct,
        metric_dict,
        write_csv,
    )


Metric = dict[str, float | int]

POLICY_SPECS: tuple[dict[str, Any], ...] = (
    {
        "policy": "strong_model_only",
        "description": "single-model score >= 0.95 and margin >= 0.50",
        "min_score": 0.95,
        "min_margin": 0.50,
        "max_entropy": None,
        "require_weak_region": False,
        "require_weak_top1": False,
    },
    {
        "policy": "sure_model_only",
        "description": "single-model score >= 0.98, margin >= 0.75, entropy <= 0.60",
        "min_score": 0.98,
        "min_margin": 0.75,
        "max_entropy": 0.60,
        "require_weak_region": False,
        "require_weak_top1": False,
    },
    {
        "policy": "ultra_sure_model_only",
        "description": "single-model score >= 0.99, margin >= 0.85, entropy <= 0.40",
        "min_score": 0.99,
        "min_margin": 0.85,
        "max_entropy": 0.40,
        "require_weak_region": False,
        "require_weak_top1": False,
    },
    {
        "policy": "sure_with_weak_region",
        "description": "sure_model_only plus weak/path region agrees with the model region",
        "min_score": 0.98,
        "min_margin": 0.75,
        "max_entropy": 0.60,
        "require_weak_region": True,
        "require_weak_top1": False,
    },
    {
        "policy": "sure_with_weak_top1",
        "description": "sure_model_only plus weak/path country code equals the model top1",
        "min_score": 0.98,
        "min_margin": 0.75,
        "max_entropy": 0.60,
        "require_weak_region": False,
        "require_weak_top1": True,
    },
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def to_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def to_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def bool_decoded(value: object) -> bool:
    return str(value or "") == "1"


def tier_for(score: float | None, margin: float | None, thresholds: tuple[float, float, float, float]) -> str:
    strong_score, strong_margin, usable_score, usable_margin = thresholds
    if score is None or margin is None:
        return "low"
    if score >= strong_score and margin >= strong_margin:
        return "strong"
    if score >= usable_score and margin >= usable_margin:
        return "usable"
    return "low"


def policy_match(
    spec: dict[str, Any],
    *,
    score: float | None,
    margin: float | None,
    entropy: float | None,
    weak_region: bool,
    weak_top1: bool,
) -> bool:
    if score is None or margin is None:
        return False
    if score < float(spec["min_score"]) or margin < float(spec["min_margin"]):
        return False
    max_entropy = spec.get("max_entropy")
    if max_entropy is not None and (entropy is None or entropy > float(max_entropy)):
        return False
    if spec.get("require_weak_region") and not weak_region:
        return False
    if spec.get("require_weak_top1") and not weak_top1:
        return False
    return True


def get_cell(row: list[str], indices: dict[str, int], name: str) -> str:
    idx = indices.get(name)
    if idx is None or idx >= len(row):
        return ""
    return row[idx]


def merge_metric(dst: Metric, src: dict[str, Any]) -> None:
    for key, value in src.items():
        dst[key] += value


def freeze_metrics(mapping: dict[str, Metric]) -> dict[str, dict[str, float | int]]:
    return {key: dict(value) for key, value in mapping.items()}


def process_csv(task: dict[str, Any]) -> dict[str, Any]:
    csv_path = Path(task["csv_path"])
    thresholds = tuple(task["thresholds"])
    overall = metric_dict()
    by_pred: dict[str, Metric] = defaultdict(metric_dict)
    by_weak_cc: dict[str, Metric] = defaultdict(metric_dict)
    by_weak_region: dict[str, Metric] = defaultdict(metric_dict)
    by_pred_region: dict[str, Metric] = defaultdict(metric_dict)
    by_policy: dict[str, Metric] = defaultdict(metric_dict)
    by_policy_pred: dict[str, dict[str, Metric]] = defaultdict(lambda: defaultdict(metric_dict))
    mismatches: Counter[str] = Counter()
    strong_mismatches: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    dialect_counts: Counter[str] = Counter()
    region_counts: Counter[str] = Counter()

    file_rows = 0
    file_decoded = 0
    file_failed = 0
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise RuntimeError(f"Empty prediction CSV: {csv_path}") from exc
        indices = {name: idx for idx, name in enumerate(header)}
        required = {
            "decoded",
            "weak_cc",
            "top1_cc",
            "top2_cc",
            "top3_cc",
            "top1_score",
            "margin",
            "entropy",
            "duration",
        }
        missing = sorted(required - set(indices))
        if missing:
            raise RuntimeError(f"Missing required columns in {csv_path}: {missing}")

        for row in reader:
            file_rows += 1
            decoded = bool_decoded(get_cell(row, indices, "decoded"))
            file_decoded += int(decoded)
            file_failed += int(not decoded)
            if not decoded:
                continue

            weak = get_cell(row, indices, "weak_cc").upper()
            pred = get_cell(row, indices, "top1_cc").upper()
            pred2 = get_cell(row, indices, "top2_cc").upper()
            pred3 = get_cell(row, indices, "top3_cc").upper()
            score = to_float(get_cell(row, indices, "top1_score"))
            margin = to_float(get_cell(row, indices, "margin"))
            entropy = to_float(get_cell(row, indices, "entropy"))
            duration = float(to_float(get_cell(row, indices, "duration")) or 0.0)
            tier = tier_for(score, margin, thresholds)
            weak_region_name = CODE_TO_REGION.get(weak, "unknown")
            pred_region_name = get_cell(row, indices, "top1_region") or CODE_TO_REGION.get(pred, "unknown") or "unknown"
            weak_top1 = bool(weak and weak == pred)
            weak_top2 = bool(weak and weak in {pred, pred2})
            weak_top3 = bool(weak and weak in {pred, pred2, pred3})
            weak_region = bool(weak_region_name != "unknown" and weak_region_name == pred_region_name)
            kwargs = {
                "duration": duration,
                "score": score,
                "margin": margin,
                "tier": tier,
                "weak_top1": weak_top1,
                "weak_top2": weak_top2,
                "weak_top3": weak_top3,
                "weak_region": weak_region,
            }
            add_metric(overall, **kwargs)
            add_metric(by_pred[pred or "MISSING"], **kwargs)
            add_metric(by_weak_cc[weak or "MISSING"], **kwargs)
            add_metric(by_weak_region[weak_region_name], **kwargs)
            add_metric(by_pred_region[pred_region_name], **kwargs)
            for spec in POLICY_SPECS:
                if policy_match(
                    spec,
                    score=score,
                    margin=margin,
                    entropy=entropy,
                    weak_region=weak_region,
                    weak_top1=weak_top1,
                ):
                    policy = str(spec["policy"])
                    add_metric(by_policy[policy], **kwargs)
                    add_metric(by_policy_pred[policy][pred or "MISSING"], **kwargs)
            tier_counts[tier] += 1
            dialect_counts[pred or "MISSING"] += 1
            region_counts[pred_region_name] += 1
            if weak and pred and weak != pred:
                key = f"{weak}\t{pred}"
                mismatches[key] += 1
                if tier == "strong":
                    strong_mismatches[key] += 1

    expected_rows = to_int(task.get("csv_rows"), to_int(task.get("expanded_rows")))
    expected_decoded = to_int(task.get("decoded_rows"))
    expected_failed = to_int(task.get("failed_rows"))
    if file_rows != expected_rows or file_decoded != expected_decoded or file_failed != expected_failed:
        raise RuntimeError(
            f"CSV/marker count mismatch for {csv_path}: "
            f"csv rows={file_rows}, decoded={file_decoded}, failed={file_failed}; "
            f"marker rows={expected_rows}, decoded={expected_decoded}, failed={expected_failed}"
        )

    return {
        "csv_rows": file_rows,
        "decoded_rows": file_decoded,
        "failed_rows": file_failed,
        "overall": dict(overall),
        "by_pred": freeze_metrics(by_pred),
        "by_weak_cc": freeze_metrics(by_weak_cc),
        "by_weak_region": freeze_metrics(by_weak_region),
        "by_pred_region": freeze_metrics(by_pred_region),
        "by_policy": freeze_metrics(by_policy),
        "by_policy_pred": {policy: freeze_metrics(values) for policy, values in by_policy_pred.items()},
        "mismatches": dict(mismatches),
        "strong_mismatches": dict(strong_mismatches),
        "tier_counts": dict(tier_counts),
        "dialect_counts": dict(dialect_counts),
        "region_counts": dict(region_counts),
    }


def load_tasks(out_root: Path, *, max_shards: int | None, thresholds: tuple[float, float, float, float]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for done_path in sorted((out_root / "state").glob("shard_*.done")):
        marker = read_json(done_path)
        csv_path = Path(str(marker.get("output_csv", "")))
        if not csv_path.is_file():
            raise SystemExit(f"Missing prediction CSV from done marker {done_path}: {csv_path}")
        tasks.append(
            {
                "done_path": str(done_path),
                "csv_path": str(csv_path),
                "csv_rows": to_int(marker.get("csv_rows"), to_int(marker.get("expanded_rows"))),
                "expanded_rows": to_int(marker.get("expanded_rows")),
                "decoded_rows": to_int(marker.get("decoded_rows")),
                "failed_rows": to_int(marker.get("failed_rows")),
                "thresholds": thresholds,
            }
        )
        if max_shards is not None and len(tasks) >= max_shards:
            break
    return tasks


def add_counter(dst: Counter[str], src: dict[str, int]) -> None:
    for key, value in src.items():
        dst[key] += int(value)


def merge_result(
    result: dict[str, Any],
    *,
    overall: Metric,
    by_pred: dict[str, Metric],
    by_weak_cc: dict[str, Metric],
    by_weak_region: dict[str, Metric],
    by_pred_region: dict[str, Metric],
    by_policy: dict[str, Metric],
    by_policy_pred: dict[str, dict[str, Metric]],
    mismatches: Counter[str],
    strong_mismatches: Counter[str],
    tier_counts: Counter[str],
    dialect_counts: Counter[str],
    region_counts: Counter[str],
) -> tuple[int, int, int]:
    merge_metric(overall, result["overall"])
    for key, metrics in result["by_pred"].items():
        merge_metric(by_pred[key], metrics)
    for key, metrics in result["by_weak_cc"].items():
        merge_metric(by_weak_cc[key], metrics)
    for key, metrics in result["by_weak_region"].items():
        merge_metric(by_weak_region[key], metrics)
    for key, metrics in result["by_pred_region"].items():
        merge_metric(by_pred_region[key], metrics)
    for key, metrics in result["by_policy"].items():
        merge_metric(by_policy[key], metrics)
    for policy, pred_values in result["by_policy_pred"].items():
        for pred, metrics in pred_values.items():
            merge_metric(by_policy_pred[policy][pred], metrics)
    add_counter(mismatches, result["mismatches"])
    add_counter(strong_mismatches, result["strong_mismatches"])
    add_counter(tier_counts, result["tier_counts"])
    add_counter(dialect_counts, result["dialect_counts"])
    add_counter(region_counts, result["region_counts"])
    return int(result["csv_rows"]), int(result["decoded_rows"]), int(result["failed_rows"])


def expected_shards(manifest_dir: Path | None) -> int | None:
    if manifest_dir is None:
        return None
    return sum(1 for _ in manifest_dir.glob("shard_*.jsonl"))


def write_report(
    *,
    out_dir: Path,
    payload: dict[str, Any],
    by_pred: dict[str, Metric],
    by_weak_cc: dict[str, Metric],
    by_weak_region: dict[str, Metric],
    by_pred_region: dict[str, Metric],
    by_policy: dict[str, Metric],
    by_policy_pred: dict[str, dict[str, Metric]],
    mismatches: Counter[str],
    strong_mismatches: Counter[str],
    top_mismatches: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_rows = [
        {
            "pred_cc": code,
            "country": CODE_TO_COUNTRY.get(code, ""),
            "region": CODE_TO_REGION.get(code, "unknown"),
            **compact(metrics),
        }
        for code, metrics in by_pred.items()
    ]
    pred_rows.sort(key=lambda row: (-int(row["strong"]), -int(row["n"]), str(row["pred_cc"])))
    weak_rows = [
        {
            "weak_cc": code,
            "country": CODE_TO_COUNTRY.get(code, ""),
            "region": CODE_TO_REGION.get(code, "unknown"),
            **compact(metrics),
        }
        for code, metrics in by_weak_cc.items()
    ]
    weak_rows.sort(key=lambda row: (-float(row["weak_top1_pct"]), -int(row["n"]), str(row["weak_cc"])))
    weak_region_rows = [{**compact(metrics), "weak_region": region} for region, metrics in by_weak_region.items()]
    weak_region_rows.sort(key=lambda row: (-float(row["weak_region_pct"]), -int(row["n"]), str(row["weak_region"])))
    pred_region_rows = [{"pred_region": region, **compact(metrics)} for region, metrics in by_pred_region.items()]
    pred_region_rows.sort(key=lambda row: (-int(row["strong"]), -int(row["n"]), str(row["pred_region"])))
    policy_descriptions = {str(spec["policy"]): str(spec["description"]) for spec in POLICY_SPECS}
    policy_order = {str(spec["policy"]): idx for idx, spec in enumerate(POLICY_SPECS)}
    policy_rows = [
        {
            "policy": policy,
            "description": policy_descriptions.get(policy, ""),
            "pct_of_decoded": 0.0 if not payload["decoded_rows"] else 100.0 * int(metrics["n"]) / int(payload["decoded_rows"]),
            **compact(metrics),
        }
        for policy, metrics in by_policy.items()
    ]
    policy_rows.sort(key=lambda row: policy_order.get(str(row["policy"]), 999))
    policy_pred_rows = []
    for policy, pred_values in by_policy_pred.items():
        for pred, metrics in pred_values.items():
            policy_pred_rows.append(
                {
                    "policy": policy,
                    "pred_cc": pred,
                    "country": CODE_TO_COUNTRY.get(pred, ""),
                    "region": CODE_TO_REGION.get(pred, "unknown"),
                    "pct_of_decoded": 0.0 if not payload["decoded_rows"] else 100.0 * int(metrics["n"]) / int(payload["decoded_rows"]),
                    **compact(metrics),
                }
            )
    policy_pred_rows.sort(
        key=lambda row: (
            policy_order.get(str(row["policy"]), 999),
            -int(row["n"]),
            str(row["pred_cc"]),
        )
    )
    mismatch_rows = [
        {
            "weak_cc": pair.split("\t", 1)[0],
            "pred_top1_cc": pair.split("\t", 1)[1],
            "rows": count,
            "strong_rows": strong_mismatches.get(pair, 0),
        }
        for pair, count in mismatches.most_common(top_mismatches)
    ]
    payload.update(
        {
            "by_predicted_dialect": pred_rows,
            "by_weak_cc_ranked_by_top1_pct": weak_rows,
            "by_weak_region": weak_region_rows,
            "by_predicted_region": pred_region_rows,
            "for_sure_policies": policy_rows,
            "for_sure_by_predicted_dialect": policy_pred_rows,
            "top_weak_vs_prediction_mismatches": mismatch_rows,
        }
    )
    (out_dir / "prediction_decision_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    common_fields = [
        "n",
        "hours",
        "strong",
        "strong_pct",
        "usable",
        "usable_pct",
        "low",
        "low_pct",
        "avg_score",
        "avg_margin",
        "weak_top1_pct",
        "weak_top2_pct",
        "weak_top3_pct",
        "weak_region_pct",
        "strong_weak_top1_pct",
        "strong_weak_region_pct",
    ]
    write_csv(out_dir / "by_predicted_dialect.csv", pred_rows, ["pred_cc", "country", "region", *common_fields])
    write_csv(out_dir / "by_weak_cc_ranked_by_top1_pct.csv", weak_rows, ["weak_cc", "country", "region", *common_fields])
    write_csv(out_dir / "by_weak_region.csv", weak_region_rows, ["weak_region", *common_fields])
    write_csv(out_dir / "by_predicted_region.csv", pred_region_rows, ["pred_region", *common_fields])
    write_csv(
        out_dir / "for_sure_policies.csv",
        policy_rows,
        ["policy", "description", "pct_of_decoded", *common_fields],
    )
    write_csv(
        out_dir / "for_sure_by_predicted_dialect.csv",
        policy_pred_rows,
        ["policy", "pred_cc", "country", "region", "pct_of_decoded", *common_fields],
    )
    write_csv(out_dir / "top_weak_vs_prediction_mismatches.csv", mismatch_rows, ["weak_cc", "pred_top1_cc", "rows", "strong_rows"])

    overall = payload["overall"]
    lines = [
        "# BIG-RUN Prediction Decision Summary",
        "",
        f"- Source predictions: `{payload['out_root']}`",
        f"- Completed shards summarized: `{payload['done_markers']:,}`",
        f"- Expected shards: `{payload['expected_shards'] if payload['expected_shards'] is not None else 'unknown'}`",
        f"- Partial live export: `{payload['partial_export']}`",
        f"- CSV rows seen: `{payload['csv_rows_seen']:,}`",
        f"- Decoded rows: `{payload['decoded_rows']:,}`",
        f"- Failed/non-decoded rows: `{payload['failed_rows']:,}`",
        f"- Strong threshold: `top1_score >= {payload['thresholds']['strong_score']}` and `margin >= {payload['thresholds']['strong_margin']}`",
        f"- Usable threshold: `top1_score >= {payload['thresholds']['usable_score']}` and `margin >= {payload['thresholds']['usable_margin']}`",
        f"- Strong labels: `{overall['strong']:,}` ({format_pct(overall['strong_pct'])})",
        f"- Usable labels: `{overall['usable']:,}` ({format_pct(overall['usable_pct'])})",
        f"- Low-confidence labels: `{overall['low']:,}` ({format_pct(overall['low_pct'])})",
        f"- Weak/path top1 agreement: `{overall['weak_top1']:,}` / `{overall['n']:,}` ({format_pct(overall['weak_top1_pct'])})",
        f"- Weak/path region agreement: `{overall['weak_region']:,}` / `{overall['n']:,}` ({format_pct(overall['weak_region_pct'])})",
        "",
        "Weak/path country code is metadata only; it is not treated as ground truth.",
        "The `*_with_weak_*` policies are audit/sanity subsets, not alternate labels.",
        "",
        "## For-Sure Policy Counts",
        "",
        "| Policy | Rows | Coverage | Avg score | Avg margin | Weak top1 % | Weak region % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in policy_rows:
        lines.append(
            f"| {row['policy']} | {int(row['n']):,} | {format_pct(row['pct_of_decoded'])} | "
            f"{format_num(row['avg_score'])} | {format_num(row['avg_margin'])} | "
            f"{format_pct(row['weak_top1_pct'])} | {format_pct(row['weak_region_pct'])} |"
        )
    lines.extend(
        [
            "",
            "## For-Sure Dialect Mix",
            "",
            "| Policy | Top1 dialect | Region | Rows | Coverage | Weak top1 % | Weak region % |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in policy_pred_rows:
        if int(row["n"]) <= 0:
            continue
        lines.append(
            f"| {row['policy']} | {row['pred_cc']} | {row['region']} | {int(row['n']):,} | "
            f"{format_pct(row['pct_of_decoded'])} | {format_pct(row['weak_top1_pct'])} | "
            f"{format_pct(row['weak_region_pct'])} |"
        )
    lines.extend(
        [
        "",
        "## Predicted Dialects",
        "",
        "| Top1 dialect | Region | Rows | Strong | Strong % | Avg score | Avg margin | Weak top1 % | Weak region % |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in pred_rows:
        lines.append(
            f"| {row['pred_cc']} | {row['region']} | {int(row['n']):,} | {int(row['strong']):,} | "
            f"{format_pct(row['strong_pct'])} | {format_num(row['avg_score'])} | {format_num(row['avg_margin'])} | "
            f"{format_pct(row['weak_top1_pct'])} | {format_pct(row['weak_region_pct'])} |"
        )
    lines.extend(
        [
            "",
            "## Weak/Path Dialects Ranked By Top1 Agreement",
            "",
            "| Weak CC | Region | Rows | Top1 agree | Top2 | Top3 | Region agree | Strong rows | Strong top1 agree |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in weak_rows:
        lines.append(
            f"| {row['weak_cc']} | {row['region']} | {int(row['n']):,} | {format_pct(row['weak_top1_pct'])} | "
            f"{format_pct(row['weak_top2_pct'])} | {format_pct(row['weak_top3_pct'])} | "
            f"{format_pct(row['weak_region_pct'])} | {int(row['strong']):,} | {format_pct(row['strong_weak_top1_pct'])} |"
        )
    lines.extend(
        [
            "",
            "## Largest Weak/Path vs Top1 Mismatches",
            "",
            "| Weak CC | Top1 dialect | Rows | Strong rows |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in mismatch_rows[:75]:
        lines.append(f"| {row['weak_cc']} | {row['pred_top1_cc']} | {int(row['rows']):,} | {int(row['strong_rows']):,} |")
    lines.append("")
    (out_dir / "prediction_decision_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=max(1, min(32, mp.cpu_count())))
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--strong-score", type=float, default=0.95)
    parser.add_argument("--strong-margin", type=float, default=0.50)
    parser.add_argument("--usable-score", type=float, default=0.90)
    parser.add_argument("--usable-margin", type=float, default=0.30)
    parser.add_argument("--top-mismatches", type=int, default=100)
    parser.add_argument("--progress-every-shards", type=int, default=1000)
    args = parser.parse_args()

    start = time.time()
    thresholds = (args.strong_score, args.strong_margin, args.usable_score, args.usable_margin)
    tasks = load_tasks(args.out_root, max_shards=args.max_shards, thresholds=thresholds)
    if not tasks:
        raise SystemExit(f"No completed prediction shards found under {args.out_root / 'state'}")
    expected = expected_shards(args.manifest_dir)

    overall = metric_dict()
    by_pred: dict[str, Metric] = defaultdict(metric_dict)
    by_weak_cc: dict[str, Metric] = defaultdict(metric_dict)
    by_weak_region: dict[str, Metric] = defaultdict(metric_dict)
    by_pred_region: dict[str, Metric] = defaultdict(metric_dict)
    by_policy: dict[str, Metric] = defaultdict(metric_dict)
    by_policy_pred: dict[str, dict[str, Metric]] = defaultdict(lambda: defaultdict(metric_dict))
    mismatches: Counter[str] = Counter()
    strong_mismatches: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    dialect_counts: Counter[str] = Counter()
    region_counts: Counter[str] = Counter()
    csv_rows_seen = 0
    decoded_rows = 0
    failed_rows = 0

    workers = max(1, min(args.workers, len(tasks)))
    with mp.Pool(processes=workers) as pool:
        for completed, result in enumerate(pool.imap_unordered(process_csv, tasks, chunksize=args.chunksize), start=1):
            csv_rows, decoded, failed = merge_result(
                result,
                overall=overall,
                by_pred=by_pred,
                by_weak_cc=by_weak_cc,
                by_weak_region=by_weak_region,
                by_pred_region=by_pred_region,
                by_policy=by_policy,
                by_policy_pred=by_policy_pred,
                mismatches=mismatches,
                strong_mismatches=strong_mismatches,
                tier_counts=tier_counts,
                dialect_counts=dialect_counts,
                region_counts=region_counts,
            )
            csv_rows_seen += csv_rows
            decoded_rows += decoded
            failed_rows += failed
            if args.progress_every_shards > 0 and completed % args.progress_every_shards == 0:
                elapsed = max(time.time() - start, 1e-9)
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "completed_shards": completed,
                            "total_shards": len(tasks),
                            "decoded_rows": decoded_rows,
                            "rows_per_sec": decoded_rows / elapsed,
                            "elapsed_sec": elapsed,
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )

    overall_row = compact(overall)
    payload = {
        "out_root": str(args.out_root),
        "generated_at_epoch": time.time(),
        "wall_sec": time.time() - start,
        "source": "prediction_csv_done_markers",
        "workers": workers,
        "chunksize": args.chunksize,
        "done_markers": len(tasks),
        "expected_shards": expected,
        "partial_export": expected is None or len(tasks) != expected,
        "thresholds": {
            "strong_score": args.strong_score,
            "strong_margin": args.strong_margin,
            "usable_score": args.usable_score,
            "usable_margin": args.usable_margin,
        },
        "csv_rows_seen": csv_rows_seen,
        "decoded_rows": decoded_rows,
        "failed_rows": failed_rows,
        "overall": overall_row,
        "confidence_tiers": dict(tier_counts),
        "by_final_region": dict(region_counts),
        "by_final_dialect_counts": dict(dialect_counts),
        "policy_specs": list(POLICY_SPECS),
    }
    write_report(
        out_dir=args.out_dir,
        payload=payload,
        by_pred=by_pred,
        by_weak_cc=by_weak_cc,
        by_weak_region=by_weak_region,
        by_pred_region=by_pred_region,
        by_policy=by_policy,
        by_policy_pred=by_policy_pred,
        mismatches=mismatches,
        strong_mismatches=strong_mismatches,
        top_mismatches=args.top_mismatches,
    )
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "done_markers": len(tasks),
                "decoded_rows": decoded_rows,
                "strong": overall_row["strong"],
                "strong_pct": overall_row["strong_pct"],
                "sure_model_only": compact(by_policy["sure_model_only"])["n"],
                "ultra_sure_model_only": compact(by_policy["ultra_sure_model_only"])["n"],
                "weak_top1_pct": overall_row["weak_top1_pct"],
                "weak_region_pct": overall_row["weak_region_pct"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
