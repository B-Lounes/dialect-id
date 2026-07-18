#!/usr/bin/env python3
"""Summarize final BIG-RUN dialect decisions from a synced tracker.sqlite.

The report treats the exported model prediction as the decision surface and
the row/path country code as weak metadata only. It is meant for final
single-model exports after
``python -m tools.big_run.sync_tracker --sample-mode all``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dialect_id.labels import CODE_TO_COUNTRY, CODE_TO_REGION
try:
    from .export_big_run_final_labels import SURE_POLICY_SPECS, sure_policy_flags
except ImportError:  # Direct script execution.
    from export_big_run_final_labels import SURE_POLICY_SPECS, sure_policy_flags


def metric_dict() -> dict[str, float | int]:
    return {
        "n": 0,
        "duration_sec": 0.0,
        "score_sum": 0.0,
        "score_n": 0,
        "margin_sum": 0.0,
        "margin_n": 0,
        "strong": 0,
        "usable": 0,
        "low": 0,
        "weak_top1": 0,
        "weak_top2": 0,
        "weak_top3": 0,
        "weak_region": 0,
        "strong_weak_top1": 0,
        "strong_weak_region": 0,
    }


def add_metric(
    metrics: dict[str, float | int],
    *,
    duration: float,
    score: float | None,
    margin: float | None,
    tier: str,
    weak_top1: bool,
    weak_top2: bool,
    weak_top3: bool,
    weak_region: bool,
) -> None:
    metrics["n"] += 1
    metrics["duration_sec"] += duration
    if score is not None:
        metrics["score_sum"] += score
        metrics["score_n"] += 1
    if margin is not None:
        metrics["margin_sum"] += margin
        metrics["margin_n"] += 1
    metrics[tier] += 1
    metrics["weak_top1"] += int(weak_top1)
    metrics["weak_top2"] += int(weak_top2)
    metrics["weak_top3"] += int(weak_top3)
    metrics["weak_region"] += int(weak_region)
    if tier == "strong":
        metrics["strong_weak_top1"] += int(weak_top1)
        metrics["strong_weak_region"] += int(weak_region)


def pct(num: float, den: float) -> float:
    return 0.0 if den == 0 else 100.0 * num / den


def compact(metrics: dict[str, float | int]) -> dict[str, Any]:
    n = int(metrics["n"])
    strong = int(metrics["strong"])
    usable = int(metrics["usable"])
    low = int(metrics["low"])
    return {
        "n": n,
        "hours": float(metrics["duration_sec"]) / 3600.0,
        "avg_score": None if not metrics["score_n"] else float(metrics["score_sum"]) / float(metrics["score_n"]),
        "avg_margin": None if not metrics["margin_n"] else float(metrics["margin_sum"]) / float(metrics["margin_n"]),
        "strong": strong,
        "usable": usable,
        "low": low,
        "strong_pct": pct(strong, n),
        "usable_pct": pct(usable, n),
        "low_pct": pct(low, n),
        "weak_top1": int(metrics["weak_top1"]),
        "weak_top2": int(metrics["weak_top2"]),
        "weak_top3": int(metrics["weak_top3"]),
        "weak_region": int(metrics["weak_region"]),
        "weak_top1_pct": pct(float(metrics["weak_top1"]), n),
        "weak_top2_pct": pct(float(metrics["weak_top2"]), n),
        "weak_top3_pct": pct(float(metrics["weak_top3"]), n),
        "weak_region_pct": pct(float(metrics["weak_region"]), n),
        "strong_weak_top1": int(metrics["strong_weak_top1"]),
        "strong_weak_region": int(metrics["strong_weak_region"]),
        "strong_weak_top1_pct": pct(float(metrics["strong_weak_top1"]), strong),
        "strong_weak_region_pct": pct(float(metrics["strong_weak_region"]), strong),
    }


def tier_for(score: float | None, margin: float | None, args: argparse.Namespace) -> str:
    if score is None or margin is None:
        return "low"
    if score >= args.strong_score and margin >= args.strong_margin:
        return "strong"
    if score >= args.usable_score and margin >= args.usable_margin:
        return "usable"
    return "low"


def to_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def iter_rows(conn: sqlite3.Connection, batch_size: int):
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
          weak_cc,
          decoded,
          duration,
          top1_cc,
          top1_score,
          top2_cc,
          top2_score,
          top3_cc,
          top3_score,
          margin,
          entropy,
          top1_region
        FROM samples
        """
    )
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        yield from rows


def format_pct(value: float | None) -> str:
    return "NA" if value is None else f"{value:.2f}%"


def format_num(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--strong-score", type=float, default=0.95)
    parser.add_argument("--strong-margin", type=float, default=0.50)
    parser.add_argument("--usable-score", type=float, default=0.90)
    parser.add_argument("--usable-margin", type=float, default=0.30)
    parser.add_argument("--batch-size", type=int, default=100000)
    parser.add_argument("--top-mismatches", type=int, default=100)
    args = parser.parse_args()

    start = time.time()
    conn = sqlite3.connect(f"file:{args.tracker}?mode=ro&immutable=1", uri=True)

    overall = metric_dict()
    decoded_rows = 0
    nondecoded_rows = 0
    by_pred: dict[str, dict[str, float | int]] = defaultdict(metric_dict)
    by_weak_cc: dict[str, dict[str, float | int]] = defaultdict(metric_dict)
    by_weak_region: dict[str, dict[str, float | int]] = defaultdict(metric_dict)
    by_pred_region: dict[str, dict[str, float | int]] = defaultdict(metric_dict)
    by_policy: dict[str, dict[str, float | int]] = defaultdict(metric_dict)
    by_policy_pred: dict[str, dict[str, dict[str, float | int]]] = defaultdict(lambda: defaultdict(metric_dict))
    mismatches: Counter[str] = Counter()
    strong_mismatches: Counter[str] = Counter()

    for (
        weak_cc,
        decoded,
        duration,
        top1_cc,
        top1_score,
        top2_cc,
        _top2_score,
        top3_cc,
        _top3_score,
        margin,
        entropy,
        top1_region,
    ) in iter_rows(conn, args.batch_size):
        if int(decoded or 0) != 1:
            nondecoded_rows += 1
            continue
        decoded_rows += 1
        weak = str(weak_cc or "").upper()
        pred = str(top1_cc or "").upper()
        pred2 = str(top2_cc or "").upper()
        pred3 = str(top3_cc or "").upper()
        weak_region_name = CODE_TO_REGION.get(weak, "unknown")
        pred_region_name = str(top1_region or CODE_TO_REGION.get(pred, "unknown") or "unknown")
        score = to_float(top1_score)
        margin_f = to_float(margin)
        entropy_f = to_float(entropy)
        duration_f = float(duration or 0.0)
        tier = tier_for(score, margin_f, args)
        weak_top1 = bool(weak and weak == pred)
        weak_top2 = bool(weak and weak in {pred, pred2})
        weak_top3 = bool(weak and weak in {pred, pred2, pred3})
        weak_region = bool(weak_region_name != "unknown" and pred_region_name == weak_region_name)
        policy_flags = sure_policy_flags(
            score=score,
            margin=margin_f,
            entropy=entropy_f,
            weak_region_match=weak_region,
            weak_top1_match=weak_top1,
        )

        kwargs = {
            "duration": duration_f,
            "score": score,
            "margin": margin_f,
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
        for policy, value in policy_flags.items():
            if value:
                add_metric(by_policy[policy], **kwargs)
                add_metric(by_policy_pred[policy][pred or "MISSING"], **kwargs)
        if weak and pred and weak != pred:
            key = f"{weak}\t{pred}"
            mismatches[key] += 1
            if tier == "strong":
                strong_mismatches[key] += 1

    conn.close()

    overall_row = compact(overall)
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

    weak_region_rows = [
        {**compact(metrics), "weak_region": region}
        for region, metrics in by_weak_region.items()
    ]
    weak_region_rows.sort(key=lambda row: (-float(row["weak_region_pct"]), -int(row["n"]), str(row["weak_region"])))

    pred_region_rows = [
        {"pred_region": region, **compact(metrics)}
        for region, metrics in by_pred_region.items()
    ]
    pred_region_rows.sort(key=lambda row: (-int(row["strong"]), -int(row["n"]), str(row["pred_region"])))
    policy_order = {str(spec["policy"]): idx for idx, spec in enumerate(SURE_POLICY_SPECS)}
    policy_descriptions = {str(spec["policy"]): str(spec["description"]) for spec in SURE_POLICY_SPECS}
    policy_rows = [
        {
            "policy": policy,
            "description": policy_descriptions.get(policy, ""),
            "pct_of_decoded": pct(int(metrics["n"]), decoded_rows),
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
                    "pct_of_decoded": pct(int(metrics["n"]), decoded_rows),
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
            "final_top1_cc": pair.split("\t", 1)[1],
            "rows": count,
            "strong_rows": strong_mismatches.get(pair, 0),
        }
        for pair, count in mismatches.most_common(args.top_mismatches)
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "tracker": str(args.tracker),
        "generated_at_epoch": time.time(),
        "wall_sec": time.time() - start,
        "thresholds": {
            "strong_score": args.strong_score,
            "strong_margin": args.strong_margin,
            "usable_score": args.usable_score,
            "usable_margin": args.usable_margin,
        },
        "decoded_rows": decoded_rows,
        "nondecoded_rows": nondecoded_rows,
        "overall": overall_row,
        "by_predicted_dialect": pred_rows,
        "by_weak_cc_ranked_by_top1_pct": weak_rows,
        "by_weak_region": weak_region_rows,
        "by_predicted_region": pred_region_rows,
        "for_sure_policies": policy_rows,
        "for_sure_by_predicted_dialect": policy_pred_rows,
        "top_weak_vs_final_mismatches": mismatch_rows,
    }
    (args.out_dir / "final_dialect_decision_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
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
    write_csv(args.out_dir / "by_predicted_dialect.csv", pred_rows, ["pred_cc", "country", "region", *common_fields])
    write_csv(args.out_dir / "by_weak_cc_ranked_by_top1_pct.csv", weak_rows, ["weak_cc", "country", "region", *common_fields])
    write_csv(args.out_dir / "by_weak_region.csv", weak_region_rows, ["weak_region", *common_fields])
    write_csv(args.out_dir / "by_predicted_region.csv", pred_region_rows, ["pred_region", *common_fields])
    write_csv(
        args.out_dir / "for_sure_policies.csv",
        policy_rows,
        ["policy", "description", "pct_of_decoded", *common_fields],
    )
    write_csv(
        args.out_dir / "for_sure_by_predicted_dialect.csv",
        policy_pred_rows,
        ["policy", "pred_cc", "country", "region", "pct_of_decoded", *common_fields],
    )
    write_csv(args.out_dir / "top_weak_vs_final_mismatches.csv", mismatch_rows, ["weak_cc", "final_top1_cc", "rows", "strong_rows"])

    lines = [
        "# Final BIG-RUN Dialect Decision Summary",
        "",
        f"- Tracker: `{args.tracker}`",
        f"- Decoded rows: `{decoded_rows:,}`",
        f"- Non-decoded rows: `{nondecoded_rows:,}`",
        f"- Strong threshold: `top1_score >= {args.strong_score}` and `margin >= {args.strong_margin}`",
        f"- Usable threshold: `top1_score >= {args.usable_score}` and `margin >= {args.usable_margin}`",
        f"- Strong final labels: `{overall_row['strong']:,}` ({format_pct(overall_row['strong_pct'])})",
        f"- Usable final labels: `{overall_row['usable']:,}` ({format_pct(overall_row['usable_pct'])})",
        f"- Low-confidence labels: `{overall_row['low']:,}` ({format_pct(overall_row['low_pct'])})",
        f"- Weak/path top1 agreement: `{overall_row['weak_top1']:,}` / `{overall_row['n']:,}` ({format_pct(overall_row['weak_top1_pct'])})",
        f"- Weak/path region agreement: `{overall_row['weak_region']:,}` / `{overall_row['n']:,}` ({format_pct(overall_row['weak_region_pct'])})",
        "",
        "Weak/path country code is used only as metadata here; it is not treated as ground truth.",
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
            "| Policy | Final top1 | Region | Rows | Coverage | Weak top1 % | Weak region % |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in policy_pred_rows:
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
        "| Final top1 | Region | Rows | Strong | Strong % | Avg score | Avg margin | Weak top1 % | Weak region % |",
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
            "## Weak/Path Regions",
            "",
            "| Weak region | Rows | Top1 dialect agree | Region agree | Strong rows | Strong region agree |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in weak_region_rows:
        lines.append(
            f"| {row['weak_region']} | {int(row['n']):,} | {format_pct(row['weak_top1_pct'])} | "
            f"{format_pct(row['weak_region_pct'])} | {int(row['strong']):,} | {format_pct(row['strong_weak_region_pct'])} |"
        )

    lines.extend(
        [
            "",
            "## Largest Weak/Path vs Final Top1 Mismatches",
            "",
            "| Weak CC | Final top1 | Rows | Strong rows |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in mismatch_rows[:75]:
        lines.append(f"| {row['weak_cc']} | {row['final_top1_cc']} | {int(row['rows']):,} | {int(row['strong_rows']):,} |")
    lines.append("")
    (args.out_dir / "final_dialect_decision_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out_dir / 'final_dialect_decision_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
