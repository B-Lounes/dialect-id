#!/usr/bin/env python3
"""Stream final BIG-RUN labels and decision reports from prediction CSV shards."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dialect_id.labels import CODE_TO_COUNTRY, CODE_TO_REGION
try:
    from .export_big_run_final_labels import (
        FIELDS,
        SURE_POLICY_SPECS,
        ShardedWriter,
        confidence_tier,
        format_float,
        pct,
        prepare_output_dir,
        sure_policy_flags,
        sure_policy_summary,
        to_float,
    )
    from .report_big_run_final_dialect_decisions import (
        add_metric,
        compact,
        format_num,
        format_pct,
        metric_dict,
        write_csv,
    )
except ImportError:  # Direct script execution.
    from export_big_run_final_labels import (
        FIELDS,
        SURE_POLICY_SPECS,
        ShardedWriter,
        confidence_tier,
        format_float,
        pct,
        prepare_output_dir,
        sure_policy_flags,
        sure_policy_summary,
        to_float,
    )
    from report_big_run_final_dialect_decisions import (
        add_metric,
        compact,
        format_num,
        format_pct,
        metric_dict,
        write_csv,
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_done_markers(state_dir: Path):
    for path in sorted(state_dir.glob("shard_*.done")):
        yield path, read_json(path)


def bool_decoded(value: object) -> bool:
    return str(value or "") == "1"


def to_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def update_decision_metrics(
    *,
    row: dict[str, str],
    tier: str,
    overall: dict[str, float | int],
    by_pred: dict[str, dict[str, float | int]],
    by_weak_cc: dict[str, dict[str, float | int]],
    by_weak_region: dict[str, dict[str, float | int]],
    by_pred_region: dict[str, dict[str, float | int]],
    mismatches: Counter[str],
    strong_mismatches: Counter[str],
) -> None:
    weak = str(row.get("weak_cc") or "").upper()
    pred = str(row.get("top1_cc") or "").upper()
    pred2 = str(row.get("top2_cc") or "").upper()
    pred3 = str(row.get("top3_cc") or "").upper()
    weak_region_name = CODE_TO_REGION.get(weak, "unknown")
    pred_region_name = str(row.get("top1_region") or CODE_TO_REGION.get(pred, "unknown") or "unknown")
    score = to_float(row.get("top1_score"))
    margin = to_float(row.get("margin"))
    duration = float(to_float(row.get("duration")) or 0.0)
    weak_top1 = bool(weak and weak == pred)
    weak_top2 = bool(weak and weak in {pred, pred2})
    weak_top3 = bool(weak and weak in {pred, pred2, pred3})
    weak_region = bool(weak_region_name != "unknown" and pred_region_name == weak_region_name)
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
    if weak and pred and weak != pred:
        key = f"{weak}\t{pred}"
        mismatches[key] += 1
        if tier == "strong":
            strong_mismatches[key] += 1


def write_decision_report(
    *,
    out_dir: Path,
    tracker_label: str,
    start: float,
    args: argparse.Namespace,
    decoded_rows: int,
    nondecoded_rows: int,
    done_markers: int | None = None,
    overall: dict[str, float | int],
    by_pred: dict[str, dict[str, float | int]],
    by_weak_cc: dict[str, dict[str, float | int]],
    by_weak_region: dict[str, dict[str, float | int]],
    by_pred_region: dict[str, dict[str, float | int]],
    by_policy: dict[str, dict[str, float | int]],
    by_policy_pred: dict[str, dict[str, dict[str, float | int]]],
    mismatches: Counter[str],
    strong_mismatches: Counter[str],
) -> None:
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
    weak_region_rows = [{**compact(metrics), "weak_region": region} for region, metrics in by_weak_region.items()]
    weak_region_rows.sort(key=lambda row: (-float(row["weak_region_pct"]), -int(row["n"]), str(row["weak_region"])))
    pred_region_rows = [{"pred_region": region, **compact(metrics)} for region, metrics in by_pred_region.items()]
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

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "tracker": tracker_label,
        "generated_at_epoch": time.time(),
        "wall_sec": time.time() - start,
        "source": "prediction_csv_stream",
        "thresholds": {
            "strong_score": args.strong_score,
            "strong_margin": args.strong_margin,
            "usable_score": args.usable_score,
            "usable_margin": args.usable_margin,
        },
        "done_markers": done_markers,
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
    (out_dir / "final_dialect_decision_summary.json").write_text(
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
    write_csv(out_dir / "top_weak_vs_final_mismatches.csv", mismatch_rows, ["weak_cc", "final_top1_cc", "rows", "strong_rows"])

    lines = [
        "# Final BIG-RUN Dialect Decision Summary",
        "",
        f"- Source: `{tracker_label}`",
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
            "## Largest Weak/Path vs Final Top1 Mismatches",
            "",
            "| Weak CC | Final top1 | Rows | Strong rows |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in mismatch_rows[:75]:
        lines.append(f"| {row['weak_cc']} | {row['final_top1_cc']} | {int(row['rows']):,} | {int(row['strong_rows']):,} |")
    lines.append("")
    (out_dir / "final_dialect_decision_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--decision-out-dir", type=Path, required=True)
    parser.add_argument("--strong-score", type=float, default=0.95)
    parser.add_argument("--strong-margin", type=float, default=0.50)
    parser.add_argument("--usable-score", type=float, default=0.90)
    parser.add_argument("--usable-margin", type=float, default=0.30)
    parser.add_argument("--rows-per-file", type=int, default=1_000_000)
    parser.add_argument("--gzip-level", type=int, default=1)
    parser.add_argument("--gzip-backend", choices=["python", "pigz", "auto"], default="python")
    parser.add_argument("--pigz-threads", type=int, default=0)
    parser.add_argument("--require-all-decoded", action="store_true")
    parser.add_argument("--stale-lock-seconds", type=float, default=3600.0)
    parser.add_argument("--top-mismatches", type=int, default=100)
    parser.add_argument("--progress-every-rows", type=int, default=5_000_000)
    args = parser.parse_args()

    start = time.time()
    state_dir = args.out_root / "state"
    in_progress, lock_path = prepare_output_dir(args.out_dir, args.stale_lock_seconds)
    writer = ShardedWriter(
        args.out_dir,
        rows_per_file=args.rows_per_file,
        gzip_level=args.gzip_level,
        gzip_backend=args.gzip_backend,
        pigz_threads=args.pigz_threads,
    )
    tier_counts: Counter[str] = Counter()
    dialect_counts: Counter[str] = Counter()
    region_counts: Counter[str] = Counter()
    weak_match_counts: Counter[str] = Counter()
    sure_policy_counts: Counter[str] = Counter()
    tier_by_dialect: dict[str, Counter[str]] = defaultdict(Counter)
    overall = metric_dict()
    by_pred: dict[str, dict[str, float | int]] = defaultdict(metric_dict)
    by_weak_cc: dict[str, dict[str, float | int]] = defaultdict(metric_dict)
    by_weak_region: dict[str, dict[str, float | int]] = defaultdict(metric_dict)
    by_pred_region: dict[str, dict[str, float | int]] = defaultdict(metric_dict)
    by_policy: dict[str, dict[str, float | int]] = defaultdict(metric_dict)
    by_policy_pred: dict[str, dict[str, dict[str, float | int]]] = defaultdict(lambda: defaultdict(metric_dict))
    mismatches: Counter[str] = Counter()
    strong_mismatches: Counter[str] = Counter()

    done_markers = 0
    marker_rows = 0
    marker_decoded = 0
    marker_failed = 0
    csv_rows = 0
    decoded_rows = 0
    nondecoded_rows = 0

    for _done_path, marker in iter_done_markers(state_dir):
        done_markers += 1
        marker_rows += to_int(marker.get("csv_rows"), to_int(marker.get("expanded_rows")))
        marker_decoded += to_int(marker.get("decoded_rows"))
        marker_failed += to_int(marker.get("failed_rows"))
        csv_path = Path(str(marker.get("output_csv", "")))
        if not csv_path.is_file():
            raise SystemExit(f"Missing prediction CSV from done marker: {csv_path}")
        file_rows = 0
        file_decoded = 0
        file_failed = 0
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                file_rows += 1
                csv_rows += 1
                decoded = bool_decoded(row.get("decoded"))
                file_decoded += int(decoded)
                file_failed += int(not decoded)
                if not decoded:
                    nondecoded_rows += 1
                    if args.require_all_decoded:
                        raise SystemExit(f"Non-decoded row found in {csv_path}")
                    continue
                decoded_rows += 1
                if args.progress_every_rows > 0 and decoded_rows % args.progress_every_rows == 0:
                    elapsed = max(time.time() - start, 1e-9)
                    print(
                        json.dumps(
                            {
                                "event": "progress",
                                "decoded_rows": decoded_rows,
                                "csv_rows_seen": csv_rows,
                                "done_markers_seen": done_markers,
                                "label_files_opened": writer.index + 1,
                                "rows_per_sec": decoded_rows / elapsed,
                                "elapsed_sec": elapsed,
                            },
                            sort_keys=True,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )

                weak = str(row.get("weak_cc") or "").upper()
                pred = str(row.get("top1_cc") or "").upper()
                pred2 = str(row.get("top2_cc") or "").upper()
                pred3 = str(row.get("top3_cc") or "").upper()
                score = to_float(row.get("top1_score"))
                margin = to_float(row.get("margin"))
                entropy = to_float(row.get("entropy"))
                tier = confidence_tier(score, margin, args)
                weak_region = CODE_TO_REGION.get(weak, "unknown")
                pred_region = str(row.get("top1_region") or CODE_TO_REGION.get(pred, "unknown") or "unknown")
                pred_country = str(row.get("top1_country") or CODE_TO_COUNTRY.get(pred, "") or "")
                weak_top1 = bool(weak and weak == pred)
                weak_top2 = bool(weak and weak in {pred, pred2})
                weak_top3 = bool(weak and weak in {pred, pred2, pred3})
                weak_region_match = bool(weak_region != "unknown" and weak_region == pred_region)
                policy_flags = sure_policy_flags(
                    score=score,
                    margin=margin,
                    entropy=entropy,
                    weak_region_match=weak_region_match,
                    weak_top1_match=weak_top1,
                )
                duration_f = float(to_float(row.get("duration")) or 0.0)
                metric_kwargs = {
                    "duration": duration_f,
                    "score": score,
                    "margin": margin,
                    "tier": tier,
                    "weak_top1": weak_top1,
                    "weak_top2": weak_top2,
                    "weak_top3": weak_top3,
                    "weak_region": weak_region_match,
                }
                out_row = {
                    "sample_id": row.get("sample_id", ""),
                    "manifest_row": row.get("manifest_row", ""),
                    "tar_path": row.get("tar_path", ""),
                    "member": row.get("member", ""),
                    "youtube_id": row.get("youtube_id", ""),
                    "duration": format_float(row.get("duration")),
                    "decoded": 1,
                    "weak_cc": weak,
                    "weak_region": weak_region,
                    "pseudo_cc": row.get("pseudo_cc", ""),
                    "teacher_cc": row.get("teacher_cc", ""),
                    "teacher_confidence": format_float(row.get("teacher_confidence")),
                    "teacher_margin": format_float(row.get("teacher_margin")),
                    "pseudo_label_source": row.get("pseudo_label_source", ""),
                    "whisper_language": row.get("whisper_language", ""),
                    "whisper_language_confidence": format_float(row.get("whisper_language_confidence")),
                    "final_dialect": pred,
                    "final_country": pred_country,
                    "final_region": pred_region,
                    "confidence_tier": tier,
                    **policy_flags,
                    "top1_score": format_float(score),
                    "margin": format_float(margin),
                    "entropy": format_float(entropy),
                    "top2_cc": pred2,
                    "top2_score": format_float(row.get("top2_score")),
                    "top3_cc": pred3,
                    "top3_score": format_float(row.get("top3_score")),
                    "weak_top1_match": int(weak_top1),
                    "weak_top2_match": int(weak_top2),
                    "weak_top3_match": int(weak_top3),
                    "weak_region_match": int(weak_region_match),
                    "source_prediction_csv": str(csv_path),
                }
                writer.write(out_row)
                tier_counts[tier] += 1
                dialect_counts[pred or "MISSING"] += 1
                region_counts[pred_region] += 1
                tier_by_dialect[pred or "MISSING"][tier] += 1
                weak_match_counts["top1"] += int(weak_top1)
                weak_match_counts["top2"] += int(weak_top2)
                weak_match_counts["top3"] += int(weak_top3)
                weak_match_counts["region"] += int(weak_region_match)
                for policy, value in policy_flags.items():
                    sure_policy_counts[policy] += int(value)
                    if value:
                        add_metric(by_policy[policy], **metric_kwargs)
                        add_metric(by_policy_pred[policy][pred or "MISSING"], **metric_kwargs)
                update_decision_metrics(
                    row=row,
                    tier=tier,
                    overall=overall,
                    by_pred=by_pred,
                    by_weak_cc=by_weak_cc,
                    by_weak_region=by_weak_region,
                    by_pred_region=by_pred_region,
                    mismatches=mismatches,
                    strong_mismatches=strong_mismatches,
                )
        if file_rows != to_int(marker.get("csv_rows"), to_int(marker.get("expanded_rows"))):
            raise SystemExit(f"CSV row count mismatch for {csv_path}: csv={file_rows}, marker={marker}")
        if file_decoded != to_int(marker.get("decoded_rows")) or file_failed != to_int(marker.get("failed_rows")):
            raise SystemExit(
                f"CSV decoded/failed mismatch for {csv_path}: "
                f"decoded={file_decoded}, failed={file_failed}, marker={marker}"
            )

    writer.close()

    manifest_path = args.out_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        manifest_writer = csv.DictWriter(handle, fieldnames=["file", "rows", "first_sample_id", "last_sample_id", "bytes"])
        manifest_writer.writeheader()
        manifest_writer.writerows(writer.manifest_rows)

    by_dialect = []
    for dialect, count in dialect_counts.most_common():
        tiers = tier_by_dialect[dialect]
        by_dialect.append(
            {
                "final_dialect": dialect,
                "country": CODE_TO_COUNTRY.get(dialect, ""),
                "region": CODE_TO_REGION.get(dialect, "unknown"),
                "rows": count,
                "strong": tiers.get("strong", 0),
                "usable": tiers.get("usable", 0),
                "low": tiers.get("low", 0),
                "strong_pct": pct(tiers.get("strong", 0), count),
            }
        )

    summary = {
        "tracker": str(args.out_root / "tracker.sqlite"),
        "out_dir": str(args.out_dir),
        "generated_at_epoch": time.time(),
        "wall_sec": time.time() - start,
        "source": "prediction_csv_stream",
        "done_markers": done_markers,
        "marker_rows": marker_rows,
        "marker_decoded_rows": marker_decoded,
        "marker_failed_rows": marker_failed,
        "csv_rows_seen": csv_rows,
        "thresholds": {
            "strong_score": args.strong_score,
            "strong_margin": args.strong_margin,
            "usable_score": args.usable_score,
            "usable_margin": args.usable_margin,
        },
        "source_sample_counts": {
            "samples": csv_rows,
            "decoded": decoded_rows,
            "nondecoded": nondecoded_rows,
        },
        "decoded_only": True,
        "compression": {
            "gzip_level": args.gzip_level,
            "gzip_backend": args.gzip_backend,
            "pigz_threads": args.pigz_threads,
        },
        "exported_rows": writer.total_rows,
        "files": writer.manifest_rows,
        "confidence_tiers": dict(tier_counts),
        "sure_policy_specs": list(SURE_POLICY_SPECS),
        "sure_policy_counts": sure_policy_summary(sure_policy_counts, writer.total_rows),
        "weak_match_counts": {
            "top1": weak_match_counts.get("top1", 0),
            "top1_pct": pct(weak_match_counts.get("top1", 0), writer.total_rows),
            "top2": weak_match_counts.get("top2", 0),
            "top2_pct": pct(weak_match_counts.get("top2", 0), writer.total_rows),
            "top3": weak_match_counts.get("top3", 0),
            "top3_pct": pct(weak_match_counts.get("top3", 0), writer.total_rows),
            "region": weak_match_counts.get("region", 0),
            "region_pct": pct(weak_match_counts.get("region", 0), writer.total_rows),
        },
        "by_final_dialect": by_dialect,
        "by_final_region": dict(region_counts),
        "columns": FIELDS,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    readme_lines = [
        "# BIG-RUN Final Dialect Labels",
        "",
        f"- Source predictions: `{args.out_root / 'predictions'}`",
        f"- Exported rows: `{writer.total_rows:,}`",
        f"- Output files: `{len(writer.manifest_rows):,}`",
        f"- Strong threshold: `top1_score >= {args.strong_score}` and `margin >= {args.strong_margin}`",
        f"- Usable threshold: `top1_score >= {args.usable_score}` and `margin >= {args.usable_margin}`",
        "",
        "Each row is one decoded audio sample. `final_dialect` is the single-model top1 decision.",
        "`confidence_tier` is derived only from model score and top1-top2 margin.",
        "`sure_*` columns are stricter decision-policy flags; metadata-overlay policies are audit subsets, not alternate labels.",
        "`weak_cc` is path/Qwen metadata and is not treated as ground truth.",
        "",
        "Files:",
        "",
        "- `manifest.csv`: list of label shards, row counts, first/last sample id in file order, and file size.",
        "- `summary.json`: counts, thresholds, weak-metadata agreement, and dialect totals.",
        "- `final_labels_*.csv.gz`: per-sample labels and audit metadata.",
        "",
        "Columns:",
        "",
    ]
    readme_lines.extend(f"- `{field}`" for field in FIELDS)
    (args.out_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    write_decision_report(
        out_dir=args.decision_out_dir,
        tracker_label=str(args.out_root / "predictions"),
        start=start,
        args=args,
        done_markers=done_markers,
        decoded_rows=decoded_rows,
        nondecoded_rows=nondecoded_rows,
        overall=overall,
        by_pred=by_pred,
        by_weak_cc=by_weak_cc,
        by_weak_region=by_weak_region,
        by_pred_region=by_pred_region,
        by_policy=by_policy,
        by_policy_pred=by_policy_pred,
        mismatches=mismatches,
        strong_mismatches=strong_mismatches,
    )

    (args.out_dir / "_SUCCESS").write_text(
        json.dumps({"completed_at_epoch": time.time(), "exported_rows": writer.total_rows}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    in_progress.unlink(missing_ok=True)
    lock_path.unlink(missing_ok=True)
    print(json.dumps({"exported_rows": writer.total_rows, "out_dir": str(args.out_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
