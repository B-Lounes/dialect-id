#!/usr/bin/env python3
"""Build post-hoc diagnostics for the selected 23-way DID checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dialect_id.labels import DIALECTS, ID_TO_CODE, ID_TO_REGION


CONFIDENCE_BUCKETS = (
    (0.0, 0.50, "<0.50"),
    (0.50, 0.70, "0.50-0.70"),
    (0.70, 0.85, "0.70-0.85"),
    (0.85, 0.95, "0.85-0.95"),
    (0.95, 0.98, "0.95-0.98"),
    (0.98, 1.0000001, ">=0.98"),
)
DURATION_BUCKETS = (
    (0.0, 3.0, "<3s"),
    (3.0, 10.0, "3-10s"),
    (10.0, 20.0, "10-20s"),
    (20.0, 30.0, "20-30s"),
    (30.0, 60.0, "30-60s"),
    (60.0, float("inf"), ">=60s"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-confusions", type=int, default=25)
    return parser.parse_args()


def prediction_split(path: Path) -> str:
    name = path.name
    marker = "_predictions"
    if marker not in name:
        return path.stem
    return name.split(marker, 1)[0]


def iter_prediction_files(report_dirs: list[Path]) -> dict[str, list[Path]]:
    by_split: dict[str, list[Path]] = defaultdict(list)
    for report_dir in report_dirs:
        for path in sorted(report_dir.glob("*_predictions*.jsonl")):
            by_split[prediction_split(path)].append(path)
    return dict(sorted(by_split.items()))


def safe_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    xs = sorted(values)
    out = {}
    for pct in (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100):
        if len(xs) == 1:
            value = xs[0]
        else:
            pos = (len(xs) - 1) * pct / 100.0
            lo = int(math.floor(pos))
            hi = int(math.ceil(pos))
            if lo == hi:
                value = xs[lo]
            else:
                frac = pos - lo
                value = xs[lo] * (1.0 - frac) + xs[hi] * frac
        out[f"p{pct:02d}"] = float(value)
    return out


def confidence_bucket(score: float) -> str:
    for low, high, label in CONFIDENCE_BUCKETS:
        if low <= score < high:
            return label
    return "unknown"


def chunk_bucket(count: int) -> str:
    if count <= 1:
        return "1"
    if count == 2:
        return "2"
    if count <= 4:
        return "3-4"
    if count <= 8:
        return "5-8"
    return "9+"


def duration_bucket(seconds: float | None) -> str:
    if seconds is None:
        return "missing"
    for low, high, label in DURATION_BUCKETS:
        if low <= seconds < high:
            return label
    return "missing"


def entropy(probs: list[float]) -> float:
    total = 0.0
    for prob in probs:
        p = max(float(prob), 1e-12)
        total -= p * math.log(p)
    return total


def metric_row(total: int, correct: int) -> dict[str, Any]:
    return {
        "n": total,
        "correct": correct,
        "accuracy": (correct / total) if total else None,
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    correct = sum(int(row["truth"] == row["pred"]) for row in rows)
    return metric_row(n, correct)


def compact_metric(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def split_safe_name(split: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in split)


def load_split_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                topk_probs = row.get("topk_probs") or []
                top1 = safe_float(topk_probs[0], 0.0) if topk_probs else 0.0
                top2 = safe_float(topk_probs[1], 0.0) if len(topk_probs) > 1 else 0.0
                probs = row.get("probs") if isinstance(row.get("probs"), list) else []
                duration = safe_float(row.get("duration_seconds"), None)
                effective_duration = safe_float(row.get("effective_duration_seconds"), None)
                rows.append(
                    {
                        "truth": int(row["truth"]),
                        "pred": int(row["pred"]),
                        "top1_score": top1,
                        "margin": top1 - top2,
                        "entropy": entropy([safe_float(x, 0.0) or 0.0 for x in probs]),
                        "crop_count": int(row.get("crop_count") or 1),
                        "chunk_count": int(row.get("chunk_count") or row.get("crop_count") or 1),
                        "duration_seconds": duration,
                        "effective_duration_seconds": effective_duration,
                        "source": row.get("source") or "",
                        "country_code": row.get("country_code") or "",
                    }
                )
    return rows


def per_dialect(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_truth: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_truth[int(row["truth"])].append(row)
    out: list[dict[str, Any]] = []
    for label in DIALECTS:
        group = by_truth.get(label.id, [])
        pred_counts = Counter(int(row["pred"]) for row in group if int(row["pred"]) != label.id)
        top_confusions = ", ".join(
            f"{ID_TO_CODE.get(pred, str(pred))}:{count}" for pred, count in pred_counts.most_common(5)
        )
        metric = summarize_group(group)
        out.append(
            {
                "dialect": label.code,
                "country": label.country,
                "region": label.region,
                "support": metric["n"],
                "correct": metric["correct"],
                "accuracy": metric["accuracy"],
                "top_confusions": top_confusions,
            }
        )
    return out


def per_region(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_region[ID_TO_REGION.get(int(row["truth"]), "unknown")].append(row)
    out = []
    for region, group in sorted(by_region.items()):
        exact_correct = sum(int(row["truth"] == row["pred"]) for row in group)
        region_correct = sum(
            int(ID_TO_REGION.get(int(row["truth"])) == ID_TO_REGION.get(int(row["pred"])))
            for row in group
        )
        pred_regions = Counter(ID_TO_REGION.get(int(row["pred"]), "unknown") for row in group)
        n = len(group)
        out.append(
            {
                "region": region,
                "support": n,
                "dialect_correct": exact_correct,
                "dialect_accuracy": (exact_correct / n) if n else None,
                "region_correct": region_correct,
                "region_accuracy": (region_correct / n) if n else None,
                "top_pred_regions": ", ".join(f"{key}:{value}" for key, value in pred_regions.most_common(5)),
            }
        )
    return out


def bucket_rows(rows: list[dict[str, Any]], *, kind: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if kind == "confidence":
            key = confidence_bucket(float(row["top1_score"]))
        elif kind == "chunk_count":
            key = chunk_bucket(int(row["chunk_count"]))
        elif kind == "duration":
            key = duration_bucket(row.get("duration_seconds"))
        else:
            raise ValueError(kind)
        groups[key].append(row)
    if kind == "confidence":
        order = [label for _, _, label in CONFIDENCE_BUCKETS] + ["unknown"]
    elif kind == "duration":
        order = [label for _, _, label in DURATION_BUCKETS] + ["missing"]
    else:
        order = ["1", "2", "3-4", "5-8", "9+"]
    out = []
    for key in order:
        group = groups.get(key, [])
        if not group:
            continue
        duration_values = [float(row["duration_seconds"]) for row in group if row.get("duration_seconds") is not None]
        metric = summarize_group(group)
        out.append(
            {
                "bucket": key,
                "support": metric["n"],
                "correct": metric["correct"],
                "accuracy": metric["accuracy"],
                "avg_top1_score": sum(float(row["top1_score"]) for row in group) / len(group),
                "avg_margin": sum(float(row["margin"]) for row in group) / len(group),
                "avg_entropy": sum(float(row["entropy"]) for row in group) / len(group),
                "avg_crop_count": sum(int(row["crop_count"]) for row in group) / len(group),
                "avg_chunk_count": sum(int(row["chunk_count"]) for row in group) / len(group),
                "avg_duration_seconds": (sum(duration_values) / len(duration_values)) if duration_values else None,
            }
        )
    return out


def top_confusions(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    supports = Counter(int(row["truth"]) for row in rows)
    counts: Counter[tuple[int, int]] = Counter()
    for row in rows:
        truth = int(row["truth"])
        pred = int(row["pred"])
        if truth != pred:
            counts[(truth, pred)] += 1
    out = []
    for (truth, pred), count in counts.most_common(limit):
        support = supports[truth]
        out.append(
            {
                "truth": ID_TO_CODE.get(truth, str(truth)),
                "pred": ID_TO_CODE.get(pred, str(pred)),
                "truth_region": ID_TO_REGION.get(truth, "unknown"),
                "pred_region": ID_TO_REGION.get(pred, "unknown"),
                "count": count,
                "truth_support": support,
                "pct_of_truth": (count / support) if support else None,
            }
        )
    return out


def split_summary(split: str, rows: list[dict[str, Any]], top_confusion_limit: int) -> dict[str, Any]:
    top1_scores = [float(row["top1_score"]) for row in rows]
    margins = [float(row["margin"]) for row in rows]
    entropies = [float(row["entropy"]) for row in rows]
    crop_counts = [float(row["crop_count"]) for row in rows]
    chunk_counts = [float(row["chunk_count"]) for row in rows]
    durations = [float(row["duration_seconds"]) for row in rows if row.get("duration_seconds") is not None]
    exact = summarize_group(rows)
    region_correct = sum(
        int(ID_TO_REGION.get(int(row["truth"])) == ID_TO_REGION.get(int(row["pred"])))
        for row in rows
    )
    n = len(rows)
    return {
        "split": split,
        "overall": {
            **exact,
            "region_correct": region_correct,
            "region_accuracy": (region_correct / n) if n else None,
        },
        "score_quantiles": {
            "top1_score": quantiles(top1_scores),
            "margin": quantiles(margins),
            "entropy": quantiles(entropies),
            "crop_count": quantiles(crop_counts),
            "chunk_count": quantiles(chunk_counts),
            "duration_seconds": quantiles(durations),
        },
        "per_dialect": per_dialect(rows),
        "per_region": per_region(rows),
        "confidence_buckets": bucket_rows(rows, kind="confidence"),
        "chunk_count_buckets": bucket_rows(rows, kind="chunk_count"),
        "duration_buckets": bucket_rows(rows, kind="duration"),
        "top_confusions": top_confusions(rows, top_confusion_limit),
    }


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if value is None:
        return ""
    return str(value)


def md_table(headers: list[str], rows: list[dict[str, Any]], fields: list[str], *, max_rows: int | None = None) -> list[str]:
    shown = rows[:max_rows] if max_rows else rows
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in shown:
        lines.append("| " + " | ".join(fmt(row.get(field)) for field in fields) + " |")
    return lines


def write_split_csvs(out_dir: Path, split: str, summary: dict[str, Any]) -> None:
    prefix = split_safe_name(split)
    write_csv(
        out_dir / f"{prefix}_per_dialect.csv",
        summary["per_dialect"],
        ["dialect", "country", "region", "support", "correct", "accuracy", "top_confusions"],
    )
    write_csv(
        out_dir / f"{prefix}_per_region.csv",
        summary["per_region"],
        [
            "region",
            "support",
            "dialect_correct",
            "dialect_accuracy",
            "region_correct",
            "region_accuracy",
            "top_pred_regions",
        ],
    )
    write_csv(
        out_dir / f"{prefix}_confidence_buckets.csv",
        summary["confidence_buckets"],
        [
            "bucket",
            "support",
            "correct",
            "accuracy",
            "avg_top1_score",
            "avg_margin",
            "avg_entropy",
            "avg_crop_count",
            "avg_chunk_count",
            "avg_duration_seconds",
        ],
    )
    write_csv(
        out_dir / f"{prefix}_chunk_count_buckets.csv",
        summary["chunk_count_buckets"],
        [
            "bucket",
            "support",
            "correct",
            "accuracy",
            "avg_top1_score",
            "avg_margin",
            "avg_entropy",
            "avg_crop_count",
            "avg_chunk_count",
            "avg_duration_seconds",
        ],
    )
    write_csv(
        out_dir / f"{prefix}_duration_buckets.csv",
        summary["duration_buckets"],
        [
            "bucket",
            "support",
            "correct",
            "accuracy",
            "avg_top1_score",
            "avg_margin",
            "avg_entropy",
            "avg_crop_count",
            "avg_chunk_count",
            "avg_duration_seconds",
        ],
    )
    write_csv(
        out_dir / f"{prefix}_top_confusions.csv",
        summary["top_confusions"],
        ["truth", "pred", "truth_region", "pred_region", "count", "truth_support", "pct_of_truth"],
    )


def write_markdown(out_dir: Path, summaries: dict[str, Any], report_dirs: list[Path]) -> None:
    lines = [
        "# Selected Step400 Diagnostics",
        "",
        "Derived from the listed prediction JSONL shards.",
        "",
        "Input reports:",
        "",
    ]
    for path in report_dirs:
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "Duration buckets are reported when `duration_seconds` is present in the prediction shards. "
            "Older shards without that field show a `missing` duration bucket.",
            "",
            "## Split Summary",
            "",
            "| Split | Rows | Accuracy | Region accuracy | Top1 p50 | Top1 p90 | Margin p50 | Entropy p50 | Duration p50 | Chunk-count p90 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split, summary in summaries.items():
        q = summary["score_quantiles"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{split}`",
                    f"{summary['overall']['n']}",
                    fmt(summary["overall"]["accuracy"]),
                    fmt(summary["overall"]["region_accuracy"]),
                    fmt(q["top1_score"].get("p50")),
                    fmt(q["top1_score"].get("p90")),
                    fmt(q["margin"].get("p50")),
                    fmt(q["entropy"].get("p50")),
                    fmt(q["duration_seconds"].get("p50")),
                    fmt(q["chunk_count"].get("p90")),
                ]
            )
            + " |"
        )
    for split, summary in summaries.items():
        lines.extend(["", f"## {split}", "", "### Per Region", ""])
        lines.extend(
            md_table(
                ["Region", "Rows", "Dialect Acc", "Region Acc", "Top Pred Regions"],
                summary["per_region"],
                ["region", "support", "dialect_accuracy", "region_accuracy", "top_pred_regions"],
            )
        )
        lines.extend(["", "### Confidence Buckets", ""])
        lines.extend(
            md_table(
                ["Bucket", "Rows", "Accuracy", "Avg Margin", "Avg Entropy"],
                summary["confidence_buckets"],
                ["bucket", "support", "accuracy", "avg_margin", "avg_entropy"],
            )
        )
        lines.extend(["", "### Chunk Count Buckets", ""])
        lines.extend(
            md_table(
                ["Chunks", "Rows", "Accuracy", "Avg Top1", "Avg Margin", "Avg Duration"],
                summary["chunk_count_buckets"],
                ["bucket", "support", "accuracy", "avg_top1_score", "avg_margin", "avg_duration_seconds"],
            )
        )
        lines.extend(["", "### Duration Buckets", ""])
        lines.extend(
            md_table(
                ["Duration", "Rows", "Accuracy", "Avg Top1", "Avg Margin", "Avg Chunks"],
                summary["duration_buckets"],
                ["bucket", "support", "accuracy", "avg_top1_score", "avg_margin", "avg_chunk_count"],
            )
        )
        lines.extend(["", "### Lowest Per-Dialect Accuracy", ""])
        nonempty = [row for row in summary["per_dialect"] if int(row["support"]) > 0]
        nonempty.sort(key=lambda row: (float(row["accuracy"] or 0.0), -int(row["support"])))
        lines.extend(
            md_table(
                ["Dialect", "Region", "Rows", "Accuracy", "Top Confusions"],
                nonempty,
                ["dialect", "region", "support", "accuracy", "top_confusions"],
                max_rows=10,
            )
        )
        lines.extend(["", "### Top Confusions", ""])
        lines.extend(
            md_table(
                ["Truth", "Pred", "Truth Region", "Pred Region", "Count", "Pct Of Truth"],
                summary["top_confusions"],
                ["truth", "pred", "truth_region", "pred_region", "count", "pct_of_truth"],
                max_rows=15,
            )
        )
    (out_dir / "selected_model_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    report_dirs = [Path(path) for path in args.report_dir]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files_by_split = iter_prediction_files(report_dirs)
    summaries: dict[str, Any] = {}
    for split, paths in files_by_split.items():
        rows = load_split_rows(paths)
        summaries[split] = split_summary(split, rows, args.top_confusions)
        write_split_csvs(out_dir, split, summaries[split])
    payload = {
        "report_dirs": [str(path) for path in report_dirs],
        "splits": summaries,
        "notes": [
            "Derived from saved prediction JSONL shards.",
            "Duration and true full-audio chunk count are reported when prediction shards include duration_seconds and chunk_count.",
        ],
    }
    (out_dir / "selected_model_diagnostics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(out_dir, summaries, report_dirs)
    print(json.dumps({"out_dir": str(out_dir), "splits": list(summaries), "rows": {k: v["overall"]["n"] for k, v in summaries.items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
