#!/usr/bin/env python3
"""Gate a legacy campaign's strict no-ACGC-old19 periodic-eval continuation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_WATCH_CODES = ("BH", "JO", "KW", "OM", "AE", "QA", "SY")
OLD19_SPLIT = "acgc_old19_train_as_eval"
TD_SPLITS = ("acgc_td_validation_23way", "acgc_td_test_23way")


@dataclass(frozen=True)
class GateThresholds:
    macro_delta_min: float
    accuracy_max_regression: float
    weighted_f1_max_regression: float
    kw_f1_min: float
    ratio_distance_tolerance: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--baseline-step", type=int, default=400000)
    parser.add_argument("--target-step", type=int, default=450000)
    parser.add_argument("--watch-codes", default=",".join(DEFAULT_WATCH_CODES))
    parser.add_argument("--macro-delta-min", type=float, default=0.002)
    parser.add_argument("--accuracy-max-regression", type=float, default=0.002)
    parser.add_argument("--weighted-f1-max-regression", type=float, default=0.001)
    parser.add_argument(
        "--kw-f1-min",
        type=float,
        default=0.55,
        help="Minimum target KW F1 for releasing continuation.",
    )
    parser.add_argument(
        "--ratio-distance-tolerance",
        type=float,
        default=0.05,
        help="Allowed increase in |pred/gold - 1| for watched dialects.",
    )
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a markdown-ish text report.",
    )
    return parser.parse_args()


def step_from_name(path: Path) -> int | None:
    match = re.search(r"_step_(\d+)_", path.name)
    if match:
        return int(match.group(1))
    return None


def find_report_dir(reports_root: Path, step: int) -> Path:
    candidates = [
        path
        for path in reports_root.iterdir()
        if path.is_dir() and step_from_name(path) == step
    ]
    valid = [
        path
        for path in candidates
        if (path / f"{OLD19_SPLIT}_metrics.json").is_file()
    ]
    if not valid:
        raise FileNotFoundError(
            f"No completed periodic eval report for step {step} under {reports_root}"
        )
    return sorted(valid, key=lambda path: path.stat().st_mtime)[-1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_report(report_dir: Path) -> dict[str, Any]:
    old = load_json(report_dir / f"{OLD19_SPLIT}_metrics.json")
    payload: dict[str, Any] = {
        "report_dir": str(report_dir),
        OLD19_SPLIT: old,
    }
    for split in TD_SPLITS:
        path = report_dir / f"{split}_metrics.json"
        if path.is_file():
            payload[split] = load_json(path)
    return payload


def per_class_by_code(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["code"]): row
        for row in report[OLD19_SPLIT]["per_class"]
        if int(row.get("support", 0)) > 0
    }


def pred_gold_ratios(report: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    old = report[OLD19_SPLIT]
    classes = old["per_class"]
    cm = old["confusion_matrix"]
    out: dict[str, dict[str, float | int]] = {}
    for row in classes:
        idx = int(row["id"])
        gold = int(sum(cm[idx]))
        pred = int(sum(cm[src][idx] for src in range(len(cm))))
        if gold <= 0 and pred <= 0:
            continue
        ratio = float(pred / gold) if gold else float("inf")
        out[str(row["code"])] = {
            "gold": gold,
            "pred": pred,
            "ratio": ratio,
            "tp": int(cm[idx][idx]),
            "fp": int(pred - cm[idx][idx]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1"]),
        }
    return out


def top_false_positive_sources(
    report: dict[str, Any],
    code: str,
    *,
    limit: int = 8,
) -> list[dict[str, int | str]]:
    old = report[OLD19_SPLIT]
    classes = old["per_class"]
    cm = old["confusion_matrix"]
    id_to_code = {int(row["id"]): str(row["code"]) for row in classes}
    code_to_id = {value: key for key, value in id_to_code.items()}
    if code not in code_to_id:
        return []
    target = code_to_id[code]
    rows: list[tuple[int, str]] = []
    for truth_idx, row in enumerate(cm):
        if truth_idx == target:
            continue
        count = int(row[target])
        if count:
            rows.append((count, id_to_code.get(truth_idx, str(truth_idx))))
    return [
        {"truth": truth, "count": count}
        for count, truth in sorted(rows, reverse=True)[:limit]
    ]


def metric_deltas(
    baseline: dict[str, Any],
    target: dict[str, Any],
) -> dict[str, float]:
    base_metrics = baseline[OLD19_SPLIT]["metrics"]
    target_metrics = target[OLD19_SPLIT]["metrics"]
    return {
        key: float(target_metrics[key]) - float(base_metrics[key])
        for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
    }


def evaluate_gate(
    baseline: dict[str, Any],
    target: dict[str, Any],
    watch_codes: list[str],
    thresholds: GateThresholds,
) -> dict[str, Any]:
    deltas = metric_deltas(baseline, target)
    base_class = per_class_by_code(baseline)
    target_class = per_class_by_code(target)
    base_ratios = pred_gold_ratios(baseline)
    target_ratios = pred_gold_ratios(target)

    per_code = {}
    ratio_failures: list[str] = []
    for code in watch_codes:
        if code not in target_class or code not in base_class:
            continue
        base_f1 = float(base_class[code]["f1"])
        target_f1 = float(target_class[code]["f1"])
        base_ratio = float(base_ratios[code]["ratio"])
        target_ratio = float(target_ratios[code]["ratio"])
        base_dist = abs(base_ratio - 1.0)
        target_dist = abs(target_ratio - 1.0)
        ratio_ok = target_dist <= base_dist + thresholds.ratio_distance_tolerance
        if not ratio_ok:
            ratio_failures.append(code)
        per_code[code] = {
            "support": int(target_class[code]["support"]),
            "baseline_f1": base_f1,
            "target_f1": target_f1,
            "f1_delta": target_f1 - base_f1,
            "baseline_pred_gold": base_ratio,
            "target_pred_gold": target_ratio,
            "pred_gold_distance_delta": target_dist - base_dist,
            "ratio_ok": ratio_ok,
            "target_gold": int(target_ratios[code]["gold"]),
            "target_pred": int(target_ratios[code]["pred"]),
            "top_false_positive_sources": top_false_positive_sources(target, code),
        }

    target_metrics = target[OLD19_SPLIT]["metrics"]
    base_metrics = baseline[OLD19_SPLIT]["metrics"]
    checks = {
        "macro_f1_improved": deltas["macro_f1"] >= thresholds.macro_delta_min,
        "accuracy_not_regressed": deltas["accuracy"] >= -thresholds.accuracy_max_regression,
        "weighted_f1_not_regressed": deltas["weighted_f1"]
        >= -thresholds.weighted_f1_max_regression,
        "kw_recovered": (
            "KW" in target_class and float(target_class["KW"]["f1"]) >= thresholds.kw_f1_min
        ),
        "watched_pred_gold_not_worse": not ratio_failures,
    }
    release_recommended = all(checks.values())

    return {
        "release_recommended": release_recommended,
        "checks": checks,
        "ratio_failures": ratio_failures,
        "baseline_metrics": base_metrics,
        "target_metrics": target_metrics,
        "metric_deltas": deltas,
        "per_code": per_code,
        "td_metrics": {
            split: target[split]["metrics"]
            for split in TD_SPLITS
            if split in target
        },
    }


def fmt(value: float) -> str:
    return f"{value:.6f}"


def render_markdown(
    result: dict[str, Any],
    *,
    baseline_step: int,
    target_step: int,
    baseline_dir: Path,
    target_dir: Path,
) -> str:
    lines = [
        f"# Strict No-ACGC-Old19 Eval Gate: {target_step}",
        "",
        f"Baseline step: `{baseline_step}`",
        f"Target step: `{target_step}`",
        f"Baseline report: `{baseline_dir}`",
        f"Target report: `{target_dir}`",
        "",
        f"Recommendation: `{'RELEASE' if result['release_recommended'] else 'HOLD'}`",
        "",
        "## Checks",
        "",
        "| Check | Pass |",
        "|---|---:|",
    ]
    for key, ok in result["checks"].items():
        lines.append(f"| `{key}` | {'yes' if ok else 'no'} |")

    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Metric | Baseline | Target | Delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"):
        base = float(result["baseline_metrics"][key])
        target = float(result["target_metrics"][key])
        delta = float(result["metric_deltas"][key])
        lines.append(f"| `{key}` | {fmt(base)} | {fmt(target)} | {fmt(delta)} |")

    if result["td_metrics"]:
        lines.extend(["", "## TD Metrics", "", "| Split | Accuracy | Macro F1 |", "|---|---:|---:|"])
        for split, metrics in result["td_metrics"].items():
            lines.append(
                f"| `{split}` | {fmt(float(metrics['accuracy']))} | "
                f"{fmt(float(metrics['macro_f1']))} |"
            )

    lines.extend(
        [
            "",
            "## Watched Dialects",
            "",
            "| Dialect | Support | F1 Base | F1 Target | F1 Delta | Pred/Gold Base | Pred/Gold Target | Ratio OK |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for code, row in result["per_code"].items():
        lines.append(
            f"| `{code}` | {row['support']} | {fmt(row['baseline_f1'])} | "
            f"{fmt(row['target_f1'])} | {fmt(row['f1_delta'])} | "
            f"{row['baseline_pred_gold']:.2f} | {row['target_pred_gold']:.2f} | "
            f"{'yes' if row['ratio_ok'] else 'no'} |"
        )

    lines.extend(["", "## Top False-Positive Sources"])
    for code, row in result["per_code"].items():
        sources = row["top_false_positive_sources"]
        summary = ", ".join(f"{src['truth']} {src['count']}" for src in sources) or "-"
        lines.append(f"- `{code}`: {summary}")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    watch_codes = [code.strip().upper() for code in args.watch_codes.split(",") if code.strip()]
    thresholds = GateThresholds(
        macro_delta_min=args.macro_delta_min,
        accuracy_max_regression=args.accuracy_max_regression,
        weighted_f1_max_regression=args.weighted_f1_max_regression,
        kw_f1_min=args.kw_f1_min,
        ratio_distance_tolerance=args.ratio_distance_tolerance,
    )

    try:
        baseline_dir = find_report_dir(args.reports_root, args.baseline_step)
        target_dir = find_report_dir(args.reports_root, args.target_step)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    baseline = load_report(baseline_dir)
    target = load_report(target_dir)
    result = evaluate_gate(baseline, target, watch_codes, thresholds)
    result["baseline_step"] = args.baseline_step
    result["target_step"] = args.target_step
    result["baseline_report"] = str(baseline_dir)
    result["target_report"] = str(target_dir)
    result["thresholds"] = thresholds.__dict__

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        rendered = render_markdown(
            result,
            baseline_step=args.baseline_step,
            target_step=args.target_step,
            baseline_dir=baseline_dir,
            target_dir=target_dir,
        )
        print(rendered, end="")
        if args.out_md is not None:
            args.out_md.parent.mkdir(parents=True, exist_ok=True)
            args.out_md.write_text(rendered, encoding="utf-8")

    return 0 if result["release_recommended"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
