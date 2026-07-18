#!/usr/bin/env python3
"""Gate a legacy campaign's completed no-ACGC-old19 calibration probes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_WATCH_CODES = ("BH", "JO", "KW", "OM", "AE", "QA", "SY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--target-summary", type=Path, required=True)
    parser.add_argument("--baseline-step", type=int, default=400000)
    parser.add_argument("--target-step", type=int, default=450000)
    parser.add_argument("--watch-codes", default=",".join(DEFAULT_WATCH_CODES))
    parser.add_argument("--macro-delta-min", type=float, default=0.001)
    parser.add_argument("--accuracy-max-regression", type=float, default=0.0015)
    parser.add_argument("--weighted-f1-max-regression", type=float, default=0.0015)
    parser.add_argument("--balanced-max-regression", type=float, default=0.002)
    parser.add_argument("--kw-f1-min", type=float, default=0.65)
    parser.add_argument("--ratio-distance-tolerance", type=float, default=0.05)
    parser.add_argument("--min-holdout-rows", type=int, default=100000)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def parse_codes(raw: str) -> list[str]:
    return [part.strip().upper() for part in raw.split(",") if part.strip()]


def load_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing calibration summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def metric_block(summary: dict[str, Any]) -> dict[str, float]:
    return summary["metrics"]["holdout"]["calibrated"]


def metric_deltas(baseline: dict[str, Any], target: dict[str, Any]) -> dict[str, float]:
    base = metric_block(baseline)
    current = metric_block(target)
    return {
        key: float(current[key]) - float(base[key])
        for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
    }


def watched_rows(
    baseline: dict[str, Any],
    target: dict[str, Any],
    watch_codes: list[str],
    ratio_distance_tolerance: float,
) -> tuple[dict[str, dict[str, float]], list[str]]:
    base_rows = baseline.get("watched_holdout") or {}
    target_rows = target.get("watched_holdout") or {}
    out: dict[str, dict[str, float]] = {}
    ratio_failures: list[str] = []
    for code in watch_codes:
        if code not in base_rows or code not in target_rows:
            continue
        base = base_rows[code]
        current = target_rows[code]
        base_ratio = float(base["calibrated_pred_gold"])
        target_ratio = float(current["calibrated_pred_gold"])
        base_dist = abs(base_ratio - 1.0)
        target_dist = abs(target_ratio - 1.0)
        ratio_ok = target_dist <= base_dist + ratio_distance_tolerance
        if not ratio_ok:
            ratio_failures.append(code)
        out[code] = {
            "baseline_f1": float(base["calibrated_f1"]),
            "target_f1": float(current["calibrated_f1"]),
            "f1_delta": float(current["calibrated_f1"]) - float(base["calibrated_f1"]),
            "baseline_pred_gold": base_ratio,
            "target_pred_gold": target_ratio,
            "pred_gold_distance_delta": target_dist - base_dist,
            "ratio_ok": ratio_ok,
        }
    return out, ratio_failures


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    baseline = load_summary(args.baseline_summary)
    target = load_summary(args.target_summary)
    watch_codes = parse_codes(args.watch_codes)
    deltas = metric_deltas(baseline, target)
    per_code, ratio_failures = watched_rows(
        baseline,
        target,
        watch_codes,
        args.ratio_distance_tolerance,
    )
    target_metrics = metric_block(target)
    checks = {
        "enough_holdout_rows": int(target.get("n_holdout", 0)) >= args.min_holdout_rows,
        "macro_f1_improved": deltas["macro_f1"] >= args.macro_delta_min,
        "accuracy_not_regressed": deltas["accuracy"] >= -args.accuracy_max_regression,
        "weighted_f1_not_regressed": deltas["weighted_f1"] >= -args.weighted_f1_max_regression,
        "balanced_accuracy_not_regressed": (
            deltas["balanced_accuracy"] >= -args.balanced_max_regression
        ),
        "kw_calibrated_ok": (
            "KW" in per_code and per_code["KW"]["target_f1"] >= args.kw_f1_min
        ),
        "watched_pred_gold_not_worse": not ratio_failures,
    }
    return {
        "release_recommended": all(checks.values()),
        "checks": checks,
        "ratio_failures": ratio_failures,
        "baseline_summary": str(args.baseline_summary),
        "target_summary": str(args.target_summary),
        "baseline_step": args.baseline_step,
        "target_step": args.target_step,
        "baseline_metrics": metric_block(baseline),
        "target_metrics": target_metrics,
        "metric_deltas": deltas,
        "per_code": per_code,
        "n_holdout": int(target.get("n_holdout", 0)),
    }


def fmt(value: float) -> str:
    return f"{value:.6f}"


def render_markdown(result: dict[str, Any]) -> str:
    baseline_step = result["baseline_step"]
    target_step = result["target_step"]
    action = "RELEASE" if result["release_recommended"] else "HOLD"
    lines = [
        f"# Strict No-ACGC-Old19 Calibration Gate: {target_step}",
        "",
        f"Recommendation: `{action}`",
        "",
        f"Baseline summary: `{result['baseline_summary']}`",
        f"Target summary: `{result['target_summary']}`",
        f"Target holdout rows: `{result['n_holdout']}`",
        "",
        "This gate uses the held-out half of the post-hoc calibration probe. It does not",
        "train the model and does not run inference; it only decides whether the next",
        "one training segment can be released when calibrated evidence is better than",
        f"the calibrated `{baseline_step}` baseline.",
        "",
        "## Metrics",
        "",
        "| Metric | Baseline Cal Holdout | Target Cal Holdout | Delta |",
        "|---|---:|---:|---:|",
    ]
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"):
        lines.append(
            f"| `{key}` | {fmt(result['baseline_metrics'][key])} | "
            f"{fmt(result['target_metrics'][key])} | {fmt(result['metric_deltas'][key])} |"
        )
    lines.extend(["", "## Checks", ""])
    for key, value in result["checks"].items():
        lines.append(f"- `{key}`: `{value}`")
    if result["ratio_failures"]:
        failures = ", ".join(f"`{code}`" for code in result["ratio_failures"])
        lines.append(f"- ratio failures: {failures}")
    lines.extend(
        [
            "",
            "## Watched Dialects",
            "",
            "| Dialect | Baseline F1 | Target F1 | Delta | Baseline Pred/Gold | Target Pred/Gold | Ratio OK |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for code, row in result["per_code"].items():
        lines.append(
            f"| `{code}` | {fmt(row['baseline_f1'])} | {fmt(row['target_f1'])} | "
            f"{fmt(row['f1_delta'])} | {row['baseline_pred_gold']:.2f} | "
            f"{row['target_pred_gold']:.2f} | `{row['ratio_ok']}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    try:
        result = evaluate(args)
    except FileNotFoundError as exc:
        payload = {
            "release_recommended": False,
            "error": str(exc),
            "baseline_summary": str(args.baseline_summary),
            "target_summary": str(args.target_summary),
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(payload["error"], file=sys.stderr)
        return 2

    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(render_markdown(result), encoding="utf-8")
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("RELEASE" if result["release_recommended"] else "HOLD")
        print(f"report: {args.out_md}")
    return 0 if result["release_recommended"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
