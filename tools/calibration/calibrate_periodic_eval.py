#!/usr/bin/env python3
"""Probe additive logit-bias calibration on a completed periodic eval report.

This is a post-hoc diagnostic. It reads prediction JSONL shards only; it does
not run model inference and does not touch training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from dialect_id.labels import CODE_TO_ID, ID_TO_CODE, num_dialects  # noqa: E402
from dialect_id.metrics import confusion_matrix_np, metrics_from_confusion, per_class_metrics  # noqa: E402


DEFAULT_OUT_DIR = Path("calibration-output")
DEFAULT_REPORTS_ROOT = Path(".")
DEFAULT_SPLIT = "acgc_old19_train_as_eval"
DEFAULT_EXCLUDED = "DJ,KM,SO,TD"
DEFAULT_WATCH = "BH,JO,KW,OM,AE,QA,SY"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=DEFAULT_REPORTS_ROOT,
        help="Periodic eval root used with --step.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Find the completed periodic eval report for this step.",
    )
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--excluded-codes", default=DEFAULT_EXCLUDED)
    parser.add_argument(
        "--calibrate-codes",
        default="",
        help="Comma-separated dialects to tune. Default: all non-excluded codes with support.",
    )
    parser.add_argument("--watch-codes", default=DEFAULT_WATCH)
    parser.add_argument("--steps", default="1.0,0.5,0.25,0.1,0.05,0.02,0.01")
    parser.add_argument("--passes-per-step", type=int, default=3)
    parser.add_argument("--bias-l2", type=float, default=0.001)
    parser.add_argument("--balanced-weight", type=float, default=10.0)
    parser.add_argument("--macro-weight", type=float, default=2.0)
    parser.add_argument("--accuracy-weight", type=float, default=1.0)
    parser.add_argument("--weighted-f1-weight", type=float, default=1.0)
    parser.add_argument("--max-bias-abs", type=float, default=3.0)
    return parser.parse_args()


def parse_codes(raw: str) -> list[str]:
    return [part.strip().upper() for part in raw.split(",") if part.strip()]


def find_report_dir(reports_root: Path, step: int, split: str) -> Path:
    needle = f"_step_{step}_"
    candidates = [
        path
        for path in reports_root.iterdir()
        if path.is_dir()
        and needle in path.name
        and any(path.glob(f"{split}_predictions.rank*.jsonl"))
    ]
    if not candidates:
        raise FileNotFoundError(f"No completed periodic eval report for step {step} under {reports_root}")
    return sorted(candidates, key=lambda path: path.stat().st_mtime)[-1]


def stable_bucket(row: dict[str, Any]) -> int:
    key = "|".join(
        str(row.get(name, ""))
        for name in ("sample_id", "source", "country_code", "duration_seconds")
    )
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little") % 2


def load_predictions(report_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = sorted(report_dir.glob(f"{split}_predictions.rank*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No prediction shards for {split} under {report_dir}")
    probs: list[list[float]] = []
    truth: list[int] = []
    bucket: list[int] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                probs.append(row["probs"])
                truth.append(int(row["truth"]))
                bucket.append(stable_bucket(row))
    return (
        np.asarray(probs, dtype=np.float64),
        np.asarray(truth, dtype=np.int64),
        np.asarray(bucket, dtype=np.int8),
    )


def predict(probs: np.ndarray, bias: np.ndarray, excluded_ids: list[int]) -> np.ndarray:
    scores = np.log(np.maximum(probs, 1e-45)) + bias[None, :]
    scores[:, excluded_ids] = -1e30
    return scores.argmax(axis=1).astype(np.int64)


def bundle(
    probs: np.ndarray,
    truth: np.ndarray,
    bias: np.ndarray,
    excluded_ids: list[int],
) -> tuple[dict[str, float], np.ndarray, list[dict[str, Any]]]:
    pred = predict(probs, bias, excluded_ids)
    matrix = confusion_matrix_np(truth, pred, labels=num_dialects())
    return metrics_from_confusion(matrix), matrix, per_class_metrics(matrix)


def objective(metrics: dict[str, float], bias: np.ndarray, args: argparse.Namespace) -> float:
    return (
        args.balanced_weight * float(metrics["balanced_accuracy"])
        + args.macro_weight * float(metrics["macro_f1"])
        + args.accuracy_weight * float(metrics["accuracy"])
        + args.weighted_f1_weight * float(metrics["weighted_f1"])
        - args.bias_l2 * float(np.dot(bias, bias))
    )


def tune_bias(
    probs: np.ndarray,
    truth: np.ndarray,
    *,
    calibrate_ids: list[int],
    excluded_ids: list[int],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float]]:
    bias = np.zeros(num_dialects(), dtype=np.float64)
    best_metrics, _, _ = bundle(probs, truth, bias, excluded_ids)
    best_score = objective(best_metrics, bias, args)
    steps = [float(step) for step in args.steps.split(",") if step.strip()]
    calibrate_ids_arr = np.asarray(calibrate_ids, dtype=np.int64)

    for step in steps:
        improved = True
        passes = 0
        while improved and passes < args.passes_per_step:
            improved = False
            passes += 1
            for cls in calibrate_ids:
                local_bias = bias
                local_metrics = best_metrics
                local_score = best_score
                for delta in (-step, step):
                    candidate = bias.copy()
                    candidate[cls] += delta
                    candidate[calibrate_ids_arr] -= candidate[calibrate_ids_arr].mean()
                    if args.max_bias_abs > 0:
                        candidate[calibrate_ids_arr] = np.clip(
                            candidate[calibrate_ids_arr],
                            -args.max_bias_abs,
                            args.max_bias_abs,
                        )
                    candidate[excluded_ids] = 0.0
                    metrics, _, _ = bundle(probs, truth, candidate, excluded_ids)
                    score = objective(metrics, candidate, args)
                    if score > local_score + 1e-12:
                        local_score = score
                        local_bias = candidate
                        local_metrics = metrics
                if local_score > best_score + 1e-12:
                    bias = local_bias
                    best_metrics = local_metrics
                    best_score = local_score
                    improved = True
    return bias, best_metrics


def pred_gold_summary(matrix: np.ndarray, per_class: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    gold = matrix.sum(axis=1)
    pred = matrix.sum(axis=0)
    for row in per_class:
        idx = int(row["id"])
        if int(gold[idx]) <= 0 and int(pred[idx]) <= 0:
            continue
        out[str(row["code"])] = {
            "gold": int(gold[idx]),
            "pred": int(pred[idx]),
            "pred_gold_ratio": float(pred[idx] / gold[idx]) if gold[idx] else None,
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1"]),
        }
    return out


def top_false_positive_sources(
    matrix: np.ndarray,
    per_class: list[dict[str, Any]],
    code: str,
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    id_to_code = {int(row["id"]): str(row["code"]) for row in per_class}
    code_to_id = {value: key for key, value in id_to_code.items()}
    if code not in code_to_id:
        return []
    target = code_to_id[code]
    rows = []
    for truth_idx in range(matrix.shape[0]):
        if truth_idx == target:
            continue
        count = int(matrix[truth_idx, target])
        if count:
            rows.append((count, id_to_code.get(truth_idx, str(truth_idx))))
    return [
        {"truth": truth_code, "count": count}
        for count, truth_code in sorted(rows, reverse=True)[:limit]
    ]


def write_metrics(out_dir: Path, prefix: str, matrix: np.ndarray, per_class: list[dict[str, Any]]) -> None:
    payload = {
        "metrics": metrics_from_confusion(matrix),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }
    (out_dir / f"{prefix}_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (out_dir / f"{prefix}_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["truth\\pred", *[ID_TO_CODE.get(i, str(i)) for i in range(matrix.shape[1])]])
        for idx, row in enumerate(matrix.tolist()):
            writer.writerow([ID_TO_CODE.get(idx, str(idx)), *row])


def fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6f}"


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Strict No-ACGC-Old19 Calibration Probe",
        "",
        f"Source report: `{summary['source_report']}`",
        f"Split: `{summary['split']}`",
        f"Output dir: `{summary['out_dir']}`",
        f"Tune rows: `{summary['n_tune']}`",
        f"Holdout rows: `{summary['n_holdout']}`",
        "",
        "This is a diagnostic post-hoc additive logit-bias calibration. It uses",
        "one deterministic half of the eval predictions for tuning and reports",
        "the other half as holdout. It does not retrain the model.",
        "",
        "## Metrics",
        "",
        "| Subset | Variant | Accuracy | Balanced Acc | Macro F1 | Weighted F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for subset in ("tune", "holdout", "all"):
        for variant in ("baseline", "calibrated"):
            metrics = summary["metrics"][subset][variant]
            lines.append(
                f"| `{subset}` | `{variant}` | {fmt(metrics['accuracy'])} | "
                f"{fmt(metrics['balanced_accuracy'])} | {fmt(metrics['macro_f1'])} | "
                f"{fmt(metrics['weighted_f1'])} |"
            )
    lines.extend(
        [
            "",
            "## Holdout Watched Dialects",
            "",
            "| Dialect | Base F1 | Cal F1 | F1 Delta | Base Pred/Gold | Cal Pred/Gold |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for code, row in summary["watched_holdout"].items():
        lines.append(
            f"| `{code}` | {fmt(row['baseline_f1'])} | {fmt(row['calibrated_f1'])} | "
            f"{fmt(row['f1_delta'])} | {row['baseline_pred_gold']:.2f} | "
            f"{row['calibrated_pred_gold']:.2f} |"
        )
    lines.extend(["", "## Learned Bias"])
    for code, value in summary["bias_by_code"].items():
        if abs(value) >= 0.005:
            lines.append(f"- `{code}`: {value:+.4f}")
    lines.extend(["", "## Holdout False-Positive Sources After Calibration"])
    for code, sources in summary["holdout_calibrated_false_positive_sources"].items():
        rendered = ", ".join(f"{item['truth']} {item['count']}" for item in sources) or "-"
        lines.append(f"- `{code}`: {rendered}")
    return "\n".join(lines) + "\n"


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def main() -> int:
    args = parse_args()
    if args.step is not None:
        args.report_dir = find_report_dir(args.reports_root, args.step, args.split)
        if args.out_dir == DEFAULT_OUT_DIR:
            short = f"{args.step // 1000}k" if args.step % 1000 == 0 else str(args.step)
            args.out_dir = args.out_dir.parent / f"calibration_probe_noacgcold19_step{short}_half_split_v1"
    elif args.report_dir is None:
        raise SystemExit("Pass --report-dir, or pass --step with --reports-root.")
    excluded_codes = parse_codes(args.excluded_codes)
    excluded_ids = [CODE_TO_ID[code] for code in excluded_codes if code in CODE_TO_ID]
    watch_codes = parse_codes(args.watch_codes)

    probs, truth, bucket = load_predictions(args.report_dir, args.split)
    tune_mask = bucket == 0
    holdout_mask = ~tune_mask

    baseline_all_metrics, baseline_all_matrix, baseline_all_per_class = bundle(
        probs,
        truth,
        np.zeros(num_dialects(), dtype=np.float64),
        excluded_ids,
    )
    support_codes = {
        str(row["code"])
        for row in baseline_all_per_class
        if int(row.get("support", 0)) > 0 and str(row["code"]) not in excluded_codes
    }
    calibrate_codes = parse_codes(args.calibrate_codes) or sorted(support_codes)
    calibrate_ids = [CODE_TO_ID[code] for code in calibrate_codes if code in CODE_TO_ID]
    if not calibrate_ids:
        raise SystemExit("No calibrate ids selected")

    bias, tune_best_metrics = tune_bias(
        probs[tune_mask],
        truth[tune_mask],
        calibrate_ids=calibrate_ids,
        excluded_ids=excluded_ids,
        args=args,
    )

    subsets = {
        "tune": tune_mask,
        "holdout": holdout_mask,
        "all": np.ones_like(tune_mask, dtype=bool),
    }
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    matrices: dict[str, dict[str, np.ndarray]] = {}
    per_classes: dict[str, dict[str, list[dict[str, Any]]]] = {}
    zero = np.zeros(num_dialects(), dtype=np.float64)
    for name, mask in subsets.items():
        metrics[name] = {}
        matrices[name] = {}
        per_classes[name] = {}
        for variant, variant_bias in (("baseline", zero), ("calibrated", bias)):
            m, matrix, per_class = bundle(probs[mask], truth[mask], variant_bias, excluded_ids)
            metrics[name][variant] = m
            matrices[name][variant] = matrix
            per_classes[name][variant] = per_class

    baseline_holdout = pred_gold_summary(
        matrices["holdout"]["baseline"],
        per_classes["holdout"]["baseline"],
    )
    calibrated_holdout = pred_gold_summary(
        matrices["holdout"]["calibrated"],
        per_classes["holdout"]["calibrated"],
    )
    watched_holdout: dict[str, dict[str, float]] = {}
    for code in watch_codes:
        if code not in baseline_holdout or code not in calibrated_holdout:
            continue
        watched_holdout[code] = {
            "baseline_f1": baseline_holdout[code]["f1"],
            "calibrated_f1": calibrated_holdout[code]["f1"],
            "f1_delta": calibrated_holdout[code]["f1"] - baseline_holdout[code]["f1"],
            "baseline_pred_gold": baseline_holdout[code]["pred_gold_ratio"],
            "calibrated_pred_gold": calibrated_holdout[code]["pred_gold_ratio"],
        }

    bias_by_code = {
        ID_TO_CODE.get(idx, str(idx)): float(value)
        for idx, value in enumerate(bias.tolist())
    }
    summary = {
        "source_report": str(args.report_dir),
        "split": args.split,
        "out_dir": str(args.out_dir),
        "n_total": int(len(truth)),
        "n_tune": int(tune_mask.sum()),
        "n_holdout": int(holdout_mask.sum()),
        "excluded_codes": excluded_codes,
        "calibrate_codes": calibrate_codes,
        "watch_codes": watch_codes,
        "args": jsonable_args(args),
        "tune_best_metrics": tune_best_metrics,
        "metrics": metrics,
        "bias_by_code": bias_by_code,
        "watched_holdout": watched_holdout,
        "holdout_calibrated_false_positive_sources": {
            code: top_false_positive_sources(
                matrices["holdout"]["calibrated"],
                per_classes["holdout"]["calibrated"],
                code,
            )
            for code in watch_codes
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "dialect_logit_bias.json").write_text(
        json.dumps({"bias_by_code": bias_by_code, "bias": bias.tolist()}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for subset in subsets:
        for variant in ("baseline", "calibrated"):
            write_metrics(
                args.out_dir,
                f"{subset}_{variant}",
                matrices[subset][variant],
                per_classes[subset][variant],
            )
    (args.out_dir / "README.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "summary": str(args.out_dir / "summary.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
