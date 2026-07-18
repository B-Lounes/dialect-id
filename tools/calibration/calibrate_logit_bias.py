#!/usr/bin/env python3
"""Tune a 20-active additive dialect-logit calibration from eval probabilities."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from dialect_id.labels import CODE_TO_ID, ID_TO_CODE, num_dialects  # noqa: E402
from dialect_id.metrics import confusion_matrix_np, metrics_from_confusion, per_class_metrics  # noqa: E402


ACTIVE_CODES = [
    "AE",
    "BH",
    "DZ",
    "EG",
    "IQ",
    "JO",
    "KW",
    "LB",
    "LY",
    "MA",
    "MR",
    "OM",
    "PS",
    "QA",
    "SA",
    "SD",
    "SY",
    "TD",
    "TN",
    "YE",
]
EXCLUDED_CODES = ["DJ", "KM", "SO"]
SPLITS = [
    "acgc_old19_validation",
    "acgc_old19_test",
    "acgc_td_validation_23way",
    "acgc_td_test_23way",
]
REQUIRED_NON_REGRESSION = ("accuracy", "balanced_accuracy", "weighted_f1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", required=True)
    parser.add_argument("--out-report", required=True)
    parser.add_argument("--base-checkpoint", default="unknown")
    parser.add_argument(
        "--steps",
        default="1.0,0.5,0.25,0.1,0.05,0.02,0.01,0.005",
        help="Comma-separated coordinate-descent step sizes.",
    )
    parser.add_argument("--passes-per-step", type=int, default=3)
    parser.add_argument("--bias-l2", type=float, default=0.002)
    parser.add_argument("--balanced-weight", type=float, default=20.0)
    parser.add_argument("--macro-weight", type=float, default=2.0)
    parser.add_argument("--regression-penalty", type=float, default=2000.0)
    return parser.parse_args()


def load_predictions(report: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    probs: list[list[float]] = []
    truth: list[int] = []
    paths = sorted(report.glob(f"{split}_predictions.rank*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No prediction shards found for {split} under {report}")
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                probs.append(row["probs"])
                truth.append(int(row["truth"]))
    return np.asarray(probs, dtype=np.float64), np.asarray(truth, dtype=np.int64)


def predict_with_bias(
    probs: np.ndarray,
    bias: np.ndarray,
    *,
    excluded_ids: list[int],
) -> np.ndarray:
    scores = np.log(np.maximum(probs, 1e-45)) + bias[None, :]
    scores[:, excluded_ids] = -1e30
    return scores.argmax(axis=1).astype(np.int64)


def metrics_bundle(
    probs: np.ndarray,
    truth: np.ndarray,
    bias: np.ndarray,
    *,
    excluded_ids: list[int],
) -> tuple[dict[str, float], np.ndarray, list[dict[str, Any]]]:
    pred = predict_with_bias(probs, bias, excluded_ids=excluded_ids)
    matrix = confusion_matrix_np(truth, pred, labels=num_dialects())
    return metrics_from_confusion(matrix), matrix, per_class_metrics(matrix)


def metric_deltas(metrics: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {key: float(metrics[key]) - float(baseline[key]) for key in baseline}


def objective(
    metrics: dict[str, float],
    baseline: dict[str, float],
    bias: np.ndarray,
    *,
    args: argparse.Namespace,
) -> float:
    regression = sum(max(0.0, baseline[key] - metrics[key]) for key in REQUIRED_NON_REGRESSION)
    macro_delta = metrics["macro_f1"] - baseline["macro_f1"]
    balanced_delta = metrics["balanced_accuracy"] - baseline["balanced_accuracy"]
    accuracy_delta = metrics["accuracy"] - baseline["accuracy"]
    regularizer = args.bias_l2 * float(np.dot(bias, bias))
    if regression > 1e-12:
        return (
            args.balanced_weight * balanced_delta
            + macro_delta
            - args.regression_penalty * regression
            - regularizer
        )
    return (
        1000.0
        + args.balanced_weight * balanced_delta
        + args.macro_weight * macro_delta
        + accuracy_delta
        - regularizer
    )


def tune_bias(
    probs: np.ndarray,
    truth: np.ndarray,
    *,
    active_old19_ids: list[int],
    excluded_ids: list[int],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
    bias = np.zeros(num_dialects(), dtype=np.float64)
    baseline, _, _ = metrics_bundle(probs, truth, bias, excluded_ids=excluded_ids)
    best_metrics = baseline
    best_score = objective(best_metrics, baseline, bias, args=args)
    steps = [float(value) for value in args.steps.split(",") if value.strip()]
    for step in steps:
        improved = True
        passes = 0
        while improved and passes < args.passes_per_step:
            improved = False
            passes += 1
            for cls in active_old19_ids:
                local_score = best_score
                local_bias = bias
                local_metrics = best_metrics
                for delta in (-step, step):
                    candidate = bias.copy()
                    candidate[cls] += delta
                    candidate[active_old19_ids] -= candidate[active_old19_ids].mean()
                    candidate[CODE_TO_ID["TD"]] = 0.0
                    metrics, _, _ = metrics_bundle(
                        probs,
                        truth,
                        candidate,
                        excluded_ids=excluded_ids,
                    )
                    score = objective(metrics, baseline, candidate, args=args)
                    if score > local_score + 1e-15:
                        local_score = score
                        local_bias = candidate
                        local_metrics = metrics
                if local_score > best_score + 1e-15:
                    best_score = local_score
                    bias = local_bias
                    best_metrics = local_metrics
                    improved = True
    return bias, baseline, best_metrics


def write_metrics(out_dir: Path, prefix: str, matrix: np.ndarray) -> dict[str, Any]:
    metrics = metrics_from_confusion(matrix)
    per_class = per_class_metrics(matrix)
    payload = {
        "metrics": metrics,
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
    return payload


def fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6f}"


def write_report_md(path: Path, metadata: dict[str, Any], summary: dict[str, Any]) -> None:
    lines = [
        "# 20-Active Step400 Balanced Calibration v1",
        "",
        f"Generated at: `{metadata['generated_at']}`",
        f"Base checkpoint: `{metadata['base_checkpoint']}`",
        f"Source report: `{metadata['source_report']}`",
        f"Bias JSON: `{metadata['bias_json']}`",
        "",
        "The bias vector is tuned on `acgc_old19_validation` only. It is applied as",
        "an additive per-dialect logit offset after the base model logits and before",
        "the active 20-way mask/softmax.",
        "",
        "## Metrics",
        "",
        "| Split | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        metrics = summary[split]["metrics"]
        lines.append(
            f"| `{split}` | {fmt(metrics.get('accuracy'))} | "
            f"{fmt(metrics.get('balanced_accuracy'))} | "
            f"{fmt(metrics.get('macro_f1'))} | {fmt(metrics.get('weighted_f1'))} |"
        )
    lines.extend(
        [
            "",
            "## Bias By Code",
            "",
            "| Code | Bias |",
            "|---|---:|",
        ]
    )
    for code, value in metadata["bias_by_code"].items():
        if abs(float(value)) > 1e-12:
            lines.append(f"| `{code}` | {float(value):+.6f} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_report = Path(args.source_report)
    out_report = Path(args.out_report)
    out_report.mkdir(parents=True, exist_ok=True)

    active_ids = [CODE_TO_ID[code] for code in ACTIVE_CODES]
    excluded_ids = [CODE_TO_ID[code] for code in EXCLUDED_CODES]
    active_old19_ids = [idx for idx in active_ids if idx != CODE_TO_ID["TD"]]

    split_data = {split: load_predictions(source_report, split) for split in SPLITS}
    bias, baseline_validation, calibrated_validation = tune_bias(
        split_data["acgc_old19_validation"][0],
        split_data["acgc_old19_validation"][1],
        active_old19_ids=active_old19_ids,
        excluded_ids=excluded_ids,
        args=args,
    )

    summary: dict[str, Any] = {}
    for split, (probs, truth) in split_data.items():
        metrics, matrix, _per_class = metrics_bundle(probs, truth, bias, excluded_ids=excluded_ids)
        summary[split] = write_metrics(out_report, split, matrix)
        summary[split]["deltas_vs_source_baseline"] = metric_deltas(
            metrics,
            metrics_bundle(probs, truth, np.zeros(num_dialects()), excluded_ids=excluded_ids)[0],
        )

    bias_by_id = {str(idx): float(bias[idx]) for idx in range(num_dialects())}
    bias_by_code = {ID_TO_CODE[idx]: float(bias[idx]) for idx in range(num_dialects())}
    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "base_checkpoint": args.base_checkpoint,
        "source_report": str(source_report),
        "out_report": str(out_report),
        "bias_json": str(out_report / "dialect_logit_bias.json"),
        "active_codes": ACTIVE_CODES,
        "excluded_codes": EXCLUDED_CODES,
        "required_non_regression_metrics": list(REQUIRED_NON_REGRESSION),
        "tuning_split": "acgc_old19_validation",
        "tuning_objective": "balanced_first_coordinate_descent",
        "tuning_args": {
            "steps": args.steps,
            "passes_per_step": args.passes_per_step,
            "bias_l2": args.bias_l2,
            "balanced_weight": args.balanced_weight,
            "macro_weight": args.macro_weight,
            "regression_penalty": args.regression_penalty,
        },
        "baseline_validation": baseline_validation,
        "calibrated_validation": calibrated_validation,
        "bias": [float(value) for value in bias.tolist()],
        "bias_by_id": bias_by_id,
        "bias_by_code": bias_by_code,
    }
    (out_report / "dialect_logit_bias.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_report / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_report / "calibration_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report_md(out_report / "calibration_report.md", metadata, summary)
    print(
        json.dumps(
            {
                "out_report": str(out_report),
                "bias_json": str(out_report / "dialect_logit_bias.json"),
                "old19_validation": summary["acgc_old19_validation"]["metrics"],
                "old19_test": summary["acgc_old19_test"]["metrics"],
                "td_validation": summary["acgc_td_validation_23way"]["metrics"],
                "td_test": summary["acgc_td_test_23way"]["metrics"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
