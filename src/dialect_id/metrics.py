from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .labels import ID_TO_CODE, num_dialects


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, labels: int = num_dialects()) -> np.ndarray:
    truth = np.asarray(y_true, dtype=np.int64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    valid = (truth >= 0) & (truth < labels) & (pred >= 0) & (pred < labels)
    flat = truth[valid] * labels + pred[valid]
    return np.bincount(flat, minlength=labels * labels).reshape(labels, labels).astype(np.int64)


def metrics_from_confusion(matrix: np.ndarray) -> dict[str, float]:
    total = matrix.sum()
    correct = np.trace(matrix)
    per_class_total = matrix.sum(axis=1)
    pred_total = matrix.sum(axis=0)
    tp = np.diag(matrix).astype(np.float64)
    recall = np.divide(tp, per_class_total, out=np.zeros_like(tp), where=per_class_total != 0)
    precision = np.divide(tp, pred_total, out=np.zeros_like(tp), where=pred_total != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) != 0,
    )
    active = per_class_total != 0
    return {
        "accuracy": float(correct / total) if total else 0.0,
        "balanced_accuracy": float(recall[active].mean()) if active.any() else 0.0,
        "macro_f1": float(f1[active].mean()) if active.any() else 0.0,
        "weighted_f1": float((f1 * per_class_total).sum() / total) if total else 0.0,
    }


def per_class_metrics(matrix: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    totals = matrix.sum(axis=1)
    preds = matrix.sum(axis=0)
    tp = np.diag(matrix).astype(np.float64)
    recall = np.divide(tp, totals, out=np.zeros_like(tp), where=totals != 0)
    precision = np.divide(tp, preds, out=np.zeros_like(tp), where=preds != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) != 0,
    )
    for idx in range(matrix.shape[0]):
        rows.append(
            {
                "id": idx,
                "code": ID_TO_CODE.get(idx, str(idx)),
                "support": int(totals[idx]),
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
            }
        )
    return rows


def save_metrics_bundle(out_dir: str | Path, matrix: np.ndarray, prefix: str) -> dict[str, float]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = metrics_from_confusion(matrix)
    payload = {
        "metrics": metrics,
        "per_class": per_class_metrics(matrix),
        "confusion_matrix": matrix.tolist(),
    }
    (out / f"{prefix}_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (out / f"{prefix}_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["truth\\pred", *[ID_TO_CODE.get(i, str(i)) for i in range(matrix.shape[1])]])
        for idx, row in enumerate(matrix.tolist()):
            writer.writerow([ID_TO_CODE.get(idx, str(idx)), *row])
    return metrics
