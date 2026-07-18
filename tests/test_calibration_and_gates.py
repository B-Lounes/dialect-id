import argparse
import json

import numpy as np

from dialect_id.labels import CODE_TO_ID, DIALECTS, num_dialects
from tools.calibration.calibrate_logit_bias import predict_with_bias
from tools.calibration.calibrate_periodic_eval import stable_bucket
from tools.evaluation.calibration_gate import evaluate as evaluate_calibration_gate
from tools.evaluation.eval_gate import GateThresholds, OLD19_SPLIT, evaluate_gate


def test_calibration_prediction_respects_excluded_labels() -> None:
    probs = np.full((1, num_dialects()), 1e-5, dtype=np.float64)
    probs[0, CODE_TO_ID["DJ"]] = 0.99
    probs[0, CODE_TO_ID["AE"]] = 0.50
    prediction = predict_with_bias(
        probs,
        np.zeros(num_dialects()),
        excluded_ids=[CODE_TO_ID["DJ"], CODE_TO_ID["KM"], CODE_TO_ID["SO"]],
    )
    assert prediction.tolist() == [CODE_TO_ID["AE"]]


def test_calibration_partition_is_deterministic() -> None:
    row = {"sample_id": 17, "source": "synthetic", "country_code": "MA", "duration_seconds": 4.2}
    assert stable_bucket(row) == stable_bucket(dict(row))
    assert stable_bucket({**row, "sample_id": 18}) in {0, 1}


def test_calibration_gate_releases_only_non_regressing_target(tmp_path) -> None:
    base_metrics = {"accuracy": 0.90, "balanced_accuracy": 0.89, "macro_f1": 0.88, "weighted_f1": 0.90}
    target_metrics = {"accuracy": 0.901, "balanced_accuracy": 0.891, "macro_f1": 0.883, "weighted_f1": 0.901}
    baseline = {
        "n_holdout": 120_000,
        "metrics": {"holdout": {"calibrated": base_metrics}},
        "watched_holdout": {"KW": {"calibrated_f1": 0.66, "calibrated_pred_gold": 1.02}},
    }
    target = {
        "n_holdout": 120_000,
        "metrics": {"holdout": {"calibrated": target_metrics}},
        "watched_holdout": {"KW": {"calibrated_f1": 0.70, "calibrated_pred_gold": 1.01}},
    }
    baseline_path = tmp_path / "baseline.json"
    target_path = tmp_path / "target.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    target_path.write_text(json.dumps(target), encoding="utf-8")
    args = argparse.Namespace(
        baseline_summary=baseline_path,
        target_summary=target_path,
        baseline_step=400_000,
        target_step=450_000,
        watch_codes="KW",
        macro_delta_min=0.001,
        accuracy_max_regression=0.0015,
        weighted_f1_max_regression=0.0015,
        balanced_max_regression=0.002,
        kw_f1_min=0.65,
        ratio_distance_tolerance=0.05,
        min_holdout_rows=100_000,
    )
    assert evaluate_calibration_gate(args)["release_recommended"] is True

    target["watched_holdout"]["KW"]["calibrated_pred_gold"] = 1.20
    target_path.write_text(json.dumps(target), encoding="utf-8")
    result = evaluate_calibration_gate(args)
    assert result["release_recommended"] is False
    assert result["ratio_failures"] == ["KW"]


def test_evaluation_gate_combines_aggregate_and_watched_checks() -> None:
    matrix = [[0 for _ in DIALECTS] for _ in DIALECTS]
    for label in DIALECTS:
        matrix[label.id][label.id] = 10

    def report(metrics: dict[str, float], kw_f1: float) -> dict:
        rows = [
            {
                "id": label.id,
                "code": label.code,
                "support": 10,
                "precision": kw_f1 if label.code == "KW" else 1.0,
                "recall": kw_f1 if label.code == "KW" else 1.0,
                "f1": kw_f1 if label.code == "KW" else 1.0,
            }
            for label in DIALECTS
        ]
        return {OLD19_SPLIT: {"metrics": metrics, "per_class": rows, "confusion_matrix": matrix}}

    baseline = report(
        {"accuracy": 0.90, "balanced_accuracy": 0.89, "macro_f1": 0.88, "weighted_f1": 0.90},
        0.65,
    )
    target = report(
        {"accuracy": 0.901, "balanced_accuracy": 0.891, "macro_f1": 0.883, "weighted_f1": 0.901},
        0.70,
    )
    thresholds = GateThresholds(
        macro_delta_min=0.002,
        accuracy_max_regression=0.002,
        weighted_f1_max_regression=0.001,
        kw_f1_min=0.70,
        ratio_distance_tolerance=0.05,
    )
    result = evaluate_gate(baseline, target, ["KW"], thresholds)
    assert result["release_recommended"] is True
    assert all(result["checks"].values())
