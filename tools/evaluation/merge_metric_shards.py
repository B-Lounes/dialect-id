from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from dialect_id.labels import num_dialects
from dialect_id.metrics import save_metrics_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge DID eval shard confusion matrices.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--eval-name", required=True)
    parser.add_argument("--shard-dir", action="append", required=True)
    return parser.parse_args()


def matrix_from_predictions(shard_dir: Path, eval_name: str) -> np.ndarray:
    matrix = np.zeros((num_dialects(), num_dialects()), dtype=np.int64)
    prediction_paths = sorted(shard_dir.glob(f"{eval_name}_predictions.rank*.jsonl"))
    if not prediction_paths:
        prediction_paths = sorted(shard_dir.glob(f"{eval_name}_predictions.jsonl"))
    if not prediction_paths:
        raise FileNotFoundError(
            f"No {eval_name} metrics JSON or prediction JSONL found in {shard_dir}"
        )
    for path in prediction_paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                truth = int(row["truth"])
                pred = int(row["pred"])
                if 0 <= truth < matrix.shape[0] and 0 <= pred < matrix.shape[1]:
                    matrix[truth, pred] += 1
                else:
                    raise ValueError(f"Invalid truth/pred in {path}:{line_no}: {truth}/{pred}")
    return matrix


def main() -> None:
    args = parse_args()
    matrix = None
    shard_payloads = []
    for raw_dir in args.shard_dir:
        shard_dir = Path(raw_dir)
        metrics_path = shard_dir / f"{args.eval_name}_metrics.json"
        if metrics_path.is_file():
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            shard_matrix = np.asarray(payload["confusion_matrix"], dtype=np.int64)
            metrics_payload = payload.get("metrics", {})
            source = str(metrics_path)
        else:
            shard_matrix = matrix_from_predictions(shard_dir, args.eval_name)
            metrics_payload = {}
            source = "predictions_fallback"
        matrix = shard_matrix if matrix is None else matrix + shard_matrix
        shard_payloads.append(
            {
                "dir": str(shard_dir),
                "metrics": metrics_payload,
                "source": source,
                "support": int(shard_matrix.sum()),
            }
        )
    if matrix is None:
        raise RuntimeError("No shard metrics loaded")

    out_dir = Path(args.out_dir)
    metrics = save_metrics_bundle(out_dir, matrix, args.eval_name)
    (out_dir / f"{args.eval_name}_shards.json").write_text(
        json.dumps(
            {
                "eval_name": args.eval_name,
                "metrics": metrics,
                "num_shards": len(shard_payloads),
                "shards": shard_payloads,
                "support": int(matrix.sum()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "summary_metrics.json").write_text(
        json.dumps({args.eval_name: metrics}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(out_dir), "metrics": metrics, "support": int(matrix.sum())}, sort_keys=True))


if __name__ == "__main__":
    main()
