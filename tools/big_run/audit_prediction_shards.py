#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PART_RE = re.compile(r"\.part(\d+)\.")


def part_index(path: Path) -> int:
    match = PART_RE.search(path.name)
    if match is None:
        raise ValueError(f"Could not parse part index from {path}")
    return int(match.group(1))


def compact_ranges(values: list[int]) -> list[str]:
    if not values:
        return []
    ranges: list[str] = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = value
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return ranges


def count_nonempty_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def prediction_status(path: Path, expected_rows: int, min_bytes: int) -> dict[str, object]:
    size = path.stat().st_size
    if size < min_bytes:
        return {"valid": False, "reason": "too_small", "bytes": size, "rows": 0}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        header = handle.readline().strip().split(",")
    if "sample_id" not in header or "top1_cc" not in header:
        return {"valid": False, "reason": "bad_header", "bytes": size, "rows": 0}
    rows = max(0, count_nonempty_lines(path) - 1)
    if rows != expected_rows:
        return {
            "valid": False,
            "reason": "row_count_mismatch",
            "bytes": size,
            "rows": rows,
            "expected_rows": expected_rows,
        }
    return {"valid": True, "reason": "ok", "bytes": size, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit manifest shards against prediction CSV outputs.")
    parser.add_argument("--chunks-dir", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--manifest-pattern", default="manifest.part*.jsonl")
    parser.add_argument("--prediction-pattern", default="predictions.part*.csv")
    parser.add_argument("--min-prediction-bytes", type=int, default=128)
    args = parser.parse_args()

    chunks_dir = Path(args.chunks_dir)
    pred_dir = Path(args.pred_dir)
    expected = {part_index(path): path for path in chunks_dir.glob(args.manifest_pattern)}
    expected_rows = {idx: count_nonempty_lines(path) for idx, path in expected.items()}
    prediction_paths = {part_index(path): path for path in pred_dir.glob(args.prediction_pattern)}
    statuses = {
        idx: prediction_status(path, expected_rows.get(idx, 0), args.min_prediction_bytes)
        for idx, path in prediction_paths.items()
    }
    present = {idx: prediction_paths[idx] for idx, status in statuses.items() if bool(status["valid"])}
    empty = {idx: prediction_paths[idx] for idx, status in statuses.items() if status["reason"] == "too_small"}
    incomplete = {
        idx: status
        for idx, status in statuses.items()
        if idx in expected and not bool(status["valid"]) and status["reason"] != "too_small"
    }
    missing = sorted(set(expected) - set(present))
    extra = sorted(set(prediction_paths) - set(expected))
    payload = {
        "chunks_dir": str(chunks_dir),
        "pred_dir": str(pred_dir),
        "expected_shards": len(expected),
        "expected_rows": int(sum(expected_rows.values())),
        "present_prediction_shards": len(present),
        "present_prediction_rows": int(sum(int(status.get("rows", 0)) for idx, status in statuses.items() if idx in present)),
        "missing_shards": missing,
        "missing_ranges": compact_ranges(missing),
        "empty_prediction_shards": sorted(empty),
        "incomplete_prediction_shards": sorted(incomplete),
        "incomplete_prediction_details": {str(idx): incomplete[idx] for idx in sorted(incomplete)[:50]},
        "extra_prediction_shards": extra,
        "complete": not missing and not extra and not incomplete,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
