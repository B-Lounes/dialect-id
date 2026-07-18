#!/usr/bin/env python3
"""Report BIG-RUN final export status from manifest/state/prediction files."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def file_age(path: Path, now: float) -> float | None:
    try:
        return max(0.0, now - path.stat().st_mtime)
    except FileNotFoundError:
        return None


def dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except FileNotFoundError:
                pass
    return total


def human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TiB"


def run_squeue(job_ids: str) -> list[str]:
    if not job_ids:
        return []
    try:
        proc = subprocess.run(
            ["squeue", "-j", job_ids, "-o", "%i %j %T %M %R %N %E"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        return []
    if proc.returncode != 0:
        return [proc.stderr.strip()]
    return [line for line in proc.stdout.splitlines() if line.strip()]


def estimate_eta(done_markers: list[Path], expected_count: int, now: float) -> dict[str, Any]:
    if len(done_markers) < 2 or expected_count <= len(done_markers):
        return {"done_per_hour_recent": None, "eta_hours_recent": None}
    recent_window_sec = 30 * 60
    recent = []
    for path in done_markers:
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            continue
        if now - mtime <= recent_window_sec:
            recent.append(mtime)
    if len(recent) >= 2:
        elapsed = max(recent) - min(recent)
        if elapsed > 0:
            rate_per_hour = (len(recent) - 1) / elapsed * 3600.0
            remaining = expected_count - len(done_markers)
            return {
                "done_per_hour_recent": rate_per_hour,
                "eta_hours_recent": remaining / rate_per_hour if rate_per_hour > 0 else None,
            }
    return {"done_per_hour_recent": None, "eta_hours_recent": None}


def summarize_done_markers(done_markers: list[Path], expected_count: int, now: float) -> dict[str, Any]:
    totals = {
        "done_marker_files_read": 0,
        "done_marker_read_errors": 0,
        "decoded_rows_done": 0,
        "expanded_rows_done": 0,
        "failed_rows_done": 0,
        "csv_rows_done": 0,
        "duration_sec_done": 0.0,
        "worker_elapsed_sec_done": 0.0,
        "avg_decoded_rows_per_done_shard": None,
        "projected_decoded_rows_total_by_done_mean": None,
        "projected_decoded_rows_remaining_by_done_mean": None,
        "decoded_rows_per_hour_recent": None,
        "row_eta_hours_recent_by_done_mean": None,
    }
    recent_window_sec = 30 * 60
    recent_decoded_rows = 0
    recent_mtimes = []
    for path in done_markers:
        try:
            mtime = path.stat().st_mtime
            marker = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            totals["done_marker_read_errors"] += 1
            continue
        decoded_rows = int(marker.get("decoded_rows") or 0)
        totals["done_marker_files_read"] += 1
        totals["decoded_rows_done"] += decoded_rows
        totals["expanded_rows_done"] += int(marker.get("expanded_rows") or 0)
        totals["failed_rows_done"] += int(marker.get("failed_rows") or 0)
        totals["csv_rows_done"] += int(marker.get("csv_rows") or 0)
        totals["duration_sec_done"] += float(marker.get("duration_sec") or 0.0)
        totals["worker_elapsed_sec_done"] += float(marker.get("elapsed_sec") or 0.0)
        if now - mtime <= recent_window_sec:
            recent_decoded_rows += decoded_rows
            recent_mtimes.append(mtime)
    if totals["done_marker_files_read"]:
        avg_rows = totals["decoded_rows_done"] / totals["done_marker_files_read"]
        projected_total = avg_rows * expected_count if expected_count else None
        totals["avg_decoded_rows_per_done_shard"] = avg_rows
        totals["projected_decoded_rows_total_by_done_mean"] = projected_total
        if projected_total is not None:
            totals["projected_decoded_rows_remaining_by_done_mean"] = max(
                0.0,
                projected_total - totals["decoded_rows_done"],
            )
    if len(recent_mtimes) >= 2:
        elapsed = max(recent_mtimes) - min(recent_mtimes)
        if elapsed > 0:
            row_rate = recent_decoded_rows / elapsed * 3600.0
            totals["decoded_rows_per_hour_recent"] = row_rate
            remaining = totals["projected_decoded_rows_remaining_by_done_mean"]
            if row_rate > 0 and remaining is not None:
                totals["row_eta_hours_recent_by_done_mean"] = remaining / row_rate
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--job-ids", default="")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--include-size", action="store_true")
    args = parser.parse_args()

    now = time.time()
    state_dir = args.out_root / "state"
    pred_dir = args.out_root / "predictions"
    expected = {path.stem for path in args.manifest_dir.glob("shard_*.jsonl")}
    done_paths = list(state_dir.glob("shard_*.done"))
    failed_paths = list(state_dir.glob("shard_*.failed"))
    lock_paths = list(state_dir.glob("shard_*.lock"))
    done = {path.stem.rsplit(".", 1)[0] for path in done_paths}
    failed = {path.stem.rsplit(".", 1)[0] for path in failed_paths}
    final_csv = {path.stem for path in pred_dir.glob("shard_*.csv") if ".tmp" not in path.name}
    tmp_csv = sorted(path.name for path in pred_dir.glob("shard_*.tmp.csv"))

    lock_ages = [age for path in lock_paths if (age := file_age(path, now)) is not None]
    final_csv_without_done = sorted(final_csv - done)
    done_without_final_csv = sorted(done - final_csv)
    payload: dict[str, Any] = {
        "created_unix": now,
        "out_root": str(args.out_root),
        "manifest_dir": str(args.manifest_dir),
        "expected_shards": len(expected),
        "done_shards": len(done),
        "final_csv_shards": len(final_csv),
        "tmp_csv_files": len(tmp_csv),
        "failed_shards": len(failed),
        "active_locks": len(lock_paths),
        "completion_pct": 0.0 if not expected else 100.0 * len(done) / len(expected),
        "remaining_shards": len(expected - done - failed),
        "final_csv_without_done_count": len(final_csv_without_done),
        "done_without_final_csv_count": len(done_without_final_csv),
        "final_csv_without_done": final_csv_without_done[:50],
        "done_without_final_csv": done_without_final_csv[:50],
        "failed_examples": sorted(failed)[:50],
        "tmp_csv_examples": tmp_csv[:50],
        "lock_age_max_sec": max(lock_ages) if lock_ages else None,
        "lock_age_min_sec": min(lock_ages) if lock_ages else None,
        "squeue": run_squeue(args.job_ids),
    }
    payload.update(summarize_done_markers(done_paths, len(expected), now))
    payload.update(estimate_eta(done_paths, len(expected), now))
    if args.include_size:
        size = dir_size_bytes(pred_dir)
        payload["prediction_dir_bytes"] = size
        payload["prediction_dir_human"] = human_bytes(size)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        avg_rows = payload.get("avg_decoded_rows_per_done_shard")
        projected_total = payload.get("projected_decoded_rows_total_by_done_mean")
        avg_rows_text = "n/a" if avg_rows is None else f"{avg_rows:,.1f}"
        projected_total_text = "n/a" if projected_total is None else f"{projected_total:,.0f}"
        lines = [
            "# BIG-RUN Export Status",
            "",
            f"- Done shards: `{payload['done_shards']:,} / {payload['expected_shards']:,}` (`{payload['completion_pct']:.3f}%`)",
            f"- Decoded rows in done shards: `{payload['decoded_rows_done']:,}`",
            f"- Expanded rows in done shards: `{payload['expanded_rows_done']:,}`",
            f"- Failed rows in done shards: `{payload['failed_rows_done']:,}`",
            f"- Mean decoded rows per done shard: `{avg_rows_text}`",
            f"- Projected total decoded rows from done-shard mean: `{projected_total_text}`",
            f"- Final prediction CSVs: `{payload['final_csv_shards']:,}`",
            f"- Failed shards: `{payload['failed_shards']:,}`",
            f"- Active locks: `{payload['active_locks']:,}`",
            f"- CSVs without `.done`: `{payload['final_csv_without_done_count']:,}`",
            f"- `.done` without CSV: `{payload['done_without_final_csv_count']:,}`",
        ]
        if payload.get("done_per_hour_recent") is not None:
            lines.append(f"- Recent throughput: `{payload['done_per_hour_recent']:.1f}` shards/hour")
            lines.append(f"- Recent ETA: `{payload['eta_hours_recent']:.2f}` hours")
        if payload.get("decoded_rows_per_hour_recent") is not None:
            lines.append(f"- Recent row throughput: `{payload['decoded_rows_per_hour_recent']:,.0f}` decoded rows/hour")
            row_eta = payload.get("row_eta_hours_recent_by_done_mean")
            if row_eta is not None:
                lines.append(f"- Recent row ETA from done-shard mean: `{row_eta:.2f}` hours")
        if payload.get("prediction_dir_human"):
            lines.append(f"- Prediction dir size: `{payload['prediction_dir_human']}`")
        if payload["squeue"]:
            lines.extend(["", "## squeue", "", "```", *payload["squeue"], "```"])
        args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
