#!/usr/bin/env python3
"""Audit checkpoint-to-evaluation coverage for the 23-way DID task."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


CORE_SPLITS = (
    "acgc_old19_validation",
    "acgc_old19_test",
    "acgc_td_validation_23way",
    "acgc_td_test_23way",
)
OLD19_SPLITS = ("acgc_old19_validation", "acgc_old19_test")
TD_SPLITS = ("acgc_td_validation_23way", "acgc_td_test_23way")
METRIC_KEYS = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--reports-root", required=True)
    parser.add_argument("--logs-root", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_sort_key(path: Path) -> tuple[int, int, str]:
    name = path.stem
    if name.startswith("step_"):
        try:
            return (0, int(name.split("_", 1)[1]), str(path))
        except ValueError:
            return (0, -1, str(path))
    if name == "final":
        return (1, 10**9, str(path))
    if name.startswith("epoch_"):
        try:
            return (2, int(name.split("_", 1)[1]), str(path))
        except ValueError:
            return (2, -1, str(path))
    return (3, -1, str(path))


def run_short_name(run_name: str) -> str:
    return run_name


def load_args_summary(run_dir: Path) -> dict[str, Any]:
    args_path = run_dir / "args.json"
    if not args_path.is_file():
        return {"args_exists": False}
    try:
        args = load_json(args_path)
    except (OSError, json.JSONDecodeError) as exc:
        return {"args_exists": True, "args_error": str(exc)}
    keep = [
        "init_from_checkpoint",
        "max_train_steps",
        "freeze_encoder_layers",
        "learning_rate",
        "head_learning_rate",
        "pseudo_manifest_glob",
        "disable_clean_train",
        "skip_validation",
    ]
    return {"args_exists": True, **{key: args.get(key) for key in keep if key in args}}


def collect_runs(run_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runs = []
    checkpoints = []
    for run_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        ckpts = sorted((run_dir / "checkpoints").glob("*.pt"), key=checkpoint_sort_key)
        run_row = {
            "run": run_dir.name,
            "short_run": run_short_name(run_dir.name),
            "path": str(run_dir),
            "checkpoint_count": len(ckpts),
            "args": load_args_summary(run_dir),
        }
        runs.append(run_row)
        for ckpt in ckpts:
            stat = ckpt.stat()
            checkpoints.append(
                {
                    "run": run_dir.name,
                    "short_run": run_row["short_run"],
                    "checkpoint": str(ckpt),
                    "checkpoint_name": ckpt.name,
                    "size_bytes": stat.st_size,
                    "mtime": stat.st_mtime,
                }
            )
    return runs, checkpoints


def split_from_metric_file(path: Path) -> str:
    suffix = "_metrics.json"
    return path.name[: -len(suffix)] if path.name.endswith(suffix) else path.stem


def extract_metrics(path: Path) -> dict[str, float]:
    try:
        payload = load_json(path)
    except (OSError, json.JSONDecodeError):
        return {}
    metrics = payload.get("metrics") if isinstance(payload, dict) else None
    if not isinstance(metrics, dict) and isinstance(payload, dict):
        metrics = payload
    if not isinstance(metrics, dict):
        return {}
    parsed = {}
    for key in METRIC_KEYS:
        value = metrics.get(key)
        if value is not None:
            try:
                parsed[key] = float(value)
            except (TypeError, ValueError):
                pass
    return parsed


def collect_reports(reports_root: Path) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for report_dir in sorted(path for path in reports_root.glob("eval_*") if path.is_dir()):
        splits = {}
        for metric_path in sorted(report_dir.glob("*_metrics.json")):
            split = split_from_metric_file(metric_path)
            splits[split] = {
                "metrics": extract_metrics(metric_path),
                "metrics_path": str(metric_path),
            }
        reports[report_dir.name] = {
            "report": report_dir.name,
            "path": str(report_dir),
            "splits": splits,
            "has_summary": (report_dir / "summary_metrics.json").is_file(),
            "has_old19_valtest": all(split in splits for split in OLD19_SPLITS),
            "has_td_valtest": all(split in splits for split in TD_SPLITS),
            "has_core_valtest_td": all(split in splits for split in CORE_SPLITS),
        }
    return reports


def parse_eval_logs(logs_root: Path) -> list[dict[str, str]]:
    rows = []
    checkpoint_re = re.compile(r"^checkpoint\s*:\s*(.+)$")
    out_dir_re = re.compile(r"^out_dir\s*:\s*(.+)$")
    for path in sorted(logs_root.glob("*.out")):
        checkpoint = ""
        out_dir = ""
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            if match := checkpoint_re.match(line):
                checkpoint = match.group(1).strip()
            elif match := out_dir_re.match(line):
                out_dir = match.group(1).strip()
        if checkpoint or out_dir:
            rows.append(
                {
                    "log": str(path),
                    "checkpoint": checkpoint,
                    "out_dir": out_dir,
                    "report": Path(out_dir).name if out_dir else "",
                }
            )
    return rows


def coverage_status(report_names: list[str], reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    split_union = set()
    old19_reports = []
    td_reports = []
    core_reports = []
    td_pass = False
    for report_name in report_names:
        report = reports.get(report_name)
        if not report:
            continue
        split_union.update(report["splits"])
        if report["has_old19_valtest"]:
            old19_reports.append(report_name)
        if report["has_td_valtest"]:
            td_reports.append(report_name)
            td_val = report["splits"]["acgc_td_validation_23way"]["metrics"].get("accuracy")
            td_test = report["splits"]["acgc_td_test_23way"]["metrics"].get("accuracy")
            td_pass = td_pass or (td_val == 1.0 and td_test == 1.0)
        if report["has_core_valtest_td"]:
            core_reports.append(report_name)
    return {
        "reports": sorted(set(report_names)),
        "splits": sorted(split_union),
        "has_old19_valtest": bool(old19_reports),
        "has_td_valtest": bool(td_reports),
        "td_pass": td_pass,
        "has_core_valtest_td": bool(core_reports),
        "old19_reports": old19_reports,
        "td_reports": td_reports,
        "core_reports": core_reports,
    }


def classify_checkpoint(row: dict[str, Any], coverage: dict[str, Any]) -> str:
    run = row["run"]
    name = row["checkpoint_name"]
    if coverage["has_core_valtest_td"]:
        return "core_covered"
    if coverage["has_old19_valtest"] and coverage["has_td_valtest"]:
        return "covered_split_across_reports"
    if coverage["has_old19_valtest"]:
        return "old19_only"
    if coverage["has_td_valtest"]:
        return "td_only"
    if "smoke" in run or name.startswith("epoch_"):
        return "debug_or_epoch_uncovered"
    return "uncovered"


def candidate_priority(row: dict[str, Any], status: str, coverage: dict[str, Any]) -> tuple[int, str]:
    run = row["run"]
    name = row["checkpoint_name"]
    if status not in {"uncovered", "old19_only", "td_only"}:
        return (99, "")
    if coverage["has_td_valtest"] and not coverage["td_pass"]:
        return (85, "TD validation/test already fails, so this checkpoint is not eligible")
    if any(marker in run.lower() for marker in ("smoke", "debug", "head_only", "row_only")):
        return (90, "debug/specialized run")
    if name in {"final.pt", "epoch_1.pt"}:
        return (80, "final/epoch duplicate candidate")
    if re.match(r"step_\d+\.pt$", name):
        return (10, "standard step checkpoint without complete coverage")
    if name.startswith("step_"):
        return (20, "intermediate step checkpoint without complete coverage")
    return (50, "uncovered checkpoint")


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    run_root = Path(args.run_root)
    reports_root = Path(args.reports_root)
    logs_root = Path(args.logs_root)
    runs, checkpoints = collect_runs(run_root)
    reports = collect_reports(reports_root)
    eval_logs = parse_eval_logs(logs_root)

    reports_by_checkpoint: dict[str, list[str]] = defaultdict(list)
    for row in eval_logs:
        if row["checkpoint"] and row["report"]:
            reports_by_checkpoint[row["checkpoint"]].append(row["report"])

    checkpoint_rows = []
    status_counts = defaultdict(int)
    recommendation_rows = []
    for row in checkpoints:
        report_names = reports_by_checkpoint.get(row["checkpoint"], [])
        coverage = coverage_status(report_names, reports)
        status = classify_checkpoint(row, coverage)
        status_counts[status] += 1
        priority, reason = candidate_priority(row, status, coverage)
        checkpoint_row = {**row, "coverage": coverage, "status": status, "recommendation_priority": priority}
        checkpoint_rows.append(checkpoint_row)
        if priority < 50:
            recommendation_rows.append({**checkpoint_row, "recommendation_reason": reason})

    recommendation_rows.sort(key=lambda row: (row["recommendation_priority"], row["run"], row["checkpoint_name"]))
    return {
        "run_root": str(run_root),
        "reports_root": str(reports_root),
        "logs_root": str(logs_root),
        "run_count": len(runs),
        "checkpoint_count": len(checkpoints),
        "report_count": len(reports),
        "eval_log_count": len(eval_logs),
        "status_counts": dict(sorted(status_counts.items())),
        "runs_without_checkpoints": [row for row in runs if row["checkpoint_count"] == 0],
        "recommended_next_evals": recommendation_rows[:25],
        "checkpoints": checkpoint_rows,
        "reports": reports,
        "eval_logs": eval_logs,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Eval Coverage Audit",
        "",
        "| Field | Value |",
        "|---|---:|",
        f"| Runs | {payload['run_count']} |",
        f"| Checkpoints | {payload['checkpoint_count']} |",
        f"| Eval reports | {payload['report_count']} |",
        f"| Eval logs parsed | {payload['eval_log_count']} |",
        "",
        "## Status Counts",
        "",
        "| Status | Checkpoints |",
        "|---|---:|",
    ]
    for status, count in payload["status_counts"].items():
        lines.append(f"| `{status}` | {count} |")

    lines.extend(["", "## Recommended Next Evals", ""])
    if not payload["recommended_next_evals"]:
        lines.append("No uncovered high-priority checkpoint was found by the audit.")
    else:
        lines.extend(
            [
                "| Priority | Run | Checkpoint | Status | Reason |",
                "|---:|---|---|---|---|",
            ]
        )
        for row in payload["recommended_next_evals"][:25]:
            lines.append(
                "| {} | `{}` | `{}` | `{}` | {} |".format(
                    row["recommendation_priority"],
                    row["short_run"],
                    row["checkpoint_name"],
                    row["status"],
                    row["recommendation_reason"],
                )
            )

    lines.extend(["", "## Runs Without Checkpoints", ""])
    if not payload["runs_without_checkpoints"]:
        lines.append("None.")
    else:
        for row in payload["runs_without_checkpoints"]:
            lines.append(f"- `{row['run']}`")

    lines.extend(["", "## Notes", ""])
    lines.append("- `core_covered` means one report has old19 validation/test and TD validation/test.")
    lines.append("- `covered_split_across_reports` means old19 and TD are both covered, possibly by companion reports.")
    lines.append("- The audit maps checkpoints to reports through SLURM log `checkpoint:` and `out_dir:` lines.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(out_md, payload)
    print(
        json.dumps(
            {
                "checkpoint_count": payload["checkpoint_count"],
                "report_count": payload["report_count"],
                "status_counts": payload["status_counts"],
                "recommended_next_evals": len(payload["recommended_next_evals"]),
                "out_json": str(out_json),
                "out_md": str(out_md),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
