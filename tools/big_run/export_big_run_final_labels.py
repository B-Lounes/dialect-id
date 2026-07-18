#!/usr/bin/env python3
"""Export per-sample final dialect labels from a synced BIG-RUN tracker."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import shutil
import sqlite3
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from dialect_id.labels import CODE_TO_COUNTRY, CODE_TO_REGION


FIELDS = [
    "sample_id",
    "manifest_row",
    "tar_path",
    "member",
    "youtube_id",
    "duration",
    "decoded",
    "weak_cc",
    "weak_region",
    "pseudo_cc",
    "teacher_cc",
    "teacher_confidence",
    "teacher_margin",
    "pseudo_label_source",
    "whisper_language",
    "whisper_language_confidence",
    "final_dialect",
    "final_country",
    "final_region",
    "confidence_tier",
    "strong_model_only",
    "sure_model_only",
    "ultra_sure_model_only",
    "sure_with_weak_region",
    "sure_with_weak_top1",
    "top1_score",
    "margin",
    "entropy",
    "top2_cc",
    "top2_score",
    "top3_cc",
    "top3_score",
    "weak_top1_match",
    "weak_top2_match",
    "weak_top3_match",
    "weak_region_match",
    "source_prediction_csv",
]


OPTIONAL_COLUMNS = {
    "pseudo_cc": "''",
    "teacher_cc": "''",
    "teacher_confidence": "NULL",
    "teacher_margin": "NULL",
    "pseudo_label_source": "''",
    "youtube_id": "''",
    "whisper_language": "''",
    "whisper_language_confidence": "NULL",
    "top1_country": "NULL",
}


SURE_POLICY_SPECS: tuple[dict[str, Any], ...] = (
    {
        "policy": "strong_model_only",
        "description": "single-model score >= 0.95 and margin >= 0.50",
        "min_score": 0.95,
        "min_margin": 0.50,
        "max_entropy": None,
        "require_weak_region": False,
        "require_weak_top1": False,
    },
    {
        "policy": "sure_model_only",
        "description": "single-model score >= 0.98, margin >= 0.75, entropy <= 0.60",
        "min_score": 0.98,
        "min_margin": 0.75,
        "max_entropy": 0.60,
        "require_weak_region": False,
        "require_weak_top1": False,
    },
    {
        "policy": "ultra_sure_model_only",
        "description": "single-model score >= 0.99, margin >= 0.85, entropy <= 0.40",
        "min_score": 0.99,
        "min_margin": 0.85,
        "max_entropy": 0.40,
        "require_weak_region": False,
        "require_weak_top1": False,
    },
    {
        "policy": "sure_with_weak_region",
        "description": "sure_model_only plus weak/path region agrees with the model region",
        "min_score": 0.98,
        "min_margin": 0.75,
        "max_entropy": 0.60,
        "require_weak_region": True,
        "require_weak_top1": False,
    },
    {
        "policy": "sure_with_weak_top1",
        "description": "sure_model_only plus weak/path country code equals the model top1",
        "min_score": 0.98,
        "min_margin": 0.75,
        "max_entropy": 0.60,
        "require_weak_region": False,
        "require_weak_top1": True,
    },
)


def acquire_export_lock(out_dir: Path, stale_lock_seconds: float) -> Path:
    lock_path = out_dir / "_EXPORT_LOCK"
    payload = {
        "created_at_epoch": time.time(),
        "pid": os.getpid(),
    }
    while True:
        try:
            fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
            except FileNotFoundError:
                continue
            if stale_lock_seconds >= 0 and age > stale_lock_seconds:
                lock_path.unlink()
                continue
            raise SystemExit(
                f"Final label export lock exists: {lock_path}. "
                f"age_sec={age:.1f}; stale_lock_seconds={stale_lock_seconds}"
            )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        return lock_path


def prepare_output_dir(out_dir: Path, stale_lock_seconds: float) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = acquire_export_lock(out_dir, stale_lock_seconds)
    for path in out_dir.glob("final_labels_*.csv.gz"):
        path.unlink()
    for name in ("manifest.csv", "summary.json", "README.md", "_SUCCESS"):
        path = out_dir / name
        if path.exists():
            path.unlink()
    in_progress = out_dir / "_EXPORT_IN_PROGRESS"
    in_progress.write_text(json.dumps({"started_at_epoch": time.time()}, sort_keys=True) + "\n", encoding="utf-8")
    return in_progress, lock_path


def to_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_float(value: object) -> str:
    parsed = to_float(value)
    if parsed is None:
        return ""
    return f"{parsed:.8f}"


def confidence_tier(score: float | None, margin: float | None, args: argparse.Namespace) -> str:
    if score is None or margin is None:
        return "low"
    if score >= args.strong_score and margin >= args.strong_margin:
        return "strong"
    if score >= args.usable_score and margin >= args.usable_margin:
        return "usable"
    return "low"


def sure_policy_flags(
    *,
    score: float | None,
    margin: float | None,
    entropy: float | None,
    weak_region_match: bool,
    weak_top1_match: bool,
) -> dict[str, int]:
    flags: dict[str, int] = {}
    for spec in SURE_POLICY_SPECS:
        policy = str(spec["policy"])
        hit = False
        if score is not None and margin is not None:
            hit = score >= float(spec["min_score"]) and margin >= float(spec["min_margin"])
            max_entropy = spec.get("max_entropy")
            if max_entropy is not None:
                hit = hit and entropy is not None and entropy <= float(max_entropy)
            if spec.get("require_weak_region"):
                hit = hit and weak_region_match
            if spec.get("require_weak_top1"):
                hit = hit and weak_top1_match
        flags[policy] = int(hit)
    return flags


def sure_policy_summary(counts: Counter[str], total: int) -> dict[str, dict[str, float | int | str]]:
    return {
        str(spec["policy"]): {
            "description": str(spec["description"]),
            "rows": counts.get(str(spec["policy"]), 0),
            "pct": pct(counts.get(str(spec["policy"]), 0), total),
        }
        for spec in SURE_POLICY_SPECS
    }


def pct(num: int, den: int) -> float:
    return 0.0 if den == 0 else 100.0 * num / den


def get_sample_columns(conn: sqlite3.Connection) -> set[str]:
    return {str(row[1]) for row in conn.execute("PRAGMA table_info(samples)")}


def select_expr(column: str, columns: set[str]) -> str:
    if column in columns:
        return column
    if column in OPTIONAL_COLUMNS:
        return f"{OPTIONAL_COLUMNS[column]} AS {column}"
    raise KeyError(column)


def iter_samples(conn: sqlite3.Connection, args: argparse.Namespace) -> Iterable[sqlite3.Row]:
    columns = get_sample_columns(conn)
    required = {
        "sample_id",
        "manifest_row",
        "tar_path",
        "member",
        "weak_cc",
        "decoded",
        "duration",
        "top1_cc",
        "top1_score",
        "top2_cc",
        "top2_score",
        "top3_cc",
        "top3_score",
        "margin",
        "entropy",
        "top1_region",
        "output_csv",
    }
    missing = sorted(required - columns)
    if missing:
        raise SystemExit(f"samples table is missing required columns: {missing}")

    selected = [
        "sample_id",
        "manifest_row",
        "tar_path",
        "member",
        "weak_cc",
        select_expr("pseudo_cc", columns),
        select_expr("teacher_cc", columns),
        select_expr("teacher_confidence", columns),
        select_expr("teacher_margin", columns),
        select_expr("pseudo_label_source", columns),
        select_expr("youtube_id", columns),
        select_expr("whisper_language", columns),
        select_expr("whisper_language_confidence", columns),
        "decoded",
        "duration",
        "top1_cc",
        "top1_score",
        "top2_cc",
        "top2_score",
        "top3_cc",
        "top3_score",
        "margin",
        "entropy",
        select_expr("top1_country", columns),
        "top1_region",
        "output_csv",
    ]
    where = "WHERE decoded = 1" if args.decoded_only else ""
    order_by = {
        "sample_id": "sample_id",
        "path": "tar_path, member",
        "none": "",
    }[args.order_by]
    order_sql = f"ORDER BY {order_by}" if order_by else ""
    query = f"SELECT {', '.join(selected)} FROM samples {where} {order_sql}"
    cursor = conn.execute(query)
    while True:
        rows = cursor.fetchmany(args.batch_size)
        if not rows:
            break
        yield from rows


class ShardedWriter:
    def __init__(
        self,
        out_dir: Path,
        *,
        rows_per_file: int,
        gzip_level: int,
        gzip_backend: str = "python",
        pigz_threads: int = 0,
    ) -> None:
        self.out_dir = out_dir
        self.rows_per_file = rows_per_file
        self.gzip_level = gzip_level
        self.gzip_backend = gzip_backend
        self.pigz_threads = pigz_threads
        self.index = -1
        self.rows_in_file = 0
        self.total_rows = 0
        self.handle: Any | None = None
        self.text_handle: Any | None = None
        self.proc: subprocess.Popen[bytes] | None = None
        self.writer: csv.DictWriter[str] | None = None
        self.current_path: Path | None = None
        self.current_first_sample_id: str = ""
        self.current_last_sample_id: str = ""
        self.manifest_rows: list[dict[str, Any]] = []

    def _open_text_handle(self, path: Path) -> Any:
        backend = self.gzip_backend
        if backend == "auto":
            backend = "pigz" if shutil.which("pigz") else "python"
        if backend == "python":
            self.handle = gzip.open(path, "wt", encoding="utf-8", newline="", compresslevel=self.gzip_level)
            return self.handle
        if backend != "pigz":
            raise ValueError(f"Unsupported gzip backend: {self.gzip_backend}")
        pigz = shutil.which("pigz")
        if pigz is None:
            raise RuntimeError("gzip backend 'pigz' requested, but pigz is not available on PATH")
        threads = self.pigz_threads if self.pigz_threads > 0 else max(1, min(8, os.cpu_count() or 1))
        self.handle = path.open("wb")
        self.proc = subprocess.Popen(
            [pigz, "-c", f"-{self.gzip_level}", "-p", str(threads)],
            stdin=subprocess.PIPE,
            stdout=self.handle,
        )
        assert self.proc.stdin is not None
        self.text_handle = io.TextIOWrapper(self.proc.stdin, encoding="utf-8", newline="")
        return self.text_handle

    def _open_next(self, first_sample_id: object) -> None:
        self.close()
        self.index += 1
        self.rows_in_file = 0
        self.current_first_sample_id = str(first_sample_id or "")
        self.current_last_sample_id = ""
        self.current_path = self.out_dir / f"final_labels_{self.index:05d}.csv.gz"
        self.writer = csv.DictWriter(self._open_text_handle(self.current_path), fieldnames=FIELDS)
        self.writer.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        if self.writer is None or self.rows_in_file >= self.rows_per_file:
            self._open_next(row.get("sample_id"))
        assert self.writer is not None
        self.writer.writerow({field: row.get(field, "") for field in FIELDS})
        self.rows_in_file += 1
        self.total_rows += 1
        self.current_last_sample_id = str(row.get("sample_id") or "")

    def close(self) -> None:
        if self.handle is None and self.text_handle is None:
            return
        assert self.current_path is not None
        if self.text_handle is not None:
            self.text_handle.close()
            self.text_handle = None
        elif self.handle is not None:
            self.handle.close()
        if self.proc is not None:
            ret = self.proc.wait()
            self.proc = None
            if self.handle is not None:
                self.handle.close()
                self.handle = None
            if ret != 0:
                raise RuntimeError(f"pigz failed for {self.current_path} with exit code {ret}")
        else:
            self.handle = None
        self.manifest_rows.append(
            {
                "file": str(self.current_path),
                "rows": self.rows_in_file,
                "first_sample_id": self.current_first_sample_id,
                "last_sample_id": self.current_last_sample_id,
                "bytes": self.current_path.stat().st_size,
            }
        )
        self.writer = None
        self.current_path = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--strong-score", type=float, default=0.95)
    parser.add_argument("--strong-margin", type=float, default=0.50)
    parser.add_argument("--usable-score", type=float, default=0.90)
    parser.add_argument("--usable-margin", type=float, default=0.30)
    parser.add_argument("--rows-per-file", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--gzip-level", type=int, default=1)
    parser.add_argument("--gzip-backend", choices=["python", "pigz", "auto"], default="python")
    parser.add_argument("--pigz-threads", type=int, default=0)
    parser.add_argument("--decoded-only", action="store_true")
    parser.add_argument("--require-all-decoded", action="store_true")
    parser.add_argument("--order-by", choices=["sample_id", "path", "none"], default="sample_id")
    parser.add_argument("--stale-lock-seconds", type=float, default=3600.0)
    args = parser.parse_args()

    start = time.time()
    in_progress, lock_path = prepare_output_dir(args.out_dir, args.stale_lock_seconds)
    conn = sqlite3.connect(f"file:{args.tracker}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    counts = {
        "samples": int(conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]),
        "decoded": int(conn.execute("SELECT COUNT(*) FROM samples WHERE decoded = 1").fetchone()[0]),
        "nondecoded": int(conn.execute("SELECT COUNT(*) FROM samples WHERE decoded != 1").fetchone()[0]),
    }
    if args.require_all_decoded and counts["nondecoded"]:
        raise SystemExit(f"Refusing export with nondecoded samples: {counts['nondecoded']}")

    writer = ShardedWriter(
        args.out_dir,
        rows_per_file=args.rows_per_file,
        gzip_level=args.gzip_level,
        gzip_backend=args.gzip_backend,
        pigz_threads=args.pigz_threads,
    )
    tier_counts: Counter[str] = Counter()
    dialect_counts: Counter[str] = Counter()
    region_counts: Counter[str] = Counter()
    weak_match_counts: Counter[str] = Counter()
    sure_policy_counts: Counter[str] = Counter()
    tier_by_dialect: dict[str, Counter[str]] = defaultdict(Counter)
    exported_rows = 0

    for row in iter_samples(conn, args):
        weak = str(row["weak_cc"] or "").upper()
        pred = str(row["top1_cc"] or "").upper()
        pred2 = str(row["top2_cc"] or "").upper()
        pred3 = str(row["top3_cc"] or "").upper()
        score = to_float(row["top1_score"])
        margin = to_float(row["margin"])
        entropy = to_float(row["entropy"])
        tier = confidence_tier(score, margin, args)
        weak_region = CODE_TO_REGION.get(weak, "unknown")
        pred_region = str(row["top1_region"] or CODE_TO_REGION.get(pred, "unknown") or "unknown")
        pred_country = str(row["top1_country"] or CODE_TO_COUNTRY.get(pred, "") or "")
        weak_top1 = bool(weak and weak == pred)
        weak_top2 = bool(weak and weak in {pred, pred2})
        weak_top3 = bool(weak and weak in {pred, pred2, pred3})
        weak_region_match = bool(weak_region != "unknown" and weak_region == pred_region)
        policy_flags = sure_policy_flags(
            score=score,
            margin=margin,
            entropy=entropy,
            weak_region_match=weak_region_match,
            weak_top1_match=weak_top1,
        )

        out_row = {
            "sample_id": row["sample_id"],
            "manifest_row": row["manifest_row"],
            "tar_path": row["tar_path"],
            "member": row["member"],
            "youtube_id": row["youtube_id"],
            "duration": format_float(row["duration"]),
            "decoded": int(row["decoded"] or 0),
            "weak_cc": weak,
            "weak_region": weak_region,
            "pseudo_cc": row["pseudo_cc"],
            "teacher_cc": row["teacher_cc"],
            "teacher_confidence": format_float(row["teacher_confidence"]),
            "teacher_margin": format_float(row["teacher_margin"]),
            "pseudo_label_source": row["pseudo_label_source"],
            "whisper_language": row["whisper_language"],
            "whisper_language_confidence": format_float(row["whisper_language_confidence"]),
            "final_dialect": pred,
            "final_country": pred_country,
            "final_region": pred_region,
            "confidence_tier": tier,
            **policy_flags,
            "top1_score": format_float(score),
            "margin": format_float(margin),
            "entropy": format_float(entropy),
            "top2_cc": pred2,
            "top2_score": format_float(row["top2_score"]),
            "top3_cc": pred3,
            "top3_score": format_float(row["top3_score"]),
            "weak_top1_match": int(weak_top1),
            "weak_top2_match": int(weak_top2),
            "weak_top3_match": int(weak_top3),
            "weak_region_match": int(weak_region_match),
            "source_prediction_csv": row["output_csv"],
        }
        writer.write(out_row)
        exported_rows += 1
        tier_counts[tier] += 1
        dialect_counts[pred or "MISSING"] += 1
        region_counts[pred_region] += 1
        tier_by_dialect[pred or "MISSING"][tier] += 1
        weak_match_counts["top1"] += int(weak_top1)
        weak_match_counts["top2"] += int(weak_top2)
        weak_match_counts["top3"] += int(weak_top3)
        weak_match_counts["region"] += int(weak_region_match)
        for policy, value in policy_flags.items():
            sure_policy_counts[policy] += int(value)

    writer.close()
    conn.close()

    manifest_path = args.out_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        manifest_writer = csv.DictWriter(handle, fieldnames=["file", "rows", "first_sample_id", "last_sample_id", "bytes"])
        manifest_writer.writeheader()
        manifest_writer.writerows(writer.manifest_rows)

    by_dialect = []
    for dialect, count in dialect_counts.most_common():
        tiers = tier_by_dialect[dialect]
        by_dialect.append(
            {
                "final_dialect": dialect,
                "country": CODE_TO_COUNTRY.get(dialect, ""),
                "region": CODE_TO_REGION.get(dialect, "unknown"),
                "rows": count,
                "strong": tiers.get("strong", 0),
                "usable": tiers.get("usable", 0),
                "low": tiers.get("low", 0),
                "strong_pct": pct(tiers.get("strong", 0), count),
            }
        )

    summary = {
        "tracker": str(args.tracker),
        "out_dir": str(args.out_dir),
        "generated_at_epoch": time.time(),
        "wall_sec": time.time() - start,
        "thresholds": {
            "strong_score": args.strong_score,
            "strong_margin": args.strong_margin,
            "usable_score": args.usable_score,
            "usable_margin": args.usable_margin,
        },
        "source_sample_counts": counts,
        "decoded_only": bool(args.decoded_only),
        "compression": {
            "gzip_level": args.gzip_level,
            "gzip_backend": args.gzip_backend,
            "pigz_threads": args.pigz_threads,
        },
        "exported_rows": exported_rows,
        "files": writer.manifest_rows,
        "confidence_tiers": dict(tier_counts),
        "sure_policy_specs": list(SURE_POLICY_SPECS),
        "sure_policy_counts": sure_policy_summary(sure_policy_counts, exported_rows),
        "weak_match_counts": {
            "top1": weak_match_counts.get("top1", 0),
            "top1_pct": pct(weak_match_counts.get("top1", 0), exported_rows),
            "top2": weak_match_counts.get("top2", 0),
            "top2_pct": pct(weak_match_counts.get("top2", 0), exported_rows),
            "top3": weak_match_counts.get("top3", 0),
            "top3_pct": pct(weak_match_counts.get("top3", 0), exported_rows),
            "region": weak_match_counts.get("region", 0),
            "region_pct": pct(weak_match_counts.get("region", 0), exported_rows),
        },
        "by_final_dialect": by_dialect,
        "by_final_region": dict(region_counts),
        "columns": FIELDS,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    readme_lines = [
        "# BIG-RUN Final Dialect Labels",
        "",
        f"- Tracker: `{args.tracker}`",
        f"- Exported rows: `{exported_rows:,}`",
        f"- Output files: `{len(writer.manifest_rows):,}`",
        f"- Strong threshold: `top1_score >= {args.strong_score}` and `margin >= {args.strong_margin}`",
        f"- Usable threshold: `top1_score >= {args.usable_score}` and `margin >= {args.usable_margin}`",
        "",
        "Each row is one audio sample. `final_dialect` is the single-model top1 decision.",
        "`confidence_tier` is derived only from model score and top1-top2 margin.",
        "`sure_*` columns are stricter decision-policy flags; metadata-overlay policies are audit subsets, not alternate labels.",
        "`weak_cc` is path/Qwen metadata and is not treated as ground truth.",
        "",
        "Files:",
        "",
        "- `manifest.csv`: list of label shards, row counts, first/last sample id in file order, and file size.",
        "- `summary.json`: counts, thresholds, weak-metadata agreement, and dialect totals.",
        "- `final_labels_*.csv.gz`: per-sample labels and audit metadata.",
        "",
        "Columns:",
        "",
    ]
    readme_lines.extend(f"- `{field}`" for field in FIELDS)
    (args.out_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    (args.out_dir / "_SUCCESS").write_text(
        json.dumps({"completed_at_epoch": time.time(), "exported_rows": exported_rows}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    in_progress.unlink(missing_ok=True)
    lock_path.unlink(missing_ok=True)

    print(json.dumps({"exported_rows": exported_rows, "out_dir": str(args.out_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
