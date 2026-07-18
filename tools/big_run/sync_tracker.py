#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def ensure_sample_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS samples (
            sample_id INTEGER,
            tar_path TEXT NOT NULL,
            member TEXT NOT NULL,
            manifest_row INTEGER,
            weak_cc TEXT,
            pseudo_cc TEXT,
            teacher_cc TEXT,
            teacher_confidence REAL,
            teacher_margin REAL,
            pseudo_label_source TEXT,
            youtube_id TEXT,
            whisper_language TEXT,
            whisper_language_confidence REAL,
            decoded INTEGER NOT NULL,
            duration REAL NOT NULL DEFAULT 0.0,
            top1_cc TEXT,
            top1_score REAL,
            top2_cc TEXT,
            top2_score REAL,
            top3_cc TEXT,
            top3_score REAL,
            margin REAL,
            entropy REAL,
            top1_country TEXT,
            top1_region TEXT,
            output_csv TEXT,
            updated_at REAL,
            PRIMARY KEY (tar_path, member)
        );
        """
    )
    existing = {row[1] for row in conn.execute("PRAGMA table_info(samples)")}
    wanted = {
        "pseudo_cc": "TEXT",
        "teacher_cc": "TEXT",
        "teacher_confidence": "REAL",
        "teacher_margin": "REAL",
        "pseudo_label_source": "TEXT",
        "youtube_id": "TEXT",
        "whisper_language": "TEXT",
        "whisper_language_confidence": "REAL",
        "top1_country": "TEXT",
    }
    for column, kind in wanted.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE samples ADD COLUMN {column} {kind}")


def drop_sample_indexes(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP INDEX IF EXISTS idx_samples_decoded;
        DROP INDEX IF EXISTS idx_samples_sample_id;
        DROP INDEX IF EXISTS idx_samples_top1_cc;
        DROP INDEX IF EXISTS idx_samples_top1_region;
        """
    )


def create_sample_indexes(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_samples_decoded ON samples(decoded);
        CREATE INDEX IF NOT EXISTS idx_samples_sample_id ON samples(sample_id);
        CREATE INDEX IF NOT EXISTS idx_samples_top1_cc ON samples(top1_cc);
        CREATE INDEX IF NOT EXISTS idx_samples_top1_region ON samples(top1_region);
        """
    )


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.loads(handle.readline())


def to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def sync_done_csv(
    conn: sqlite3.Connection,
    done_payload: dict[str, object],
    *,
    sample_mode: str,
    batch_size: int,
) -> tuple[int, int, int]:
    output_csv = Path(str(done_payload["output_csv"]))
    now = time.time()
    by_tar: dict[str, dict[str, float | int]] = defaultdict(lambda: {"sample_rows": 0, "decoded_rows": 0, "failed_rows": 0, "duration_sec": 0.0})
    sample_rows: list[tuple[object, ...]] = []
    seen = 0
    decoded_total = 0
    failed_total = 0
    with output_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            decoded = 1 if str(row.get("decoded", "")) == "1" else 0
            duration = to_float(row.get("duration"))
            tar_path = str(row.get("tar_path", ""))
            group = by_tar[tar_path]
            group["sample_rows"] = int(group["sample_rows"]) + 1
            group["decoded_rows"] = int(group["decoded_rows"]) + decoded
            group["failed_rows"] = int(group["failed_rows"]) + (1 - decoded)
            group["duration_sec"] = float(group["duration_sec"]) + duration
            seen += 1
            decoded_total += decoded
            failed_total += 1 - decoded
            if sample_mode == "none":
                continue
            if sample_mode == "failed" and decoded:
                continue
            sample_rows.append(
                (
                    to_int(row.get("sample_id")),
                    tar_path,
                    str(row.get("member", "")),
                    to_int(row.get("manifest_row")),
                    str(row.get("weak_cc", "")),
                    str(row.get("pseudo_cc", "")),
                    str(row.get("teacher_cc", "")),
                    to_float(row.get("teacher_confidence")),
                    to_float(row.get("teacher_margin")),
                    str(row.get("pseudo_label_source", "")),
                    str(row.get("youtube_id", "")),
                    str(row.get("whisper_language", "")),
                    to_float(row.get("whisper_language_confidence")),
                    decoded,
                    duration,
                    str(row.get("top1_cc", "")),
                    to_float(row.get("top1_score")),
                    str(row.get("top2_cc", "")),
                    to_float(row.get("top2_score")),
                    str(row.get("top3_cc", "")),
                    to_float(row.get("top3_score")),
                    to_float(row.get("margin")),
                    to_float(row.get("entropy")),
                    str(row.get("top1_country", "")),
                    str(row.get("top1_region", "")),
                    str(output_csv),
                    now,
                )
            )
            if len(sample_rows) >= batch_size:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO samples(
                        sample_id, tar_path, member, manifest_row, weak_cc,
                        pseudo_cc, teacher_cc, teacher_confidence, teacher_margin,
                        pseudo_label_source, youtube_id, whisper_language,
                        whisper_language_confidence, decoded, duration,
                        top1_cc, top1_score, top2_cc, top2_score, top3_cc, top3_score,
                        margin, entropy, top1_country, top1_region, output_csv, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    sample_rows,
                )
                sample_rows.clear()
    if sample_rows:
        conn.executemany(
            """
            INSERT OR REPLACE INTO samples(
                sample_id, tar_path, member, manifest_row, weak_cc,
                pseudo_cc, teacher_cc, teacher_confidence, teacher_margin,
                pseudo_label_source, youtube_id, whisper_language,
                whisper_language_confidence, decoded, duration,
                top1_cc, top1_score, top2_cc, top2_score, top3_cc, top3_score,
                margin, entropy, top1_country, top1_region, output_csv, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            sample_rows,
        )

    for tar_path, stats in by_tar.items():
        status = "done" if int(stats["failed_rows"]) == 0 else "partial"
        conn.execute(
            """
            UPDATE tars
            SET status=?, sample_rows=?, decoded_rows=?, failed_rows=?, duration_sec=?,
                output_csv=?, updated_at=?
            WHERE tar_path=?
            """,
            (
                status,
                int(stats["sample_rows"]),
                int(stats["decoded_rows"]),
                int(stats["failed_rows"]),
                float(stats["duration_sec"]),
                str(output_csv),
                now,
                tar_path,
            ),
        )
    return seen, decoded_total, failed_total


def write_retry_manifest(out_path: Path, out_root: Path) -> int:
    retry_rows = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fout:
        for failed in sorted((out_root / "state").glob("shard_*.failed")):
            payload = read_json(failed)
            manifest = Path(str(payload.get("manifest", "")))
            if manifest.is_file():
                with manifest.open("r", encoding="utf-8") as fin:
                    for line in fin:
                        if line.strip():
                            fout.write(line)
                            retry_rows += 1
        for done in sorted((out_root / "state").glob("shard_*.done")):
            payload = read_json(done)
            failed_rows = payload.get("failed_rows")
            if failed_rows is not None and to_int(failed_rows) <= 0:
                continue
            csv_path = Path(str(payload.get("output_csv", "")))
            if not csv_path.is_file():
                continue
            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if str(row.get("decoded", "")) == "1":
                        continue
                    retry_row = {
                        "sample_id": to_int(row.get("sample_id")),
                        "manifest_row": to_int(row.get("manifest_row")),
                        "tar_path": str(row.get("tar_path", "")),
                        "member": str(row.get("member", "")),
                        "weak_country_code": str(row.get("weak_cc", "")),
                        "youtube_id": str(row.get("youtube_id", "")),
                        "source": "BIG-RUN-did-retry",
                    }
                    fout.write(json.dumps(retry_row, sort_keys=True) + "\n")
                    retry_rows += 1
    return retry_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync BIG-RUN-did done markers and CSVs into tracker.sqlite.")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--sample-mode", choices=["all", "failed", "none"], default="all")
    parser.add_argument("--insert-batch-size", type=int, default=10000)
    parser.add_argument("--commit-every-shards", type=int, default=100)
    parser.add_argument("--write-retry-manifest", default=None)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    conn = connect(out_root / "tracker.sqlite")
    ensure_sample_schema(conn)
    if args.sample_mode != "none":
        drop_sample_indexes(conn)
    now = time.time()
    done_count = 0
    failed_count = 0
    sample_rows = 0
    decoded_rows = 0
    failed_rows = 0
    marked_shards: set[int] = set()

    for done in sorted((out_root / "state").glob("shard_*.done")):
        payload = read_json(done)
        shard_idx = to_int(payload.get("shard"))
        marked_shards.add(shard_idx)
        csv_rows, csv_decoded, csv_failed = sync_done_csv(
            conn,
            payload,
            sample_mode=args.sample_mode,
            batch_size=args.insert_batch_size,
        )
        sample_rows += csv_rows
        decoded_rows += csv_decoded
        failed_rows += csv_failed
        status = "done" if csv_failed == 0 else "partial"
        conn.execute(
            """
            UPDATE shards
            SET status=?, attempts=MAX(attempts, 1), expanded_rows=?, decoded_rows=?,
                failed_rows=?, duration_sec=?, output_csv=?, worker_id=?, hostname=?,
                error=NULL, updated_at=?
            WHERE shard_idx=?
            """,
            (
                status,
                to_int(payload.get("expanded_rows"), csv_rows),
                csv_decoded,
                csv_failed,
                to_float(payload.get("duration_sec")),
                str(payload.get("output_csv", "")),
                str(payload.get("worker_id", "")),
                str(payload.get("hostname", "")),
                now,
                shard_idx,
            ),
        )
        done_count += 1
        if args.commit_every_shards > 0 and done_count % args.commit_every_shards == 0:
            conn.commit()

    for failed in sorted((out_root / "state").glob("shard_*.failed")):
        payload = read_json(failed)
        shard_idx = to_int(payload.get("shard"))
        marked_shards.add(shard_idx)
        conn.execute(
            """
            UPDATE shards
            SET status='failed', attempts=MAX(attempts, 1), error=?, worker_id=?,
                hostname=?, updated_at=?
            WHERE shard_idx=?
            """,
            (
                str(payload.get("error", "")),
                str(payload.get("worker_id", "")),
                str(payload.get("hostname", "")),
                now,
                shard_idx,
            ),
        )
        failed_count += 1

    if marked_shards:
        placeholders = ",".join("?" for _ in marked_shards)
        conn.execute(
            f"""
            UPDATE shards
            SET status='pending', error=NULL, worker_id=NULL, hostname=NULL, updated_at=?
            WHERE shard_idx NOT IN ({placeholders}) AND status IN ('failed', 'partial')
            """,
            (now, *sorted(marked_shards)),
        )

    retry_rows = 0
    if args.write_retry_manifest:
        retry_rows = write_retry_manifest(Path(args.write_retry_manifest), out_root)
    if args.sample_mode != "none":
        create_sample_indexes(conn)
    conn.commit()
    summary = {
        "out_root": str(out_root),
        "done_markers": done_count,
        "failed_markers": failed_count,
        "csv_sample_rows_seen": sample_rows,
        "csv_decoded_rows_seen": decoded_rows,
        "csv_failed_rows_seen": failed_rows,
        "sample_mode": args.sample_mode,
        "retry_rows_written": retry_rows,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    conn.close()


if __name__ == "__main__":
    main()
