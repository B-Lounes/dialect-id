#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import TextIO


def stable_shard(value: str, num_shards: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % num_shards


def infer_cc(path: Path) -> str:
    for part in reversed(path.parts):
        if part.startswith("cc="):
            return part.split("=", 1)[1].upper()
    return ""


def shard_path(manifest_dir: Path, shard_idx: int) -> Path:
    return manifest_dir / f"shard_{shard_idx:05d}.jsonl"


class LruWriters:
    def __init__(self, manifest_dir: Path, max_open: int) -> None:
        self.manifest_dir = manifest_dir
        self.max_open = max(1, int(max_open))
        self.handles: OrderedDict[int, TextIO] = OrderedDict()

    def get(self, shard_idx: int) -> TextIO:
        handle = self.handles.pop(shard_idx, None)
        if handle is None:
            handle = shard_path(self.manifest_dir, shard_idx).open("a", encoding="utf-8")
        self.handles[shard_idx] = handle
        while len(self.handles) > self.max_open:
            _idx, old_handle = self.handles.popitem(last=False)
            old_handle.close()
        return handle

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


def init_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS run_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS shards (
            shard_idx INTEGER PRIMARY KEY,
            path TEXT NOT NULL,
            expected_tars INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            expanded_rows INTEGER NOT NULL DEFAULT 0,
            decoded_rows INTEGER NOT NULL DEFAULT 0,
            failed_rows INTEGER NOT NULL DEFAULT 0,
            duration_sec REAL NOT NULL DEFAULT 0.0,
            output_csv TEXT,
            worker_id TEXT,
            hostname TEXT,
            error TEXT,
            updated_at REAL
        );
        CREATE TABLE IF NOT EXISTS tars (
            tar_id INTEGER PRIMARY KEY,
            dataset TEXT NOT NULL,
            weak_country_code TEXT,
            tar_path TEXT NOT NULL UNIQUE,
            shard_idx INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            sample_rows INTEGER NOT NULL DEFAULT 0,
            decoded_rows INTEGER NOT NULL DEFAULT 0,
            failed_rows INTEGER NOT NULL DEFAULT 0,
            duration_sec REAL NOT NULL DEFAULT 0.0,
            output_csv TEXT,
            error TEXT,
            updated_at REAL
        );
        CREATE TABLE IF NOT EXISTS samples (
            sample_id INTEGER,
            tar_path TEXT NOT NULL,
            member TEXT NOT NULL,
            manifest_row INTEGER,
            weak_cc TEXT,
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
            top1_region TEXT,
            output_csv TEXT,
            updated_at REAL,
            PRIMARY KEY (tar_path, member)
        );
        CREATE INDEX IF NOT EXISTS idx_tars_shard ON tars(shard_idx);
        CREATE INDEX IF NOT EXISTS idx_tars_status ON tars(status);
        CREATE INDEX IF NOT EXISTS idx_samples_decoded ON samples(decoded);
        """
    )
    return conn


def find_tars(root: Path, max_tars: int) -> list[Path]:
    tars = sorted(root.rglob("*.tar"))
    if max_tars > 0:
        return tars[:max_tars]
    return tars


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BIG-RUN-did tar manifests and SQLite tracker.")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--vad-root", action="append", required=True)
    parser.add_argument("--num-shards", type=int, default=65536)
    parser.add_argument("--max-open-files", type=int, default=128)
    parser.add_argument("--max-tars-per-root", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")

    out_root = Path(args.out_root)
    manifest_dir = out_root / "manifests_tarhash"
    metadata_dir = out_root / "metadata"
    for directory in (manifest_dir, metadata_dir, out_root / "logs", out_root / "predictions", out_root / "state"):
        directory.mkdir(parents=True, exist_ok=True)

    existing_shards = list(manifest_dir.glob("shard_*.jsonl"))
    db_path = out_root / "tracker.sqlite"
    if (existing_shards or db_path.exists()) and not args.force:
        raise FileExistsError(f"{out_root} already has manifests or tracker.sqlite; pass --force to rebuild.")
    if args.force:
        for path in existing_shards:
            path.unlink()
        for path in (manifest_dir / "summary.json", db_path, Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm")):
            if path.exists():
                path.unlink()

    roots = [Path(path) for path in args.vad_root]
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)

    conn = init_db(db_path)
    writers = LruWriters(manifest_dir, args.max_open_files)
    by_dataset: Counter[str] = Counter()
    by_cc: Counter[str] = Counter()
    by_shard: Counter[int] = Counter()
    root_summaries: list[dict[str, object]] = []
    now = time.time()
    tar_id = 0
    pending_rows: list[tuple[object, ...]] = []

    try:
        for root in roots:
            dataset = root.name
            tar_paths = find_tars(root, args.max_tars_per_root)
            root_summaries.append({"dataset": dataset, "root": str(root), "tars": len(tar_paths)})
            for tar_path in tar_paths:
                cc = infer_cc(tar_path)
                shard_idx = stable_shard(str(tar_path), args.num_shards)
                row = {
                    "sample_id": tar_id,
                    "manifest_row": tar_id,
                    "tar_path": str(tar_path),
                    "member": "__ALL_MEMBERS__",
                    "weak_country_code": cc,
                    "youtube_id": "",
                    "source": dataset,
                    "dataset_root": str(root),
                }
                writers.get(shard_idx).write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                pending_rows.append((tar_id, dataset, cc, str(tar_path), shard_idx, now))
                by_dataset[dataset] += 1
                by_cc[cc] += 1
                by_shard[shard_idx] += 1
                tar_id += 1
                if len(pending_rows) >= 10000:
                    conn.executemany(
                        """
                        INSERT INTO tars(tar_id, dataset, weak_country_code, tar_path, shard_idx, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        pending_rows,
                    )
                    pending_rows.clear()
                    conn.commit()
        if pending_rows:
            conn.executemany(
                """
                INSERT INTO tars(tar_id, dataset, weak_country_code, tar_path, shard_idx, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                pending_rows,
            )
            conn.commit()
    finally:
        writers.close()

    shard_rows = [
        (idx, str(shard_path(manifest_dir, idx)), count, now)
        for idx, count in sorted(by_shard.items())
    ]
    conn.executemany(
        "INSERT INTO shards(shard_idx, path, expected_tars, updated_at) VALUES (?, ?, ?, ?)",
        shard_rows,
    )
    metadata = {
        "created_unix": now,
        "out_root": str(out_root),
        "tracker_sqlite": str(db_path),
        "manifest_dir": str(manifest_dir),
        "vad_roots": [str(root) for root in roots],
        "root_summaries": root_summaries,
        "num_shards_requested": args.num_shards,
        "num_shards_nonempty": len(by_shard),
        "total_tars": tar_id,
        "row_granularity": "one_manifest_row_per_tar_expanded_at_prediction_time",
        "member_sentinel": "__ALL_MEMBERS__",
        "by_dataset_tars": dict(sorted(by_dataset.items())),
        "by_weak_cc_tars": dict(sorted(by_cc.items())),
        "per_shard_tars_min": min(by_shard.values()) if by_shard else 0,
        "per_shard_tars_max": max(by_shard.values()) if by_shard else 0,
        "per_shard_tars_mean_nonempty": (sum(by_shard.values()) / len(by_shard)) if by_shard else 0.0,
    }
    for key, value in metadata.items():
        conn.execute(
            "INSERT OR REPLACE INTO run_metadata(key, value) VALUES (?, ?)",
            (key, json.dumps(value, sort_keys=True)),
        )
    conn.commit()
    conn.close()

    (manifest_dir / "summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (metadata_dir / "prepare_summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
