#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

try:
    from .prepare import init_db
except ImportError:  # Direct script execution.
    from prepare import init_db


def shard_index(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("shard_"):
        raise ValueError(f"Unexpected shard filename: {path}")
    return int(stem.split("_", 1)[1])


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def reset_tracker(path: Path) -> None:
    for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        if candidate.exists():
            candidate.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize BIG-RUN tracker.sqlite from an existing manifest directory.")
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--insert-batch-size", type=int, default=10000)
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir)
    out_root = Path(args.out_root)
    tracker = out_root / "tracker.sqlite"
    summary_path = manifest_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    shards = sorted(manifest_dir.glob("shard_*.jsonl"), key=shard_index)
    if not shards:
        raise FileNotFoundError(f"No shard_*.jsonl files under {manifest_dir}")

    out_root.mkdir(parents=True, exist_ok=True)
    if tracker.exists() and not args.force:
        raise FileExistsError(f"{tracker} already exists; pass --force to rebuild it.")
    if args.force:
        reset_tracker(tracker)

    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    conn = init_db(tracker)
    now = time.time()
    by_dataset: Counter[str] = Counter()
    by_cc: Counter[str] = Counter()
    shard_rows: list[tuple[object, ...]] = []
    pending_tars: list[tuple[object, ...]] = []
    total_tars = 0

    try:
        for shard_path in shards:
            idx = shard_index(shard_path)
            expected = 0
            for row in read_jsonl(shard_path):
                tar_id = int(row.get("sample_id", total_tars))
                dataset = str(row.get("source", ""))
                weak_cc = str(row.get("weak_country_code", ""))
                tar_path = str(row.get("tar_path", ""))
                if not tar_path:
                    raise ValueError(f"Missing tar_path in {shard_path}")
                pending_tars.append((tar_id, dataset, weak_cc, tar_path, idx, now))
                by_dataset[dataset] += 1
                by_cc[weak_cc] += 1
                expected += 1
                total_tars += 1
                if len(pending_tars) >= args.insert_batch_size:
                    conn.executemany(
                        """
                        INSERT INTO tars(tar_id, dataset, weak_country_code, tar_path, shard_idx, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        pending_tars,
                    )
                    pending_tars.clear()
                    conn.commit()
            if expected:
                shard_rows.append((idx, str(shard_path), expected, now))
        if pending_tars:
            conn.executemany(
                """
                INSERT INTO tars(tar_id, dataset, weak_country_code, tar_path, shard_idx, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                pending_tars,
            )
            conn.commit()
        conn.executemany(
            "INSERT INTO shards(shard_idx, path, expected_tars, updated_at) VALUES (?, ?, ?, ?)",
            shard_rows,
        )

        metadata = dict(source_summary)
        metadata.update(
            {
                "created_unix": now,
                "out_root": str(out_root),
                "tracker_sqlite": str(tracker),
                "manifest_dir": str(manifest_dir),
                "source_summary": str(summary_path),
                "num_shards_nonempty": len(shard_rows),
                "total_tars": total_tars,
                "by_dataset_tars": dict(sorted(by_dataset.items())),
                "by_weak_cc_tars": dict(sorted(by_cc.items())),
            }
        )
        for key, value in metadata.items():
            conn.execute(
                "INSERT OR REPLACE INTO run_metadata(key, value) VALUES (?, ?)",
                (key, json.dumps(value, sort_keys=True)),
            )
        conn.commit()
    finally:
        conn.close()

    out_summary = {
        "manifest_dir": str(manifest_dir),
        "out_root": str(out_root),
        "tracker_sqlite": str(tracker),
        "num_shards": len(shard_rows),
        "total_tars": total_tars,
    }
    print(json.dumps(out_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
