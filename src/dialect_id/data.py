from __future__ import annotations

import json
import math
import os
import random
import tarfile
import csv
import gzip
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .audio import decode_audio_bytes
from .labels import CODE_TO_ID, DIALECT_ID_TO_REGION_ID


DEFAULT_DATASET_CACHE = os.environ.get("DIALECT_ID_DATASET_ROOT", "")


def _open_text_maybe_zst(path: Path):
    if path.suffix == ".zst":
        import zstandard as zstd

        raw = path.open("rb")
        reader = zstd.ZstdDecompressor().stream_reader(raw)
        import io

        return io.TextIOWrapper(reader, encoding="utf-8")
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8")


def _row_country_code(row: dict[str, Any]) -> str:
    return str(
        row.get("country_code")
        or row.get("final_dialect")
        or row.get("consensus_cc")
        or row.get("teacher_top1")
        or row.get("qwen_country_code")
        or row.get("did_top1_cc")
        or row.get("top1_cc")
        or row.get("weak_country_code")
        or row.get("weak_cc")
        or row.get("cc")
        or ""
    ).upper()


def _manifest_is_csv(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".csv") or name.endswith(".csv.gz")


def _iter_manifest_rows(path: Path) -> Iterator[dict[str, Any]]:
    with _open_text_maybe_zst(path) as handle:
        if _manifest_is_csv(path):
            yield from csv.DictReader(handle)
            return
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def resolve_dataset_data_dir(dataset_root: str | Path) -> Path:
    root = Path(dataset_root)
    if (root / "data").is_dir():
        return root / "data"
    refs_main = root / "refs" / "main"
    if refs_main.is_file():
        snapshot = refs_main.read_text(encoding="utf-8").strip()
        data_dir = root / "snapshots" / snapshot / "data"
        if data_dir.is_dir():
            return data_dir
    snapshots = root / "snapshots"
    if snapshots.is_dir():
        candidates = sorted(path / "data" for path in snapshots.iterdir() if (path / "data").is_dir())
        if candidates:
            return candidates[-1]
    raise FileNotFoundError(f"Could not find HF parquet data directory under {root}")


def split_files(dataset_root: str | Path, split: str) -> list[Path]:
    data_dir = resolve_dataset_data_dir(dataset_root)
    files = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not files:
        compact = data_dir / f"{split}.parquet"
        if compact.is_file():
            files = [compact]
    if not files:
        raise FileNotFoundError(f"No parquet files found for split={split!r} in {data_dir}")
    return files


def distributed_context() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world


def scan_split_counts(dataset_root: str | Path, split: str) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    ids: Counter[int] = Counter()
    sources: Counter[str] = Counter()
    total = 0
    for path in split_files(dataset_root, split):
        table = pq.read_table(path, columns=["country_code", "country_id", "source"])
        payload = table.to_pydict()
        counts.update(payload["country_code"])
        ids.update(payload["country_id"])
        sources.update(payload["source"])
        total += len(payload["country_code"])
    return {
        "split": split,
        "total": total,
        "country_code": dict(sorted(counts.items())),
        "country_id": {str(k): v for k, v in sorted(ids.items())},
        "source": dict(sorted(sources.items())),
    }


def write_dataset_profile(dataset_root: str | Path, out_json: str | Path) -> dict[str, Any]:
    profile = {
        "dataset_root": str(dataset_root),
        "data_dir": str(resolve_dataset_data_dir(dataset_root)),
        "splits": {
            split: scan_split_counts(dataset_root, split)
            for split in ("train", "validation", "test")
        },
    }
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return profile


class ParquetDialectDataset(IterableDataset):
    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        *,
        seed: int = 0,
        shuffle: bool = False,
        exclude_unlabeled: bool = True,
        max_examples: int = 0,
        sample_rate: int = 16000,
        balanced_replay: bool = False,
        replay_buffer_size: int = 4,
        replay_prefill: int = 1,
        balanced_row_groups: bool = False,
        stratified_max_examples: bool = False,
        include_country_codes: set[str] | None = None,
        exclude_country_codes: set[str] | None = None,
        data_shard_index: int = 0,
        data_num_shards: int = 1,
    ) -> None:
        self.dataset_root = str(dataset_root)
        self.split = split
        self.files = split_files(dataset_root, split)
        self.seed = seed
        self.shuffle = shuffle
        self.exclude_unlabeled = exclude_unlabeled
        self.max_examples = max_examples
        self.sample_rate = sample_rate
        self.balanced_replay = balanced_replay
        self.replay_buffer_size = replay_buffer_size
        self.replay_prefill = replay_prefill
        self.balanced_row_groups = balanced_row_groups
        self.stratified_max_examples = stratified_max_examples
        self.include_country_codes = {code.upper() for code in include_country_codes} if include_country_codes else None
        self.exclude_country_codes = {code.upper() for code in exclude_country_codes} if exclude_country_codes else set()
        self.data_num_shards = max(1, int(data_num_shards))
        self.data_shard_index = int(data_shard_index)
        if self.data_shard_index < 0 or self.data_shard_index >= self.data_num_shards:
            raise ValueError(
                f"data_shard_index must be in [0, {self.data_num_shards}); got {self.data_shard_index}"
            )
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _worker_files(self) -> tuple[list[Path], random.Random]:
        rank, world_size = distributed_context()
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        global_worker_id = rank * num_workers + worker_id
        total_workers = world_size * num_workers
        files = list(self.files)
        rng = random.Random(self.seed + self.epoch * 7919 + global_worker_id)
        if self.shuffle:
            rng.shuffle(files)
        return files[global_worker_id::total_workers], rng

    def _iter_samples(self) -> Iterator[dict[str, Any]]:
        files, rng = self._worker_files()
        columns = [
            "sample_id",
            "audio",
            "country_code",
            "country_id",
            "source",
            "original_dialect",
            ]
        if self.shuffle:
            rank, world_size = distributed_context()
            worker = get_worker_info()
            worker_id = worker.id if worker is not None else 0
            num_workers = worker.num_workers if worker is not None else 1
            global_worker_id = rank * num_workers + worker_id
            total_workers = world_size * num_workers
            task_rng = random.Random(self.seed + self.epoch * 7919)
            rng = random.Random(self.seed + self.epoch * 7919 + global_worker_id)
            files = list(self.files)
            task_rng.shuffle(files)
            if self.balanced_row_groups:
                row_group_tasks = self._balanced_row_group_tasks(files, task_rng)
            else:
                row_group_tasks = []
                for path in files:
                    parquet = pq.ParquetFile(path)
                    row_group_tasks.extend((path, row_group) for row_group in range(parquet.num_row_groups))
                task_rng.shuffle(row_group_tasks)
            row_group_tasks = row_group_tasks[self.data_shard_index :: self.data_num_shards]
            row_group_tasks = row_group_tasks[global_worker_id::total_workers]
        else:
            rank, world_size = distributed_context()
            worker = get_worker_info()
            worker_id = worker.id if worker is not None else 0
            num_workers = worker.num_workers if worker is not None else 1
            global_worker_id = rank * num_workers + worker_id
            total_workers = world_size * num_workers
            row_group_tasks = []
            for path in self.files:
                parquet = pq.ParquetFile(path)
                row_group_tasks.extend((path, row_group) for row_group in range(parquet.num_row_groups))
            row_group_tasks = row_group_tasks[self.data_shard_index :: self.data_num_shards]
            row_group_tasks = row_group_tasks[global_worker_id::total_workers]

        parquet_cache: dict[Path, pq.ParquetFile] = {}
        for path, row_group in row_group_tasks:
            parquet = parquet_cache.get(path)
            if parquet is None:
                parquet = pq.ParquetFile(path)
                parquet_cache[path] = parquet
            table = parquet.read_row_group(row_group, columns=columns)
            payload = table.to_pydict()
            order = list(range(len(payload["sample_id"])))
            if self.shuffle:
                rng.shuffle(order)
            for idx in order:
                country_code = str(payload["country_code"][idx]).upper()
                if self.exclude_unlabeled and country_code not in CODE_TO_ID:
                    continue
                if self.include_country_codes is not None and country_code not in self.include_country_codes:
                    continue
                if country_code in self.exclude_country_codes:
                    continue
                country_id = CODE_TO_ID[country_code]
                audio_obj = payload["audio"][idx]
                wav_bytes = audio_obj["bytes"] if isinstance(audio_obj, dict) else audio_obj
                try:
                    waveform = decode_audio_bytes(wav_bytes, target_sample_rate=self.sample_rate)
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to decode sample_id={payload['sample_id'][idx]} "
                        f"from {path.name} row_group={row_group}"
                    ) from exc
                yield {
                    "sample_id": int(payload["sample_id"][idx]),
                    "waveform": waveform,
                    "duration_seconds": float(waveform.shape[0]) / float(self.sample_rate),
                    "country_code": country_code,
                    "country_id": country_id,
                    "region_id": DIALECT_ID_TO_REGION_ID[country_id],
                    "source": str(payload["source"][idx]),
                    "original_dialect": str(payload["original_dialect"][idx]),
                    "path": str(path),
                    "row_group": row_group,
                }

    def _balanced_row_group_tasks(
        self,
        files: list[Path],
        rng: random.Random,
    ) -> list[tuple[Path, int]]:
        grouped: dict[int, list[tuple[Path, int]]] = {label_id: [] for label_id in CODE_TO_ID.values()}
        fallback: list[tuple[Path, int]] = []
        for path in files:
            parquet = pq.ParquetFile(path)
            for row_group in range(parquet.num_row_groups):
                table = parquet.read_row_group(row_group, columns=["country_code"])
                labels = [
                    CODE_TO_ID[code]
                    for raw_code in table.column("country_code").to_pylist()
                    for code in [str(raw_code).upper()]
                    if code in CODE_TO_ID
                    and (self.include_country_codes is None or code in self.include_country_codes)
                    and code not in self.exclude_country_codes
                    and CODE_TO_ID[code] in grouped
                ]
                if not labels:
                    fallback.append((path, row_group))
                    continue
                label_id = Counter(labels).most_common(1)[0][0]
                grouped[label_id].append((path, row_group))

        for tasks in grouped.values():
            rng.shuffle(tasks)
        rng.shuffle(fallback)

        ready = [label_id for label_id, tasks in grouped.items() if tasks]
        ordered: list[tuple[Path, int]] = []
        while ready:
            label_id = rng.choice(ready)
            ordered.append(grouped[label_id].pop())
            if not grouped[label_id]:
                ready.remove(label_id)
        ordered.extend(fallback)
        return ordered

    def _iter_balanced_replay(self) -> Iterator[dict[str, Any]]:
        _, rng = self._worker_files()
        buffers: dict[int, list[dict[str, Any]]] = {label_id: [] for label_id in CODE_TO_ID.values()}
        target_label_ids = sorted(buffers)
        consumed = 0
        for item in self._iter_samples():
            consumed += 1
            label_id = int(item["country_id"])
            buffer = buffers[label_id]
            if len(buffer) < self.replay_buffer_size:
                buffer.append(item)
            elif self.shuffle:
                buffer[rng.randrange(self.replay_buffer_size)] = item
            else:
                buffer[consumed % self.replay_buffer_size] = item

            ready_labels = [idx for idx in target_label_ids if buffers[idx]]
            if len(ready_labels) < len(target_label_ids) and consumed < self.replay_prefill:
                continue
            sampled_label = rng.choice(ready_labels)
            yield rng.choice(buffers[sampled_label])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        source = self._iter_balanced_replay() if self.balanced_replay else self._iter_samples()
        emitted = 0
        if self.max_examples > 0 and self.stratified_max_examples:
            per_class_limit = max(1, math.ceil(self.max_examples / max(1, len(CODE_TO_ID))))
            emitted_by_class = {label_id: 0 for label_id in CODE_TO_ID.values()}
            for item in source:
                label_id = int(item["country_id"])
                if emitted_by_class.get(label_id, per_class_limit) >= per_class_limit:
                    continue
                yield item
                emitted += 1
                emitted_by_class[label_id] = emitted_by_class.get(label_id, 0) + 1
                if emitted >= self.max_examples:
                    return
            return
        for item in source:
            yield item
            emitted += 1
            if self.max_examples > 0 and emitted >= self.max_examples:
                return


class PseudoLabeledTarDataset(IterableDataset):
    def __init__(
        self,
        manifests: list[str | Path],
        *,
        seed: int = 0,
        shuffle: bool = True,
        max_examples: int = 0,
        sample_rate: int = 16000,
        default_loss_weight: float = 0.35,
        include_country_codes: set[str] | None = None,
        exclude_country_codes: set[str] | None = None,
        streaming: bool = False,
        balanced_replay: bool = False,
        replay_buffer_size: int = 8,
        replay_prefill: int = 1024,
    ) -> None:
        self.manifests = [Path(path) for path in manifests]
        self.seed = seed
        self.shuffle = shuffle
        self.max_examples = max_examples
        self.sample_rate = sample_rate
        self.default_loss_weight = default_loss_weight
        self.include_country_codes = {code.upper() for code in include_country_codes} if include_country_codes else None
        self.exclude_country_codes = {code.upper() for code in exclude_country_codes} if exclude_country_codes else set()
        self.streaming = streaming
        self.balanced_replay = balanced_replay
        self.replay_buffer_size = max(1, int(replay_buffer_size))
        self.replay_prefill = max(1, int(replay_prefill))
        self.epoch = 0
        self._rows: list[dict[str, Any]] | None = None
        self._manifest_label_index_cache: dict[Path, dict[str, list[str]]] = {}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _load_rows(self) -> list[dict[str, Any]]:
        if self._rows is not None:
            return self._rows
        rows: list[dict[str, Any]] = []
        for manifest in self.manifests:
            for row in _iter_manifest_rows(manifest):
                code = _row_country_code(row)
                if code not in CODE_TO_ID:
                    continue
                if self.include_country_codes is not None and code not in self.include_country_codes:
                    continue
                if code in self.exclude_country_codes:
                    continue
                row["country_code"] = code
                # Some manifests carry both a selected country_code/final_dialect
                # and an older country_id from a different weak source. The selected
                # code is the training target, so keep the numeric id consistent.
                row["country_id"] = CODE_TO_ID[code]
                rows.append(row)
        self._rows = rows
        return rows

    def _worker_rows(self) -> tuple[list[dict[str, Any]], random.Random]:
        rows = list(self._load_rows())
        rank, world_size = distributed_context()
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        global_worker_id = rank * num_workers + worker_id
        total_workers = world_size * num_workers
        rng = random.Random(self.seed + self.epoch * 7919 + global_worker_id)
        if self.shuffle:
            rng.shuffle(rows)
        return rows[global_worker_id::total_workers], rng

    def _worker_manifests(self) -> tuple[list[Path], random.Random]:
        rank, world_size = distributed_context()
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        global_worker_id = rank * num_workers + worker_id
        total_workers = world_size * num_workers
        task_rng = random.Random(self.seed + self.epoch * 7919)
        rng = random.Random(self.seed + self.epoch * 7919 + global_worker_id)
        manifests = list(self.manifests)
        if self.shuffle:
            task_rng.shuffle(manifests)
        if not manifests:
            return [], rng
        if total_workers <= len(manifests):
            return manifests[global_worker_id::total_workers], rng
        # With many distributed workers and fewer shard manifests, cycling keeps
        # every worker supplied while preserving deterministic epoch-level order.
        return [manifests[global_worker_id % len(manifests)]], rng

    def _load_manifest_label_index(self, root: Path) -> dict[str, list[str]]:
        root = root.resolve()
        cached = self._manifest_label_index_cache.get(root)
        if cached is not None:
            return cached
        index_path = root / "manifest_label_index.json"
        if not index_path.is_file():
            self._manifest_label_index_cache[root] = {}
            return {}
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        files = payload.get("files", payload)
        index = {
            str(key): [str(code).upper() for code in value]
            for key, value in files.items()
            if isinstance(value, list)
        }
        self._manifest_label_index_cache[root] = index
        return index

    def _manifest_label_ids(self, manifest: Path) -> list[int]:
        root = manifest.parent.parent
        index = self._load_manifest_label_index(root)
        if index:
            keys = [str(manifest), str(manifest.resolve()), manifest.name]
            for base in (Path.cwd(), Path(__file__).resolve().parents[1]):
                try:
                    keys.append(str(manifest.resolve().relative_to(base.resolve())))
                except ValueError:
                    pass
            codes = next((index[key] for key in keys if key in index), [])
            label_ids = []
            for code in codes:
                if code not in CODE_TO_ID:
                    continue
                if self.include_country_codes is not None and code not in self.include_country_codes:
                    continue
                if code in self.exclude_country_codes:
                    continue
                label_ids.append(CODE_TO_ID[code])
            return sorted(set(label_ids))

        for row in _iter_manifest_rows(manifest):
            code = _row_country_code(row)
            if code not in CODE_TO_ID:
                continue
            if self.include_country_codes is not None and code not in self.include_country_codes:
                continue
            if code in self.exclude_country_codes:
                continue
            return [CODE_TO_ID[code]]
        return []

    def _target_label_ids(self) -> list[int]:
        return [
            label_id
            for code, label_id in CODE_TO_ID.items()
            if (self.include_country_codes is None or code in self.include_country_codes)
            and code not in self.exclude_country_codes
        ]

    def _worker_balanced_manifests_by_label(self) -> tuple[dict[int, list[Path]], random.Random]:
        rank, world_size = distributed_context()
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        global_worker_id = rank * num_workers + worker_id
        total_workers = world_size * num_workers
        task_rng = random.Random(self.seed + self.epoch * 7919)
        rng = random.Random(self.seed + self.epoch * 7919 + global_worker_id)

        target_label_ids = set(self._target_label_ids())
        grouped: dict[int, list[Path]] = {label_id: [] for label_id in target_label_ids}
        manifests = list(self.manifests)
        if self.shuffle:
            task_rng.shuffle(manifests)
        for manifest in manifests:
            for label_id in self._manifest_label_ids(manifest):
                if label_id not in grouped:
                    continue
                grouped[label_id].append(manifest)

        assigned: dict[int, list[Path]] = {}
        for label_id, paths in grouped.items():
            if not paths:
                continue
            label_rng = random.Random(self.seed + self.epoch * 7919 + label_id * 104729)
            paths = list(paths)
            if self.shuffle:
                label_rng.shuffle(paths)
            selected = paths[global_worker_id::total_workers]
            if not selected:
                selected = [paths[global_worker_id % len(paths)]]
            assigned[label_id] = selected
        return assigned, rng

    def _iter_label_streams(
        self,
        manifests_by_label: dict[int, list[Path]],
        rng: random.Random,
    ) -> Iterator[dict[str, Any]]:
        label_iters: dict[int, Iterator[dict[str, Any]]] = {}
        for label_id, manifests in manifests_by_label.items():
            label_rng = random.Random(self.seed + self.epoch * 7919 + label_id * 65537)

            def iter_label(label: int, paths: list[Path], row_rng: random.Random) -> Iterator[dict[str, Any]]:
                for manifest in paths:
                    rows = self._load_manifest_rows(manifest, row_rng, only_label_id=label)
                    yield from self._iter_rows(rows, limit_emitted=False)

            label_iters[label_id] = iter(iter_label(label_id, manifests, label_rng))

        active = sorted(label_iters)
        while active:
            if self.shuffle:
                rng.shuffle(active)
            still_active: list[int] = []
            for label_id in active:
                try:
                    yield next(label_iters[label_id])
                    still_active.append(label_id)
                except StopIteration:
                    continue
            active = still_active

    def _load_manifest_rows(
        self,
        manifest: Path,
        rng: random.Random,
        *,
        only_label_id: int | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in _iter_manifest_rows(manifest):
            code = _row_country_code(row)
            if code not in CODE_TO_ID:
                continue
            if self.include_country_codes is not None and code not in self.include_country_codes:
                continue
            if code in self.exclude_country_codes:
                continue
            label_id = CODE_TO_ID[code]
            if only_label_id is not None and label_id != only_label_id:
                continue
            row["country_code"] = code
            row["country_id"] = label_id
            rows.append(row)
        if self.shuffle:
            rng.shuffle(rows)
        return rows

    def _iter_rows(self, rows: list[dict[str, Any]], *, limit_emitted: bool = True) -> Iterator[dict[str, Any]]:
        by_tar: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_tar.setdefault(str(row["tar_path"]), []).append(row)

        emitted = 0
        for tar_path, tar_rows in by_tar.items():
            wanted: dict[str, list[dict[str, Any]]] = {}
            for row in tar_rows:
                wanted.setdefault(str(row["member"]), []).append(row)
            try:
                with tarfile.open(tar_path, "r:*") as tar:
                    for member_info in tar:
                        member_rows = wanted.get(member_info.name)
                        if not member_rows:
                            continue
                        extracted = tar.extractfile(member_info)
                        if extracted is None:
                            continue
                        wav_bytes = extracted.read()
                        try:
                            waveform = decode_audio_bytes(wav_bytes, target_sample_rate=self.sample_rate)
                        except Exception:
                            continue
                        for row in member_rows:
                            country_id = int(row["country_id"])
                            soft_label = row.get("soft_label") or row.get("teacher_probs")
                            raw_lang_type_id = row.get("lang_type_id")
                            if raw_lang_type_id in (None, ""):
                                lang_type_id = -100
                            else:
                                lang_type_id = int(raw_lang_type_id)
                            lang_type = str(
                                row.get("lang_type", "unknown" if lang_type_id == -100 else "dialect")
                            )
                            accent_country_code = str(row.get("accent_country_code", ""))
                            accent_country_id = int(row.get("accent_country_id", -100))
                            yield {
                                "sample_id": int(row.get("sample_id", emitted)),
                                "waveform": waveform,
                                "duration_seconds": float(waveform.shape[0]) / float(self.sample_rate),
                                "country_code": str(row["country_code"]),
                                "country_id": country_id,
                                "region_id": DIALECT_ID_TO_REGION_ID[country_id],
                                "lang_type": lang_type,
                                "lang_type_id": lang_type_id,
                                "accent_country_code": accent_country_code,
                                "accent_country_id": accent_country_id,
                                "source": str(row.get("source", "ytb_pseudo")),
                                "original_dialect": str(row.get("original_dialect", row.get("member", ""))),
                                "path": tar_path,
                                "member": member_info.name,
                                "weak_country_code": str(row.get("weak_country_code", "")),
                                "soft_label": soft_label,
                                "loss_weight": float(row.get("loss_weight", self.default_loss_weight)),
                                "pseudo_confidence": float(row.get("teacher_confidence", row.get("confidence", 0.0))),
                            }
                            emitted += 1
                            if limit_emitted and self.max_examples > 0 and emitted >= self.max_examples:
                                return
            except (tarfile.TarError, OSError):
                continue

    def _iter_balanced_replay(
        self,
        source: Iterator[dict[str, Any]],
        rng: random.Random,
    ) -> Iterator[dict[str, Any]]:
        target_label_ids = self._target_label_ids()
        buffers: dict[int, list[dict[str, Any]]] = {label_id: [] for label_id in target_label_ids}
        consumed = 0
        emitted = 0
        for item in source:
            consumed += 1
            label_id = int(item["country_id"])
            if label_id not in buffers:
                continue
            buffer = buffers[label_id]
            if len(buffer) < self.replay_buffer_size:
                buffer.append(item)
            elif self.shuffle:
                buffer[rng.randrange(self.replay_buffer_size)] = item
            else:
                buffer[consumed % self.replay_buffer_size] = item

            ready_labels = [idx for idx in target_label_ids if buffers[idx]]
            if not ready_labels:
                continue
            if len(ready_labels) < len(target_label_ids) and consumed < self.replay_prefill:
                continue
            sampled_label = rng.choice(ready_labels)
            yield rng.choice(buffers[sampled_label])
            emitted += 1
            if self.max_examples > 0 and emitted >= self.max_examples:
                return

    def __iter__(self) -> Iterator[dict[str, Any]]:
        rank, world_size = distributed_context()
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        global_worker_id = rank * num_workers + worker_id
        replay_rng = random.Random(self.seed + self.epoch * 7919 + global_worker_id + 104729)
        if not self.streaming:
            rows, _rng = self._worker_rows()
            items = self._iter_rows(rows, limit_emitted=not self.balanced_replay)
            if self.balanced_replay:
                yield from self._iter_balanced_replay(items, replay_rng)
            else:
                yield from items
            return

        if self.balanced_replay:
            manifests_by_label, rng = self._worker_balanced_manifests_by_label()
            stream_items = self._iter_label_streams(manifests_by_label, rng)
            yield from self._iter_balanced_replay(stream_items, replay_rng)
            return

        manifests, rng = self._worker_manifests()
        emitted = 0
        for manifest in manifests:
            rows = self._load_manifest_rows(manifest, rng)
            for item in self._iter_rows(rows):
                yield item
                emitted += 1
                if self.max_examples > 0 and emitted >= self.max_examples:
                    return


class MixedDialectDataset(IterableDataset):
    def __init__(
        self,
        clean_dataset: ParquetDialectDataset,
        pseudo_dataset: PseudoLabeledTarDataset,
        *,
        pseudo_probability: float = 0.5,
        seed: int = 0,
        max_examples: int = 0,
    ) -> None:
        self.clean_dataset = clean_dataset
        self.pseudo_dataset = pseudo_dataset
        self.pseudo_probability = max(0.0, min(1.0, pseudo_probability))
        self.seed = seed
        self.max_examples = max_examples
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.clean_dataset.set_epoch(epoch)
        self.pseudo_dataset.set_epoch(epoch)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        rank, world_size = distributed_context()
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        global_worker_id = rank * num_workers + worker_id
        rng = random.Random(self.seed + self.epoch * 7919 + global_worker_id + world_size)
        clean_iter = iter(self.clean_dataset)
        pseudo_iter = iter(self.pseudo_dataset)
        clean_done = False
        pseudo_done = False
        emitted = 0
        while not (clean_done and pseudo_done):
            use_pseudo = (not pseudo_done) and (clean_done or rng.random() < self.pseudo_probability)
            iterator = pseudo_iter if use_pseudo else clean_iter
            try:
                item = next(iterator)
            except StopIteration:
                if use_pseudo:
                    pseudo_done = True
                else:
                    clean_done = True
                continue
            yield item
            emitted += 1
            if self.max_examples > 0 and emitted >= self.max_examples:
                return


def labeled_train_counts(
    dataset_root: str | Path,
    *,
    include_country_codes: set[str] | None = None,
    exclude_country_codes: set[str] | None = None,
) -> list[int]:
    counts = [0 for _ in range(len(CODE_TO_ID))]
    include = {code.upper() for code in include_country_codes} if include_country_codes else None
    exclude = {code.upper() for code in exclude_country_codes} if exclude_country_codes else set()
    stats = scan_split_counts(dataset_root, "train")
    for code, count in stats["country_code"].items():
        if code in CODE_TO_ID:
            if include is not None and code not in include:
                continue
            if code in exclude:
                continue
            counts[CODE_TO_ID[code]] = int(count)
    return counts
