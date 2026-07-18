from __future__ import annotations

import argparse
from datetime import timedelta
import glob
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from .checkpointing import load_checkpoint_weights
from .collation import AudioCollator
from .data import DEFAULT_DATASET_CACHE, ParquetDialectDataset, PseudoLabeledTarDataset
from .metrics import save_metrics_bundle
from .modeling import build_model
from .train import (
    aggregate_chunk_outputs,
    apply_active_dialect_mask,
    autocast_context,
    build_active_dialect_mask,
    combine_dialect_logits,
    move_batch,
)
from .labels import CODE_TO_ID, num_dialects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an Arabic dialect ID checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_CACHE)
    parser.add_argument("--manifest", action="append", default=None)
    parser.add_argument("--manifest-glob", action="append", default=None)
    parser.add_argument("--manifest-streaming", action="store_true")
    parser.add_argument("--eval-name", default=None)
    parser.add_argument("--include-country-code", action="append", default=None)
    parser.add_argument("--exclude-country-code", action="append", default=None)
    parser.add_argument(
        "--output-country-code",
        action="append",
        default=None,
        help=(
            "Optional dialect mask for predictions, independent from dataset filtering. "
            "Useful for evaluating a regional specialist over the full split."
        ),
    )
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--stratified-max-examples", action="store_true")
    parser.add_argument("--data-shard-index", type=int, default=0)
    parser.add_argument("--data-num-shards", type=int, default=1)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--full-audio-chunking", action="store_true")
    parser.add_argument("--chunk-seconds", type=float, default=30.0)
    parser.add_argument("--max-chunks-per-sample", type=int, default=0)
    parser.add_argument(
        "--num-crops",
        type=int,
        default=1,
        help="Average predictions over this many evenly-spaced crops per utterance.",
    )
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default=None)
    parser.add_argument(
        "--dialect-logit-bias-json",
        default=None,
        help="Optional JSON sidecar with per-dialect additive logit biases.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--log-every-batches", type=int, default=100)
    return parser.parse_args()


def parse_code_filter(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    codes: set[str] = set()
    for value in values:
        for code in value.replace(";", ",").split(","):
            code = code.strip().upper()
            if code:
                codes.add(code)
    return codes or None


def resolve_manifest_paths(args: argparse.Namespace) -> list[str]:
    paths = list(args.manifest or [])
    for pattern in args.manifest_glob or []:
        paths.extend(sorted(glob.glob(pattern)))
    seen: set[str] = set()
    unique_paths: list[str] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique_paths.append(path)
    return unique_paths


def init_distributed() -> tuple[int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, timeout=timedelta(hours=4))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
    return rank, world_size, device


def build_output_dialect_mask(args: argparse.Namespace, device: torch.device) -> torch.Tensor | None:
    if not args.output_country_code:
        return build_active_dialect_mask(args, device)
    output_args = argparse.Namespace(
        include_country_code=args.output_country_code,
        exclude_country_code=args.exclude_country_code,
    )
    return build_active_dialect_mask(output_args, device)


def train_logit_args(train_args: dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        region_logit_fusion_weight=float(train_args.get("region_logit_fusion_weight", 0.0) or 0.0),
        specialist_logit_fusion_weight=float(train_args.get("specialist_logit_fusion_weight", 0.0) or 0.0),
    )


def load_dialect_logit_bias(path: str | None, device: torch.device) -> torch.Tensor | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    bias = torch.zeros(num_dialects(), dtype=torch.float32, device=device)
    if isinstance(payload.get("bias_by_id"), dict):
        for raw_idx, value in payload["bias_by_id"].items():
            idx = int(raw_idx)
            if 0 <= idx < num_dialects():
                bias[idx] = float(value)
    if isinstance(payload.get("bias_by_code"), dict):
        for raw_code, value in payload["bias_by_code"].items():
            code = str(raw_code).upper()
            if code in CODE_TO_ID:
                bias[CODE_TO_ID[code]] = float(value)
    if isinstance(payload.get("bias"), list):
        values = payload["bias"]
        if len(values) != num_dialects():
            raise ValueError(
                f"{path} bias list has {len(values)} entries; expected {num_dialects()}"
            )
        bias = torch.tensor(values, dtype=torch.float32, device=device)
    return bias


def eval_crops(audio: np.ndarray, *, max_samples: int, num_crops: int) -> list[np.ndarray]:
    if num_crops <= 1 or max_samples <= 0 or audio.shape[0] <= max_samples:
        return [audio]
    max_start = audio.shape[0] - max_samples
    if num_crops == 2:
        starts = [0, max_start]
    else:
        starts = np.linspace(0, max_start, num=num_crops)
    crops: list[np.ndarray] = []
    seen: set[int] = set()
    for raw_start in starts:
        start = int(round(float(raw_start)))
        if start in seen:
            continue
        seen.add(start)
        crops.append(audio[start : start + max_samples].astype(np.float32, copy=False))
    return crops or [audio]


class MultiCropEvalCollator:
    def __init__(self, base_collator: AudioCollator, *, num_crops: int) -> None:
        self.base_collator = base_collator
        self.num_crops = max(1, int(num_crops))

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        if self.num_crops <= 1:
            output = self.base_collator(batch)
            output["crop_counts"] = torch.ones(len(batch), dtype=torch.long)
            return output

        expanded: list[dict[str, Any]] = []
        crop_counts: list[int] = []
        max_samples = int(self.base_collator.max_samples)
        for item in batch:
            crops = eval_crops(
                item["waveform"],
                max_samples=max_samples,
                num_crops=self.num_crops,
            )
            crop_counts.append(len(crops))
            for crop in crops:
                copied = dict(item)
                copied["waveform"] = crop
                expanded.append(copied)
        output = self.base_collator(expanded)
        output["crop_counts"] = torch.tensor(crop_counts, dtype=torch.long)
        return output


def average_crop_probs(probs: torch.Tensor, crop_counts: torch.Tensor) -> torch.Tensor:
    if crop_counts.numel() == probs.shape[0] and bool((crop_counts == 1).all()):
        return probs
    rows = []
    offset = 0
    for count in crop_counts.tolist():
        count = int(count)
        rows.append(probs[offset : offset + count].mean(dim=0))
        offset += count
    return torch.stack(rows, dim=0)


def first_crop_items(values: torch.Tensor, crop_counts: torch.Tensor) -> torch.Tensor:
    if crop_counts.numel() == values.shape[0] and bool((crop_counts == 1).all()):
        return values
    indices = []
    offset = 0
    for count in crop_counts.tolist():
        indices.append(offset)
        offset += int(count)
    return values[torch.tensor(indices, dtype=torch.long, device=values.device)]


def first_crop_meta(values: list[Any], crop_counts: torch.Tensor) -> list[Any]:
    if crop_counts.numel() == len(values) and bool((crop_counts == 1).all()):
        return values
    selected = []
    offset = 0
    for count in crop_counts.tolist():
        selected.append(values[offset])
        offset += int(count)
    return selected


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.data_num_shards < 1:
        raise ValueError("--data-num-shards must be >= 1")
    if args.data_shard_index < 0 or args.data_shard_index >= args.data_num_shards:
        raise ValueError("--data-shard-index must be in [0, --data-num-shards)")
    rank, world_size, device = init_distributed()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    train_args = checkpoint["args"]
    precision = args.precision or train_args.get("precision", "bf16")

    model = build_model(
        train_args["model_type"],
        train_args.get("model_name_or_path"),
        cache_dir=train_args.get("cache_dir"),
        llm_name_or_path=train_args.get("llm_name_or_path"),
        llm_cache_dir=train_args.get("llm_cache_dir"),
        local_files_only=args.local_files_only or train_args.get("local_files_only", False),
        freeze_encoder_layers=train_args.get("freeze_encoder_layers", 0),
        freeze_feature_encoder=not train_args.get("no_freeze_feature_encoder", False),
        freeze_llm=not train_args.get("no_freeze_llm", False),
        llm_trainable_layers=train_args.get("llm_trainable_layers", 0),
        gradient_checkpointing=False,
        audio_llm_tokens=train_args.get("audio_llm_tokens", 32),
        classifier_hidden_dim=train_args.get("classifier_hidden_dim", 512),
        classifier_head_type=train_args.get("classifier_head_type", "legacy"),
        dropout=train_args.get("dropout", 0.1),
    )
    if precision == "fp32":
        model = model.float()
    model = model.to(device)
    load_checkpoint_weights(model, checkpoint, map_location="cpu")
    model.eval()
    active_dialect_mask = build_output_dialect_mask(args, device)
    dialect_logit_bias = load_dialect_logit_bias(args.dialect_logit_bias_json, device)
    logit_args = train_logit_args(train_args)
    eval_name = args.eval_name or args.split

    include_country_codes = parse_code_filter(args.include_country_code)
    exclude_country_codes = parse_code_filter(args.exclude_country_code)
    manifest_paths = resolve_manifest_paths(args)
    if manifest_paths:
        dataset = PseudoLabeledTarDataset(
            manifest_paths,
            shuffle=False,
            max_examples=args.max_examples,
            include_country_codes=include_country_codes,
            exclude_country_codes=exclude_country_codes,
            streaming=args.manifest_streaming,
        )
    else:
        dataset = ParquetDialectDataset(
            args.dataset_root,
            args.split,
            shuffle=False,
            exclude_unlabeled=True,
            max_examples=args.max_examples,
            stratified_max_examples=args.stratified_max_examples,
            include_country_codes=include_country_codes,
            exclude_country_codes=exclude_country_codes,
            data_shard_index=args.data_shard_index,
            data_num_shards=args.data_num_shards,
        )
    collator = AudioCollator(
        train_args["model_type"],
        train_args.get("model_name_or_path"),
        cache_dir=train_args.get("cache_dir"),
        train=False,
        max_seconds=args.max_seconds,
        local_files_only=args.local_files_only or train_args.get("local_files_only", False),
        full_audio_chunking=args.full_audio_chunking or bool(train_args.get("full_audio_chunking", False)),
        chunk_seconds=args.chunk_seconds or float(train_args.get("chunk_seconds", 30.0) or 30.0),
        max_chunks_per_sample=args.max_chunks_per_sample or int(train_args.get("max_chunks_per_sample", 0) or 0),
    )
    eval_collator = MultiCropEvalCollator(collator, num_crops=args.num_crops)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=eval_collator,
        pin_memory=False,
    )
    matrix = torch.zeros((num_dialects(), num_dialects()), dtype=torch.long, device=device)
    predictions_name = f"{eval_name}_predictions"
    if world_size > 1:
        predictions_name += f".rank{rank:03d}"
    predictions_path = Path(args.out_dir) / f"{predictions_name}.jsonl"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    seen = 0
    with predictions_path.open("w", encoding="utf-8") as handle:
        for batch_idx, batch in enumerate(loader, start=1):
            batch = move_batch(batch, device)
            with autocast_context(device, precision):
                outputs = model(**batch)
                outputs = aggregate_chunk_outputs(outputs, batch)
            dialect_logits = combine_dialect_logits(outputs, logit_args)
            if dialect_logit_bias is not None:
                dialect_logits = dialect_logits + dialect_logit_bias.to(dialect_logits.dtype)
            dialect_logits = apply_active_dialect_mask(dialect_logits, active_dialect_mask)
            probs = average_crop_probs(torch.softmax(dialect_logits, dim=-1), batch["crop_counts"].to(device))
            preds = probs.argmax(dim=-1)
            labels = first_crop_items(batch["labels"], batch["crop_counts"].to(device))
            sample_ids = first_crop_items(batch["sample_ids"], batch["crop_counts"].to(device))
            sources = first_crop_meta(batch["meta"]["source"], batch["crop_counts"])
            country_codes = first_crop_meta(batch["meta"]["country_code"], batch["crop_counts"])
            duration_seconds = first_crop_meta(batch["meta"].get("duration_seconds", []), batch["crop_counts"])
            effective_duration_seconds = first_crop_meta(
                batch["meta"].get("effective_duration_seconds", []),
                batch["crop_counts"],
            )
            if "sample_chunk_counts" in batch:
                chunk_counts = first_crop_items(batch["sample_chunk_counts"], batch["crop_counts"].to(device)).cpu().tolist()
            else:
                chunk_counts = [1] * int(labels.shape[0])
            crop_counts = batch["crop_counts"].cpu().tolist()
            for idx, (truth, pred) in enumerate(zip(labels, preds)):
                matrix[int(truth), int(pred)] += 1
                topk = torch.topk(probs[idx], k=min(5, probs.shape[-1]))
                handle.write(
                    json.dumps(
                        {
                            "sample_id": int(sample_ids[idx].cpu()),
                            "truth": int(truth.cpu()),
                            "pred": int(pred.cpu()),
                            "probs": [float(x) for x in probs[idx].cpu().tolist()],
                            "topk_ids": [int(x) for x in topk.indices.cpu().tolist()],
                            "topk_probs": [float(x) for x in topk.values.cpu().tolist()],
                            "source": sources[idx],
                            "country_code": country_codes[idx],
                            "crop_count": int(crop_counts[idx]),
                            "chunk_count": int(chunk_counts[idx]),
                            "duration_seconds": float(duration_seconds[idx]) if duration_seconds else None,
                            "effective_duration_seconds": (
                                float(effective_duration_seconds[idx]) if effective_duration_seconds else None
                            ),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                seen += 1
            if args.log_every_batches > 0 and batch_idx % args.log_every_batches == 0:
                handle.flush()
                if rank == 0:
                    print(
                        json.dumps(
                            {
                                "split": eval_name,
                                "batches": batch_idx,
                                "examples_rank": seen,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(matrix, op=dist.ReduceOp.SUM)
    if rank == 0:
        metrics = save_metrics_bundle(args.out_dir, matrix.cpu().numpy(), eval_name)
        print(json.dumps(metrics, indent=2, sort_keys=True))
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
