from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import glob
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .collation import AudioCollator
from .checkpointing import load_checkpoint_weights, load_compatible_state_dict, unwrap_model
from .data import DEFAULT_DATASET_CACHE, MixedDialectDataset, ParquetDialectDataset, PseudoLabeledTarDataset, labeled_train_counts
from .labels import CODE_TO_ID, DIALECT_ID_TO_REGION_ID, SPECIALIST_DIALECT_IDS, SPECIALIST_ID_TO_LOCAL, label_payload, num_dialects, num_regions
from .metrics import save_metrics_bundle
from .modeling import build_model, dialect_region_marginal_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Arabic dialect ID classifier.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_CACHE)
    parser.add_argument("--include-country-code", action="append", default=None)
    parser.add_argument("--exclude-country-code", action="append", default=None)
    parser.add_argument("--clean-train-include-country-code", action="append", default=None)
    parser.add_argument("--clean-train-exclude-country-code", action="append", default=None)
    parser.add_argument("--disable-clean-train", action="store_true")
    parser.add_argument("--pseudo-manifest", action="append", default=None)
    parser.add_argument("--pseudo-manifest-glob", action="append", default=None)
    parser.add_argument("--pseudo-streaming", action="store_true")
    parser.add_argument("--pseudo-sample-prob", type=float, default=0.0)
    parser.add_argument("--pseudo-loss-weight", type=float, default=0.35)
    parser.add_argument("--pseudo-max-examples", type=int, default=0)
    parser.add_argument("--pseudo-val-manifest", action="append", default=None)
    parser.add_argument("--pseudo-val-manifest-glob", action="append", default=None)
    parser.add_argument("--pseudo-val-streaming", action="store_true")
    parser.add_argument("--pseudo-val-max-examples", type=int, default=0)
    parser.add_argument("--pseudo-balanced-replay", action="store_true")
    parser.add_argument("--pseudo-replay-buffer-size", type=int, default=8)
    parser.add_argument("--pseudo-replay-prefill", type=int, default=1024)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--model-type",
        choices=[
            "whisper",
            "whisper_llm",
            "whisper_llm_fusion",
            "wav2vec",
            "wavlm",
            "xlsr",
            "w2vbert",
            "qwen3_asr_audio",
            "ecapa",
        ],
        required=True,
    )
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--llm-name-or-path", default=None)
    parser.add_argument("--llm-cache-dir", default=None)
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HUB_CACHE"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--data-epoch-offset",
        type=int,
        default=0,
        help=(
            "Offset added to the dataset epoch seed. This is useful for long "
            "resume chains where the model/optimizer should continue from a "
            "checkpoint but the streaming manifest order should move to a new "
            "deterministic shuffle."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument(
        "--lr-decay-steps",
        type=int,
        default=0,
        help="Cosine-decay LR from warmup end to this step (0 = constant LR after warmup).",
    )
    parser.add_argument("--min-lr-ratio", type=float, default=0.0)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-eval-examples", type=int, default=0)
    parser.add_argument("--stratified-max-eval-examples", action="store_true")
    parser.add_argument("--eval-every-steps", type=int, default=2000)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--save-every-steps", type=int, default=2000)
    parser.add_argument("--log-every-steps", type=int, default=50)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--freeze-encoder-layers", type=int, default=0)
    parser.add_argument("--no-freeze-feature-encoder", action="store_true")
    parser.add_argument("--no-freeze-llm", action="store_true")
    parser.add_argument("--llm-trainable-layers", type=int, default=0)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--audio-llm-tokens", type=int, default=32)
    parser.add_argument("--classifier-hidden-dim", type=int, default=512)
    parser.add_argument(
        "--classifier-head-type",
        choices=["legacy", "clean_taxonomy"],
        default="legacy",
        help="Classifier head layout. clean_taxonomy removes specialist heads and derives region from dialect mass.",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--region-loss-weight", type=float, default=0.2)
    parser.add_argument("--within-region-loss-weight", type=float, default=0.0)
    parser.add_argument("--specialist-loss-weight", type=float, default=0.0)
    parser.add_argument("--lang-type-loss-weight", type=float, default=0.0)
    parser.add_argument("--accent-loss-weight", type=float, default=0.0)
    parser.add_argument("--region-logit-fusion-weight", type=float, default=0.0)
    parser.add_argument("--specialist-logit-fusion-weight", type=float, default=0.0)
    parser.add_argument("--confusion-loss-weight", type=float, default=0.0)
    parser.add_argument("--confusion-metrics-json", default=None)
    parser.add_argument("--confusion-top-k", type=int, default=3)
    parser.add_argument("--confusion-min-count", type=int, default=50)
    parser.add_argument(
        "--confusion-pair",
        action="append",
        default=None,
        help="Directional hard confusions as TRUE:PRED1,PRED2, e.g. OM:AE,SA,QA.",
    )
    parser.add_argument("--contrastive-loss-weight", type=float, default=0.0)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--hard-negative-loss-weight", type=float, default=0.0)
    parser.add_argument("--hard-negative-margin", type=float, default=0.2)
    parser.add_argument("--arcface-loss-weight", type=float, default=0.0)
    parser.add_argument("--arcface-margin", type=float, default=0.2)
    parser.add_argument("--arcface-scale", type=float, default=30.0)
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Optional dialect-label smoothing; smoothing mass is spread only over active dialect classes.",
    )
    parser.add_argument("--class-weight", choices=["none", "balanced", "sqrt_balanced"], default="sqrt_balanced")
    parser.add_argument(
        "--class-weight-min",
        type=float,
        default=0.0,
        help="Optional floor applied to active class weights after normalization; 0 disables.",
    )
    parser.add_argument(
        "--class-weight-max",
        type=float,
        default=0.0,
        help="Optional ceiling applied to active class weights after normalization; 0 disables.",
    )
    parser.add_argument(
        "--class-weight-counts-json",
        default=None,
        help=(
            "Optional JSON count source for class weights. Accepts either a flat "
            "{CODE: count} mapping or common wrapper keys such as train_by_code."
        ),
    )
    parser.add_argument("--max-seconds", type=float, default=15.0)
    parser.add_argument("--eval-max-seconds", type=float, default=20.0)
    parser.add_argument("--full-audio-chunking", action="store_true")
    parser.add_argument("--chunk-seconds", type=float, default=30.0)
    parser.add_argument("--max-chunks-per-sample", type=int, default=0)
    parser.add_argument(
        "--max-train-input-feature-elements",
        type=int,
        default=0,
        help="Skip a distributed training microbatch if any rank creates an input_features tensor above this size.",
    )
    parser.add_argument("--balanced-replay", action="store_true")
    parser.add_argument("--replay-buffer-size", type=int, default=4)
    parser.add_argument("--replay-prefill", type=int, default=1)
    parser.add_argument("--balanced-row-groups", action="store_true")
    parser.add_argument("--noise-prob", type=float, default=0.15)
    parser.add_argument("--gain-prob", type=float, default=0.15)
    parser.add_argument("--dropout-prob", type=float, default=0.05)
    parser.add_argument("--specaug", action="store_true")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init-from-checkpoint", default=None)
    parser.add_argument(
        "--ddp-find-unused-parameters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forwarded to DistributedDataParallel. Disable for clean graphs where all parameters are used.",
    )
    parser.add_argument(
        "--freeze-inactive-task-heads",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze auxiliary heads whose loss weight is zero so DDP sees a clean DID-only graph.",
    )
    parser.add_argument("--distill-teacher-checkpoint", default=None)
    parser.add_argument("--distill-loss-weight", type=float, default=0.0)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument(
        "--distill-label-code",
        action="append",
        default=None,
        help="Restrict teacher distillation to samples whose label is in these dialect codes/ids.",
    )
    parser.add_argument(
        "--distill-class-code",
        action="append",
        default=None,
        help="Restrict teacher/student KL classes to these dialect codes/ids.",
    )
    parser.add_argument(
        "--train-only-dialect-code",
        action="append",
        default=None,
        help="Freeze the model except selected rows of heads.dialect; accepts codes or ids, comma-separated.",
    )
    parser.add_argument("--save-trainable-only", action="store_true")
    parser.add_argument("--save-frozen-encoder", action="store_true")
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


def init_distributed() -> tuple[int, int, int, torch.device]:
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
    return rank, world_size, local_rank, device


def is_main(rank: int) -> bool:
    return rank == 0


def startup_marker(run_dir: Path, rank: int, message: str) -> None:
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "rank": rank,
            "message": message,
        }
        with (run_dir / f"startup_rank{rank}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    except Exception:
        pass


def set_seed(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def build_class_weights(args: argparse.Namespace, device: torch.device) -> torch.Tensor | None:
    if args.class_weight == "none":
        return None
    if args.class_weight_counts_json:
        counts_by_code = load_class_weight_counts(args.class_weight_counts_json)
        include = parse_code_filter(args.include_country_code)
        exclude = parse_code_filter(args.exclude_country_code) or set()
        counts_list = [0 for _ in range(num_dialects())]
        for code, count in counts_by_code.items():
            code = code.upper()
            if code not in CODE_TO_ID:
                continue
            if include is not None and code not in include:
                continue
            if code in exclude:
                continue
            counts_list[CODE_TO_ID[code]] = int(count)
        counts = torch.tensor(counts_list, dtype=torch.float32)
    else:
        counts = torch.tensor(
            labeled_train_counts(
                args.dataset_root,
                include_country_codes=parse_code_filter(args.include_country_code),
                exclude_country_codes=parse_code_filter(args.exclude_country_code),
            ),
            dtype=torch.float32,
        )
    active_counts = counts > 0
    counts = counts.clamp_min(1.0)
    if args.class_weight == "balanced":
        weights = counts.sum() / (counts * len(counts))
    else:
        weights = torch.sqrt(counts.sum() / (counts * len(counts)))
    if bool(active_counts.any()):
        weights = weights / weights[active_counts].mean()
        if args.class_weight_min > 0 or args.class_weight_max > 0:
            active_weights = weights[active_counts]
            if args.class_weight_min > 0:
                active_weights = active_weights.clamp_min(float(args.class_weight_min))
            if args.class_weight_max > 0:
                active_weights = active_weights.clamp_max(float(args.class_weight_max))
            weights = weights.clone()
            weights[active_counts] = active_weights
        weights = torch.where(active_counts, weights, torch.zeros_like(weights))
    else:
        weights = weights / weights.mean()
        if args.class_weight_min > 0:
            weights = weights.clamp_min(float(args.class_weight_min))
        if args.class_weight_max > 0:
            weights = weights.clamp_max(float(args.class_weight_max))
    return weights.to(device)


def load_class_weight_counts(path: str | Path) -> dict[str, int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("train_by_code", "counts_by_code", "counts", "labels"):
        nested = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(nested, dict):
            payload = nested
            break
    if not isinstance(payload, dict):
        raise ValueError(f"class weight counts JSON must contain a mapping: {path}")
    counts: dict[str, int] = {}
    for code, value in payload.items():
        code = str(code).strip().upper()
        if not code:
            continue
        try:
            counts[code] = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid count for {code!r} in {path}: {value!r}") from exc
    return counts


def build_active_dialect_mask(args: argparse.Namespace, device: torch.device) -> torch.Tensor | None:
    include = parse_code_filter(args.include_country_code)
    exclude = parse_code_filter(args.exclude_country_code) or set()
    active = torch.ones(num_dialects(), dtype=torch.bool)
    if include is not None:
        active.zero_()
        for code in include:
            if code in CODE_TO_ID:
                active[CODE_TO_ID[code]] = True
    for code in exclude:
        if code in CODE_TO_ID:
            active[CODE_TO_ID[code]] = False
    if bool(active.all()):
        return None
    if not bool(active.any()):
        raise ValueError("Active dialect mask is empty after include/exclude filtering")
    return active.to(device)


def apply_active_dialect_mask(logits: torch.Tensor | None, active_mask: torch.Tensor | None) -> torch.Tensor | None:
    if logits is None or active_mask is None:
        return logits
    return logits.masked_fill(~active_mask.to(device=logits.device).unsqueeze(0), torch.finfo(logits.dtype).min)


def mean_by_sample(tensor: torch.Tensor, chunk_to_sample: torch.Tensor, num_samples: int) -> torch.Tensor:
    if tensor.shape[0] != int(chunk_to_sample.numel()):
        return tensor
    out = tensor.new_zeros((num_samples, *tensor.shape[1:]))
    index = chunk_to_sample.to(device=tensor.device, dtype=torch.long)
    view_shape = (index.shape[0],) + (1,) * (tensor.ndim - 1)
    expanded = index.view(view_shape).expand_as(tensor)
    out.scatter_add_(0, expanded, tensor)
    counts = torch.bincount(index, minlength=num_samples).to(device=tensor.device, dtype=tensor.dtype).clamp_min(1)
    return out / counts.view((num_samples,) + (1,) * (tensor.ndim - 1))


def aggregate_chunk_outputs(outputs: dict[str, Any], batch: dict[str, Any]) -> dict[str, Any]:
    chunk_to_sample = batch.get("chunk_to_sample")
    if chunk_to_sample is None:
        return outputs
    num_samples_value = batch.get("num_samples")
    num_samples = int(num_samples_value.item()) if torch.is_tensor(num_samples_value) else int(num_samples_value)
    aggregated: dict[str, Any] = {}
    for key, value in outputs.items():
        if torch.is_tensor(value) and value.ndim > 0:
            aggregated[key] = mean_by_sample(value, chunk_to_sample, num_samples)
        elif isinstance(value, dict):
            aggregated[key] = {
                subkey: mean_by_sample(subvalue, chunk_to_sample, num_samples)
                if torch.is_tensor(subvalue) and subvalue.ndim > 0
                else subvalue
                for subkey, subvalue in value.items()
            }
        else:
            aggregated[key] = value
    return aggregated


def apply_active_soft_target_mask(
    soft_targets: torch.Tensor | None,
    labels: torch.Tensor,
    active_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if soft_targets is None or active_mask is None:
        return soft_targets
    mask = active_mask.to(device=soft_targets.device, dtype=soft_targets.dtype).unsqueeze(0)
    targets = soft_targets * mask
    normalizer = targets.sum(dim=-1, keepdim=True)
    valid = normalizer.squeeze(-1) > 1e-6
    targets = targets / normalizer.clamp_min(1e-6)
    if bool(valid.all()):
        return targets
    fallback = F.one_hot(labels.to(device=soft_targets.device), num_classes=soft_targets.shape[-1]).to(
        dtype=soft_targets.dtype
    )
    return torch.where(valid.unsqueeze(-1), targets, fallback)


def combine_dialect_logits(outputs: dict[str, Any], args: argparse.Namespace) -> torch.Tensor:
    logits = outputs["dialect_logits"]
    if args.region_logit_fusion_weight > 0:
        region_logits = outputs.get("region_logits")
        if region_logits is not None:
            region_ids = torch.tensor(
                [DIALECT_ID_TO_REGION_ID[idx] for idx in range(num_dialects())],
                dtype=torch.long,
                device=logits.device,
            ).clamp_max(num_regions() - 1)
            logits = logits + args.region_logit_fusion_weight * region_logits.index_select(1, region_ids)
    if args.specialist_logit_fusion_weight > 0:
        specialist_logits = outputs.get("specialist_logits") or {}
        fused = torch.zeros_like(logits)
        counts = torch.zeros((num_dialects(),), dtype=logits.dtype, device=logits.device)
        for name, local_logits in specialist_logits.items():
            dialect_ids = SPECIALIST_DIALECT_IDS.get(name)
            if not dialect_ids:
                continue
            idx = torch.tensor(dialect_ids, dtype=torch.long, device=logits.device)
            fused.index_add_(1, idx, local_logits.to(dtype=logits.dtype))
            counts.index_add_(0, idx, torch.ones((len(dialect_ids),), dtype=logits.dtype, device=logits.device))
        fused = fused / counts.clamp_min(1.0).unsqueeze(0)
        logits = logits + args.specialist_logit_fusion_weight * fused
    return logits


def weighted_mean(loss: torch.Tensor, sample_weights: torch.Tensor | None = None) -> torch.Tensor:
    if sample_weights is None:
        return loss.mean()
    weights = sample_weights.to(device=loss.device, dtype=loss.dtype)
    return (loss * weights).sum() / weights.sum().clamp_min(1e-6)


def optional_cross_entropy_loss(
    logits: torch.Tensor | None,
    labels: torch.Tensor | None,
    *,
    sample_weights: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    if logits is None or labels is None:
        device = labels.device if labels is not None else None
        return torch.zeros((), device=device)
    valid = labels != ignore_index
    if not bool(valid.any()):
        return logits.new_zeros(())
    loss = F.cross_entropy(logits[valid], labels[valid], reduction="none")
    weights = sample_weights[valid] if sample_weights is not None else None
    return weighted_mean(loss, weights)


def classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_weights: torch.Tensor | None,
    focal_gamma: float,
    sample_weights: torch.Tensor | None = None,
    soft_targets: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    smoothing_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    smoothing = max(0.0, min(float(label_smoothing), 1.0))
    if smoothing > 0:
        active = (
            smoothing_mask.to(device=logits.device, dtype=torch.bool)
            if smoothing_mask is not None
            else torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
        )
        denom = active.sum().clamp_min(1).to(dtype=logits.dtype)
        uniform = active.to(dtype=logits.dtype) / denom
        if soft_targets is None:
            targets = F.one_hot(labels, num_classes=logits.shape[-1]).to(device=logits.device, dtype=logits.dtype)
        else:
            targets = soft_targets.to(device=logits.device, dtype=logits.dtype)
        soft_targets = targets * (1.0 - smoothing) + uniform.unsqueeze(0) * smoothing
    if soft_targets is None:
        loss = F.cross_entropy(logits, labels, weight=class_weights, reduction="none")
    else:
        targets = soft_targets.to(device=logits.device, dtype=logits.dtype)
        log_probs = F.log_softmax(logits.float(), dim=-1).to(dtype=logits.dtype)
        loss = -(targets * log_probs).sum(dim=-1)
        if class_weights is not None:
            target_weights = (targets * class_weights.to(device=logits.device, dtype=logits.dtype).unsqueeze(0)).sum(dim=-1)
            loss = loss * target_weights
    if focal_gamma > 0:
        pt = torch.softmax(logits.float(), dim=-1).gather(1, labels[:, None]).squeeze(1).detach().clamp(1e-6, 1.0)
        loss = ((1.0 - pt) ** focal_gamma) * loss
    return weighted_mean(loss, sample_weights)


def _label_id(value: str) -> int:
    key = value.strip().upper()
    if key in CODE_TO_ID:
        return CODE_TO_ID[key]
    idx = int(key)
    if idx < 0 or idx >= num_dialects():
        raise ValueError(f"Dialect id out of range: {value!r}")
    return idx


def parse_label_ids(values: list[str] | None) -> list[int] | None:
    if not values:
        return None
    ids: list[int] = []
    seen: set[int] = set()
    for value in values:
        for item in value.replace(";", ",").split(","):
            item = item.strip()
            if not item:
                continue
            idx = _label_id(item)
            if idx not in seen:
                ids.append(idx)
                seen.add(idx)
    return ids or None


def build_confusion_penalty_matrix(
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor | None:
    if args.confusion_loss_weight <= 0 and args.hard_negative_loss_weight <= 0:
        return None
    matrix = torch.zeros((num_dialects(), num_dialects()), dtype=torch.float32)
    if args.confusion_metrics_json:
        payload = json.loads(Path(args.confusion_metrics_json).read_text(encoding="utf-8"))
        confusion = np.asarray(payload["confusion_matrix"], dtype=np.int64)
        for truth in range(min(num_dialects(), confusion.shape[0])):
            candidates = []
            for pred in range(min(num_dialects(), confusion.shape[1])):
                if pred == truth:
                    continue
                count = int(confusion[truth, pred])
                if count >= args.confusion_min_count:
                    candidates.append((count, pred))
            candidates.sort(reverse=True)
            for _count, pred in candidates[: max(0, args.confusion_top_k)]:
                matrix[truth, pred] = 1.0
    for spec in args.confusion_pair or []:
        if ":" not in spec:
            raise ValueError(f"--confusion-pair must look like TRUE:PRED1,PRED2, got {spec!r}")
        truth_code, pred_codes = spec.split(":", 1)
        truth = _label_id(truth_code)
        for pred_code in pred_codes.split(","):
            if pred_code.strip():
                pred = _label_id(pred_code)
                if pred != truth:
                    matrix[truth, pred] = 1.0
    if float(matrix.sum()) <= 0:
        return None
    return matrix.to(device)


def confusion_probability_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    penalty_matrix: torch.Tensor | None,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if penalty_matrix is None:
        return logits.new_zeros(())
    probs = torch.softmax(logits.float(), dim=-1)
    penalties = penalty_matrix[labels]
    return weighted_mean((probs * penalties).sum(dim=-1), sample_weights)


def within_region_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    region_labels: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    losses = []
    dialect_regions = torch.tensor(
        [DIALECT_ID_TO_REGION_ID[idx] for idx in range(num_dialects())],
        dtype=torch.long,
        device=logits.device,
    )
    for region_id in region_labels.unique(sorted=True).tolist():
        dialect_ids = torch.nonzero(dialect_regions == int(region_id), as_tuple=False).flatten()
        if dialect_ids.numel() <= 1:
            continue
        sample_mask = region_labels == int(region_id)
        if not sample_mask.any():
            continue
        region_logits = logits[sample_mask][:, dialect_ids]
        region_labels_global = labels[sample_mask]
        local_targets = (region_labels_global[:, None] == dialect_ids[None, :]).long().argmax(dim=-1)
        local_loss = F.cross_entropy(region_logits, local_targets, reduction="none")
        local_weights = sample_weights[sample_mask] if sample_weights is not None else None
        losses.append(weighted_mean(local_loss, local_weights))
    if not losses:
        return logits.new_zeros(())
    return torch.stack(losses).mean()


def specialist_head_loss(
    specialist_logits: dict[str, torch.Tensor] | None,
    labels: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if not specialist_logits:
        return labels.new_zeros((), dtype=torch.float32)
    losses = []
    for name, logits in specialist_logits.items():
        dialect_ids = SPECIALIST_DIALECT_IDS.get(name)
        id_to_local = SPECIALIST_ID_TO_LOCAL.get(name)
        if not dialect_ids or not id_to_local:
            continue
        mask = torch.zeros_like(labels, dtype=torch.bool)
        local_targets = torch.empty_like(labels)
        for dialect_id in dialect_ids:
            local_mask = labels == int(dialect_id)
            if local_mask.any():
                mask |= local_mask
                local_targets[local_mask] = int(id_to_local[int(dialect_id)])
        if not mask.any():
            continue
        local_loss = F.cross_entropy(logits[mask], local_targets[mask], reduction="none")
        local_weights = sample_weights[mask] if sample_weights is not None else None
        losses.append(weighted_mean(local_loss, local_weights))
    if not losses:
        first = next(iter(specialist_logits.values()))
        return first.new_zeros(())
    return torch.stack(losses).mean()


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if embeddings.shape[0] <= 1:
        return embeddings.new_zeros(())
    embeddings, labels = gather_embeddings_and_labels(embeddings, labels)
    z = F.normalize(embeddings.float(), dim=-1)
    logits = torch.matmul(z, z.transpose(0, 1)) / max(temperature, 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    eye = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~eye
    anchor_mask = positive_mask.any(dim=1)
    if not anchor_mask.any():
        return embeddings.new_zeros(())
    logits_mask = ~eye
    exp_logits = torch.exp(logits) * logits_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1)
    return -mean_log_prob_pos[anchor_mask].mean()


def hard_negative_margin_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    penalty_matrix: torch.Tensor | None,
    *,
    margin: float,
) -> torch.Tensor:
    if penalty_matrix is None or embeddings.shape[0] <= 1:
        return embeddings.new_zeros(())
    embeddings, labels = gather_embeddings_and_labels(embeddings, labels)
    z = F.normalize(embeddings.float(), dim=-1)
    sim = torch.matmul(z, z.transpose(0, 1))
    hard_mask = penalty_matrix[labels][:, labels].bool()
    hard_mask = hard_mask | hard_mask.transpose(0, 1)
    hard_mask.fill_diagonal_(False)
    if not hard_mask.any():
        return embeddings.new_zeros(())
    return F.relu(sim[hard_mask] - margin).mean()


def gather_embeddings_and_labels(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (dist.is_available() and dist.is_initialized()):
        return embeddings, labels
    world_size = dist.get_world_size()
    if world_size <= 1:
        return embeddings, labels
    local_size = torch.tensor([embeddings.shape[0]], device=embeddings.device, dtype=torch.long)
    sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(sizes, local_size)
    if len({int(size.item()) for size in sizes}) != 1:
        return embeddings, labels
    try:
        from torch.distributed.nn.functional import all_gather

        gathered_embeddings = torch.cat(tuple(all_gather(embeddings)), dim=0)
    except Exception:
        return embeddings, labels
    gathered_labels = [torch.zeros_like(labels) for _ in range(world_size)]
    dist.all_gather(gathered_labels, labels)
    return gathered_embeddings, torch.cat(gathered_labels, dim=0)


def arcface_loss(
    model: nn.Module,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_weights: torch.Tensor | None,
    focal_gamma: float,
    margin: float,
    scale: float,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    raw_model = unwrap_model(model)
    heads = getattr(raw_model, "heads", None)
    dialect_head = getattr(heads, "dialect", None)
    if dialect_head is None:
        return embeddings.new_zeros(())
    z = F.normalize(embeddings.float(), dim=-1)
    weights = F.normalize(dialect_head.weight.float(), dim=-1)
    logits = torch.matmul(z, weights.transpose(0, 1))
    one_hot = F.one_hot(labels, num_classes=logits.shape[-1]).float()
    logits = (logits - one_hot * margin) * scale
    return classification_loss(
        logits,
        labels,
        class_weights=class_weights,
        focal_gamma=focal_gamma,
        sample_weights=sample_weights,
    )


def optimizer_for(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    head_params = []
    encoder_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            ".heads." in name
            or name.startswith("heads")
            or ".pool." in name
            or name.startswith("pool")
            or name.startswith("audio_")
            or ".audio_" in name
        ):
            head_params.append(param)
        else:
            encoder_params.append(param)
    groups = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": args.learning_rate})
    if head_params:
        groups.append({"params": head_params, "lr": args.head_learning_rate})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def freeze_inactive_task_heads(model: nn.Module, args: argparse.Namespace) -> dict[str, Any]:
    if not args.freeze_inactive_task_heads:
        return {"enabled": False, "frozen_heads": []}
    raw_model = unwrap_model(model)
    heads = getattr(raw_model, "heads", None)
    if heads is None:
        return {"enabled": True, "frozen_heads": []}
    frozen: list[str] = []
    candidates = [
        ("region", args.region_loss_weight),
        ("lang_type", args.lang_type_loss_weight),
        ("accent", args.accent_loss_weight),
        ("specialists", args.specialist_loss_weight),
    ]
    for name, loss_weight in candidates:
        if float(loss_weight) > 0:
            continue
        head = getattr(heads, name, None)
        if not isinstance(head, nn.Module):
            continue
        params = list(head.parameters())
        if not params:
            continue
        for param in params:
            param.requires_grad_(False)
        frozen.append(name)
    return {"enabled": True, "frozen_heads": frozen}


def configure_trainable_parameters(model: nn.Module, args: argparse.Namespace) -> dict[str, Any]:
    only_codes = parse_code_filter(args.train_only_dialect_code)
    if not only_codes:
        return {"mode": "default"}
    raw_model = unwrap_model(model)
    selected_ids = sorted(CODE_TO_ID[code] for code in only_codes if code in CODE_TO_ID)
    if not selected_ids:
        raise ValueError(f"No valid dialect codes in --train-only-dialect-code={args.train_only_dialect_code!r}")
    for param in raw_model.parameters():
        param.requires_grad = False
    heads = getattr(raw_model, "heads", None)
    dialect_head = getattr(heads, "dialect", None)
    if dialect_head is None or not hasattr(dialect_head, "weight"):
        raise ValueError("--train-only-dialect-code requires a model with heads.dialect")
    row_mask = torch.zeros((dialect_head.weight.shape[0],), dtype=torch.bool, device=dialect_head.weight.device)
    row_mask[selected_ids] = True
    dialect_head.weight.requires_grad = True
    dialect_head.weight.register_hook(lambda grad: grad * row_mask.to(device=grad.device, dtype=grad.dtype).unsqueeze(1))
    row_freeze_state: dict[str, Any] = {
        "row_mask": row_mask.detach().clone(),
        "weight": dialect_head.weight.detach().clone(),
    }
    if dialect_head.bias is not None:
        dialect_head.bias.requires_grad = True
        dialect_head.bias.register_hook(lambda grad: grad * row_mask.to(device=grad.device, dtype=grad.dtype))
        row_freeze_state["bias"] = dialect_head.bias.detach().clone()
    raw_model._dialect_row_freeze = row_freeze_state  # type: ignore[attr-defined]
    id_to_code = {idx: code for code, idx in CODE_TO_ID.items()}
    return {
        "mode": "train_only_dialect_rows",
        "dialect_codes": [id_to_code[idx] for idx in selected_ids],
        "dialect_ids": selected_ids,
    }


@torch.no_grad()
def restore_frozen_dialect_rows(model: nn.Module) -> None:
    raw_model = unwrap_model(model)
    state = getattr(raw_model, "_dialect_row_freeze", None)
    if not state:
        return
    heads = getattr(raw_model, "heads", None)
    dialect_head = getattr(heads, "dialect", None)
    if dialect_head is None:
        return
    row_mask = state["row_mask"].to(device=dialect_head.weight.device)
    frozen = ~row_mask
    dialect_head.weight.data[frozen] = state["weight"].to(
        device=dialect_head.weight.device,
        dtype=dialect_head.weight.dtype,
    )[frozen]
    if dialect_head.bias is not None and "bias" in state:
        dialect_head.bias.data[frozen] = state["bias"].to(
            device=dialect_head.bias.device,
            dtype=dialect_head.bias.dtype,
        )[frozen]


def lr_scale(step: int, warmup_steps: int, decay_steps: int = 0, min_lr_ratio: float = 0.0) -> float:
    if warmup_steps > 0 and step + 1 < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    if decay_steps > warmup_steps:
        progress = min(1.0, float(step + 1 - warmup_steps) / float(decay_steps - warmup_steps))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return 1.0


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def local_oversized_train_batch(batch: dict[str, Any], args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    limit = int(args.max_train_input_feature_elements or 0)
    if limit <= 0 or "input_features" not in batch or not torch.is_tensor(batch["input_features"]):
        return False, {}

    input_features = batch["input_features"]
    elements = int(input_features.numel())
    if elements <= limit:
        return False, {}

    meta = batch.get("meta") or {}
    info: dict[str, Any] = {
        "event": "skip_oversized_train_batch",
        "input_features_shape": [int(dim) for dim in input_features.shape],
        "input_features_elements": elements,
        "limit": limit,
        "sample_ids": batch.get("sample_ids").tolist() if torch.is_tensor(batch.get("sample_ids")) else None,
        "labels": batch.get("labels").tolist() if torch.is_tensor(batch.get("labels")) else None,
        "country_code": meta.get("country_code"),
        "source": meta.get("source"),
        "duration_seconds": meta.get("duration_seconds"),
        "effective_duration_seconds": meta.get("effective_duration_seconds"),
    }
    if torch.is_tensor(batch.get("sample_chunk_counts")):
        info["sample_chunk_counts"] = batch["sample_chunk_counts"].tolist()
    return True, info


def distributed_should_skip_train_batch(local_skip: bool, device: torch.device) -> bool:
    if not (dist.is_available() and dist.is_initialized()):
        return local_skip
    flag_device = device if device.type == "cuda" else torch.device("cpu")
    flag = torch.tensor(1 if local_skip else 0, dtype=torch.int32, device=flag_device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(int(flag.item()))


def forward_loss(
    model: nn.Module,
    batch: dict[str, Any],
    args: argparse.Namespace,
    class_weights: torch.Tensor | None,
    confusion_penalty: torch.Tensor | None,
    active_dialect_mask: torch.Tensor | None = None,
    teacher_model: nn.Module | None = None,
    distill_label_ids: torch.Tensor | None = None,
    distill_class_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    outputs = model(**batch)
    outputs = aggregate_chunk_outputs(outputs, batch)
    dialect_logits = apply_active_dialect_mask(combine_dialect_logits(outputs, args), active_dialect_mask)
    region_logits = outputs["region_logits"]
    if outputs.get("region_logits_are_dialect_marginals") and active_dialect_mask is not None:
        region_logits = dialect_region_marginal_logits(dialect_logits)
    sample_weights = batch.get("sample_weights")
    soft_labels = apply_active_soft_target_mask(
        batch.get("soft_labels"),
        batch["labels"],
        active_dialect_mask,
    )
    dialect_loss = classification_loss(
        dialect_logits,
        batch["labels"],
        class_weights=class_weights,
        focal_gamma=args.focal_gamma,
        sample_weights=sample_weights,
        soft_targets=soft_labels,
        label_smoothing=args.label_smoothing,
        smoothing_mask=active_dialect_mask,
    )
    region_loss = weighted_mean(
        F.cross_entropy(region_logits, batch["region_labels"], reduction="none"),
        sample_weights,
    )
    intra_region_loss = within_region_loss(
        dialect_logits,
        batch["labels"],
        batch["region_labels"],
        sample_weights=sample_weights,
    )
    specialist_loss = specialist_head_loss(
        outputs.get("specialist_logits"),
        batch["labels"],
        sample_weights=sample_weights,
    )
    lang_type_loss = optional_cross_entropy_loss(
        outputs.get("lang_type_logits"),
        batch.get("lang_type_labels"),
        sample_weights=sample_weights,
    )
    accent_logits = apply_active_dialect_mask(outputs.get("accent_logits"), active_dialect_mask)
    accent_loss = optional_cross_entropy_loss(
        accent_logits,
        batch.get("accent_labels"),
        sample_weights=sample_weights,
    )
    confusion_loss = confusion_probability_loss(
        dialect_logits,
        batch["labels"],
        confusion_penalty,
        sample_weights=sample_weights,
    )
    embeddings = outputs.get("embedding")
    contrastive_loss = (
        supervised_contrastive_loss(
            embeddings,
            batch["labels"],
            temperature=args.contrastive_temperature,
        )
        if embeddings is not None and args.contrastive_loss_weight > 0
        else dialect_logits.new_zeros(())
    )
    hard_negative_loss = (
        hard_negative_margin_loss(
            embeddings,
            batch["labels"],
            confusion_penalty,
            margin=args.hard_negative_margin,
        )
        if embeddings is not None and args.hard_negative_loss_weight > 0
        else dialect_logits.new_zeros(())
    )
    arc_loss = (
        arcface_loss(
            model,
            embeddings,
            batch["labels"],
            class_weights=class_weights,
            focal_gamma=args.focal_gamma,
            margin=args.arcface_margin,
            scale=args.arcface_scale,
            sample_weights=sample_weights,
        )
        if embeddings is not None and args.arcface_loss_weight > 0
        else dialect_logits.new_zeros(())
    )
    distill_loss = dialect_logits.new_zeros(())
    if teacher_model is not None and args.distill_loss_weight > 0:
        sample_mask = torch.ones_like(batch["labels"], dtype=torch.bool)
        if distill_label_ids is not None:
            label_ids = distill_label_ids.to(device=batch["labels"].device)
            sample_mask = batch["labels"].unsqueeze(1).eq(label_ids.unsqueeze(0)).any(dim=1)
        if bool(sample_mask.any()):
            with torch.no_grad():
                teacher_outputs = teacher_model(**batch)
                teacher_outputs = aggregate_chunk_outputs(teacher_outputs, batch)
                teacher_logits = apply_active_dialect_mask(
                    combine_dialect_logits(teacher_outputs, args),
                    active_dialect_mask,
                )
            student_logits = dialect_logits
            if distill_class_ids is not None:
                class_ids = distill_class_ids.to(device=student_logits.device)
                student_logits = student_logits.index_select(1, class_ids)
                teacher_logits = teacher_logits.index_select(1, class_ids)
            temperature = max(float(args.distill_temperature), 1e-6)
            student_log_probs = F.log_softmax(student_logits[sample_mask].float() / temperature, dim=-1)
            teacher_probs = F.softmax(teacher_logits[sample_mask].float() / temperature, dim=-1)
            per_sample = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
            per_sample = per_sample * (temperature * temperature)
            distill_weights = sample_weights[sample_mask] if sample_weights is not None else None
            distill_loss = weighted_mean(per_sample.to(dtype=dialect_logits.dtype), distill_weights)
    loss = (
        dialect_loss
        + args.region_loss_weight * region_loss
        + args.within_region_loss_weight * intra_region_loss
        + args.specialist_loss_weight * specialist_loss
        + args.lang_type_loss_weight * lang_type_loss
        + args.accent_loss_weight * accent_loss
        + args.confusion_loss_weight * confusion_loss
        + args.contrastive_loss_weight * contrastive_loss
        + args.hard_negative_loss_weight * hard_negative_loss
        + args.arcface_loss_weight * arc_loss
        + args.distill_loss_weight * distill_loss
    )
    pred = dialect_logits.argmax(dim=-1)
    acc = (pred == batch["labels"]).float().mean()
    return loss, {
        "loss": float(loss.detach().cpu()),
        "dialect_loss": float(dialect_loss.detach().cpu()),
        "region_loss": float(region_loss.detach().cpu()),
        "within_region_loss": float(intra_region_loss.detach().cpu()),
        "specialist_loss": float(specialist_loss.detach().cpu()),
        "lang_type_loss": float(lang_type_loss.detach().cpu()),
        "accent_loss": float(accent_loss.detach().cpu()),
        "confusion_loss": float(confusion_loss.detach().cpu()),
        "contrastive_loss": float(contrastive_loss.detach().cpu()),
        "hard_negative_loss": float(hard_negative_loss.detach().cpu()),
        "arcface_loss": float(arc_loss.detach().cpu()),
        "distill_loss": float(distill_loss.detach().cpu()),
        "acc": float(acc.detach().cpu()),
    }, pred


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    rank: int,
    out_dir: Path,
    prefix: str,
    active_dialect_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    model.eval()
    matrix = torch.zeros((num_dialects(), num_dialects()), dtype=torch.long, device=device)
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            with autocast_context(device, args.precision):
                outputs = model(**batch)
                outputs = aggregate_chunk_outputs(outputs, batch)
            dialect_logits = apply_active_dialect_mask(combine_dialect_logits(outputs, args), active_dialect_mask)
            preds = dialect_logits.argmax(dim=-1)
            labels = batch["labels"]
            for truth, pred in zip(labels, preds):
                if 0 <= int(truth) < num_dialects() and 0 <= int(pred) < num_dialects():
                    matrix[int(truth), int(pred)] += 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(matrix, op=dist.ReduceOp.SUM)
    metrics = {}
    if is_main(rank):
        metrics = save_metrics_bundle(out_dir, matrix.cpu().numpy(), prefix)
    model.train()
    return metrics


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    step: int,
    epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = unwrap_model(model)
    state = raw_model.state_dict()
    state_type = "full"
    if args.save_trainable_only:
        trainable_names = {
            name for name, param in raw_model.named_parameters() if param.requires_grad
        }
        state = {
            name: tensor
            for name, tensor in state.items()
            if name in trainable_names or (args.save_frozen_encoder and name.startswith("encoder."))
        }
        state_type = "trainable_only"
    torch.save(
        {
            "model": state,
            "state_type": state_type,
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "step": step,
            "epoch": epoch,
            "labels": label_payload(),
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[int, int]:
    payload = torch.load(path, map_location=map_location)
    load_checkpoint_weights(model, payload, map_location=map_location)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return int(payload.get("step", 0)), int(payload.get("epoch", 0))


def main() -> None:
    args = parse_args()
    preinit_rank = int(os.environ.get("RANK", "0"))
    startup_marker(Path(args.run_dir), preinit_rank, "before_init_distributed")
    rank, world_size, _local_rank, device = init_distributed()
    set_seed(args.seed, rank)
    run_dir = Path(args.run_dir)
    startup_marker(run_dir, rank, f"distributed_initialized world_size={world_size} device={device}")
    if is_main(rank):
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")
        (run_dir / "labels.json").write_text(json.dumps(label_payload(), indent=2, sort_keys=True) + "\n")

    startup_marker(run_dir, rank, "before_model_build")
    model = build_model(
        args.model_type,
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        llm_name_or_path=args.llm_name_or_path,
        llm_cache_dir=args.llm_cache_dir,
        local_files_only=args.local_files_only,
        freeze_encoder_layers=args.freeze_encoder_layers,
        freeze_feature_encoder=not args.no_freeze_feature_encoder,
        freeze_llm=not args.no_freeze_llm,
        llm_trainable_layers=args.llm_trainable_layers,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        audio_llm_tokens=args.audio_llm_tokens,
        classifier_hidden_dim=args.classifier_hidden_dim,
        classifier_head_type=args.classifier_head_type,
        dropout=args.dropout,
    )
    startup_marker(run_dir, rank, "after_model_build_before_to_device")
    if args.precision == "fp32":
        model = model.float()
    model = model.to(device)
    startup_marker(run_dir, rank, "after_model_to_device")

    global_step = 0
    start_epoch = 0
    if args.init_from_checkpoint:
        payload = torch.load(args.init_from_checkpoint, map_location="cpu")
        load_compatible_state_dict(
            model,
            payload["model"],
            strict=False,
            source_labels=payload.get("labels"),
        )
    teacher_model = None
    if args.distill_teacher_checkpoint and args.distill_loss_weight > 0:
        startup_marker(run_dir, rank, "before_teacher_model_build")
        teacher_model = build_model(
            args.model_type,
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            llm_name_or_path=args.llm_name_or_path,
            llm_cache_dir=args.llm_cache_dir,
            local_files_only=args.local_files_only,
            freeze_encoder_layers=args.freeze_encoder_layers,
            freeze_feature_encoder=True,
            freeze_llm=True,
            llm_trainable_layers=0,
            gradient_checkpointing=False,
            audio_llm_tokens=args.audio_llm_tokens,
            classifier_hidden_dim=args.classifier_hidden_dim,
            classifier_head_type=args.classifier_head_type,
            dropout=args.dropout,
        )
        if args.precision == "fp32":
            teacher_model = teacher_model.float()
        teacher_model = teacher_model.to(device)
        teacher_payload = torch.load(args.distill_teacher_checkpoint, map_location="cpu")
        load_compatible_state_dict(
            teacher_model,
            teacher_payload["model"],
            strict=False,
            source_labels=teacher_payload.get("labels"),
        )
        for param in teacher_model.parameters():
            param.requires_grad_(False)
        teacher_model.eval()
        startup_marker(run_dir, rank, "after_teacher_model_ready")
    inactive_task_head_config = freeze_inactive_task_heads(model, args)
    trainable_config = configure_trainable_parameters(model, args)
    trainable_config["inactive_task_heads"] = inactive_task_head_config
    if is_main(rank):
        (run_dir / "trainable_config.json").write_text(
            json.dumps(trainable_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if world_size > 1:
        startup_marker(run_dir, rank, "before_ddp_wrap")
        model = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=args.ddp_find_unused_parameters,
        )
        startup_marker(run_dir, rank, "after_ddp_wrap")

    optimizer = optimizer_for(model, args)
    startup_marker(run_dir, rank, "after_optimizer")
    class_weights = build_class_weights(args, device)
    active_dialect_mask = build_active_dialect_mask(args, device)
    confusion_penalty = build_confusion_penalty_matrix(args, device)
    distill_label_id_list = parse_label_ids(args.distill_label_code)
    distill_class_id_list = parse_label_ids(args.distill_class_code)
    distill_label_ids = (
        torch.tensor(distill_label_id_list, dtype=torch.long, device=device)
        if distill_label_id_list is not None
        else None
    )
    distill_class_ids = (
        torch.tensor(distill_class_id_list, dtype=torch.long, device=device)
        if distill_class_id_list is not None
        else None
    )
    startup_marker(run_dir, rank, "after_loss_helpers")
    if args.resume:
        global_step, start_epoch = load_checkpoint(args.resume, model, optimizer, map_location=device)

    include_codes = parse_code_filter(args.include_country_code)
    exclude_codes = parse_code_filter(args.exclude_country_code)
    clean_train_include_codes = parse_code_filter(args.clean_train_include_country_code)
    clean_train_exclude_codes = parse_code_filter(args.clean_train_exclude_country_code)
    if clean_train_include_codes is None:
        clean_train_include_codes = include_codes
    if clean_train_exclude_codes is None:
        clean_train_exclude_codes = exclude_codes
    startup_marker(run_dir, rank, "before_clean_train_dataset")
    clean_train_ds = None
    if not args.disable_clean_train:
        clean_train_ds = ParquetDialectDataset(
            args.dataset_root,
            "train",
            seed=args.seed,
            shuffle=True,
            exclude_unlabeled=True,
            max_examples=0 if (args.pseudo_manifest or args.pseudo_manifest_glob) else args.max_train_examples,
            balanced_replay=args.balanced_replay,
            replay_buffer_size=max(1, args.replay_buffer_size),
            replay_prefill=max(1, args.replay_prefill),
            balanced_row_groups=args.balanced_row_groups,
            include_country_codes=clean_train_include_codes,
            exclude_country_codes=clean_train_exclude_codes,
        )
    startup_marker(run_dir, rank, "after_clean_train_dataset")
    pseudo_manifests = list(args.pseudo_manifest or [])
    if args.pseudo_manifest_glob:
        for pattern in args.pseudo_manifest_glob:
            pseudo_manifests.extend(sorted(glob.glob(pattern)))
    if pseudo_manifests:
        startup_marker(run_dir, rank, "before_pseudo_dataset")
        pseudo_ds = PseudoLabeledTarDataset(
            pseudo_manifests,
            seed=args.seed + 17,
            shuffle=True,
            max_examples=args.pseudo_max_examples,
            default_loss_weight=args.pseudo_loss_weight,
            include_country_codes=include_codes,
            exclude_country_codes=exclude_codes,
            streaming=args.pseudo_streaming,
            balanced_replay=args.pseudo_balanced_replay,
            replay_buffer_size=args.pseudo_replay_buffer_size,
            replay_prefill=args.pseudo_replay_prefill,
        )
        startup_marker(run_dir, rank, "after_pseudo_dataset")
        if clean_train_ds is None or args.pseudo_sample_prob >= 1.0:
            train_ds = pseudo_ds
        else:
            train_ds = MixedDialectDataset(
                clean_train_ds,
                pseudo_ds,
                pseudo_probability=args.pseudo_sample_prob,
                seed=args.seed,
                max_examples=args.max_train_examples,
            )
    else:
        if clean_train_ds is None:
            raise ValueError("--disable-clean-train requires at least one pseudo manifest")
        train_ds = clean_train_ds
    startup_marker(run_dir, rank, "before_validation_dataset")
    val_ds = ParquetDialectDataset(
        args.dataset_root,
        "validation",
        seed=args.seed,
        shuffle=False,
        exclude_unlabeled=True,
        max_examples=args.max_eval_examples,
        stratified_max_examples=args.stratified_max_eval_examples,
        include_country_codes=include_codes,
        exclude_country_codes=exclude_codes,
    )
    startup_marker(run_dir, rank, "after_validation_dataset")
    pseudo_val_manifests = list(args.pseudo_val_manifest or [])
    if args.pseudo_val_manifest_glob:
        for pattern in args.pseudo_val_manifest_glob:
            pseudo_val_manifests.extend(sorted(glob.glob(pattern)))
    pseudo_val_ds = None
    if pseudo_val_manifests:
        startup_marker(run_dir, rank, "before_pseudo_validation_dataset")
        pseudo_val_ds = PseudoLabeledTarDataset(
            pseudo_val_manifests,
            seed=args.seed + 29,
            shuffle=False,
            max_examples=args.pseudo_val_max_examples,
            default_loss_weight=1.0,
            include_country_codes=include_codes,
            exclude_country_codes=exclude_codes,
            streaming=args.pseudo_val_streaming,
        )
        startup_marker(run_dir, rank, "after_pseudo_validation_dataset")
    startup_marker(run_dir, rank, "before_collators")
    train_collator = AudioCollator(
        args.model_type,
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        train=True,
        max_seconds=args.max_seconds,
        noise_prob=args.noise_prob,
        gain_prob=args.gain_prob,
        dropout_prob=args.dropout_prob,
        specaug=args.specaug,
        local_files_only=args.local_files_only,
        full_audio_chunking=args.full_audio_chunking,
        chunk_seconds=args.chunk_seconds,
        max_chunks_per_sample=args.max_chunks_per_sample,
    )
    eval_collator = AudioCollator(
        args.model_type,
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        train=False,
        max_seconds=args.eval_max_seconds,
        noise_prob=0.0,
        gain_prob=0.0,
        dropout_prob=0.0,
        specaug=False,
        local_files_only=args.local_files_only,
        full_audio_chunking=args.full_audio_chunking,
        chunk_seconds=args.chunk_seconds,
        max_chunks_per_sample=args.max_chunks_per_sample,
    )
    startup_marker(run_dir, rank, "after_collators_before_loaders")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=train_collator,
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=eval_collator,
        pin_memory=False,
    )
    pseudo_val_loader = (
        DataLoader(
            pseudo_val_ds,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            collate_fn=eval_collator,
            pin_memory=False,
        )
        if pseudo_val_ds is not None
        else None
    )
    startup_marker(run_dir, rank, "after_loaders_before_training_loop")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    last_log = time.time()
    for epoch in range(start_epoch, args.epochs):
        data_epoch = epoch + args.data_epoch_offset
        train_ds.set_epoch(data_epoch)
        ddp_join = model.join() if isinstance(model, DistributedDataParallel) else None
        if ddp_join is not None:
            ddp_join.__enter__()
        try:
            if is_main(rank):
                startup_marker(run_dir, rank, f"training_epoch_{epoch + 1}_before_iter")
            train_iter = enumerate(train_loader)
            for batch_idx, batch in train_iter:
                trace_batch = is_main(rank) and epoch == start_epoch and batch_idx < 2
                if trace_batch:
                    startup_marker(run_dir, rank, f"training_epoch_{epoch + 1}_batch_{batch_idx}_fetched")
                local_skip, skip_info = local_oversized_train_batch(batch, args)
                should_skip = distributed_should_skip_train_batch(local_skip, device)
                if should_skip:
                    if local_skip:
                        skip_info.update(
                            {
                                "rank": rank,
                                "epoch": epoch + 1,
                                "batch_idx": batch_idx,
                            }
                        )
                        with (run_dir / "skipped_train_batches.jsonl").open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(skip_info, sort_keys=True) + "\n")
                    if trace_batch:
                        startup_marker(run_dir, rank, f"training_epoch_{epoch + 1}_batch_{batch_idx}_skipped")
                    continue
                for group in optimizer.param_groups:
                    group.setdefault("base_lr", group["lr"])
                    group["lr"] = group["base_lr"] * lr_scale(
                        global_step,
                        args.warmup_steps,
                        decay_steps=args.lr_decay_steps,
                        min_lr_ratio=args.min_lr_ratio,
                    )
                batch = move_batch(batch, device)
                if trace_batch:
                    startup_marker(run_dir, rank, f"training_epoch_{epoch + 1}_batch_{batch_idx}_moved")
                with autocast_context(device, args.precision):
                    loss, logs, _pred = forward_loss(
                        model,
                        batch,
                        args,
                        class_weights,
                        confusion_penalty,
                        active_dialect_mask,
                        teacher_model=teacher_model,
                        distill_label_ids=distill_label_ids,
                        distill_class_ids=distill_class_ids,
                    )
                    loss = loss / args.gradient_accumulation_steps
                if trace_batch:
                    startup_marker(run_dir, rank, f"training_epoch_{epoch + 1}_batch_{batch_idx}_forward_done")
                sync_gradients = (batch_idx + 1) % args.gradient_accumulation_steps == 0
                backward_context = (
                    model.no_sync()
                    if isinstance(model, DistributedDataParallel) and not sync_gradients
                    else nullcontext()
                )
                with backward_context:
                    loss.backward()
                if trace_batch:
                    startup_marker(run_dir, rank, f"training_epoch_{epoch + 1}_batch_{batch_idx}_backward_done")
                if sync_gradients:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    restore_frozen_dialect_rows(model)
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if trace_batch:
                        startup_marker(run_dir, rank, f"training_epoch_{epoch + 1}_batch_{batch_idx}_optimizer_step")

                    if is_main(rank) and global_step % args.log_every_steps == 0:
                        elapsed = max(1e-6, time.time() - last_log)
                        last_log = time.time()
                        local_chunks = int(batch["input_features"].shape[0]) if "input_features" in batch else args.batch_size
                        local_samples = int(batch["labels"].shape[0]) if "labels" in batch else args.batch_size
                        examples_since_log = args.batch_size * args.gradient_accumulation_steps * args.log_every_steps
                        msg = {
                            "step": global_step,
                            "epoch": epoch + 1,
                            "data_epoch": data_epoch + 1,
                            "rank": rank,
                            "examples_per_sec_local": examples_since_log / elapsed,
                            "examples_per_sec_global": examples_since_log * world_size / elapsed,
                            "samples_local": local_samples,
                            "chunks_local": local_chunks,
                            **logs,
                        }
                        with (run_dir / "train_log.jsonl").open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(msg, sort_keys=True) + "\n")
                        print(json.dumps(msg, sort_keys=True), flush=True)

                    if is_main(rank) and args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                        save_checkpoint(run_dir / "checkpoints" / f"step_{global_step}.pt", model, optimizer, args, global_step, epoch)

                    if (
                        not args.skip_validation
                        and args.eval_every_steps > 0
                        and global_step % args.eval_every_steps == 0
                    ):
                        metrics = evaluate(
                            model,
                            val_loader,
                            args,
                            device,
                            rank,
                            run_dir,
                            prefix=f"validation_step_{global_step}",
                            active_dialect_mask=active_dialect_mask,
                        )
                        if is_main(rank):
                            print(json.dumps({"step": global_step, "validation": metrics}, sort_keys=True), flush=True)
                        if pseudo_val_loader is not None:
                            pseudo_metrics = evaluate(
                                model,
                                pseudo_val_loader,
                                args,
                                device,
                                rank,
                                run_dir,
                                prefix=f"pseudo_validation_step_{global_step}",
                                active_dialect_mask=active_dialect_mask,
                            )
                            if is_main(rank):
                                print(json.dumps({"step": global_step, "pseudo_validation": pseudo_metrics}, sort_keys=True), flush=True)

                    if is_main(rank) and args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                        save_checkpoint(run_dir / "checkpoints" / f"step_{global_step}.pt", model, optimizer, args, global_step, epoch)

                    if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                        break
        finally:
            if ddp_join is not None:
                ddp_join.__exit__(None, None, None)
        if is_main(rank):
            save_checkpoint(run_dir / "checkpoints" / f"epoch_{epoch + 1}.pt", model, optimizer, args, global_step, epoch + 1)
        if not args.skip_validation:
            metrics = evaluate(
                model,
                val_loader,
                args,
                device,
                rank,
                run_dir,
                prefix=f"validation_epoch_{epoch + 1}",
                active_dialect_mask=active_dialect_mask,
            )
            if is_main(rank):
                print(json.dumps({"epoch": epoch + 1, "validation": metrics}, sort_keys=True), flush=True)
            if pseudo_val_loader is not None:
                pseudo_metrics = evaluate(
                    model,
                    pseudo_val_loader,
                    args,
                    device,
                    rank,
                    run_dir,
                    prefix=f"pseudo_validation_epoch_{epoch + 1}",
                    active_dialect_mask=active_dialect_mask,
                )
                if is_main(rank):
                    print(json.dumps({"epoch": epoch + 1, "pseudo_validation": pseudo_metrics}, sort_keys=True), flush=True)
        if args.max_train_steps > 0 and global_step >= args.max_train_steps:
            break

    if is_main(rank):
        save_checkpoint(run_dir / "checkpoints" / "final.pt", model, optimizer, args, global_step, args.epochs)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
