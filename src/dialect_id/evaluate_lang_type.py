from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.distributed as dist

from .evaluate import (
    MultiCropEvalCollator,
    average_crop_probs,
    first_crop_items,
    first_crop_meta,
    init_distributed,
    parse_args,
    resolve_manifest_paths,
)
from .collation import AudioCollator
from .data import ParquetDialectDataset, PseudoLabeledTarDataset
from .checkpointing import load_checkpoint_weights
from .modeling import build_model
from .train import aggregate_chunk_outputs, autocast_context, move_batch
from .evaluate import parse_code_filter

from torch.utils.data import DataLoader


LANG_TYPE_NAMES = ("dialectal", "msa")


def matrix_metrics(matrix: torch.Tensor) -> dict[str, object]:
    matrix_cpu = matrix.detach().cpu().to(torch.long)
    total = int(matrix_cpu.sum().item())
    correct = int(torch.trace(matrix_cpu).item())
    per_class = []
    recalls = []
    f1s = []
    for idx, name in enumerate(LANG_TYPE_NAMES):
        tp = int(matrix_cpu[idx, idx].item())
        support = int(matrix_cpu[idx, :].sum().item())
        predicted = int(matrix_cpu[:, idx].sum().item())
        precision = float(tp / predicted) if predicted else 0.0
        recall = float(tp / support) if support else 0.0
        f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
        if support:
            recalls.append(recall)
            f1s.append(f1)
        per_class.append(
            {
                "id": idx,
                "name": name,
                "support": support,
                "predicted": predicted,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return {
        "accuracy": float(correct / total) if total else 0.0,
        "balanced_accuracy": float(sum(recalls) / len(recalls)) if recalls else 0.0,
        "macro_f1": float(sum(f1s) / len(f1s)) if f1s else 0.0,
        "total": total,
        "correct": correct,
        "confusion_matrix": matrix_cpu.tolist(),
        "classes": list(LANG_TYPE_NAMES),
        "per_class": per_class,
    }


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

    eval_name = args.eval_name or args.split
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_name = f"{eval_name}_lang_type_predictions"
    if world_size > 1:
        predictions_name += f".rank{rank:03d}"
    predictions_path = out_dir / f"{predictions_name}.jsonl"
    matrix = torch.zeros((2, 2), dtype=torch.long, device=device)
    seen = 0
    ignored = 0

    with predictions_path.open("w", encoding="utf-8") as handle:
        for batch_idx, batch in enumerate(loader, start=1):
            batch = move_batch(batch, device)
            with autocast_context(device, precision):
                outputs = model(**batch)
                outputs = aggregate_chunk_outputs(outputs, batch)
            lang_logits = outputs.get("lang_type_logits")
            if lang_logits is None:
                raise RuntimeError("Checkpoint/model did not produce lang_type_logits")
            probs = average_crop_probs(torch.softmax(lang_logits, dim=-1), batch["crop_counts"].to(device))
            preds = probs.argmax(dim=-1)
            labels = first_crop_items(batch["lang_type_labels"], batch["crop_counts"].to(device))
            sample_ids = first_crop_items(batch["sample_ids"], batch["crop_counts"].to(device))
            sources = first_crop_meta(batch["meta"]["source"], batch["crop_counts"])
            country_codes = first_crop_meta(batch["meta"]["country_code"], batch["crop_counts"])
            lang_types = first_crop_meta(batch["meta"].get("lang_type", []), batch["crop_counts"])

            for idx, (truth, pred) in enumerate(zip(labels, preds)):
                truth_int = int(truth.cpu())
                pred_int = int(pred.cpu())
                if truth_int < 0:
                    ignored += 1
                    continue
                if truth_int >= 2:
                    raise ValueError(f"Invalid lang_type label {truth_int}")
                matrix[truth_int, pred_int] += 1
                handle.write(
                    json.dumps(
                        {
                            "sample_id": int(sample_ids[idx].cpu()),
                            "truth": truth_int,
                            "truth_name": LANG_TYPE_NAMES[truth_int],
                            "pred": pred_int,
                            "pred_name": LANG_TYPE_NAMES[pred_int],
                            "probs": [float(x) for x in probs[idx].cpu().tolist()],
                            "source": sources[idx],
                            "country_code": country_codes[idx],
                            "lang_type": lang_types[idx] if lang_types else None,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                seen += 1

            if args.log_every_batches > 0 and batch_idx % args.log_every_batches == 0 and rank == 0:
                print(
                    json.dumps(
                        {
                            "split": eval_name,
                            "batches": batch_idx,
                            "examples_rank": seen,
                            "ignored_rank": ignored,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    ignored_tensor = torch.tensor([ignored], dtype=torch.long, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(matrix, op=dist.ReduceOp.SUM)
        dist.all_reduce(ignored_tensor, op=dist.ReduceOp.SUM)

    if rank == 0:
        metrics = matrix_metrics(matrix)
        metrics["ignored"] = int(ignored_tensor.item())
        metrics_path = out_dir / f"{eval_name}_lang_type_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
