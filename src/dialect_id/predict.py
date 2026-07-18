from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch

from .checkpointing import load_checkpoint_weights
from .collation import AudioCollator
from .labels import ID_TO_CODE, ID_TO_COUNTRY, ID_TO_REGION
from .modeling import build_model
from .train import autocast_context, combine_dialect_logits, move_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Arabic dialect for WAV files.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("audio", nargs="+")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def read_wav(path: str) -> torch.Tensor:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != 16000:
        raise ValueError(f"{path} has sample_rate={sr}; resample to 16 kHz first.")
    return audio


@torch.no_grad()
def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_args = ckpt["args"]
    precision = args.precision or train_args.get("precision", "bf16")
    logit_args = argparse.Namespace(
        region_logit_fusion_weight=float(train_args.get("region_logit_fusion_weight", 0.0) or 0.0),
        specialist_logit_fusion_weight=float(train_args.get("specialist_logit_fusion_weight", 0.0) or 0.0),
    )
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
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
    load_checkpoint_weights(model, ckpt, map_location="cpu")
    model.eval()
    collator = AudioCollator(
        train_args["model_type"],
        train_args.get("model_name_or_path"),
        cache_dir=train_args.get("cache_dir"),
        train=False,
        max_seconds=args.max_seconds,
        local_files_only=args.local_files_only or train_args.get("local_files_only", False),
    )
    batch = []
    for idx, path in enumerate(args.audio):
        batch.append(
            {
                "sample_id": idx,
                "waveform": read_wav(path),
                "country_id": 0,
                "region_id": 0,
                "country_code": "NA",
                "source": "file",
                "original_dialect": Path(path).name,
            }
        )
    tensor_batch = move_batch(collator(batch), device)
    with autocast_context(device, precision):
        outputs = model(**tensor_batch)
    probs = torch.softmax(combine_dialect_logits(outputs, logit_args), dim=-1)
    for path, row in zip(args.audio, probs):
        topk = torch.topk(row, k=min(args.top_k, row.shape[-1]))
        result = []
        for score, idx in zip(topk.values.cpu().tolist(), topk.indices.cpu().tolist()):
            result.append(
                {
                    "id": int(idx),
                    "code": ID_TO_CODE[int(idx)],
                    "country": ID_TO_COUNTRY[int(idx)],
                    "region": ID_TO_REGION[int(idx)],
                    "prob": float(score),
                }
            )
        print(json.dumps({"audio": path, "predictions": result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
