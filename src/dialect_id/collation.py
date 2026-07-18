from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import AutoFeatureExtractor, Wav2Vec2FeatureExtractor, WhisperFeatureExtractor

from .audio import augment_waveform, crop_or_pad, pad_waveforms
from .labels import num_dialects


def _spec_augment(features: torch.Tensor, freq_masks: int, time_masks: int) -> torch.Tensor:
    if features.ndim != 3:
        return features
    output = features.clone()
    batch, freq, time = output.shape
    for bidx in range(batch):
        for _ in range(freq_masks):
            width = random.randint(0, max(1, freq // 8))
            if width <= 0 or width >= freq:
                continue
            start = random.randint(0, freq - width)
            output[bidx, start : start + width, :] = output[bidx].mean()
        for _ in range(time_masks):
            width = random.randint(0, max(1, time // 20))
            if width <= 0 or width >= time:
                continue
            start = random.randint(0, time - width)
            output[bidx, :, start : start + width] = output[bidx].mean()
    return output


@dataclass
class AudioCollator:
    model_type: str
    model_name_or_path: str | None = None
    cache_dir: str | None = None
    train: bool = True
    sample_rate: int = 16000
    max_seconds: float = 15.0
    min_seconds: float = 1.0
    noise_prob: float = 0.0
    gain_prob: float = 0.0
    dropout_prob: float = 0.0
    specaug: bool = False
    local_files_only: bool = False
    full_audio_chunking: bool = False
    chunk_seconds: float = 30.0
    max_chunks_per_sample: int = 0

    def __post_init__(self) -> None:
        self.max_samples = int(self.max_seconds * self.sample_rate) if self.max_seconds > 0 else 0
        self.min_samples = int(self.min_seconds * self.sample_rate) if self.min_seconds > 0 else 0
        self.chunk_samples = int(self.chunk_seconds * self.sample_rate) if self.chunk_seconds > 0 else 0
        self.feature_extractor = None
        if self.model_type in {"whisper", "whisper_llm", "whisper_llm_fusion"}:
            if not self.model_name_or_path:
                raise ValueError("Whisper collator requires model_name_or_path")
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
                self.model_name_or_path,
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
            )
        elif self.model_type == "qwen3_asr_audio":
            if not self.model_name_or_path:
                raise ValueError("qwen3_asr_audio collator requires model_name_or_path")
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.model_name_or_path,
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
            )
        elif self.model_type in {"wav2vec", "wavlm", "xlsr", "w2vbert"}:
            if not self.model_name_or_path:
                raise ValueError("wav2vec/wavlm/xlsr/w2vbert collator requires model_name_or_path")
            try:
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                    self.model_name_or_path,
                    cache_dir=self.cache_dir,
                    local_files_only=self.local_files_only,
                )
            except OSError:
                # Some SSL checkpoints in the shared cache only contain config/weights.
                self.feature_extractor = Wav2Vec2FeatureExtractor(
                    feature_size=1,
                    sampling_rate=self.sample_rate,
                    padding_value=0.0,
                    do_normalize=True,
                    return_attention_mask=True,
                )

    def _prepare_waveforms(self, batch: list[dict[str, Any]]) -> list[np.ndarray]:
        waves = []
        for item in batch:
            audio = crop_or_pad(
                item["waveform"],
                max_samples=self.max_samples,
                min_samples=self.min_samples,
                train=self.train,
            )
            if self.train:
                audio = augment_waveform(
                    audio,
                    noise_prob=self.noise_prob,
                    gain_prob=self.gain_prob,
                    dropout_prob=self.dropout_prob,
                )
            waves.append(audio)
        return waves

    def _feature_waveforms(self, waves: list[np.ndarray]) -> tuple[list[np.ndarray], torch.Tensor | None]:
        if not self.full_audio_chunking or self.chunk_samples <= 0:
            return waves, None
        chunks: list[np.ndarray] = []
        chunk_to_sample: list[int] = []
        for sample_idx, wave in enumerate(waves):
            num_chunks = max(1, (max(1, len(wave)) + self.chunk_samples - 1) // self.chunk_samples)
            if self.max_chunks_per_sample > 0 and num_chunks > self.max_chunks_per_sample:
                if self.train:
                    start_idx = random.randint(0, num_chunks - self.max_chunks_per_sample)
                else:
                    start_idx = 0
                end_idx = start_idx + self.max_chunks_per_sample
            else:
                start_idx = 0
                end_idx = num_chunks
            sample_chunks = [
                wave[chunk_idx * self.chunk_samples : (chunk_idx + 1) * self.chunk_samples]
                for chunk_idx in range(start_idx, end_idx)
            ]
            chunks.extend(sample_chunks)
            chunk_to_sample.extend([sample_idx] * len(sample_chunks))
        return chunks, torch.tensor(chunk_to_sample, dtype=torch.long)

    def _add_chunk_metadata(
        self,
        result: dict[str, Any],
        chunk_to_sample: torch.Tensor | None,
        batch_size: int,
    ) -> dict[str, Any]:
        if chunk_to_sample is None:
            return result
        result["chunk_to_sample"] = chunk_to_sample
        result["num_samples"] = torch.tensor(batch_size, dtype=torch.long)
        result["sample_chunk_counts"] = torch.bincount(chunk_to_sample, minlength=batch_size)
        return result

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        original_durations = [
            float(item.get("duration_seconds", len(item["waveform"]) / float(self.sample_rate)))
            for item in batch
        ]
        waves = self._prepare_waveforms(batch)
        effective_durations = [float(len(wave)) / float(self.sample_rate) for wave in waves]
        feature_waves, chunk_to_sample = self._feature_waveforms(waves)
        labels = torch.tensor([item["country_id"] for item in batch], dtype=torch.long)
        region_labels = torch.tensor([item["region_id"] for item in batch], dtype=torch.long)
        sample_ids = torch.tensor([item["sample_id"] for item in batch], dtype=torch.long)
        sample_weights = torch.tensor([float(item.get("loss_weight", 1.0)) for item in batch], dtype=torch.float32)
        lang_type_labels = torch.tensor([int(item.get("lang_type_id", -100)) for item in batch], dtype=torch.long)
        accent_labels = torch.tensor([int(item.get("accent_country_id", -100)) for item in batch], dtype=torch.long)
        meta = {
            "country_code": [item["country_code"] for item in batch],
            "lang_type": [item.get("lang_type", "unknown") for item in batch],
            "accent_country_code": [item.get("accent_country_code", "") for item in batch],
            "source": [item["source"] for item in batch],
            "original_dialect": [item["original_dialect"] for item in batch],
            "duration_seconds": original_durations,
            "effective_duration_seconds": effective_durations,
        }
        soft_labels = None
        if any(item.get("soft_label") is not None for item in batch):
            rows = []
            for item in batch:
                soft_label = item.get("soft_label")
                if soft_label is None:
                    row = torch.zeros(num_dialects(), dtype=torch.float32)
                    row[int(item["country_id"])] = 1.0
                else:
                    row = torch.tensor(soft_label, dtype=torch.float32)
                    if row.numel() != num_dialects():
                        raise ValueError(f"soft_label must have {num_dialects()} entries, got {row.numel()}")
                    row = row.clamp_min(0.0)
                    total = row.sum()
                    if float(total) <= 0.0:
                        row[int(item["country_id"])] = 1.0
                    else:
                        row = row / total
                rows.append(row)
            soft_labels = torch.stack(rows, dim=0)

        if self.model_type in {"whisper", "whisper_llm", "whisper_llm_fusion"}:
            encoded = self.feature_extractor(
                feature_waves,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                truncation=True,
            )
            input_features = encoded["input_features"]
            if self.train and self.specaug:
                input_features = _spec_augment(input_features, freq_masks=2, time_masks=2)
            return self._add_chunk_metadata({
                "input_features": input_features,
                "labels": labels,
                "region_labels": region_labels,
                "lang_type_labels": lang_type_labels,
                "accent_labels": accent_labels,
                "sample_ids": sample_ids,
                "sample_weights": sample_weights,
                **({"soft_labels": soft_labels} if soft_labels is not None else {}),
                "meta": meta,
            }, chunk_to_sample, len(batch))

        if self.model_type == "qwen3_asr_audio":
            encoded = self.feature_extractor(
                feature_waves,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_attention_mask=True,
            )
            input_features = encoded["input_features"]
            if self.train and self.specaug:
                input_features = _spec_augment(input_features, freq_masks=2, time_masks=2)
            result = {
                "input_features": input_features,
                "labels": labels,
                "region_labels": region_labels,
                "lang_type_labels": lang_type_labels,
                "accent_labels": accent_labels,
                "sample_ids": sample_ids,
                "sample_weights": sample_weights,
                **({"soft_labels": soft_labels} if soft_labels is not None else {}),
                "meta": meta,
            }
            if "attention_mask" in encoded:
                result["feature_attention_mask"] = encoded["attention_mask"]
            return self._add_chunk_metadata(result, chunk_to_sample, len(batch))

        padded, lengths = pad_waveforms(feature_waves)
        if self.model_type == "w2vbert":
            encoded = self.feature_extractor(
                feature_waves,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            )
            result = {
                "input_features": encoded["input_features"],
                "labels": labels,
                "region_labels": region_labels,
                "lang_type_labels": lang_type_labels,
                "accent_labels": accent_labels,
                "sample_ids": sample_ids,
                "sample_weights": sample_weights,
                **({"soft_labels": soft_labels} if soft_labels is not None else {}),
                "meta": meta,
            }
            if "attention_mask" in encoded:
                result["attention_mask"] = encoded["attention_mask"]
            return self._add_chunk_metadata(result, chunk_to_sample, len(batch))

        if self.model_type in {"wav2vec", "wavlm", "xlsr"}:
            encoded = self.feature_extractor(
                feature_waves,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            )
            result = {
                "input_values": encoded["input_values"],
                "labels": labels,
                "region_labels": region_labels,
                "lang_type_labels": lang_type_labels,
                "accent_labels": accent_labels,
                "sample_ids": sample_ids,
                "sample_weights": sample_weights,
                **({"soft_labels": soft_labels} if soft_labels is not None else {}),
                "meta": meta,
            }
            if "attention_mask" in encoded:
                result["attention_mask"] = encoded["attention_mask"]
            return self._add_chunk_metadata(result, chunk_to_sample, len(batch))

        return self._add_chunk_metadata({
            "input_values": torch.from_numpy(padded),
            "input_lengths": torch.from_numpy(lengths),
            "labels": labels,
            "region_labels": region_labels,
            "lang_type_labels": lang_type_labels,
            "accent_labels": accent_labels,
            "sample_ids": sample_ids,
            "sample_weights": sample_weights,
            **({"soft_labels": soft_labels} if soft_labels is not None else {}),
            "meta": meta,
        }, chunk_to_sample, len(batch))
