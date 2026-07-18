from __future__ import annotations

import io
import random
import wave
from typing import Iterable

import numpy as np
import soundfile as sf


def wav_duration_seconds(wav_bytes: bytes) -> float | None:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as handle:
            frames = handle.getnframes()
            sample_rate = handle.getframerate()
        if sample_rate <= 0:
            return None
        return frames / float(sample_rate)
    except Exception:
        return None


def decode_audio_bytes(wav_bytes: bytes, target_sample_rate: int = 16000) -> np.ndarray:
    audio, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != target_sample_rate:
        raise ValueError(
            f"Expected {target_sample_rate} Hz audio, got {sample_rate} Hz. "
            "Add resampling before training if this occurs."
        )
    if not np.isfinite(audio).all():
        audio = np.nan_to_num(audio)
    return np.asarray(audio, dtype=np.float32)


def crop_or_pad(
    audio: np.ndarray,
    max_samples: int,
    train: bool,
    min_samples: int = 0,
) -> np.ndarray:
    if max_samples > 0 and audio.shape[0] > max_samples:
        if train:
            start = random.randint(0, audio.shape[0] - max_samples)
        else:
            start = max(0, (audio.shape[0] - max_samples) // 2)
        audio = audio[start : start + max_samples]
    if min_samples > 0 and audio.shape[0] < min_samples:
        pad = min_samples - audio.shape[0]
        audio = np.pad(audio, (0, pad), mode="constant")
    return audio.astype(np.float32, copy=False)


def augment_waveform(
    audio: np.ndarray,
    noise_prob: float = 0.0,
    gain_prob: float = 0.0,
    dropout_prob: float = 0.0,
) -> np.ndarray:
    if gain_prob > 0.0 and random.random() < gain_prob:
        gain_db = random.uniform(-6.0, 6.0)
        audio = audio * float(10.0 ** (gain_db / 20.0))
    if noise_prob > 0.0 and random.random() < noise_prob:
        rms = float(np.sqrt(np.mean(np.square(audio))) + 1e-7)
        snr_db = random.uniform(10.0, 30.0)
        noise_rms = rms / float(10.0 ** (snr_db / 20.0))
        audio = audio + np.random.normal(0.0, noise_rms, size=audio.shape).astype(np.float32)
    if dropout_prob > 0.0 and random.random() < dropout_prob and audio.shape[0] > 4000:
        width = random.randint(400, min(4000, max(401, audio.shape[0] // 8)))
        start = random.randint(0, audio.shape[0] - width)
        audio = audio.copy()
        audio[start : start + width] = 0.0
    return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)


def pad_waveforms(waveforms: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    waves = list(waveforms)
    lengths = np.asarray([wave.shape[0] for wave in waves], dtype=np.int64)
    max_len = int(lengths.max()) if len(lengths) else 0
    batch = np.zeros((len(waves), max_len), dtype=np.float32)
    for idx, wave_arr in enumerate(waves):
        batch[idx, : wave_arr.shape[0]] = wave_arr
    return batch, lengths
