# Arabic Dialect Identification

Research and engineering toolkit for training, evaluating,
calibrating, and operationalizing a 23-label Arabic dialect-identification
(DID) system. The package supports clean Parquet supervision, pseudo-labeled
audio in tar manifests, distributed training, full-audio chunk aggregation,
checkpoint migration, post-hoc logit-bias calibration, acceptance gates, and
large-run label export/verification.

The fixed label space is AE, BH, DJ, DZ, EG, IQ, JO, KM, KW, LB, LY, MA, MR,
OM, PS, QA, SA, SD, SO, SY, TD, TN, and YE. A historically deployed 20-active
scope masked DJ, KM, and SO while keeping the 23-row head and checkpoint format.

## What is implemented

- Model backends for Whisper, Whisper+LLM, Wav2Vec2/WavLM/XLS-R,
  Wav2Vec2-BERT, ECAPA, and a Qwen3-ASR audio tower.
- A clean taxonomy head: dialect is primary, region logits are exact marginals
  of dialect probabilities, and language type is a separate task.
- DDP training with class balancing, streaming/balanced replay, pseudo-label
  mixing, focal and label-smoothed loss, specialist/confusion objectives,
  contrastive and margin losses, distillation, selective row tuning, and
  resumable trainable-only checkpoints.
- Long-utterance training/evaluation by fixed-duration chunks followed by
  per-sample mean-logit aggregation.
- Sharded metrics, diagnostics, additive calibration, and explicit release
  gates for regression-sensitive dialects.
- Deterministic tar sharding, SQLite tracking, decision summaries, confidence
  policies, final-label export, and artifact verification for large labeling
  runs. The model-serving/inference worker is deliberately not included; plug
  these tools into an approved inference backend.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,compression]'
```

PyTorch builds are hardware-specific; on GPU clusters, install the organization-
approved CUDA or ROCm build first. The Qwen3-ASR backend is optional; install it
with `python -m pip install -e '.[qwen]'`. See
[`docs/qwen-integration.md`](docs/qwen-integration.md).

## Minimal commands

Single-process training:

```bash
export DIALECT_ID_DATASET_ROOT=/path/to/parquet-dataset
dialect-id-train \
  --dataset-root "$DIALECT_ID_DATASET_ROOT" \
  --run-dir /path/to/run \
  --model-type wavlm \
  --model-name-or-path microsoft/wavlm-base-plus \
  --classifier-head-type clean_taxonomy \
  --full-audio-chunking --chunk-seconds 30
```

Evaluation and optional calibrated inference:

```bash
dialect-id-evaluate \
  --checkpoint /path/to/checkpoint.pt \
  --dataset-root /path/to/parquet-dataset \
  --split test --out-dir /path/to/report \
  --full-audio-chunking --chunk-seconds 30 \
  --dialect-logit-bias-json /path/to/logit_bias.json

dialect-id-predict --checkpoint /path/to/checkpoint.pt audio.wav
```

For portable multi-node launch examples, see [`recipes/slurm`](recipes/slurm).
Data formats and reproducibility contracts are in
[`docs/data-contracts.md`](docs/data-contracts.md).

The calibration, evaluation-gate, and large-run modules are included in built
wheels as the top-level `tools` package. For example:

```bash
python -m tools.big_run.prepare --help
python -m tools.evaluation.eval_gate --help
```

The top-level name is retained for compatibility with the original scripts;
new library code should normally import `dialect_id`.

## Historical evidence and limitations

One calibrated old-19 test report recorded accuracy 0.960042, balanced accuracy
0.954191, macro-F1 0.935519, and weighted-F1 0.960435; its paired TD
validation/test checks recorded 1.0 accuracy. These values are retained as
historical, self-reported internal engineering evidence, not as a reproducible
benchmark: the underlying data, model weights, exact environment, and raw
reports are not included.
DJ/KM/SO evidence was insufficient for production-quality claims. A later
strict no-old19 continuation was held after a step-450k branch regressed to
0.840303 raw accuracy. See [`docs/results.md`](docs/results.md).

## Legacy campaign identifiers

Names such as `acgc_*`, `old19`, and `qwenft_*` are preserved compatibility
keys for artifacts produced by one historical evaluation/labeling campaign.
They are not generic dataset names and are not requirements of the core
training package. Utilities that consume those exact keys are campaign-specific
adapters; the model, data, training, and evaluation primitives are reusable.

## License and public-release scope

Public source release of this repository has been authorized. It remains
all-rights-reserved rather than open source: viewing the source does not grant
permission to reuse or redistribute it. Datasets, weights, logs, raw reports,
hostnames, credentials, and host-specific paths remain outside the release.
See [`LICENSE.md`](LICENSE.md) and [`NOTICE.md`](NOTICE.md).
