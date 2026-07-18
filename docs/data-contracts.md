# Data Contracts

## Clean Parquet dataset

The dataset root must contain either `data/`, a Hugging Face cache-style
`refs/main` plus `snapshots/<id>/data/`, or a `snapshots/*/data/` directory.
Splits are discovered as `<split>-*.parquet` or `<split>.parquet`.

Training reads these columns:

- `sample_id`: stable integer identifier.
- `audio`: bytes or an Arrow audio object containing `bytes`.
- `country_code`: one of the 23 codes in `dialect_id.labels`.
- `country_id`: source numeric ID; the package derives the canonical target from
  `country_code`.
- `source` and `original_dialect`: provenance metadata.

Audio is decoded and resampled to 16 kHz. Dataset Parquet and audio files are
outside this source release and must not be committed.

## Pseudo-label manifests

Manifests may be JSONL, JSONL.zst, CSV, or CSV.gz. Each usable row needs:

- `tar_path`: tar archive path.
- `member`: member name inside the tar.
- A selected dialect field. Resolution order is `country_code`,
  `final_dialect`, `consensus_cc`, `teacher_top1`, `qwen_country_code`,
  `did_top1_cc`, `top1_cc`, `weak_country_code`, `weak_cc`, then `cc`.

Optional fields include `sample_id`, `source`, `loss_weight`, `soft_label` or
`teacher_probs`, `teacher_confidence`, `lang_type_id`, and accent targets. A
`manifest_label_index.json` sidecar can map manifest paths or basenames to label
codes so streaming workers can balance shards without a full pre-scan.

## Checkpoints

Training checkpoints are PyTorch dictionaries containing at least `model`,
`args`, and `labels`; full training checkpoints also contain optimizer/scheduler
state and step metadata. `state_type=trainable_only` checkpoints may reference
an initialization checkpoint and are not standalone unless that dependency is
preserved. Never commit checkpoint files.

## Evaluation outputs

An evaluation directory contains `<eval_name>_metrics.json`, per-class metrics,
a confusion matrix, and optionally rank-sharded prediction JSONL. Calibration
consumes prediction rows with `truth` and `probs`; deterministic split tooling
also uses `sample_id`, `source`, `country_code`, and `duration_seconds`.

Several retained campaign adapters expect split keys beginning with `acgc_` or
containing `old19`. Those strings are legacy artifact-schema identifiers, not
generic split names; see the compatibility note in the README.

## Large-run handoff

`tools/big_run/prepare.py` writes deterministic `shard_*.jsonl` manifests and a
SQLite tracker. An external inference adapter is expected to write prediction
CSV shards plus `state/shard_*.done` JSON markers describing completed outputs.
The retained audit/export scripts then verify row counts, summarize decisions,
and produce confidence-filtered label manifests.
