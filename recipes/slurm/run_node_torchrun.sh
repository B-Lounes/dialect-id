#!/usr/bin/env bash

set -euo pipefail

: "${REPO_ROOT:?}"
: "${DATASET_ROOT:?}"
: "${RUN_DIR:?}"
: "${MODEL_TYPE:?}"
: "${NNODES:?}"
: "${NPROC_PER_NODE:?}"
: "${MASTER_ADDR:?}"
: "${MASTER_PORT:?}"

cd "$REPO_ROOT"

train_args=(
  --dataset-root "$DATASET_ROOT"
  --run-dir "$RUN_DIR"
  --model-type "$MODEL_TYPE"
  --classifier-head-type "${CLASSIFIER_HEAD_TYPE:-clean_taxonomy}"
  --batch-size "${BATCH_SIZE:-4}"
  --eval-batch-size "${EVAL_BATCH_SIZE:-8}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-4}"
  --learning-rate "${LEARNING_RATE:-2e-5}"
  --head-learning-rate "${HEAD_LEARNING_RATE:-2e-4}"
  --max-train-steps "${MAX_TRAIN_STEPS:-100000}"
  --save-every-steps "${SAVE_EVERY_STEPS:-5000}"
  --eval-every-steps "${EVAL_EVERY_STEPS:-5000}"
  --full-audio-chunking
  --chunk-seconds "${CHUNK_SECONDS:-30}"
  --precision "${PRECISION:-bf16}"
)

if [[ -n "${MODEL_NAME_OR_PATH:-}" ]]; then
  train_args+=(--model-name-or-path "$MODEL_NAME_OR_PATH")
fi
if [[ "${ACTIVE_SCOPE:-23}" == "20" ]]; then
  train_args+=(
    --exclude-country-code DJ --exclude-country-code KM --exclude-country-code SO
    --clean-train-exclude-country-code DJ --clean-train-exclude-country-code KM --clean-train-exclude-country-code SO
  )
fi
if [[ -n "${PSEUDO_MANIFEST_GLOB:-}" ]]; then
  train_args+=(--pseudo-manifest-glob "$PSEUDO_MANIFEST_GLOB" --pseudo-streaming)
fi

torchrun \
  --nnodes="$NNODES" \
  --nproc-per-node="$NPROC_PER_NODE" \
  --node-rank="${SLURM_NODEID:-0}" \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  -m dialect_id.train "${train_args[@]}"
