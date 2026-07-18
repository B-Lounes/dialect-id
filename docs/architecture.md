# Architecture

## Training path

1. `ParquetDialectDataset` streams clean examples by Parquet row group and
   shards work across distributed ranks and data-loader workers.
2. `PseudoLabeledTarDataset` streams JSONL/CSV manifests, groups requested tar
   members, decodes audio, and can balance replay by target dialect.
3. `MixedDialectDataset` samples the two streams with a configurable pseudo
   probability and independent sample weights.
4. `AudioCollator` performs crop/pad, augmentation, feature extraction, and
   optional full-audio chunking.
5. A selected encoder produces a pooled embedding and classifier outputs.
6. Chunk-level tensors are averaged back to sample-level tensors before losses
   and metrics are computed.

## Taxonomy

The canonical head contains 23 dialect rows. `clean_taxonomy` treats dialect as
the source of truth: a region probability is the sum of probabilities for all
dialects in that region. The implementation uses `logsumexp` over dialect
logits, so a softmax over region logits exactly matches that marginal. Language
type remains an independent task. The legacy head is retained for checkpoint
compatibility and controlled comparisons.

## Objectives and adaptation

The base objective is dialect cross-entropy with optional class weights, focal
scaling, active-class label smoothing, and soft pseudo targets. Optional
objectives include region and within-region loss, specialist heads, directional
confusion penalties, supervised contrastive loss, hard-negative margin loss,
ArcFace, teacher distillation, language type, and accent classification.

Fine-tuning controls cover encoder freezing, an unfrozen LLM tail, inactive-head
freezing, and selected dialect-row updates. Checkpoints can store the full state
or trainable parameters only. Label-aware loading maps source rows by dialect
code when expanding an older head into the 23-label layout.

## Evaluation and release

Evaluation supports distributed data shards, multiple crops, full-audio chunks,
active output masks, and additive per-dialect logit bias. Metrics are generated
from confusion matrices. Separate tools merge metric shards, build coverage and
error diagnostics, tune calibration on deterministic partitions, and gate a
candidate using aggregate regressions plus watched-dialect F1 and prediction-to-
gold ratios.

## Large labeling runs

The large-run utilities deterministically hash tar paths into manifests, record
work in SQLite, audit prediction shards, summarize confidence/weak-label
agreement, export confidence-policy manifests, compare exports in parallel, and
verify final artifacts. Inference is an integration boundary: an approved model
service must write the expected prediction shards and completion markers.
