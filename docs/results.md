# Historical Results and Limitations

These self-reported numbers summarize internal reports produced during
development. They are included to explain design decisions, not to claim a
self-contained or independently verified benchmark.

| Evaluation | Accuracy | Balanced accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: | ---: |
| Calibrated old-19 test | 0.960042 | 0.954191 | 0.935519 | 0.960435 |
| TD validation | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| TD test | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

The selected candidate used a Qwen3-ASR 1.7B audio path, clean-taxonomy heads,
30-second full-audio chunks with mean aggregation, and additive calibration.
Its operational output scope masked DJ, KM, and SO, so it should be described as
20-active rather than as a validated production 23-way system.

A later from-scratch strict branch that excluded old-19 clean supervision was
not promoted. At step 450k its raw accuracy was 0.840303, a material regression
relative to the accepted candidate evidence. The calibration and evaluation
gate tools in this repository encode the resulting hold/release discipline.

Limitations:

- Datasets, checkpoints, raw predictions, and exact result bundles are omitted.
- The historical split construction may not be independently reproducible.
- Perfect TD results may reflect a small or narrow evaluation and should not be
  generalized without reviewing support and provenance.
- DJ, KM, and SO lacked sufficient verified evidence for production claims.
- No fairness, robustness, privacy, or external-domain benchmark is bundled.
- Dependency versions and accelerator kernels can change numerical results.

Any CV or publication claim should say “historical, self-reported internal
evaluation” and name the active label scope, rather than presenting these
values as an open benchmark result.
