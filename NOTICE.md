# Notice

This repository is an allowlisted public-source extraction of Arabic dialect-ID
work developed during an internship. It contains reusable implementation code,
documentation, synthetic tests, and portable launch examples.

It intentionally excludes datasets, audio, checkpoints, predictions,
experiment outputs, credentials, scheduler state, host-specific settings, and
production-serving adapters. Third-party model weights and datasets are not
redistributed. Their own licenses and access terms still apply.

The optional Qwen3-ASR audio integration is compatible with the official
`qwen-asr==0.0.6` PyPI wheel. Its two required audio-tower modules were verified
in that wheel; the dependency is exposed through the `qwen` optional extra and
is not vendored here. Third-party package and model licenses still apply.

Public release of this source extraction has been authorized. That authorization
does not grant an open-source license or redistribute excluded data and models;
see `LICENSE.md` and the applicable third-party terms.
