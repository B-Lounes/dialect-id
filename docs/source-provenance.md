# Source Provenance

This repository was assembled non-destructively from two development areas:
the canonical home-workspace package and a later shared 23-way campaign
workspace. The reusable package modules were taken from the canonical source;
the later workspace contributed calibration, evaluation-gate, diagnostics, and
large-labeling utilities. The newest parallel final-export comparison utility
was selected from the large-run tools.

The extraction used an allowlist. It excludes data, audio, model weights,
predictions, reports, logs, scheduler state, generated manifests, databases,
retry controls, and dangerous job-control scripts. Hard-coded mounts, hosts,
usernames, and source-tree `sys.path` injection were removed. CLI locations are
now explicit arguments or environment variables.

Synthetic unit tests were added for taxonomy mappings, active masking,
full-audio aggregation, label-aware checkpoint expansion, deterministic
calibration splits, regression gates, and sharding. Documentation and portable
packaging were added in the staging repository; the original development trees
were not modified.

This provenance note is technical, not a legal ownership determination. Public
release of the extraction has been authorized; reuse remains governed by
`LICENSE.md`, `NOTICE.md`, and applicable third-party terms.
