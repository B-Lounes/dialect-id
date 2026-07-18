# Operations

## Environment

Set data, run, and cache locations outside the repository:

```bash
export DIALECT_ID_DATASET_ROOT=/path/to/dataset
export HF_HUB_CACHE=/path/to/model-cache
export RUN_DIR=/path/to/run
```

Keep checkpoints, predictions, logs, reports, SQLite trackers, and scheduler
outputs in those external locations. The repository `.gitignore` also rejects
the common forms of these artifacts.

## Distributed launch

[`recipes/slurm/train_ddp.sbatch`](../recipes/slurm/train_ddp.sbatch) is a
portable template: it deliberately has no account, partition, QoS, hostname, or
site-specific module commands. Submit it with site settings, for example:

```bash
sbatch --account=<account> --partition=<partition> \
  --nodes=2 --gpus-per-node=8 \
  --export=ALL,DATASET_ROOT=/path/to/data,RUN_DIR=/path/to/run,MODEL_TYPE=wavlm,MODEL_NAME_OR_PATH=microsoft/wavlm-base-plus \
  recipes/slurm/train_ddp.sbatch
```

The template uses `torchrun`; tune CPU, memory, accelerator, and network
settings for the cluster. A local dry run should start with one process and a
tiny `--max-train-examples`/`--max-train-steps` configuration.

## Release discipline

1. Evaluate each candidate on the same named splits and retain rank shards.
2. Merge confusion matrices and build per-dialect diagnostics.
3. Tune logit bias on a deterministic calibration partition only.
4. Run the evaluation and calibration gates against the accepted baseline.
5. Treat any missing watched dialect as missing evidence, not a pass.
6. Record active output masks and calibration sidecars with the checkpoint.

The tools print machine-readable summaries and return nonzero status from gate
commands when release is not recommended.
