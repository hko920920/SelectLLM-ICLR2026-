# Step 89 stage-0 protocol: executable QNLI adapter audit

Date locked: 2026-08-12 (Asia/Seoul)

## Purpose

This stage creates an outcome-separated development/holdout split for a new
experiment that was not part of Steps 1--88.  The experiment asks whether
independently executable, answer-preserving parameter adapters can change an
active model-selection transcript and terminal root regret on QNLI without
reading any holdout reference or using an item lookup table.

This is not yet the confirmatory analysis plan.  Stage A may develop and select
one model/adapter/selector configuration using only QNLI train and the
calibration half created here.  Before Stage B opens the holdout outcomes, the
complete configuration, code, model revision, adapter weights, roster, pool,
budget, seeds, estimands, and decision gates must be written and SHA-256
locked.  Stage B may run exactly that configuration once; every primary result
must be retained irrespective of sign.

## Frozen sources

- Dataset: `nyu-mll/glue`, configuration `qnli`, revision
  `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c`.
- Official MODEL SELECTOR QNLI matrices:
  `external/model-selector/resources/datasets/glue/qnli`.
- Official QNLI acquisition parameter: `epsilon=0.44` from
  `external/model-selector/resources/experiment_configs/method_comparison/glue_qnli.json`.
- Candidate executable parent: `cross-encoder/qnli-distilroberta-base`, revision
  `7dd04ee0a6040c06fb381ad7edcb8585f4d937fd`, loaded from safetensors.

The parent revision is named here before any parent prediction is computed.
Changing it after Stage 0 invalidates the confirmatory label.

## Frozen split

For each QNLI validation item with official integer `idx`, compute

`SHA256("step89-qnli-v1|" + decimal(idx))`.

The item is calibration when the unsigned first eight digest bytes interpreted
big-endian are even and sealed holdout when they are odd.  Source order is
preserved within both partitions.  The rule may not be changed after seeing
counts, model predictions, or selection outcomes.

The Stage-0 program writes:

- a labeled calibration package;
- an input-only holdout package;
- holdout outcomes in a separately named sealed directory;
- a manifest containing source hashes, split counts, and output hashes.

No per-item holdout label or prediction may be printed by Stage 0.

## Information inspected before this lock

Before this file was created, the following were inspected: repository source
code, filenames, file sizes, SHA-256 hashes, array dimensions, class labels,
aggregate class counts, the official epsilon, dataset feature names, and model
metadata.  No prediction from the frozen parent or any proposed adapter was
computed.  No clean/refined QNLI acquisition path, selected root, regret, or
holdout attack effect was computed or printed.  The official 90-column matrix
has not previously been used as an empirical task in this project.

This is an analyst-side outcome lock, not a claim that the public QNLI labels
were cryptographically inaccessible before today.

## Allowed Stage-A development

Stage A may use:

- QNLI train examples and labels;
- the labeled calibration package;
- calibration rows of the official candidate matrix;
- the frozen parent model's calibration outputs and hidden states;
- finite, declared searches over roster size, adapter construction, pool size,
  budget, and random seed.

Stage A may not use holdout labels, holdout official-candidate rows, holdout
parent/adapter outputs, or holdout selector trajectories.  Its complete search
ledger and all calibration outcomes must be retained.

## Claim boundary

Even a successful Stage B would establish an independently executable
parameter-adapter bridge on one untouched task.  It would not establish live
production-registry admission, hidden-task compromise, or universal harm.

