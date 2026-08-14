# Step 90 stage-0 protocol: high-confidence executable-adapter transfer

Date locked: 2026-08-12 (Asia/Seoul)

## Motivation and separation from Step 89

Step 89 remains a failed confirmatory QNLI result: answer-preserving scalar
calibration adapters changed 96.9% of paths but improved terminal regret by
0.068 percentage point on average.  That result may not be retargeted or
discarded.

This follow-up tests a scientifically different, owner-input-only mechanism on
two untouched tasks.  Rather than emitting a confidence-bearing response on
almost every item, each adapter emits a distinct style view only when the
parent's own confidence exceeds a calibration-frozen threshold.  The intended
mechanism is diversion toward parent-easy items; the hard answer remains
unchanged by architecture.  MNLI and QQP are both locked before any prediction
from their frozen parents or any selector outcome is computed.  Both tasks must
pass the later confirmatory gate for a strong transfer claim.

## Frozen sources and parents

- Dataset: `nyu-mll/glue` revision
  `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c`.
- MNLI official MODEL SELECTOR matrix: 9,815 matched-validation items, 82
  candidates, official epsilon 0.43.
- QQP official MODEL SELECTOR matrix: 40,430 validation items, 101 candidates,
  official epsilon 0.47.
- MNLI parent: `cross-encoder/nli-distilroberta-base`, revision
  `b14d131f9d32668a5e6a982729b57ff6ed5dfcbd`, safetensors SHA-256
  `9df3eb5d37118f952f4ba4fb46fde6889e3a9ccedeee0bad09b0110fc64c5c29`.
- QQP parent: `cross-encoder/quora-distilroberta-base`, revision
  `f62e7a4b20b97195c2868e53ec59126df5eac743`, safetensors SHA-256
  `73fd14ad7d08f3ef30eb25841c8f4ba89e91230f48159279233b37015ccb33fb`.

Parent identity and revision cannot change after this file.

## Frozen splits

For task `t` and official integer item `idx`, compute

`SHA256("step90-high-confidence-v1|" + t + "|" + decimal(idx))`.

Calibration receives items whose unsigned first eight digest bytes are even;
sealed holdout receives odd items.  Source order is preserved.  Stage 0 writes
a labeled calibration package, an input-only holdout package, and labels plus
official candidate rows to a separately named sealed NPZ for each task.

## Information status

Before this lock, only source code, model/dataset metadata, array shapes,
available label symbols, file hashes, and Step 89's complete negative result
were inspected.  No MNLI or QQP parent prediction, confidence, adapter output,
active path, selected root, regret, or attack effect was computed.  These two
official task matrices have not been used in earlier project experiments.

## Stage-A permission and Stage-B prohibition

Stage A may use each task's labeled calibration half and calibration official
candidate rows to choose one configuration from a finite, fully retained grid.
The adapter runtime rule itself may read only the pinned parent's raw-input
confidence and its own frozen threshold/style parameters.  Stage A may not
read either task's holdout label, peer row, parent output, adapter output, pool,
or selector trajectory.

Before Stage B, all code, model revisions, task configurations, thresholds,
adapter files, rosters, seeds, estimands, multiplicity correction, and pass
gates must be SHA-256 locked.  Stage B opens both tasks once and retains every
result, including nulls and improvements.

## Claim boundary

Even if both tasks pass, this remains an executable parameter-adapter attack on
public benchmark inputs.  It is not live registry admission, hidden-task
compromise, or an independently trained full checkpoint.

