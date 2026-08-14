# Step 100 preregistration Amendment A: geometry defaults and quality scope

## Timing

This amendment is written after deterministic data separation but before any
Step 100 root prediction, root accuracy, parent gap, oracle effect, adapter
training, adapter score, grid effect, or held-out outcome is computed or
inspected.  The original protocol and stage-0 hashes remain immutable.

## Clarifications

1. The pre-grid oracle geometry screen on both `selector_search` and
   `selector_verify` uses budget 10, temperature .025, pool size 500, and the
   already frozen search/verification seed ranges.  These are the central grid
   values and were omitted as explicit geometry defaults in the original text.
2. The one-point coordinate-wise quality gate is applied conservatively on all
   five labeled development partitions, including `adapter_train`, rather than
   only the four partitions downstream of training.  This strengthens rather
   than relaxes the original scientific gate.
3. No dataset, row, label, roster, revision, model, mapping, training seed,
   threshold grid, selector grid, outcome, or success criterion changes.
