# Step 90 confirmatory preregistration: high-confidence executable adapters

Date locked: 2026-08-12 (Asia/Seoul)

## Confirmatory question

Can four answer-preserving parameter adapters, each executable from a raw text
pair through a pinned public parent and containing no item lookup table, divert
Select-LLM toward parent-high-confidence items and significantly increase
terminal root regret on both sealed MNLI and sealed QQP holdouts?

This is a prospective two-task transfer test.  The earlier QNLI adapter failure
remains part of the evidence and may not be omitted: it changed 96.9% of paths
but had terminal delta -0.068 percentage point with an interval crossing zero.

## Frozen endpoint and intervention

The MNLI parent is `cross-encoder/nli-distilroberta-base` at revision
`b14d131f9d32668a5e6a982729b57ff6ed5dfcbd`; the QQP parent is
`cross-encoder/quora-distilroberta-base` at revision
`f62e7a4b20b97195c2868e53ec59126df5eac743`.  Exact safetensors hashes are in
the execution lock.

Each adapter is a standalone safetensors parameter file containing:

- a calibration-frozen parent-confidence threshold;
- one of four style IDs;
- the task class count.

At runtime, the endpoint loads the pinned parent, computes its hard answer and
confidence from the raw text pair, and returns the canonical hard answer unless
confidence is at least the threshold.  Above threshold, it returns the same
hard answer with its adapter-specific style view.  The utility parser extracts
the unchanged hard answer.  Therefore every alias has exactly the parent's
per-item hard-utility vector on any input, without reading a reference.

Runtime inputs are the raw text pair and the pinned parent's own logits only.
The endpoint reads no item ID, lookup table, reference, peer output, pool,
trajectory, or selector state.  These are calibration adapters sharing one
parent root, not independently trained full checkpoints.

## Complete calibration development

For each task separately, all 135 combinations of the following finite grid
were run on seeds 0--19 and retained:

- top-roster size `{8,12,20}`;
- budget `{20,30,50}`;
- Select-LLM temperature `{0.25,1.0,4.0}`;
- high-confidence trigger fraction `{0.05,0.10,0.20,0.30,0.50}`.

For a trigger fraction, the adapter threshold is the corresponding calibration
confidence quantile.  The 12 configurations with the greatest pilot terminal
delta were rerun on seeds 0--299.  Selection is maximum verified terminal
delta, then path-change rate, cumulative delta, and serialized configuration.
Every pilot and verification result remains in the task ledger.

The calibration-frozen conditions are:

- **MNLI:** official roots `[37,77,49,34,73,2,57,17]`, pool 400, budget 50,
  `tau=0.25`, trigger fraction 0.30, threshold
  `0.9867420665230863`.  Calibration terminal delta +1.4542 pp, approximate
  paired-bootstrap 95% interval `[+1.1667,+1.7575]` pp, path change 100%.
- **QQP:** official roots `[31,35,15,20,34,10,16,5]`, pool 400, budget 50,
  `tau=1.0`, trigger fraction 0.20, threshold
  `0.9997407698405986`.  Calibration terminal delta +1.3408 pp, approximate
  interval `[+1.2017,+1.4817]` pp, path change 100%.

Calibration condition selection used calibration labels and peer rows.  This is
disclosed and is distinct from runtime access.  No holdout label, peer row,
parent/adapter output, path, selected root, or regret was used.

## Frozen holdout protocol

- MNLI: 4,882 SHA-256-sealed matched-validation items.
- QQP: 20,121 SHA-256-sealed validation items.
- Seeds: 0--999 for each task.
- Per seed: 400 items sampled without replacement by
  `random.Random(seed).sample` and sorted in source position order.
- Budget: 50 on both tasks.
- Clean registry: eight frozen official roots plus the pinned parent.
- Refined registry: clean plus four adapter entries mapped to the parent root.
- Acquisition: exact-response Select-LLM with the task-frozen temperature and
  uniform initial entry mass.
- Stable SHA-256 query and unique-root tie policies are paired between clean and
  refined conditions.
- Deployment uses cumulative hard correctness and collapses tied entries to
  unique roots before tie resolution.
- Terminal root regret is best-original-root pool accuracy minus the deployed
  root's pool accuracy at budget 50.  Cumulative regret is secondary.

## Causal control and estimand

Force the clean query path on both registries and recompute every deployment.
Because adapter hard-utility vectors equal the parent's coordinate-wise and
ties are root-collapsed, complete fixed-query root paths plus terminal and
cumulative regret arrays must be exactly equal.  Any nonzero element invalidates
the implementation.

The primary per-task estimand is refined-active minus clean-active terminal root
regret.  The separately reported active-minus-fixed estimand subtracts the
corresponding fixed-query difference and must satisfy the same materiality
gate.

## Frozen statistics and multiplicity

For each task:

- 50,000 paired percentile-bootstrap resamples;
- 100,000 paired sign flips for the one-sided harmful alternative;
- all raw paired terminal/cumulative/fixed-query arrays retained.

Holm correction is applied across the two terminal sign-flip tests.  Both task
outcomes remain in the artifact regardless of sign.

## Per-task pass gate

Each task passes only if all are true:

1. parent and four adapter hashes match;
2. all alias hard-answer and utility vectors equal the parent's coordinate-wise;
3. fixed-query root paths and terminal/cumulative effects are exactly equal;
4. ordered path-change rate is at least 50%;
5. mean terminal-regret increase is at least 0.005 (0.50 percentage point);
6. terminal paired-bootstrap 95% lower endpoint is above zero;
7. Holm-adjusted one-sided sign-flip `p <= 0.05`;
8. mean active-minus-fixed terminal increase is at least 0.005 and its
   bootstrap lower endpoint is above zero.

Decision labels:

- both tasks pass: `GO_STRONG_EXECUTABLE_ADAPTER_TRANSFER`;
- exactly one passes: `GO_LIMITED_EXECUTABLE_ADAPTER_TRANSFER`, appendix-only;
- neither passes: `NO_CONFIRMATORY_EXECUTABLE_ADAPTER_TRANSFER`.

Only the strong decision permits a new main-text transfer claim.  No task,
threshold, roster, budget, temperature, seed, endpoint, statistic, or gate may
be changed after the execution lock.

## Claim boundary

A strong result would connect exact answer preservation, raw-input executable
adapter parameters, acquisition mediation, and terminal cost on two untouched
tasks.  It would still not show live registry admission, independent full
checkpoints, secret-label compromise, or universal harm.

