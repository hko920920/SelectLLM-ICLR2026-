# Step 89 confirmatory preregistration: executable answer-preserving adapters

Date locked: 2026-08-12 (Asia/Seoul)

## Question and claim boundary

This is a single-condition confirmatory test on the sealed QNLI holdout created
before any parent or adapter prediction was computed.  It tests whether four
parameter files that are independently executable with a pinned public parent,
read raw question/sentence input, and preserve the parent's hard answer on
every item can change Select-LLM acquisition and increase terminal root regret.

A pass supports an executable calibration-adapter bridge on one untouched task.
It is not evidence of live production admission, hidden-task compromise, an
independently trained full checkpoint, or universal harm.

## Frozen model and adapter endpoint

- Parent: `cross-encoder/qnli-distilroberta-base` at revision
  `7dd04ee0a6040c06fb381ad7edcb8585f4d937fd`.
- Parent safetensors SHA-256:
  `8b0cfff5547faac6a89c58eca8e0d26b3c4df0022937d59accb417957747ce6c`.
- The one parent logit is mapped to QNLI label 0 when nonnegative and label 1
  otherwise, as fixed using calibration only.
- Each adapter stores one `log_scale` and the frozen confidence-bin count in a
  standalone safetensors file.  On raw input it computes the pinned parent's
  logit `z`, preserves `sign(z)` exactly, and serializes the same hard answer
  together with a quantized confidence obtained from
  `sigmoid(abs(z) * exp(log_scale))`.
- The four log scales are `[-2.0, -1.5, 1.5, 2.0]`, with eight confidence bins.
- The adapter response contains no item ID and performs no prompt lookup.  At
  runtime it reads no reference, peer output, pool, trajectory, or selector
  state.  Because its positive scale cannot change the logit sign, every
  adapter's per-item hard utility is equal to the parent's by construction.

The four adapter file hashes are recorded in the execution lock.  Calling these
full independent checkpoints is forbidden; they are independently executable
calibration adapters sharing the pinned parent root.

## Calibration development retained in full

The development search used only the 2,691-item labeled calibration split and
its official candidate rows.  It evaluated all 405 combinations of:

- top-roster size in `{8, 12, 20}`;
- budget in `{20, 30, 50}`;
- temperature in `{0.25, 1.0, 4.0}`;
- confidence bins in `{4, 8, 16}`;
- five declared four-adapter scale sets.

Every configuration used seeds 0--19.  The 12 configurations with the largest
pilot terminal delta were rerun on seeds 0--299.  The frozen selection rule was
maximum verified mean terminal delta, then path-change rate, cumulative delta,
and serialized configuration.  The complete ledger is retained.

The selected calibration condition has 12 official roots (source columns
`[24,58,15,87,19,63,41,77,27,42,66,71]`) plus the pinned parent, pool size
400, budget 20, `tau=0.25`, eight confidence bins, and the extreme scale set
above.  On 300 calibration runs its terminal-regret delta was 0.0022417, its
paired bootstrap 95% interval was approximately `[0.000675,0.003833]`, and
98.67% of ordered paths changed.  These are development outcomes, not
confirmatory evidence.

Condition selection used the calibration peer matrix; this must be disclosed.
No holdout peer row, label, parent output, adapter output, path, or regret was
used.  All 405 development results remain available, including null and
improving configurations.

## Frozen holdout protocol

- Holdout: all 2,772 items assigned by the Stage-0 SHA-256 split.
- Seeds: integers 0 through 999.
- Per seed: sample 400 holdout items without replacement using
  `random.Random(seed).sample`, sorted into source position order.
- Budget: 20 labels.
- Selector: the paper's exact-response Select-LLM acquisition with
  `tau=0.25`, uniform initial mass per submitted entry, and stable SHA-256 query
  and root tie policies.
- Clean registry: the 12 frozen official roots plus the parent, each returning
  its hard QNLI label.
- Refined registry: clean plus the four adapter entries, all mapped back to the
  parent root.
- The evaluator parses the hard QNLI answer for utility but the acquisition
  view retains the adapter's confidence-bearing serialized response.
- Deployment is root-level.  At each step it chooses among maximum cumulative
  hard-correctness entries and resolves ties over unique roots, so duplicate
  tie mass cannot create the primary effect.
- Terminal root regret is best-original-root pool accuracy minus deployed-root
  pool accuracy at budget 20.  Cumulative root regret is secondary.

The parent model is run once over the input-only holdout after this plan and all
code/input hashes are locked.  The sealed file is then opened for the single
confirmatory evaluation.

## Causal control

Force the clean query sequence on both registries and recompute deployment.
Because all aliases have the parent's complete hard-correctness vector and
deployment ties are collapsed to roots, refined-minus-clean fixed-query
terminal and cumulative arrays, and complete selected-root paths, must be
exactly identical.  Any deviation invalidates the implementation.

The primary active-minus-fixed estimand is

`(refined_active - clean_active) - (refined_fixed - clean_fixed)`

for terminal root regret.  Under the mandatory fixed-query equality it equals
the active terminal delta, but both are reported independently.

## Frozen inference

- Primary paired statistic: refined-active minus clean-active terminal root
  regret across the 1,000 paired pools.
- 50,000 paired percentile-bootstrap resamples, seed 890100.
- 100,000 paired sign flips for the one-sided harmful alternative, seed
  890101.
- The active-minus-fixed terminal statistic receives a separate 50,000
  bootstrap with seed 890300.
- There is exactly one confirmatory condition, so no across-condition
  multiplicity correction is required.  The 405 calibration conditions are
  development and cannot be relabeled as holdout tests.

## Frozen pass gate

`GO_EXECUTABLE_ADAPTER_BRIDGE` requires every item below:

1. parent and adapter file hashes match the execution lock;
2. all four adapter hard-answer and hard-utility vectors equal the parent's
   coordinate-wise on the complete holdout;
3. fixed-query terminal and cumulative deltas and root paths are exactly zero;
4. at least 50% of paired ordered acquisition paths change;
5. mean terminal-regret increase is at least 0.0015 (0.15 percentage point);
6. terminal paired-bootstrap 95% lower endpoint is above zero;
7. one-sided paired sign-flip `p <= 0.05`;
8. mean active-minus-fixed terminal increase is at least 0.0015 and its
   bootstrap lower endpoint is above zero.

Otherwise the decision is `NO_CONFIRMATORY_EXECUTABLE_ADAPTER_BRIDGE`.  A null
or improving result remains in the artifact and may not be replaced by another
holdout condition.

