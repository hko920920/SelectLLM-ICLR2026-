# Step 92 preregistration: learned contextual adapters on a new raw-input task

Locked: 2026-08-12T22:46:44+09:00

## Decision-changing question

Can a genuinely trained, independently executable parameter adapter, built
without item-level test references or prompt lookup, alter active acquisition
and increase terminal root regret on a previously unused task while the
fixed-query effect remains exactly zero?

This experiment addresses the score-changing condition recovered in Step 91.
It is not a request for a favorable re-review. The sealed result is retained
whether it is harmful, null, or beneficial.

## Pre-outcome status and task choice

The frozen task is AG News (`fancyzhx/ag_news`) at dataset revision
`eb185aade064a813bc0b7f42de02595523103ca4`. The public card reports 120,000
train and 7,600 test examples with four labels. A repository-wide text audit
found no prior AG News experiment or result in this project.

SST-2 was considered using metadata only, then rejected before any new model or
selector outcome was computed because the project history showed that SST-2
had already appeared in the earlier CODA audits. This correction is part of
the record and AG News may not be replaced after its outcomes are opened.

Before this lock, the AG News Hub identifier, revision, public split sizes, and
file names were inspected. No AG News row, label array, model prediction,
adapter response, active path, selected root, or regret value was downloaded,
deserialized, printed, or used for design selection.

## Frozen data separation

Every train example is assigned from its source index by
`SHA256("step92-learned-adapter-v1|train|<idx>")`:

- residues 0--6 modulo 10: clean-root model training;
- residues 7--8: learned calibration-adapter training;
- residue 9: Stage-A attack/configuration calibration.

The complete official test split is the sealed confirmatory holdout. Stage 0
writes its raw text into an input-only package and its labels into a separately
named sealed NPZ. Stage A may not import, open, hash-derived-index, or otherwise
read the sealed labels. It may use all labels in the three train partitions.

## Frozen raw-input model family

All endpoints use the public encoder
`distilbert/distilbert-base-uncased`, revision
`12040accade4e8a0f71eabdb258fecc2e7e948be`. The cached base
`model.safetensors` SHA-256 is
`5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063`.
The encoder is frozen.

The clean registry contains 12 independently optimized bottleneck
classification adapters. Each has LayerNorm, a 768-to-128 GELU bottleneck,
and a four-class output layer. Root `j` is trained with seed `92100+j` on the
first fraction of its own SHA-256 ordering of the model-training partition:

`[.02,.04,.08,.12,.18,.25,.35,.50,.65,.80,.90,1.00]`.

There is no artificial label corruption. Optimization is AdamW, 25 epochs,
batch 512, learning rate `1e-3`, weight decay `1e-4`, gradient clipping 1.0.
The finite parent candidates are roots 4--8. Parent choice is permitted only
from Stage-A calibration and every parent/configuration result is retained.
All 12 clean roots remain in every registry; no root is filtered by observed
accuracy.

## Genuine learned refinement

For each eligible parent, four contextual temperature adapters are optimized
independently on the adapter-training partition with seeds 92200--92203 and
deterministic bootstrap samples. Each contains LayerNorm, a 768-to-64 GELU
bottleneck, and a scalar log-temperature head. It is trained for 30 epochs
with AdamW (batch 512, learning rate `1e-3`, weight decay `1e-4`) to minimize
the parent's calibrated cross-entropy plus `1e-3` squared-log-temperature
regularization.

At runtime an adapter receives only the raw text, the frozen encoder state,
and its parent logits. Its strictly positive contextual temperature rescales
all parent logits, so its hard class is mathematically identical to the parent
on every input. Its response is the hard label alone below a frozen confidence
threshold and the hard label plus a quantized calibrated-confidence field
above threshold. The utility parser reads only the hard label.

Each saved safetensors file must contain at least 49,000 learned floating-point
parameters, nonzero tensor variance, a distinct hash, and no row-, item-, text-,
label-, or reference-indexed table. These are learned neural adapters, not the
three-value configuration files used in Step 90.

## Frozen Stage-A search

Stage A evaluates the complete Cartesian product on the train-calibration
partition:

- parent root: `{4,5,6,7,8}`;
- clean roster: all 12 preregistered roots;
- budget: `50`;
- Select-LLM temperature: `{0.25,1.0,4.0}`;
- response confidence bins: `{8,16,32,64}`;
- per-adapter high-confidence trigger fraction: `{.05,.10,.20,.30,.50}`.

Every configuration uses seeds 0--19 and a pool of at most 400 calibration
items. The 12 configurations ranked highest by mean terminal-regret delta,
then path-change rate, cumulative-regret delta, and serialized configuration
are rerun on seeds 0--299. The same ordering selects exactly one configuration.
All pilot and verification rows remain in the ledger. No test item or outcome
may be used for parent, roster, threshold, bin, budget, or temperature choice.

## Frozen confirmatory protocol

- Holdout: all 7,600 source-order AG News test inputs.
- Paired seeds: 0--999.
- Per-seed pool: 400 items sampled without replacement by
  `random.Random(seed).sample`, then sorted.
- Parent, temperature, confidence bins, and four thresholds:
  the single Stage-A-frozen configuration.
- Clean roster: all 12 roots; budget: 50, both fixed before Stage A.
- Clean registry: the frozen clean roots in the selected roster.
- Refined registry: clean plus the four learned calibration adapters mapped to
  the selected parent root.
- Acquisition: the equation-faithful exact-response Select-LLM implementation
  already independently audited in the project, with stable SHA-256 query and
  root tie rules.
- Deployment: cumulative acquired hard-label correctness; ties collapse to
  unique roots before deterministic root selection.
- Terminal root regret: best clean root's pool accuracy minus the deployed
  root's pool accuracy at the frozen budget.

The causal control forces the clean query sequence on both registries. Because
every adapter's hard answer equals its parent coordinate-wise and deployment is
root-collapsed, clean and refined fixed-query root paths, terminal regret, and
cumulative regret must be byte-identical. Any nonzero value invalidates the
implementation.

## Frozen inference and pass gate

The primary paired estimand is refined-active minus clean-active terminal root
regret. Active-minus-fixed terminal harm is reported separately. Statistics:

- 50,000 paired percentile-bootstrap resamples;
- 100,000 paired sign flips for the one-sided harmful alternative;
- all raw paths, roots, terminal, cumulative, and fixed-query arrays retained.

The single confirmatory condition passes only if all are true:

1. dataset, base encoder, clean-root, learned-adapter, code, and lock hashes
   match;
2. every learned adapter satisfies the parameter, variance, distinct-hash, and
   no-lookup requirements above;
3. adapter and parent hard labels are coordinate-wise identical on the entire
   holdout;
4. fixed-query root paths and terminal/cumulative effects are exactly equal;
5. ordered path-change rate is at least 50%;
6. mean terminal-regret increase is at least 0.005 (0.50 percentage point);
7. the paired-bootstrap 95% terminal lower endpoint is above zero;
8. one-sided paired sign-flip `p <= .05`; and
9. mean active-minus-fixed terminal increase is at least 0.005 and its
   bootstrap lower endpoint is above zero.

Pass decision:
`GO_SCORE_CHANGING_LEARNED_ADAPTER_BRIDGE`.

Otherwise:
`NO_SCORE_CHANGING_LEARNED_ADAPTER_BRIDGE`, unless an integrity invariant
fails, in which case the label is `INVALID_STEP92_IMPLEMENTATION`. Nulls and
improvements may not be retargeted, replaced, or omitted.

## Claim boundary

A pass would establish a preregistered, raw-input, no-test-lookup transfer to a
new task using nontrivially learned and independently executable adapter
weights, with terminal cost mediated entirely by active acquisition. It would
not establish live registry admission, secret-task compromise, universal
instability, or deployment prevalence. A failure leaves V12 centered near a
strong ICLR 6 and must remain in the artifact.
