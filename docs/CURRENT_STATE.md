# Current state and exact handoff boundary

Date: 2026-08-14 (Asia/Seoul)

## Submission evidence

The validated V20 artifact is the submission-grade evidence boundary.  Its
paper PDF has SHA-256
`5774968956ba2144414756a8c0cdba6d4ad7ac494b1cc9088d7a99c0fb5cff4b`.
Its release manifest has SHA-256
`b405b9d72a7795415e143a273e292df1dd07fc48ebb35b1dd434cdb790bdc0fc`.
The original V20 ZIP has SHA-256
`14ce5abfd32091e4219c89308670ce64c1d8effcb59606db54c0729382ba8388`.

### Step 104: promoted held-out learned confirmation

Literal decision:
`GO_STEP104_IMDB_HELDOUT_LEARNED_SAME_S_TERMINAL_PRIMARY`.

- disjoint held-out rows: 13,000;
- paired runs: 3,000;
- terminal and active-minus-fixed effect: +0.7024 percentage points,
  bootstrap 95% interval approximately [+0.6155,+0.7882];
- cumulative effect: +0.0279467;
- path-change rate: 99.933%;
- final-root-change rate: 60.667%;
- fixed-query terminal and cumulative effects: exactly zero;
- all locked gates passed.

This is development-informed held-out evidence, not a fully untouched
cross-task preregistration and not a live registry attack.

### Step 105: retained prospective negative

Literal decision: `NO_GO_RETAIN_STEP105_PROSPECTIVE_NEGATIVE`.

- Yelp terminal and active-minus-fixed effect: +0.3527 points with positive
  inference;
- cumulative effect: +0.0179247;
- path-change rate: 99.9%;
- fixed-query effect: exactly zero;
- failed only the two predeclared +0.5-point materiality gates.

No threshold relaxation or replacement outcome was authorized.

## Post-V20 research record

These experiments are retained for audit and future research.  They are not
promoted into the current manuscript claim.

### Step 113 BoolQ

Literal decision: `NO_GO_STEP113_PRIMARY_CONFIRMATION`.

- terminal and active-minus-fixed effect: +3.9619 points;
- all effect, mediation, directionality, and inference gates passed;
- coordinate-wise non-improvement passed;
- alias losses were 49, 53, 56, and 49 of 3,270 rows
  (1.50--1.71 points), so the locked one-point quality cap failed;
- an explicitly post-outcome rank-cap diagnostic cannot rescue the primary
  decision.

The complete ledger, arrays, heads, and independent validation are under
`post_v20/step113_boolq/`.

### Step 114 Amazon Polarity

Literal decision: `STOP_STEP114_DEVELOPMENT_NOT_CERTIFIED`.

- the exact public data revision was partitioned and the outcome labels were
  sealed;
- a Transformers compatibility shim was separately documented and locked
  before any outcome inference;
- all alias, quality, checkpoint, and non-improvement gates passed;
- the sole failed development gate was `designated_parent_unique_best`:
  the parent and one challenger both scored 0.94950 on the 4,000-row safety
  block;
- no primary-outcome model inference, prediction inspection, label opening, or
  outcome analysis occurred.

The protocol forbids a replacement task, roster, quantile, or budget after a
stop.  The outcome is intentionally omitted from this handoff.

## Remaining scientific boundary

The remaining uncertainty is whether a completely untouched task and registry
can show a material terminal effect under a one-time, outcome-free,
source-faithful pipeline.  This affects external generality and the difference
between a marginal and confident accept; it does not invalidate the causal
mechanism already established.

No new experiment should be presented as a retroactive rescue of Step 105,
Step 113, or Step 114.  A future study requires a new scientific rationale,
new lock, and explicit accounting for the accumulated failed attempts.
