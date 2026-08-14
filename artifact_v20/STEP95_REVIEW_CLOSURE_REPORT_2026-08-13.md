# Step 95 MNLI terminal-bridge review closure

## Decision-changing condition audited

Step 95 prospectively tested the remaining stronger reviewer condition on one
new task: four independently trained, raw-input, no-test-reference learned
adapters; one literal exact-match similarity for acquisition and posterior
evidence; material terminal and active-minus-fixed harm; and exact-zero
fixed-query effects.  MultiNLI, eight public checkpoint revisions, all data
partitions, the six-root roster rule, training recipe, complete 48-cell grid,
top-eight verification, gates, and a one-test stopping rule were fixed before
any MultiNLI outcome was inspected.

## What passed on development verification

- The deterministic rule selected `typeform_distilbert` and a six-root roster
  spanning DistilBERT, DeBERTa, RoBERTa, MiniLM, BERT-mini, and BERT-tiny.
- Each of four checkpoints contains 14,767,874 learned parameters and has a
  distinct SHA-256 digest.
- Runtime consumes premise/hypothesis text only; no test reference, item
  lookup, peer output, pool, trajectory, posterior, selector state, or separate
  evidence matrix is accepted.
- The same literal `1[a=b]` function drives acquisition and posterior evidence.
- Aliases are coordinate-wise non-improving; on verification they lose only
  0.724--0.895 accuracy points.
- Ordered paths change in 99.33%, query sets in 99.20%, and terminal roots in
  21.13% of 1,500 verification runs.
- Cumulative regret rises by 0.050557 with paired-bootstrap interval
  [0.042609, 0.058891] and one-sided sign-flip p=9.9999e-6.
- Terminal and active-minus-fixed effects equal +0.04987 percentage points,
  with interval [+0.02640, +0.07307] and one-sided p=1.99998e-5.
- Fixed-query terminal/cumulative effects are exactly zero and fixed root
  histories are bitwise identical.

## What failed

The terminal and active-minus-fixed means are only +0.04987 percentage points,
one tenth of the preregistered +0.5-point materiality gate.  The matching search
means are +0.0375 points.  In addition, replayed search loss exceeds the
one-point cap for three aliases (maximum 1.3867 points), despite threshold and
verification losses staying within one point.

The literal decision is `NO_GO_DEVELOPMENT_GATE_STOP`.  No confirmatory
execution lock was created and the sealed `validation_matched` outcome was not
opened.  `validation_mismatched` remains unused and is not a replacement test.

## Independent closure

An independent validator re-hashes the authorities, code, arrays, and four
checkpoints; reconstructs the parent and roster; checks the complete 48-cell
grid and top-eight selection; reruns the selected 400 search and 1,500
verification trajectories; reconstructs bootstrap, sign-flip, all development
gates, and the no-go decision; and verifies the complete absence of Step 95
confirmatory outputs.  All checks pass.

The study therefore strengthens the evidence that learned source-faithful
aliases can alter nearly every acquisition path and increase anytime cost, but
it does not close the material terminal or operational bridge.  The result is
retained rather than replaced by another task.
