# Step 114: final prospective Amazon Polarity terminal bridge

Date: 2026-08-14 (Asia/Seoul)

## Status

Step 114 is a separate, final prospective task.  It retains the Step 113 BoolQ
NO_GO: that study passed every causal/effect gate but exceeded the prespecified
one-point alias-quality cap because a train-fixed p99 threshold triggered on
2.26--3.03% of the shifted outcome inputs.  A labeled post-outcome rank-cap
diagnostic is not confirmatory and is not used as evidence.

The one design correction tested here is outcome-blind target-distribution
threshold calibration plus a pre-outcome trigger-count certificate.  No Step
114 row, model prediction, or selector effect was accessed before this protocol
and its implementation were fixed.  If development, pre-outcome, or outcome
gates fail, the family stops; no replacement task, roster, quantile, or budget
is allowed.

## Metadata-only task and registry

Amazon Polarity was selected before row access because it was absent from all
prior project studies, has a 400,000-row official test population, and has at
least four immutable public classifiers with a unique public-card parent in the
chosen roster.  Dataset revision:
`9d9c45c18f8c3cf1b23a3c27917b60cbf28f3289`.

Clean registry order:

1. `AdamCodd/distilbert-base-uncased-finetuned-sentiment-amazon` at
   `af15ca2e0c7a2779f19dc242a7e87817393ab797` (parent; card accuracy .961);
2. `Etelis/amazonPolarity_ALBERT_5E` at
   `099a705d29cbaf47df683b10070cb87c0560fb8b` (card accuracy .9533);
3. `fabriceyhc/bert-base-uncased-amazon_polarity` at
   `36abc4b1e41b52cb0904d666ea522e9434bca998` (card accuracy .9465);
4. `quintonpyx/distilbert-amazon-polarity` at
   `d8abbd64ad8206b7e680bcf6a8b0a7f2938321b3` (card accuracy .940).

All roots receive only the raw `content` field, truncated to 256 tokens.
Dataset-native label 0 is negative and label 1 is positive.  Models with
semantic config labels expose that map directly; the Etelis card is trained on
the same native Amazon Polarity labels and uses the identity index map.

## Single test-population split and sealing

Before inspecting any row, a fixed NumPy PCG64 seed selects 24,000 source
indices without replacement from the 400,000-row test parquet and assigns them
without replacement to:

- 8,000 labeled `error_head_train` rows;
- 4,000 labeled `safety` rows;
- 4,000 label-free `target_threshold` rows;
- 8,000 `primary_outcome` rows whose labels are sealed separately.

Only content and opaque UID are stored for the last two blocks.  The target-
threshold labels are never read by any later stage.  Outcome labels and label
statistics remain unopened until the final pre-outcome lock.

## Frozen endpoint and structural quality certificate

The error-head recipe is exactly the already-locked Step 113 recipe: freeze the
parent, concatenate final first-token representation and two logits, standardize
on error-head-train rows, and train four seed-specific balanced-bootstrap linear
error heads for 30 epochs with AdamW (`lr=1e-3`, `weight_decay=1e-3`, batch 256).
The recipe, architecture, and loss are not searched on Amazon Polarity.

Each threshold is the empirical 99.3rd percentile (`higher`) of its head scores
on the 4,000 label-free target-threshold inputs.  This percentile was fixed
after diagnosing Step 113's train-to-outcome score shift but before accessing
any Amazon row.  A triggered alias emits private abstention code 2, 3, 4, or 5;
otherwise it emits the parent hard label.  It therefore cannot improve
coordinate-wise exact-match utility.

Development GO requires immutable models, distinct clean responses, a unique
best parent on safety, four distinct nonempty learned heads, nonempty safety
triggers, at most one-point safety loss per alias, and exact checkpoint replay.
No selector is run on development, safety, or target-threshold rows.

After all 8,000 primary root and alias predictions are produced without labels,
pre-outcome GO additionally requires each alias to trigger between 1 and 80
rows.  This bound is label-free and mathematically guarantees that an abstaining
alias can lose at most 80/8,000 = 1 percentage point, whatever the sealed
answers are.  Failure stops before label opening.

## Frozen selector and primary decision

The exact same literal equality defines output-output acquisition groups and
reference-output posterior feedback.  The clean registry has four roots; the
refined registry appends four aliases mapped to root zero.

- pool size 500;
- annotation budget 2;
- temperature .05;
- 3,000 paired seeds 1,216,000 through 1,218,999;
- canonical locked Step-93 query/root tie rules;
- 10,000 paired bootstrap replicates and 100,000 one-sided paired sign flips.

The primary decision directly matches the review's remaining terminal bridge.
GO requires:

1. all predata, split, model, head, threshold, prediction, code, and outcome
   bindings pass;
2. aliases are response-distinct, coordinate-wise non-improving, and lose no
   more than one point each;
3. at least half of acquisition paths change;
4. fixed-query root histories are exactly identical and fixed terminal and
   cumulative effects are exactly zero;
5. terminal and active-minus-fixed terminal means are each at least +.5
   percentage points, have positive 95% paired-bootstrap lower bounds, and have
   one-sided paired sign-flip p-values at most .05.

Cumulative regret, switch-count direction, trigger precision, AUROC, and two
hash-defined outcome halves are reported diagnostics, not gates.  This is
deliberate: Step 114 tests the reviewer's terminal score-changing condition,
while the manuscript's main T2 experiments already establish cumulative harm.

## Claim boundary

A GO establishes one fully fixed cross-task raw-input endpoint whose outcome
uses no item lookup, outcome reference, peer output, or selector state, uses the
same similarity for acquisition and evidence, produces material acquisition-
mediated terminal harm, and obeys a one-point quality cap.  It still does not
establish live admission, secret-benchmark compromise, universal harm, or an
independently trained full base model.
