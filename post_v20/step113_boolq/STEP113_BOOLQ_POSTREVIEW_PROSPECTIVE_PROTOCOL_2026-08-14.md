# Step 113: post-review prospective BoolQ confirmation

Date: 2026-08-14 (Asia/Seoul)

## Status and purpose

This is a new, post-review experimental chain.  It does not relabel any result
from Steps 105--112 and it does not erase their stops or no-go outcomes.  It is
motivated by the remaining review question: can a raw-input, no-item-lookup,
same-similarity endpoint produce material terminal harm on a task and registry
that were untouched by the preceding development chain?

The answer is determined by one frozen BoolQ outcome.  If any pre-outcome
validity gate fails, or if the confirmatory outcome fails, Step 113 stops.  No
alternative BoolQ roster, threshold, budget, seed block, or replacement task is
allowed after this lock.

## Metadata-only task and registry selection

Before accessing any BoolQ row, the task was selected by the following public
metadata rule:

1. the task token did not occur in the project's prior study source or reports;
2. it is a paired-text, finite-label task with a public official train split and
   a distinct official validation split;
3. at least four public sequence-classification checkpoints have immutable
   revisions and an explicit `False=0, True=1` label map;
4. their public model cards identify a unique strongest parent and at least one
   runner-up separated by more than two percentage points.

BoolQ is the first retained task under that rule.  The immutable dataset
revision is `35b264d03638db9f4ce671b711558bf7ff0f80d5`.  The clean registry order is:

1. `nfliu/deberta-v3-large_boolq` at
   `9df67c219a86b2611c177f288b8ccc82d7b97707` (parent; card accuracy .8835);
2. `nfliu/roberta-large_boolq` at
   `efc939f590968d9b5055127d4aeb8a930ffa0826` (card accuracy .8569);
3. `nfliu/MiniLMv2-L6-H768-distilled-from-RoBERTa-Large_boolq` at
   `f31b1a94395e97a0472054793e5236fd8b7b2416` (card accuracy .7379);
4. `andi611/distilbert-base-uncased-qa-boolq` at
   `168ef0953f2bf4083670a8519a14db558a6f043c` (card accuracy .7315).

All candidates receive `(question, passage)` through their native tokenizer,
truncated to 256 tokens.  Native label index zero is `False` and index one is
`True`.  The public card values choose the registry but are not project-side
row-level outcome inspection.

## Data sealing

The official train split has 9,427 rows and the official validation split has
3,270 rows.  A stage-0 program, bound by the predata lock, downloads the exact
parquet blobs.  It writes validation questions/passages and opaque UIDs to an
input-only JSON package and writes answers separately to a sealed NPZ.  No
validation answer, label statistic, root prediction, or selector effect may be
printed or inspected before the pre-outcome lock.

Train rows are ordered by SHA-256 of the immutable dataset revision, row UID,
and `step113-boolq-train-partition-v1`, then split without replacement into:

- 6,599 `error_head_train` rows;
- 1,414 label-free `threshold_calibration` rows;
- 1,414 `safety` rows.

No selector replay is run on any train partition.

## Frozen raw-input endpoint

The parent checkpoint is frozen.  On each raw `(question, passage)` pair it
produces its two logits and final-layer first-token representation.  Their
concatenation is standardized using only `error_head_train` rows.  Four
independent linear error heads are trained to predict whether the parent hard
answer is wrong.  Each head uses a seed-specific balanced bootstrap, 30 fixed
epochs, AdamW, learning rate `1e-3`, weight decay `1e-3`, and batch size 256.
No selector output or validation row is available to training.

For each head, its threshold is the empirical 99th percentile (`higher`
quantile) of scores on the label-free threshold-calibration inputs.  A triggered
alias emits a private abstention code (`2`, `3`, `4`, or `5`); otherwise it emits
the parent's hard label.  Thus every endpoint is executable from raw input,
aliases are response-distinct, and abstention can never improve coordinate-wise
exact-match correctness.

The pre-outcome development decision is GO if and only if:

1. all exact revisions load and all four clean response vectors are distinct;
2. the designated parent is the unique best clean root on the safety partition;
3. every alias triggers at least once and at most 2% of safety rows;
4. every alias is coordinate-wise non-improving and loses at most 1 percentage
   point of safety accuracy relative to the parent;
5. all four learned checkpoint files are nonempty, mutually hash-distinct, and
   reconstruct their recorded safety scores exactly.

Error AUROC, error precision, trigger overlap, and every safety statistic not
listed above are diagnostics only.  In particular, there is no Step-112-style
minimum precision gate and no effect-based configuration selection.

## Frozen selector and confirmatory outcome

The selector is the equation-faithful literal-exact-match specialization used
elsewhere in the paper.  The same equality function defines output-output
acquisition groups and reference-output posterior feedback.  The clean registry
has four entries.  The refined registry appends the four parent aliases and maps
all four to accountable root zero.

- pool size: 500;
- annotation budget: 2;
- temperature: .05;
- paired runs: 3,000, seeds 1,213,000 through 1,215,999;
- canonical query/root tie rules: the locked Step-93 rules;
- bootstrap replicates: 10,000;
- paired one-sided sign flips: 100,000.

Before opening validation answers, the code, task inputs, all four clean root
predictions, parent representations/logits, all four learned heads, thresholds,
alias responses, and seed block are hash-bound in a pre-outcome lock.

The single confirmatory outcome is GO if and only if all of the following hold:

1. all bindings and prediction replays agree;
2. aliases are response-distinct, coordinate-wise non-improving, and each loses
   no more than 1 percentage point of outcome accuracy;
3. at least half the paired acquisition paths change;
4. clean-parent to refined-challenger switches are more frequent than the
   reverse;
5. clean and fixed-query refined root histories are identical and both fixed
   terminal and cumulative effects are exactly zero;
6. terminal and active-minus-fixed terminal means are each at least +.5
   percentage points, their 95% paired bootstrap lower bounds are positive, and
   their one-sided paired sign-flip p-values are at most .05;
7. cumulative regret mean and its 95% paired bootstrap lower bound are positive.

Hash-defined validation halves are reported as diagnostics but are not gates.
No gate may be relaxed after the answers are opened.

## Claim boundary

A GO would close the narrow empirical criticism by establishing one prospective
cross-task, raw-input, no-validation-reference, same-similarity, acquisition-
mediated terminal effect.  It would not establish live registry admission,
secret-benchmark compromise, universal harm, or independent full-base model
training.  A STOP or NO_GO remains evidence and cannot be replaced inside this
experimental chain.
