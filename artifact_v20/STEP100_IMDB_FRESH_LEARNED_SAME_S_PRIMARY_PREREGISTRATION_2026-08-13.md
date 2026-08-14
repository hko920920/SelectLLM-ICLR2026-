# Step 100 IMDB fresh learned same-s terminal primary preregistration

## Status and purpose

This protocol is written before any IMDB example, label vector, model
prediction, adapter score, development effect, or held-out outcome is loaded or
inspected in this project.  Only the public dataset card and split metadata,
model cards, configurations, repository revisions, and prior project source
audit have been inspected.

The prior outcomes are not reopened or relabeled.  Step 96 remains a locked
primary quality-cap NO-GO with a post-confirmatory conservative secondary.
Step 95 remains a MultiNLI development stop.  Step 98 remains a development
NO-GO.  Step 99 remains a development-informed sealed ANLI negative.  IMDB has
not appeared as an outcome-bearing task in the manuscript or project search.

Step 100 is the single fresh task requested by external review: a learned
raw-input endpoint, no held-out reference or item lookup, the same literal
similarity for acquisition and evidence, terminal and active-minus-fixed harm,
and exact fixed-query mediation.  A failure ends Step 100; there is no alternate
task, split, roster, model, adapter family, or held-out condition.

## Frozen dataset and outcome separation

- Dataset: `stanfordnlp/imdb`, config `plain_text`.
- Dataset revision:
  `e6281661ce1c48d982bc483cf8a173c1bbeb5d31`.
- Public metadata: `text` and binary label `(negative, positive)=(0,1)`;
  official test has 25,000 examples, 12,500 per class.
- Salt: `step100-imdb-fresh-learned-same-s-primary-v1`.

Within each class of the official test split, SHA-256 of revision, source index,
text, label, and salt orders rows.  The first 6,000 rows per class form the
12,000-row development package; the remaining 6,500 per class form one 13,000-
row outcome.  The latter is written immediately as separate label-free inputs
and a sealed label file.  No held-out label, class-specific prediction, or
effect is inspected until the final pre-outcome lock is complete.

The development rows are assigned in hash order within each class:

| partition | rows per class | total |
|---|---:|---:|
| `adapter_train` | 3,000 | 6,000 |
| `threshold_calibration` | 500 | 1,000 |
| `threshold_safety` | 1,000 | 2,000 |
| `selector_search` | 500 | 1,000 |
| `selector_verify` | 1,000 | 2,000 |

All partitions and the sealed outcome are disjoint.  The official train and
unsupervised splits are not used.

## Frozen four-root registry

All roots use max length 256 and hard-label argmax.  Label mapping is identity.

1. parent: `lvwerra/distilbert-imdb`, revision
   `0fc02cd68445b599a9cb2da2368050e7fb31d29a`;
2. challenger: `textattack/distilbert-base-uncased-imdb`, revision
   `5b0f46c2fc4b86bf21f0ec0409bed77ee142b332`;
3. challenger: `textattack/albert-base-v2-imdb`, revision
   `e377b81678ba240cd835375c5853bb590e10e75a`;
4. challenger: `Harsha901/tinybert-imdb-sentiment-analysis-model`, revision
   `0cd5d1ac6c06eb0f5f81b7022f95691528c981fa`.

The public cards report 92.8% for the parent and approximately 88.0--89.2% for
the challengers.  Cards, not project outcomes, motivate this fixed strong-parent
registry.  No root, revision, mapping, tokenizer, or length can be replaced.

## Four independently trained raw-input endpoints

Four binary error adapters independently initialize from the pinned parent.
Embeddings and DistilBERT layers 0--3 remain frozen; layers 4--5,
`pre_classifier`, and `classifier` train to predict
`1[parent hard label != development reference]` from review text.

- training partition: `adapter_train` only;
- seeds: `100100+a`, `a in {0,1,2,3}`;
- two epochs and class-balanced bootstrap;
- max length 256;
- batch size 8, gradient accumulation 4;
- AdamW learning rate `2e-5`, weight decay `.01`;
- 10% linear warmup then decay;
- gradient norm cap 1.0;
- deterministic CUDA settings and mixed precision only on CUDA.

Each saved endpoint must contain at least ten million learned parameters and a
distinct SHA-256.  At runtime alias `a` emits the parent hard label unless its
error score exceeds its frozen threshold, when it emits unique out-of-label
token `ABSTAIN_a`.  Runtime cannot receive a label, reference, item ID, lookup,
peer output, pool, posterior, trajectory, selector state, split, or dataset ID.

## Development grid and one-condition selection

The grid is fixed before data.  Every cell uses four aliases, pool size 500,
the same literal `s(a,b)=1[a=b]` for output--output acquisition and
reference--output posterior evidence, and root-level regret.

- calibration correct-coordinate loss cap per alias:
  `.0025`, `.0050`, `.0075`;
- trigger cap per alias: `.025`, `.05`, `.10`;
- budget: `5`, `10`, `20`;
- temperature: `.01`, `.025`, `.05`.

This is the complete 81-cell grid.  For each loss/trigger pair, a threshold is
estimated only on `threshold_calibration`, then raised (never lowered) by the
smallest amount necessary on `threshold_safety` until realized loss is at most
.5%, the exact one-sided 99% Clopper--Pearson upper bound is at most 1%, and the
trigger cap holds.  Ties at a threshold are backed off completely.  Thresholds
then remain unchanged on search, verification, and outcome.

Before the grid, both selector partitions must show:

1. parent accuracy exceeds every challenger by at least 2.5 points;
2. parent error prevalence is at least 5%;
3. some challenger is correct on at least 15% of parent errors;
4. reference-aware oracle aliases produce at least +1 point terminal harm;
5. oracle parent-to-challenger transitions exceed the reverse; and
6. oracle fixed-query terminal/cumulative effects are exactly zero.

Failure stops before adapter/grid evaluation.  The oracle is a geometry screen,
not evidence for the reference-free intervention.

Search uses seeds `1000000:1000399`.  A cell is eligible only if it satisfies:

1. coordinate-wise non-improvement and loss at most one point per alias on all
   four labeled development partitions;
2. path-change rate at least 50%;
3. parent-to-challenger transitions exceed the reverse;
4. terminal and active-minus-fixed mean harm at least +.5 points;
5. cumulative regret mean positive; and
6. bitwise-equal fixed-query root histories with exactly zero fixed terminal and
   cumulative effect.

Select the eligible cell maximizing search terminal harm, then cumulative harm,
path-change rate, smaller calibration loss cap, smaller trigger cap, smaller
budget, and smaller temperature.  If none is eligible, stop without opening the
outcome.  All 81 cells remain reported.

The selected cell alone runs on `selector_verify` with seeds
`1001000:1002499`.  It must reproduce every search gate and additionally have
positive paired-bootstrap 95% lower bounds and one-sided paired sign-flip
`p<=.05` for terminal and active-minus-fixed terminal harm, plus a positive
cumulative lower bound.  Bootstrap repetitions are 10,000 and sign-flip
repetitions are 100,000, with seeds `100700/100701`, `100701/100702`, and
`100702/100703`.  Failure stops without opening held-out labels; no second cell
is tried.

## One-time sealed primary confirmation

After development GO, all four root predictions, four error-score vectors, four
alias vectors, and thresholds are generated from the 13,000 label-free inputs
and hashed.  A lock binds this protocol, dataset/input/sealed hashes, complete
grid and selected cell, all code and model hashes, raw prediction package, and
exactly 3,000 paired seeds `1003000:1005999`.  Only then is the sealed outcome
opened once.

Every implementation, independence, geometry-independent quality, mechanism,
and outcome gate used for verification is mandatory on the held-out outcome,
except that the development-only parent-gap/oracle screen is not reimposed.
Test quality uses exact counts `loss_count <= floor(.01*13000)=130` per alias.
Test inference uses seeds `100800/100801`, `100801/100802`, and
`100802/100803` for terminal, active-minus-fixed terminal, and cumulative
effects.

The only success label is
`GO_STEP100_IMDB_FRESH_LEARNED_SAME_S_TERMINAL_PRIMARY`.
Any failed gate yields `NO_GO_RETAIN_STEP100_NEGATIVE`.  No failed outcome may be
replaced by another split, task, roster, condition, threshold, or seed range.

## Interpretation boundary

A pass is a before-data, fresh-task, held-out-outcome confirmation for four
independently learned raw-text adapters under a source-faithful same-similarity
selector, with terminal and fixed-query mediation gates.  It is stronger than a
lookup wrapper or decoupled-view intervention.  It does not establish live
registry admission, secret-label compromise, a separately trained base model,
average harm, or production attack.
