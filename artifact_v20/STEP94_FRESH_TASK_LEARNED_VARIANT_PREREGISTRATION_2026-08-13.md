# Step 94 fresh-task learned-variant preregistration

Locked: `2026-08-13T12:08:46.9960290+09:00`

Status at lock: no row from the three official test splits, no test label, no
test prediction, and no selector outcome for these tasks has been downloaded or
inspected in this workspace.  A case-insensitive workspace search immediately
before lock returned no prior project use of `20_newsgroups`,
`amazon_massive_intent`, or `SetFit/sst5`.  Public repository metadata, file
lists, split counts, and a few dataset-card **training** examples were inspected
only to establish feasibility.  They play no role in configuration selection.

This is a single, finite, score-targeted confirmation following the retained
Banking77 negative.  It addresses that negative's diagnosed failure mode: the
attack changed 99.8% of paths and increased cumulative regret, but the top clean
roots were too quality-dense for a material terminal effect.  Step 94 therefore
predeclares a calibration-only root-spacing and task-selection rule before any
test outcome is materialized.  It is not authorization to keep opening tasks
until one succeeds.

## 1. Scientific question

Can an independently executable, learned raw-input variant of one accountable
root, built without test references, item lookup, peer output, or selector
state, cause material terminal root-selection harm in a source-faithful
single-similarity active model selector on one previously unused task?

The one similarity is, literally and everywhere,

\[
s(a,b)=\mathbf 1[a=b].
\]

The same function is used for candidate--candidate acquisition and
reference--candidate posterior evidence.  No confidence view, soft utility,
type-specific parser, or hidden deployment score is allowed.

## 2. Frozen finite task menu

Exactly three candidates are admitted.  They are ordered below only for
deterministic tie breaking.

1. `SetFit/20_newsgroups`, revision
   `f1b91292074e7cfb69be58b642d583ec262f30ed`; 20 classes; official train and
   test splits.  This mirror removes headers, signatures, and quotations as
   described in its dataset card.
2. `SetFit/amazon_massive_intent_en-US`, revision
   `f7672a018e8ceb37fc0184dcfbb7e665155ffea6`; 60 classes; official train,
   validation, and test splits.  Its upstream source is
   `AmazonScience/massive`, revision
   `ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740` (CC BY 4.0).
3. `SetFit/sst5`, revision
   `e51bdcd8cd3a30da231967c1a249ba59361279a3`; 5 classes; official train,
   dev, and test splits.

For tasks with an official validation/dev split, train and validation/dev are
concatenated in that order to form the development population.  The official
test split is never used for task, roster, parent, threshold, temperature, or
stopping decisions.

Frozen encoder: `distilbert/distilbert-base-uncased`, revision
`12040accade4e8a0f71eabdb258fecc2e7e948be`, with the locally cached
`model.safetensors` SHA-256
`5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063`.
The encoder is frozen and emits a 768-dimensional CLS representation from at
most 192 tokens.

## 3. Development partitions and test sealing

For each task, development index `i` is assigned by SHA-256 of
`step94-fresh-learned-variant-v1`, the immutable task ID, `development`, and
`i`.  Residues modulo 20 define:

- `0..10`: clean-root training (55%);
- `11..13`: variant-gate training (15%);
- `14..16`: Stage-A search (15%);
- `17..19`: Stage-A verification (15%).

All official test texts are saved separately as input-only files.  Their labels
are written to `external_data/step94_sealed/` and may be opened only after a
confirmatory execution lock binds the selected task and every learned weight.
Stage A is forbidden from importing, opening, hashing through an API that
returns, or otherwise accessing the sealed outcome arrays.  Stage 0 may report
only split counts and cryptographic hashes, never a row, label distribution,
prediction, accuracy, path, root, or regret.

## 4. Candidate-bank and calibration-only root spacing

Each task has a bank of 16 independently optimized bottleneck classifiers:

`LayerNorm(768) -> Linear(768,128) -> GELU -> Linear(128,C)`.

Training fractions are

`[.03,.05,.075,.10,.15,.20,.30,.40,.50,.60,.70,.80,.90,1,1,1]`.

Seeds are `94400 + 100*task_index + root_index`.  The final three use distinct
bootstrap orderings.  Every checkpoint must contain nonconstant learned tensors
and at least `98,000` trainable parameters.  A root is a real learned endpoint:
at replay it consumes raw text plus its frozen encoder and saved adapter only.

The clean registry contains exactly eight roots chosen from the bank using only
Stage-A search labels.  Three deterministic quality-gap templates are tried:

- dense: `[.005,.010,.020,.030,.050,.080,.120]`;
- mixed: `[.010,.025,.050,.080,.120,.180,.250]`;
- wide: `[.020,.050,.100,.150,.220,.300,.400]`.

For a template, the highest-search-accuracy bank root is included first.  Each
remaining target gap is filled, without replacement, by the root whose search
accuracy lies closest to `best_accuracy - target_gap`; ties use smaller bank
root ID.  Duplicate eight-root rosters are collapsed but retained in the
ledger.  The eligible alias parent set is the three highest-search-accuracy
members of each roster.  No test-set spacing or test accuracy is used.

## 5. Independently executable learned variants

For every eligible parent, four independently trained error gates are fit on
the gate-training partition.  A gate receives the frozen CLS representation
concatenated with the parent's internally computed logits and uses

`LayerNorm(768+C) -> Linear(768+C,128) -> GELU -> Linear(128,1)`.

Its binary target is whether the parent prediction is wrong on that training
example.  Seeds are `94600 + 100*task_index + 10*parent_bank_id + alias_id`;
each gate uses an independently salted bootstrap and class-balanced binary
cross-entropy.

Each submitted variant checkpoint contains a copy of the selected parent
classifier and its own learned gate.  It is replayable as one raw-text endpoint.
If the learned gate score exceeds its frozen threshold, variant `a` emits the
literal token `ABSTAIN_a`; otherwise it emits the parent's class label.  The
four tokens are outside the task's reference alphabet.  Thus, structurally and
without test labels,

\[
s(r,f_{variant}(x))\le s(r,f_{parent}(x))
\]

at every coordinate.  This makes fixed-query mediation an exact falsification
control while still giving four response-distinct learned model variants.

Forbidden runtime inputs are test reference, item index/ID, prompt lookup,
peer output, pool seed, trajectory, acquisition score, posterior, or selector
state.  A replay audit must call the saved endpoint from raw text and compare it
bitwise with the matrix used by the selector.

## 6. Frozen Stage-A search and task selection

For each task, the complete finite grid is:

- all distinct rosters from the three frozen gap templates;
- parent: the top three search-accuracy roots in that roster;
- selector temperature: `.05,.10,.25,.50,1,2`;
- maximum search utility loss per alias: `.0025,.0050,.0100`;
- maximum trigger fraction: `.05,.10,.20,.40`.

Thresholds are the largest score prefixes satisfying both caps, with midpoint
thresholds between distinct adjacent scores.  Selector pool size is 500 and
budget is 50.  Search seeds are `940000..940039`.  Within each task, the best
24 eligible conditions by mean active-minus-fixed terminal regret, then
terminal regret, cumulative regret, path-change rate, smaller utility-loss cap,
smaller trigger cap, template order, parent ID, and temperature are replayed on
verification seeds `940100..940599`.

Exactly one verified condition per task is retained using the same ordering.
The selected confirmatory task is the task whose retained condition has the
largest verification active-minus-fixed terminal regret, provided all of the
following calibration-only eligibility checks hold:

1. search and verification mean active-minus-fixed terminal effects are both
   positive;
2. verification path-change rate is at least 50%;
3. all four verification alias losses are at most 1.00 percentage point;
4. fixed-query terminal and cumulative differences are exactly zero; and
5. the selected parent is the best member of its eight-root roster on both
   search and verification partitions.

If no task is eligible, Step 94 stops before opening any test label with
`NO_GO_NO_CALIBRATION_ELIGIBLE_TASK`.  Otherwise the largest verified effect is
selected; ties follow the task order in Section 2.  Test-set task switching is
forbidden.

## 7. Confirmatory lock and one-time test

Before the selected official test label array is opened, a lock binds:

- this preregistration and Stage-0 manifests;
- every development/input/sealed-outcome hash;
- every root, gate, and integrated variant checkpoint;
- the complete search and verification ledgers;
- the selected task, roster, parent, thresholds, temperature, pool size, and
  budget;
- source hashes for exact-match similarity, active, fixed-query, endpoint, and
  statistical code;
- 2,000 confirmatory seeds `941000..942999`;
- bootstrap seed `94900` and sign-flip seed `94901`.

The lock must assert that no Step-94 confirmatory raw or result file exists.
Only then may the confirmatory runner import the selected sealed label array.
Unselected task labels remain sealed and are never analyzed.

The paired clean/refined runs share pools, references, temperature, budget,
query tie policy, and root tie policy.  The fixed-query run feeds the refined
registry the clean run's ordered queries.

## 8. Frozen success gates

The decision is `GO_FRESH_SOURCE_FAITHFUL_LEARNED_VARIANT_BRIDGE` only if every
gate passes:

1. all pre-outcome hashes and execution-lock assertions match;
2. one literal exact-match function is used in acquisition and posterior
   feedback;
3. four integrated variant checkpoints have distinct hashes, nonconstant
   learned parent and gate tensors, and bitwise raw-input endpoint replay;
4. no forbidden runtime input is accepted or accessed;
5. every alias is coordinate-wise non-improving relative to its parent;
6. each alias loses at most 1.00 percentage point of full-test accuracy;
7. at least 50% of ordered clean/refined paths differ;
8. mean terminal root-regret increase is at least `.005` (0.50 percentage
   point);
9. its 10,000-repetition paired-bootstrap 95% lower endpoint is above zero;
10. its one-sided paired sign-flip p-value is at most `.05`;
11. mean active-minus-fixed terminal harm is at least `.005` and its bootstrap
    lower endpoint is above zero;
12. mean cumulative regret increases and its bootstrap lower endpoint is above
    zero; and
13. every fixed-query terminal and cumulative root-regret difference is exactly
    zero, with bitwise identical root histories.

All statistics use float64.  No gate, task menu, task-selection rule, roster,
seed, threshold grid, or endpoint may change after test outcomes are opened.
Failure is retained as `NO_GO_RETAIN_FRESH_TASK_NEGATIVE`.

## 9. Claim boundary and stopping rule

A pass would close the narrow score-changing condition: on one fresh task, a
no-test-reference, learned, independently replayable raw-input variant causes
material terminal and active-minus-fixed harm under a source-faithful same-s
selector, with exact fixed-query zero.  It would not establish live registry
admission, secret-task compromise, universal harm, or production prevalence.

Step 94 permits one selected test split and one confirmatory execution.  A
failure cannot be replaced by opening either unselected test split, modifying
the gate, or starting another score-targeted task in this review cycle.
