# Step 105 Yelp prospective cross-benchmark one-shot protocol

## Status and purpose

This protocol is written before any Yelp Polarity dataset row, project-side
label vector, root prediction, learned-adapter score, acquisition trajectory,
or effect is loaded or inspected.  Before this lock, only public dataset/model
cards, repository metadata, tiny model configuration files, and file-size
metadata were inspected.  No model weight was run and no dataset row was
loaded.  Yelp Polarity has not been an outcome-bearing task anywhere in this
project.

Step 105 is the single prospective cross-benchmark test requested after the
Step 104 IMDB held-out result.  It transfers one source-faithful recipe to a
new dataset and a wholly new public checkpoint registry without target-task
selector search, geometry screening, grid search, condition replacement, or a
second outcome attempt.  A failure is retained and ends this line; no other
task, registry, parent, split, threshold rule, budget, temperature, adapter
family, or seed range may replace it.

## Frozen target and separation

- Dataset: `fancyzhx/yelp_polarity`, config `plain_text`.
- Revision: `bbf1c97a1f0cf005e5aded43839fd814654a1557`.
- Public metadata only: binary English sentiment; fields `text,label`;
  560,000 official train and 38,000 official test rows, balanced by class.
- Label mapping: `(negative, positive)=(0,1)`.
- Salt: `step105-yelp-prospective-cross-task-one-shot-v1`.

Within each train label, SHA-256 of revision, source index, text, label, and
salt orders rows.  Exactly the first 3,000 rows per class are `adapter_train`,
the next 500 per class are `threshold_calibration`, and the next 1,000 per
class are `threshold_safety`.  No selector search or selector verification
partition exists.  The complete official test is immediately separated into
a label-free input file and a sealed label file.  Test labels are not opened
until the final pre-outcome lock binds all predictions and authorities.

## Frozen public admission rule and registry

At the pre-data snapshot time, an admissible root is a public, non-private,
native Hugging Face `transformers` sequence-classification repository tagged
for `yelp_polarity`, loadable without custom code, with a binary identity label
map and a pinned immutable revision.  Adapter-only, embedding-only,
zero-shot-only, non-binary, custom-code, and non-Transformers repositories are
ineligible.  The four admissible repositories with the largest public download
counts in the frozen metadata snapshot form the registry.  No project-side
prediction or label is used by this rule.

The resulting order is fixed as follows.  The download-rank-one endpoint is
the parent.  Reported accuracies are unverified model-card metadata retained
only for transparency; neither roster admission nor parent designation uses
them:

1. parent: `VictorSanh/roberta-base-finetuned-yelp-polarity`, revision
   `709b46dd24f08e9b310fd7630cd0728f6d52b570`, 1,095 downloads,
   reported test accuracy 98.08%;
2. `fabriceyhc/bert-base-uncased-yelp_polarity`, revision
   `5793d97b55d24e604d9157704e9d3a76735515c1`, 333 downloads,
   reported accuracy 95.1605%;
3. `randellcotta/distilbert-base-uncased-finetuned-yelp-polarity`, revision
   `a70e629510f730cdaa8407db4928705cfedb257c`, 80 downloads,
   reported accuracy 96.1741%;
4. `JiaqiLee/robust-bert-yelp`, revision
   `a40e8daa515ff7cd09fed21c3d56f50ab7da8f25`, 42 downloads,
   reported test accuracy 95.32%.

All roots use hard-label argmax, identity mapping, and maximum length 256.  The
parent is the download-rank-one admitted root.  Neither the roster nor the
parent can change after any target row is loaded.  The admission rule and all
metadata are frozen in
`STEP105_PUBLIC_METADATA_SNAPSHOT_2026-08-13.json`.

## Frozen learned endpoint recipe

The recipe is transferred from Step 104 without target selector tuning.  Four
binary error adapters independently initialize from the pinned parent.
Embeddings and RoBERTa encoder layers 0--9 remain frozen; layers 10--11 and the
classification head train to predict
`1[parent hard label != train reference]` from raw review text.

- training rows: `adapter_train` only;
- seeds: `105100+a`, `a in {0,1,2,3}`;
- two epochs with the same class-balanced bootstrap construction as Step 104;
- max length 256, batch 8, gradient accumulation 4;
- AdamW learning rate `2e-5`, weight decay `.01`;
- 10% linear warmup then decay, gradient norm cap 1.0;
- deterministic CUDA and CUDA-only mixed precision.

Each saved adapter must be hash-distinct and contain at least ten million
learned parameters.  Runtime accepts raw text and the frozen parent hard label
only.  Alias `a` emits the parent label unless its learned error score exceeds
its threshold, when it emits unique out-of-label token `2+a`.  Runtime cannot
receive any test reference, label, item ID/lookup, peer output, pool,
posterior, trajectory, selector state, split, or dataset ID.

## Frozen threshold rule and selector

There is exactly one threshold rule and one selector condition; no grid is
evaluated.

- calibration correct-coordinate loss cap: `.0075`;
- trigger cap: `.025`;
- the threshold is fitted on `threshold_calibration` by the Step 104 stable
  score-order rule;
- it is only raised on `threshold_safety` until realized correct-coordinate
  loss is at most `.005`, its exact one-sided 99% Clopper--Pearson upper bound
  is at most `.01`, and the trigger cap is met;
- ties at the boundary are backed off completely;
- four aliases, pool size 500, budget 5, temperature `.05`;
- the same literal `s(a,b)=1[a=b]` is used for output--output acquisition and
  reference--output posterior evidence;
- root-level terminal and cumulative regret;
- exactly 3,000 paired seeds `1053000:1055999`.

Training/calibration may report parent errors, thresholds, and alias quality.
It may not run an active selector, inspect a test label, choose another root,
change a threshold cap, or stop on target geometry.  Even zero triggers or
unfavorable train geometry proceed to the single sealed outcome.

## Pre-data and pre-outcome locks

Before any dataset row is loaded, a pre-data lock binds this protocol, the
public metadata snapshot, all root IDs/revisions, the exact code, constants,
admission rule, adapter recipe, threshold rule, selector condition, gates, and
seeds.

After deterministic training/calibration, all four test root predictions,
four test error-score vectors, trigger vectors, alias vectors, thresholds,
adapter weights, and audits are produced from the label-free test inputs and
hashed.  A second lock binds the pre-data authority, data/input/sealed hashes,
training artifacts, model revisions, code, predictions, and outcome seeds.
Only then may the sealed test labels be opened once.

## Mandatory one-shot outcome gates

### Integrity and independence

1. Every protocol, metadata, pre-data, data, code, model, adapter, prediction,
   outcome, and lock hash matches.
2. No target selector search/verification/grid artifact exists.
3. Raw-input re-inference reproduces every locked root and alias prediction.
4. Four adapter hashes are distinct and each has at least ten million learned
   parameters.
5. Endpoint source/signature contains no forbidden test outcome or selector
   input.
6. Acquisition and evidence use the same literal exact-match implementation.

### Quality and mediation

7. Every alias is coordinate-wise non-improving relative to the parent.
8. Each alias loses at most `floor(.01*38000)=380` correct coordinates.
9. At least 50% of ordered acquisition paths change.
10. Parent-to-challenger final transitions exceed the reverse.
11. Fixed-query root histories are bitwise identical to clean histories and
    fixed terminal/cumulative effects are exactly zero.

### Outcome

12. Mean terminal harm is at least +.5 percentage point.
13. Mean active-minus-fixed terminal harm is at least +.5 point.
14. Terminal and active-minus-fixed paired-bootstrap 95% lower bounds exceed
    zero and one-sided paired sign-flip p-values are at most `.05`.
15. Cumulative regret mean and paired-bootstrap lower bound exceed zero.

Bootstrap repetitions are 10,000 and sign-flip repetitions are 100,000.
Inference seeds are `105800/105801`, `105801/105802`, and `105802/105803` for
terminal, active-minus-fixed terminal, and cumulative effects.

The only success label is
`GO_STEP105_YELP_PROSPECTIVE_CROSS_TASK_ONE_SHOT`.
Any failed gate yields `NO_GO_RETAIN_STEP105_PROSPECTIVE_NEGATIVE`; no failed
outcome can be followed by task-internal tuning or a replacement outcome.

## Interpretation boundary

A pass is a one-shot transfer of the Step 104 learned same-similarity recipe to
a previously untouched benchmark and independently versioned public registry.
It would strengthen prospective cross-benchmark generality and an
admission-style public-endpoint interpretation.  Yelp and IMDB are both binary
sentiment tasks, so it is cross-domain/cross-registry rather than broad task-
family transfer.  It remains offline and does not establish live admission,
secret-label compromise, average harm, or production compromise.
