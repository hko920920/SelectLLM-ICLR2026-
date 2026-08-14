# Step 95 MNLI full-adapter terminal-bridge preregistration

## Status and separation from Step 94

This document begins a new prospective study cycle after Step 94 retained its
prespecified terminal negative. Step 94 is not reopened, rerun, relabeled, or
used to authorize a replacement task. Step 95 uses a new task, new public model
roster, new learned-adapter family, new development partitions, a lower fixed
annotation budget, and a stricter development-to-test gate.

At this lock, no MultiNLI data file, validation row, candidate-model weight, or
candidate prediction has been downloaded or evaluated for Step 95. Public
dataset/model cards, repository commit identifiers, configuration files, label
maps, and file listings have been inspected. Search results exposed dataset
metadata and one training example but no validation outcome. No test label or
candidate accuracy/effect is known.

## Question

Can four independently trained, raw-input, no-test-reference adapters of one
accountable parent change a source-faithful single-similarity active-selection
path enough to cause material terminal root regret on one fresh task, with the
effect disappearing under the clean query matrix?

## Frozen task and data revisions

- Dataset: `nyu-mll/multi_nli`
- Revision: `da70db2af9d09693783c3320c4249840212ee221`
- Development source: the labeled `train` split only.
- One-time test: `validation_matched` only.
- `validation_mismatched` is not a replacement test and will not be analyzed.
- Utility: exact three-class correctness in dataset order
  `(entailment, neutral, contradiction) = (0,1,2)`.
- Runtime input: premise and hypothesis text only.

Rows with labels outside `{0,1,2}` are discarded. A SHA-256 rule over revision,
pair identifier, premise, hypothesis, genre, and label orders development rows
within each `(genre,label)` stratum. The first 800 rows per stratum form
`adapter_train`, the next 200 `parent_select`, the next 250 `threshold`, the
next 250 `selector_search`, and the next 350 `selector_verify`. A stratum with
fewer than 1,850 valid rows makes the protocol ineligible before model
training. The official matched-validation rows are written as a text-only input
file and a separate sealed outcome file; Stage A cannot import the latter.

## Frozen public root bank

The ordered candidate bank is:

1. `textattack/distilbert-base-uncased-MNLI`, revision
   `2cee56ec53fc7935042c094638345db757eece0d`;
2. `typeform/distilbert-base-uncased-mnli`, revision
   `cfa538a0fddbbd978fefe8966c1aeff7ad409c90`;
3. `ishan/distilbert-base-uncased-mnli`, revision
   `5b5436f6f59086b00ac829afecc16d1bd926cbfb`;
4. `M-FAC/bert-mini-finetuned-mnli`, revision
   `780061727f47254ff763de653920bb8b7e2fd5f2`;
5. `M-FAC/bert-tiny-finetuned-mnli`, revision
   `618f766f89b50853abc1bea92fd38e1973818f0b`;
6. `cross-encoder/nli-distilroberta-base`, revision
   `b14d131f9d32668a5e6a982729b57ff6ed5dfcbd`;
7. `MoritzLaurer/MiniLM-L6-mnli`, revision
   `6e0917f1a395b7a6c0f054a56b91c45d8e3af92f`;
8. `cross-encoder/nli-deberta-v3-small`, revision
   `fa2804872c3b4bd748f38c0185cc85775361e735`.

For roots 1--5 and 7, raw output indices map identically to dataset labels.
For roots 6 and 8, raw indices `(0,1,2)` mean
`(contradiction, entailment, neutral)` and are mapped to dataset indices
`(2,0,1)`. No outcome-dependent remapping is permitted.

If a root cannot be loaded at its pinned revision before any label-dependent
root evaluation, its load failure is recorded. The study stops unless all
eight load; no substitute repository is allowed.

## Frozen parent and six-root roster rule

Only roots 1--3 are eligible accountable parents because the learned-adapter
implementation is locked to their DistilBERT architecture. On `parent_select`,
choose the eligible parent with the highest exact accuracy; ties follow bank
order. Form a six-root clean roster from that parent plus the five remaining
roots with the highest accuracy not exceeding the parent's accuracy. If fewer
than five such roots exist, append the closest more-accurate roots by absolute
accuracy gap, then bank order. This is a deterministic development rule, not a
test-dependent roster search.

The study can advance to test only if the selected parent is the unique or tied
best clean root on both `selector_search` and `selector_verify`, and at least
one other roster root has global verification accuracy at least 0.5 percentage
points below the parent. Per-pool regret continues to use the best original
root in that pool, not the development parent.

## Learned adapter family

For each alias `a in {0,1,2,3}`, initialize a binary sequence classifier from
the selected public parent revision. Freeze embeddings and the first four of
six DistilBERT transformer blocks. Independently train the final two blocks,
pre-classifier, and binary classifier to predict whether the frozen parent is
wrong from premise/hypothesis text alone.

- Training partition: `adapter_train` only.
- Target: `1[parent_prediction != reference]`.
- Each alias uses a separately seeded class-balanced bootstrap.
- Seeds: `95100+a`.
- Maximum length: 128 wordpieces.
- Epochs: 2.
- Batch size: 16; gradient accumulation: 2.
- Optimizer: AdamW, learning rate `2e-5`, weight decay `.01`.
- Linear warmup: 10% of optimizer steps, then linear decay.
- Gradient norm cap: 1.0.
- Mixed precision is allowed only on CUDA; inference is float32.
- No early stopping or outcome-dependent hyperparameter change.

Only the trainable tensors and a binding to the public parent revision are
saved as an adapter checkpoint. Each adapter must contain at least ten million
nonconstant learned parameters and have a distinct SHA-256 digest. A raw-input
endpoint loads the frozen parent plus one adapter and emits the parent's class
unless the adapter error probability exceeds its frozen threshold, in which
case it emits its unique out-of-label symbol `ABSTAIN_a`.

Adapter training and runtime cannot receive test references, item identifiers,
exact-text lookup tables, peer-model outputs, pools, selector state,
trajectories, posteriors, or a separate evidence matrix. Development reference
labels are allowed only as the binary parent-error training target and for the
prespecified threshold/selector gates.

## Single-similarity selector and development grid

The same literal function `s(a,b)=1[a=b]` must construct candidate agreement
groups and reference feedback. No confidence view or second feedback matrix is
accepted. The registry contains the six clean roots plus four aliases mapped
to the selected parent. Root regret is computed against the six original roots.

Fixed constants:

- pool size: 500;
- annotation budget: 20;
- aliases: 4;
- temperatures: `{0.05, 0.10, 0.25}`;
- per-alias correct-coordinate loss caps:
  `{0.0025, 0.0050, 0.0075, 0.0100}`;
- per-alias trigger caps: `{0.05, 0.10, 0.15, 0.20}`;
- search seeds: 400 consecutive seeds beginning at 953000;
- verification seeds: 1,500 consecutive seeds beginning at 954000.

For every loss-cap/trigger-cap pair, each alias threshold is determined on
`threshold` by descending error score, choosing the largest prefix satisfying
both caps, and backing off all exact-score ties. That threshold is replayed
unchanged on search, verification, and test. The complete 48-cell grid is run
on `selector_search`. Rank cells lexicographically by:

1. mean active-minus-fixed terminal-regret increase;
2. mean terminal-regret increase;
3. mean cumulative-regret increase;
4. path-change rate;
5. lower temperature, lower loss cap, lower trigger cap.

The top eight cells are replayed on `selector_verify`. The highest-ranked
verification cell is frozen only if every development go gate below passes.
Otherwise Step 95 stops without opening the sealed matched-validation outcome.

## Development go gates

All are mandatory:

1. all authority, source, public revision, input, model, adapter, and output
   hashes match;
2. all four adapters have distinct hashes and at least ten million learned
   parameters;
3. raw-input endpoint inspection and replay confirm the forbidden runtime
   inputs are absent;
4. the same literal exact-match similarity drives acquisition and evidence;
5. selected parent is a best original root on both search and verification;
6. every alias is coordinate-wise non-improving on threshold, search, and
   verification;
7. every alias loses at most one percentage point on threshold, search, and
   verification;
8. search and verification path-change rates are each at least 50%;
9. search and verification mean terminal and active-minus-fixed terminal
   effects are each at least +0.5 percentage points;
10. on 1,500 verification runs, both terminal and active-minus-fixed paired
    95% bootstrap lower bounds exceed zero and both one-sided paired sign-flip
    p-values are at most .05;
11. verification cumulative mean and bootstrap lower bound exceed zero;
12. fixed-query terminal/cumulative effects are exactly zero and fixed-query
    root histories are bitwise equal to clean histories.

Bootstrap repetitions are 10,000 and sign-flip repetitions 100,000. Seeds are
`95900/95901` for terminal, `95901/95902` for active-minus-fixed, and
`95902/95903` for cumulative.

## One-time sealed confirmation

If and only if the development go gates pass, an execution lock binds every
authority file, code file, model/adapter, development ledger, selected cell,
test input/outcome hash, and 3,000 consecutive test seeds beginning at 955000.
The selected matched-validation outcome is then opened once.

Confirmatory success requires all implementation/quality/mechanism gates above
plus:

- mean terminal harm at least +0.5 percentage points;
- terminal paired-bootstrap lower bound above zero;
- terminal one-sided paired sign-flip p-value at most .05;
- the same three requirements for active-minus-fixed terminal harm;
- positive cumulative mean and bootstrap lower bound;
- exact-zero fixed-query terminal/cumulative effects and bitwise-equal fixed
  root histories.

The only success label is
`GO_SOURCE_FAITHFUL_FULL_ADAPTER_TERMINAL_BRIDGE`. Any failed gate yields
`NO_GO_RETAIN_STEP95_NEGATIVE`. No alternate split, model substitution,
remapping, budget, roster, threshold, seed range, or post-hoc task is permitted.

## Interpretation and stopping rule

A pass would supply the still-missing fresh-task, learned-adapter, raw-input,
single-similarity terminal bridge. It would remain an offline controlled active
evaluation result, not evidence of live registry admission or production
compromise.

A development stop or sealed failure is retained as a negative result. Step 95
authorizes one task and at most one sealed confirmation. It does not authorize
another score-targeted task in this study cycle.
