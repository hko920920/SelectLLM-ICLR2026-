# Step 96 SNLI directional terminal-bridge preregistration

## Status, motivation, and separation from earlier studies

This document begins one new prospective study after the Step 94 and Step 95
terminal negatives were retained. Neither study is reopened, relabeled, or
used to authorize an alternate test split. Step 96 tests a mechanism diagnosed
from Step 95 development data: nearly all paths changed, but the final contest
was almost entirely between roots separated by about one global accuracy point,
and transitions occurred in both directions. The observed root-change rate
therefore multiplied a small, partly cancelling conditional quality drop.

Before this lock, the public SNLI dataset card, schema, split sizes, revision,
one card example, candidate-model cards/configurations, repository revisions,
and DeBERTa parameter names were inspected. No SNLI data shard, train row beyond
the card example, validation/test outcome, candidate prediction on SNLI, or
Step 96 effect was downloaded or evaluated. The Step 95 post-hoc diagnostics
used only its frozen `selector_verify` development partition and did not open
the Step 95 sealed outcome. Those diagnostics are design motivation and cannot
be reported as confirmatory evidence.

## Question

Can four independently trained, raw-input, no-test-reference learned adapters
of a strong accountable parent create material terminal root regret on one
fresh task under a source-faithful single-similarity selector, when the clean
registry is prospectively fixed to include three meaningfully lower-quality
compact challengers and the annotation budget is ten?

## Frozen task, data revision, and separation

- Dataset: `stanfordnlp/snli`, configuration `plain_text`.
- Revision: `cdb5c3d5eed6ead6e5a341c8e56e669bb666725b`.
- Dataset labels: `(entailment, neutral, contradiction) = (0,1,2)`.
- Runtime input: premise and hypothesis text only.
- Rows with labels outside `{0,1,2}` or empty text are discarded.
- Learned-adapter training and all search partitions come only from `train`.
- The official `validation` split is the independent development-verification
  partition.
- The official `test` split is the sole one-time sealed confirmation.
- No validation row is moved into training/search, and validation cannot replace
  a failed test. No alternate corpus or split is authorized.

Within each train label, rows are ordered by SHA-256 over the dataset revision,
premise, hypothesis, label, source index, and salt
`step96-snli-directional-v1`. The first 3,000 per label form
`adapter_train`, the next 1,000 `roster_gate`, the next 1,500 `threshold`, and
the next 2,000 `selector_search`. All remaining train rows are unused. If any
label lacks 7,500 valid rows, the study is ineligible before prediction or
training.

Stage 0 writes validation rows with labels to a development file. Test premise
and hypothesis strings are written to a label-free input file; test labels are
written separately to a sealed outcome file. Stage A is forbidden to import or
hash-decode the sealed labels beyond comparing the already recorded file hash.

## Frozen four-root registry

The clean registry, order, revisions, and raw-index mappings are fixed before
SNLI outcomes:

1. accountable parent: `cross-encoder/nli-deberta-v3-small`, revision
   `fa2804872c3b4bd748f38c0185cc85775361e735`, raw indices
   `(contradiction, entailment, neutral)` mapped to dataset indices `(2,0,1)`;
2. `MoritzLaurer/MiniLM-L6-mnli`, revision
   `6e0917f1a395b7a6c0f054a56b91c45d8e3af92f`, raw indices
   `(entailment, neutral, contradiction)` mapped to `(0,1,2)`;
3. `M-FAC/bert-mini-finetuned-mnli`, revision
   `780061727f47254ff763de653920bb8b7e2fd5f2`, frozen dataset-order mapping
   `(0,1,2)`;
4. `M-FAC/bert-tiny-finetuned-mnli`, revision
   `618f766f89b50853abc1bea92fd38e1973818f0b`, frozen dataset-order mapping
   `(0,1,2)`.

There is no outcome-dependent root inclusion, exclusion, permutation, label
remapping, or substitute repository. This is deliberately a strong-parent plus
three compact-challenger registry, fixed to test the directional bottleneck
rather than average performance over arbitrary rosters.

Before adapter training, the protocol stops if on `roster_gate`:

- the DeBERTa parent is not the unique best root by at least 2.5 percentage
  points;
- its error prevalence is below 3%; or
- no challenger is correct on at least 15% of parent-wrong coordinates.

The same parent-best gap must be at least 2.5 points on `selector_search` and
validation for a test lock, and on sealed test for confirmatory success.

## Learned raw-input adapter family

Each alias is an independently optimized binary DeBERTa sequence classifier
initialized from the pinned parent. Embeddings and encoder layers 0--3 are
frozen. Encoder layers 4--5, pooler, and the binary classifier are trained to
predict `1[parent prediction != reference]` from premise/hypothesis text.

- Training partition: `adapter_train` only.
- Four separately seeded class-balanced bootstraps; seeds `96100+a` for
  `a in {0,1,2,3}`.
- Maximum length: 128 wordpieces.
- Epochs: 2.
- Micro-batch: 8; gradient accumulation: 4.
- AdamW learning rate `2e-5`, weight decay `.01`.
- Linear warmup over 10% of optimizer steps followed by linear decay.
- Gradient norm cap 1.0.
- Mixed precision only on CUDA; inference float32.
- No early stopping or outcome-dependent training change.

Only trained tensors plus the immutable parent binding are saved. Every adapter
must contain at least ten million learned parameters and have a distinct
SHA-256 digest. The executable endpoint loads the public parent and one adapter,
emits the parent's hard class unless the frozen adapter error score exceeds its
threshold, and otherwise emits its alias-specific out-of-label symbol
`ABSTAIN_a`.

Training and runtime may not receive test references, item IDs, exact-text
lookup tables, peer-model outputs, pools, selector state, trajectories,
posteriors, a dataset identifier, or a separate feedback/evidence matrix.
Development labels are used only for the binary parent-error target, frozen
threshold selection, root-geometry gate, and prespecified development tests.

## Source-faithful selector and development search

The exact same literal function `s(a,b)=1[a=b]` constructs candidate agreement
groups and reference feedback. No confidence channel, soft score, or second
utility view is allowed. Root regret is evaluated against the four original
roots; all aliases map to the DeBERTa parent.

Fixed constants:

- pool size: 500;
- annotation budget: 10;
- aliases: 4;
- temperatures: `{0.025, 0.05, 0.10}`;
- per-alias correct-coordinate loss caps:
  `{0.0025, 0.0050, 0.0075, 0.0100}`;
- per-alias trigger caps: `{0.05, 0.10, 0.20, 0.30}`;
- search seeds: 400 consecutive seeds beginning at 963000;
- validation-verification seeds: 1,500 consecutive seeds beginning at 964000;
- test seeds, if unlocked: 3,000 consecutive seeds beginning at 966000.

For each loss/trigger cap pair, each threshold is determined on `threshold` by
descending error score, choosing the largest prefix satisfying both caps, and
backing off exact-score ties. It is replayed unchanged on search, validation,
and test. The complete 48-cell grid is evaluated on `selector_search`. Cells
are ranked lexicographically by:

1. mean active-minus-fixed terminal-regret increase;
2. mean terminal-regret increase;
3. DeBERTa-to-challenger terminal transition rate minus the reverse rate;
4. mean cumulative-regret increase;
5. path-change rate;
6. lower temperature, lower loss cap, lower trigger cap.

The top eight search cells are replayed on validation. The validation cell with
the same ranking is frozen only if every gate below passes. Otherwise Step 96
stops without opening test labels.

## Mandatory development gates

All gates must pass on the prescribed partitions:

1. all authorities, revisions, inputs, code, models, adapters, arrays, and
   outputs match their hashes;
2. the four adapters have distinct hashes and at least ten million learned
   parameters each;
3. endpoint inspection and raw-text replay confirm forbidden runtime inputs are
   absent;
4. one literal exact-match similarity drives both acquisition and evidence;
5. all frozen root-geometry gates above pass on roster/search/validation;
6. each alias is coordinate-wise non-improving on threshold, search, and
   validation;
7. each alias loses at most one percentage point on all three partitions;
8. path-change rates are at least 50% on search and validation;
9. clean terminal selection chooses the DeBERTa parent in at least 60% of search
   and validation runs;
10. parent-to-challenger terminal transitions exceed reverse transitions on
    search and validation;
11. search and validation mean terminal and active-minus-fixed terminal effects
    are each at least +0.5 percentage points;
12. for 1,500 validation runs, terminal and active-minus-fixed paired-bootstrap
    95% lower bounds exceed zero and one-sided paired sign-flip p-values are at
    most .05;
13. validation cumulative mean and paired-bootstrap lower bound exceed zero;
14. fixed-query terminal/cumulative effects are exactly zero and fixed-query
    root histories are bitwise equal to the clean histories.

Bootstrap repetitions are 10,000 and sign-flip repetitions 100,000. Seeds are
`96900/96901` for terminal, `96901/96902` for active-minus-fixed, and
`96902/96903` for cumulative.

## One-time sealed confirmation

Only an all-gates development pass authorizes an execution lock binding every
authority, code file, public revision, adapter, development array/ledger,
selected cell, label-free test input, sealed outcome hash, and test seed.
The SNLI test labels are then opened once.

Confirmatory success requires:

- the parent remains the unique best global test root by at least 2.5 points;
- every alias is coordinate-wise non-improving and loses at most one point;
- path change is at least 50%, clean selects the parent at least 60%, and
  parent-to-challenger terminal transitions exceed the reverse direction;
- mean terminal harm is at least +0.5 percentage points, its paired-bootstrap
  lower bound exceeds zero, and one-sided sign-flip `p<=.05`;
- the same three requirements hold for active-minus-fixed terminal harm;
- cumulative mean and lower bound exceed zero;
- fixed-query terminal/cumulative effects are exactly zero and fixed-query root
  histories are bitwise equal.

The only success label is
`GO_SOURCE_FAITHFUL_DIRECTIONAL_TERMINAL_BRIDGE`. Any failure yields
`NO_GO_RETAIN_STEP96_NEGATIVE`.

## Interpretation and stopping rule

A pass would close the specific fresh-task, no-test-reference, learned raw-input,
single-similarity, material-terminal bridge requested by the external review.
It would remain a targeted offline registry condition, not evidence of a live
admission bypass, universal vulnerability, or production compromise.

A development stop or sealed failure is retained. Step 96 authorizes one task,
one fixed four-root roster, one development grid, and at most one sealed test.
It does not authorize SNLI validation as a replacement test, a second roster,
another dataset, a remapping, or a post-hoc threshold/budget change.
