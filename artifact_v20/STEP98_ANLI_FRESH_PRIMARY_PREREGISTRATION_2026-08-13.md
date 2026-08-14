# Step 98 ANLI fresh learned terminal confirmation preregistration

## Status and purpose

This protocol is written before any ANLI example, label, candidate prediction,
round-specific accuracy, adapter score, development effect, or dev outcome is
loaded or inspected.  Only the public dataset card, split names, feature names,
label names, repository revision, model cards, and already pinned public model
revisions have been inspected.

Step 96 is not reopened or relabeled: its selected primary remains
`NO_GO_RETAIN_STEP96_NEGATIVE`, and its conservative top-eight result remains
post-confirmatory secondary evidence.  Step 97 remains a pretraining geometry
stop with unopened SciTail test.  Step 98 asks for the missing literal primary
success on one entirely new dataset family and one sealed outcome.

## Frozen dataset family and one-round selection

- Dataset: `facebook/anli`
- Dataset revision: `8e4813d81f46d313dac7892e1c28076917cfcdf9`
- Config: `plain_text`
- Candidate train/dev pairs in fixed order:
  `train_r1/dev_r1`, `train_r2/dev_r2`, `train_r3/dev_r3`.
- Labels in dataset order:
  `(entailment, neutral, contradiction) = (0,1,2)`.
- Runtime inputs: premise and hypothesis text only.

All three train splits are eligible for a single deterministic train-only
round-selection rule.  No dev split is accessed until the rule selects one
round.  Only that round's dev inputs and sealed labels are materialized.  A
failed development or sealed gate ends Step 98; no alternate round, split,
dataset, model, mapping, roster, budget, threshold, or seed range is allowed.

Within each train split and each label, SHA-256 of the fixed dataset revision,
round, uid, premise, hypothesis, label, and salt
`step98-anli-fresh-primary-v1` orders rows.  The first rows per label are:

| partition | rows per label | total |
|---|---:|---:|
| `geometry_search` | 250 | 750 |
| `geometry_verify` | 250 | 750 |
| `adapter_train` | 1800 | 5400 |
| `threshold_calibration` | 400 | 1200 |
| `threshold_safety` | 500 | 1500 |
| `selector_search` | 600 | 1800 |
| `selector_verify` | 1000 | 3000 |

A round with fewer than 4,800 valid rows in any class is ineligible before
model inference.  The complete selected train rows are stored; ANLI dev labels
are absent from that package.

## Frozen four-root registry

The parent and three challengers are fixed before ANLI data access:

1. parent: `cross-encoder/nli-deberta-v3-small`, revision
   `fa2804872c3b4bd748f38c0185cc85775361e735`, raw labels mapped
   `(0,1,2)->(2,0,1)`;
2. `MoritzLaurer/MiniLM-L6-mnli`, revision
   `6e0917f1a395b7a6c0f054a56b91c45d8e3af92f`, identity mapping;
3. `M-FAC/bert-mini-finetuned-mnli`, revision
   `780061727f47254ff763de653920bb8b7e2fd5f2`, identity mapping;
4. `M-FAC/bert-tiny-finetuned-mnli`, revision
   `618f766f89b50853abc1bea92fd38e1973818f0b`, identity mapping.

No root or mapping can be replaced.  Root predictions are functions only of
raw text and the pinned public weights.

## Train-only round-selection rule

For each round, the fixed four-root roster is evaluated separately on
`geometry_search` and `geometry_verify`.  Four oracle aliases of the parent
abstain on every parent-wrong coordinate; this reference-aware construction is
used only as a train-only directional-geometry screen, never as evidence for a
reference-free attack.

Using pool size 500, budget 10, temperature .025, four aliases, seeds
`980000:980299` for search and `980300:980599` for verification, a round is
eligible only if on both partitions:

1. the parent exceeds every challenger by at least 2.5 percentage points;
2. parent error prevalence is at least 10%;
3. some challenger is correct on at least 15% of parent errors;
4. oracle terminal harm is at least +1.0 percentage point;
5. parent-to-challenger terminal transitions exceed the reverse direction;
6. oracle fixed-query terminal and cumulative effects are exactly zero.

Among eligible rounds, select the one maximizing the smaller of its two oracle
terminal effects; ties use the larger verification effect, larger smaller
parent gap, then fixed order R1, R2, R3.  If none is eligible, stop before any
dev access.  This menu and rule are fixed before ANLI train labels are loaded.

## Four independently learned raw-input endpoints

For the selected round, four binary error adapters independently initialize
from the pinned DeBERTa parent.  Embeddings and encoder layers 0--3 remain
frozen; layers 4--5, pooler, and classifier train to predict
`1[parent prediction != train reference]` from premise/hypothesis text.

- train partition: `adapter_train` only;
- seeds: `98100+a`, `a in {0,1,2,3}`;
- two epochs;
- class-balanced bootstrap;
- max length 128;
- batch size 8, gradient accumulation 4;
- AdamW learning rate `2e-5`, weight decay `.01`;
- 10% linear warmup then decay;
- gradient norm cap 1.0;
- mixed precision only on CUDA; inference float32.

Each saved endpoint must contain at least ten million learned parameters and a
distinct SHA-256 digest.  At runtime alias `a` emits the parent label unless
its learned error probability exceeds its frozen threshold, when it emits the
unique out-of-label token `ABSTAIN_a`.  Runtime cannot receive a label,
reference, item ID, lookup table, peer output, pool, posterior, trajectory,
selector state, round identifier, or separate evidence view.

## Single conservative condition

There is no effect grid and no post-development condition selection.

- clean roots: the fixed four-root registry above;
- aliases: four, all mapped to the parent root;
- pool size: 500;
- budget: 10;
- temperature: .025;
- acquisition and posterior evidence:
  the same literal `s(a,b)=1[a=b]`;
- initial construction correct-coordinate loss cap: .5% per alias on
  `threshold_calibration`;
- trigger cap: 10% per alias;
- safety backoff: raise each threshold, never lower it, until on
  `threshold_safety` both its realized correct-coordinate loss is at most .25%
  and its exact one-sided 99% Clopper--Pearson upper bound is at most 1%;
- ties at the threshold are backed off completely.

The final four thresholds are replayed unchanged on selector search,
selector verification, and the sealed dev outcome.

## Development gates

Search uses seeds `982000:982799`; verification uses
`983000:984499`.  All gates are mandatory on both selector partitions unless a
gate explicitly names verification only:

1. all authority, dataset, input, code, root, adapter, and output hashes match;
2. the round-selection rule is exactly reconstructed and only one round/dev is
   selected;
3. all four adapters are hash-distinct and contain at least ten million learned
   parameters;
4. endpoint signature and source inspection show only raw text at runtime;
5. acquisition and evidence call the same literal exact-match function;
6. the parent exceeds every challenger by at least 2.5 points;
7. aliases are coordinate-wise non-improving;
8. every alias loses at most one point on calibration, safety, search, and
   verification, using integer counts `loss_count <= floor(.01*N)`;
9. path-change rate is at least 50%;
10. clean terminal parent-selection rate is at least 60%;
11. parent-to-challenger transitions exceed challenger-to-parent transitions;
12. terminal and active-minus-fixed mean harm are each at least +.5 points;
13. verification terminal and active-minus-fixed 95% paired-bootstrap lower
    bounds exceed zero and one-sided paired sign-flip p-values are at most .05;
14. verification cumulative mean and lower bound exceed zero;
15. fixed-query root histories are bitwise equal to clean histories and all
    fixed terminal/cumulative effects are exactly zero.

Bootstrap repetitions are 10,000; sign-flip repetitions are 100,000.  Seeds
are `98800/98801`, `98801/98802`, and `98802/98803` for terminal,
active-minus-fixed, and cumulative respectively.  Any failed development gate
stops without accessing dev labels.

## One-time sealed primary confirmation

After development GO, a label-free prediction package for the selected dev
split is generated and hashed.  A lock binds this protocol, train-only round
selection, selected round, root and adapter weights, thresholds, code, all
development ledgers, label-free dev inputs/predictions, sealed-outcome hash,
and exactly 3,000 seeds `985000:987999`.  Only then is the selected sealed dev
outcome opened once.

The primary succeeds only if every implementation, quality, mechanism, and
outcome gate above holds on dev, with integer test quality
`loss_count <= floor(.01*N)` per alias.  Test inference seeds are
`98900/98901`, `98901/98902`, and `98902/98903`.

The only success label is
`GO_STEP98_FRESH_ANLI_PRIMARY_TERMINAL_BRIDGE`.
Any failed gate yields `NO_GO_RETAIN_STEP98_NEGATIVE`.  A failure cannot be
replaced by another ANLI round or another task.

## Interpretation boundary

A pass is a fresh-dataset, learned raw-input, no-dev-reference-at-runtime,
single-similarity, terminal and acquisition-mediated primary confirmation.  It
does not demonstrate live registry admission, secret-label compromise, a full
independently fine-tuned base checkpoint, average harm, or production attack.
