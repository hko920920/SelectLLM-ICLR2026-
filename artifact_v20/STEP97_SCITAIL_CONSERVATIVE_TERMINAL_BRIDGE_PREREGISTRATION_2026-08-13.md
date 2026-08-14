# Step 97 SciTail conservative terminal-bridge preregistration

## Status and separation

Step 96 remains `NO_GO_RETAIN_STEP96_NEGATIVE`: its sealed SNLI test passed the
material terminal, active-minus-fixed, inference, direction, and fixed-query
gates, but one of four aliases lost 100/9,824 correct coordinates when the
locked one-point cap allowed 98. Step 97 does not alter or reanalyze that test.
It performs one independent replication on a new corpus with a conservative
0.5-point development construction cap while retaining the one-point test
quality gate.

Before this lock, only the SciTail dataset card, schema, split sizes, task
description, repository file listing, and revision were inspected. No SciTail
data shard, row, reference label, candidate prediction, adapter score, or
active-selection outcome was downloaded or evaluated.

## Frozen task and data

- Dataset: `allenai/scitail`, configuration `tsv_format`.
- Revision: `0cc4353235b289165dfde1c7c5d1be983f99ce44`.
- Runtime fields: premise and hypothesis text only.
- Binary labels: `entails -> 0`, `neutral -> 1`; any other label is ineligible.
- Train is used for adapter training and search partitions.
- Official validation is independent development verification.
- Official test is the sole sealed confirmation.

Within each train label, SHA-256 over revision, premise, hypothesis, label,
source index, and salt `step97-scitail-conservative-v1` fixes row order. The
first 2,500 rows per label form `adapter_train`, the next 500 `roster_gate`, the
next 750 `threshold`, and the next 1,000 `selector_search`. Remaining train rows
are unused. Fewer than 4,750 valid rows in either label makes the protocol
ineligible before model prediction.

Stage 0 stores development rows with labels. Test text and source indices are
stored separately from a sealed label array. Stage A cannot open the latter.

## Frozen registry and binary mappings

The four-root order and revisions equal Step 96, but their semantic NLI output
is deterministically collapsed to SciTail's binary task:

1. parent `cross-encoder/nli-deberta-v3-small`, revision
   `fa2804872c3b4bd748f38c0185cc85775361e735`, raw
   `(contradiction, entailment, neutral) -> (1,0,1)`;
2. `MoritzLaurer/MiniLM-L6-mnli`, revision
   `6e0917f1a395b7a6c0f054a56b91c45d8e3af92f`, raw
   `(entailment, neutral, contradiction) -> (0,1,1)`;
3. `M-FAC/bert-mini-finetuned-mnli`, revision
   `780061727f47254ff763de653920bb8b7e2fd5f2`, frozen raw mapping `(0,1,1)`;
4. `M-FAC/bert-tiny-finetuned-mnli`, revision
   `618f766f89b50853abc1bea92fd38e1973818f0b`, frozen raw mapping `(0,1,1)`.

No root, mapping, order, or substitute is selected from SciTail outcomes. Before
training, the study stops unless on `roster_gate` the parent is uniquely best by
at least 2.5 percentage points, has error prevalence at least 3%, and at least
one challenger is correct on 15% or more of parent errors. The parent gap must
also exceed 2.5 points on search, validation, and sealed test.

## Learned endpoint

Four independently optimized binary DeBERTa error adapters are initialized
from the fixed parent. Embeddings and encoder layers 0--3 are frozen; layers
4--5, pooler, and binary classifier are trained on
`1[parent prediction != reference]`.

- Seeds: `97100+a`, `a in {0,1,2,3}`.
- Two epochs; micro-batch 8; gradient accumulation 4.
- Maximum length 128.
- AdamW `2e-5`, weight decay `.01`, 10% linear warmup then decay.
- Gradient norm cap 1.0; CUDA mixed precision only; float32 inference.
- Class-balanced bootstrap; no early stopping or outcome-dependent change.

Each adapter must contain at least ten million learned parameters and have a
distinct digest. At runtime it receives raw premise/hypothesis text and emits
the parent's binary class unless its score exceeds its frozen threshold, then
emits unique symbol `ABSTAIN_a`.

No test reference, item/text lookup, peer output, pool, selector state,
trajectory, posterior, dataset identifier, or separate evidence view may enter
training/runtime. Training labels are used only for the parent-error target and
the explicitly frozen development gates.

## Single fixed selector condition

The same literal `s(a,b)=1[a=b]` is used for output agreement and reference
feedback. There is no hyperparameter grid or post-outcome cell selection.

- clean roots: 4; aliases: 4;
- pool size: 500;
- annotation budget: 10;
- temperature: `0.025`;
- threshold construction loss cap: `0.005` per alias;
- threshold trigger cap: `0.05` per alias;
- search seeds: 400 consecutive seeds beginning at 973000;
- validation seeds: 1,500 consecutive seeds beginning at 974000;
- test seeds if unlocked: 3,000 consecutive seeds beginning at 976000.

Each threshold is fixed on `threshold` by taking the largest descending-score
prefix satisfying both construction caps and backing off score ties. The same
four thresholds are replayed unchanged on search, validation, and test.

## Development gates

All must pass:

1. authority, revision, data, code, model, adapter, and output hashes match;
2. adapters are distinct and each has at least ten million learned parameters;
3. runtime input inspection/replay finds no forbidden information;
4. one literal exact-match similarity drives acquisition and evidence;
5. all root-geometry gates pass;
6. aliases are coordinate-wise non-improving on threshold/search/validation;
7. threshold loss counts obey the 0.5-point construction cap and every alias
   loses at most one point on search and validation;
8. search and validation path-change rates are at least 50%;
9. clean selects the parent terminally in at least 60% of runs;
10. parent-to-challenger terminal transitions exceed the reverse direction;
11. terminal and active-minus-fixed means are at least +0.5 points on search
    and validation;
12. validation terminal and active-minus-fixed 95% paired-bootstrap lower
    bounds exceed zero and one-sided sign-flip `p<=.05`;
13. validation cumulative mean and lower bound exceed zero;
14. fixed-query terminal/cumulative effects are exactly zero and root histories
    are bitwise equal.

Bootstrap repetitions are 10,000; sign flips 100,000. Development inference
seeds are `97900/97901`, `97901/97902`, and `97902/97903` for terminal,
active-minus-fixed, and cumulative respectively.

## One-time test and stopping rule

An all-gates development pass creates a hash-bound execution lock before any
test label is opened. Confirmatory inference seeds are `97910/97911`,
`97911/97912`, and `97912/97913`. Test success requires the same mechanism and
root-geometry gates, coordinate-wise non-improvement, every alias loss count at
most `floor(.01*n)`, terminal and active-minus-fixed mean at least +0.5 points
with positive lower bounds and `p<=.05`, positive cumulative mean/lower bound,
and exact-zero fixed-query effects.

The sole success label is
`GO_CONSERVATIVE_SOURCE_FAITHFUL_TERMINAL_BRIDGE`; otherwise the result is
`NO_GO_RETAIN_STEP97_NEGATIVE`.

A pass is evidence for a targeted offline learned raw-input bridge, not a live
registry admission or universal effect. This protocol authorizes one SciTail
condition and at most one sealed test. It authorizes no alternate split, task,
root, mapping, threshold, budget, or subsequent score-targeted study.
