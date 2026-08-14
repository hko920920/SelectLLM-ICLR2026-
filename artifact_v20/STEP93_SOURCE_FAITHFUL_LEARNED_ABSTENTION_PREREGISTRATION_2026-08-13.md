# Step 93 source-faithful learned-abstention preregistration

Locked: `2026-08-13T01:30:32.0904006+09:00`

Status at lock: **no Banking77 row, label, model output, selector outcome, or
test statistic had been downloaded or inspected in this workspace.** A complete
case-insensitive workspace search immediately before this lock returned zero
`banking77`/`banking_77` hits. Only public repository metadata (dataset identity,
split sizes, filenames, and commit hashes) was inspected.

This protocol is a one-condition repair of the method-fidelity concern raised
after Step 92. It is not a post-hoc reinterpretation of AG News. Because the AG
News test outcomes have already been opened, an outcome-blind re-confirmation on
that test would be invalid. Step 93 therefore uses a previously unused task and
seals its official test split before any outcome analysis.

## 1. Scientific question

Does a no-test-reference, genuinely learned, same-root refinement cause
terminal and active-minus-fixed root-selection harm when the implementation
uses one and the same similarity function in both places required by the source
Select-LLM algorithm?

The single similarity is

\[
s(a,b)=\mathbf 1[a=b].
\]

It is used without parsing exceptions or type-dependent behavior:

1. acquisition uses
   \(\sum_{j,k}p_t(j)p_t(k)s(f_j(x),f_k(x))\);
2. posterior evidence uses
   \(s(r_t,f_j(x_t))\) inside the exponential update; and
3. deployment selects the maximum-posterior entry and then maps it to its
   accountable root for root-level regret.

No confidence-only field, weak view, or separate deployment utility is allowed.

## 2. Fresh task and immutable public dependencies

- Task: Banking77 English intent classification, 77 classes.
- Executable parquet mirror: `mteb/banking77`.
- Dataset revision:
  `18072d2685ea682290f7b8924d94c62acc19c0b2`.
- Upstream source: `PolyAI/banking77`.
- Upstream source revision:
  `90d4e2ee5521c04fc1488f065b8b083658768c57`.
- Public metadata reports 10,003 train and 3,080 test examples.
- Frozen encoder: `distilbert/distilbert-base-uncased`.
- Encoder revision:
  `12040accade4e8a0f71eabdb258fecc2e7e948be`.
- Encoder `model.safetensors` SHA-256:
  `5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063`.

The official test texts may be materialized as input-only data after this lock;
their labels must be stored separately under `external_data/step93_sealed/` and
may be opened only by the confirmatory runner after the execution lock exists.

## 3. Outcome-blind train partitions

Each official training index is assigned by SHA-256 of the fixed salt
`step93-source-faithful-abstention-v1`, the string `train`, and the integer
index. Residues modulo ten define:

- `0..5`: clean-root training;
- `6..7`: learned abstention-adapter training;
- `8..9`: Stage A attack calibration.

The official test split is not partitioned or searched. Stage B uses all 3,080
test items only after one configuration is frozen.

## 4. Clean registry

The frozen encoder produces one 768-dimensional CLS vector per raw utterance.
Twelve independently optimized classification adapters form the clean registry:

`LayerNorm(768) -> Linear(768,128) -> GELU -> Linear(128,77)`.

The root-training fractions are

`[.10,.15,.20,.30,.40,.50,.60,.70,.80,.90,1.00,1.00]`.

Seeds are `93300+j`, `j=0,...,11`. The final two roots use distinct seeds and
bootstrap orderings. Every saved root file must contain nonconstant learned
tensors and at least 109,000 trainable parameters. Roots `4..10` are the only
eligible parents; this set is fixed before outcomes.

## 5. Learned same-root abstention adapters

For every eligible parent, four adapters are trained independently on the
adapter-training split to predict whether that parent's hard intent prediction
is wrong. Each has architecture

`LayerNorm(768) -> Linear(768,64) -> GELU -> Linear(64,1)`

and therefore 50,817 trainable parameters. Seeds are `93400,93401,93402,93403`;
each uses an independent bootstrap and class-balanced binary cross-entropy.

At runtime an adapter sees only raw utterance text, its frozen hidden vector,
and the parent logits. If its learned error score crosses the frozen threshold,
alias `a` emits the literal response token `ABSTAIN_a`; otherwise it emits the
parent's integer intent label. The four abstention tokens are outside the set of
77 possible reference labels.

Consequently, for every coordinate and without seeing its reference,

\[
s(r,f_{alias}(x))\le s(r,f_{parent}(x)).
\]

Thus an alias can never improve on its parent, and on any fixed query sequence
its cumulative evidence can never exceed its parent's. This is a structural
control, not a test-set observation.

Forbidden runtime inputs are: test reference, item index or ID, exact-prompt
lookup table, peer output, pool seed, trajectory, acquisition score, posterior,
or selector state. No rule may branch on dataset row identity.

## 6. Stage A frozen search

All search occurs only on the Stage A calibration partition. For each adapter,
calibration error scores are sorted. A threshold is chosen as the largest
score-prefix that satisfies both a maximum trigger fraction and a maximum
calibration utility loss, with a midpoint between adjacent distinct scores used
to avoid row-dependent tie handling.

The complete grid is:

- parent root: `4,5,6,7,8,9,10`;
- Select-LLM temperature: `.10,.25,.50,1,2,4`;
- maximum calibration utility loss:
  `.0025,.0050,.0075`;
- maximum trigger fraction: `.05,.10,.20,.30`.

This gives 504 conditions. Every condition, including ineligible conditions, is
retained. A condition is eligible only if all four aliases satisfy its two
calibration caps and every threshold is finite.

Selector settings are pool size 400 and budget 50. Pilot seeds are
`930000..930019`. Conditions are ranked by mean active-minus-fixed terminal
regret, then terminal regret, cumulative regret, path-change rate, smaller
quality-loss cap, smaller trigger cap, smaller parent ID, smaller temperature.
The top 16 eligible conditions are replayed on verification seeds
`930100..930399`. The same ranking freezes exactly one condition.

The frozen Stage A file must contain the complete 504-condition ledger, all
thresholds, learned-weight hashes, calibration outcome hashes, and the selected
configuration. Stage A may not read any official test file.

## 7. Confirmatory lock and Stage B

Before test labels or confirmatory results are opened, a lock binds:

- this preregistration;
- dataset/input/sealed-outcome hashes;
- the complete Stage A ledger and selected configuration;
- all 12 root and 28 learned-adapter files;
- all execution code;
- the exact same-s assertion;
- 1,000 test seeds `931000..931999`;
- pool size 400 and budget 50.

The lock must also assert that confirmatory raw/result files do not yet exist.

The primary paired comparison uses the clean 12-root registry and a refined
registry that adds four selected-parent aliases. Pools, references, temperature,
budget, and deterministic query/root tie policies are shared. The fixed-query
control feeds each refined run the clean run's ordered query sequence.

## 8. Frozen confirmatory gates

The promoted decision is
`GO_SOURCE_FAITHFUL_LEARNED_ABSTENTION_BRIDGE` only if every gate passes:

1. every pre-outcome hash and execution-lock assertion matches;
2. the identical exact-match function is used for candidate--candidate
   acquisition and reference--candidate posterior evidence;
3. four selected adapter files have distinct hashes, 50,817 parameters each,
   nonzero variance, and at least 95% nonzero learned values;
4. raw-input endpoint replay uses none of the forbidden runtime inputs;
5. every alias is coordinate-wise non-improving relative to the parent;
6. each alias loses at most 1.00 percentage point of full-test accuracy relative
   to the parent;
7. at least 50% of ordered clean/refined acquisition paths differ;
8. mean terminal root-regret increase is at least .005 (0.50 percentage point);
9. its 10,000-repetition paired-bootstrap 95% lower endpoint is above zero;
10. its one-sided paired sign-flip p-value is at most .05;
11. mean active-minus-fixed terminal harm is at least .005 and its bootstrap
    lower endpoint is above zero; and
12. every fixed-query terminal and cumulative root-regret difference is exactly
    zero.

All statistics use float64. The bootstrap seed is `93500`; the sign-flip seed is
`93501` with 100,000 Monte Carlo draws unless the exact distribution is smaller.
No gate, seed, task, threshold grid, or endpoint may be changed after test
outcomes are opened.

Failure produces `NO_GO_RETAIN_SOURCE_FAITHFUL_NEGATIVE`; the complete negative
record must remain in the artifact and manuscript. No replacement task may be
opened under Step 93.

## 9. Claim boundary and stopping rule

A pass establishes one previously unused task in which a learned, no-lookup,
quality-controlled same-root refinement harms a source-faithful single-s
Select-LLM specialization through acquisition. It does not establish live
registry admission, an independently fine-tuned full language model, secret-task
compromise, universal harm, or deployment prevalence.

Step 93 is the final score-targeted experiment for this review cycle. After its
pass or retained failure, new experiments are authorized only for a verified
factual error, direct novelty collision, or broken core causal inference—not for
unbounded requests for more tasks, selectors, or production deployments.
