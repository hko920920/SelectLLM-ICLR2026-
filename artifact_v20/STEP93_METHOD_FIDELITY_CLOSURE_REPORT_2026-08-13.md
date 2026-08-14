# Step 93 method-fidelity closure report

Date: 2026-08-13

## Outcome

The reviewer concern was valid: the original Select-LLM algorithm uses the
same similarity in candidate--candidate acquisition and reference--candidate
posterior evidence, whereas the prospective F1/BLEU conditions and the Step 92
AG News confidence condition used decoupled acquisition and scoring views.

This factual issue is now closed in two ways:

1. the manuscript explicitly labels those results as decoupled-view selector
   audits and restricts the source-faithful claim to the exact-answer primary
   specialization; and
2. a fresh preregistered Banking77 confirmation implements one literal
   `s(a,b)=1[a=b]` in both places and accepts no separate feedback matrix.

The new experiment is a valid retained negative for its terminal gate:
`NO_GO_RETAIN_SOURCE_FAITHFUL_NEGATIVE`.

## Frozen design and separation

- Fresh task: pinned processed `mteb/banking77` revision `18072d26...`.
- Frozen encoder: DistilBERT revision `12040acc...`.
- Hash partitions: 6,015 root-training, 1,996 error-adapter training, and
  1,982 calibration rows.
- Clean registry: 12 independently optimized 109,901-parameter intent roots.
- Refinement: four independently bootstrapped 50,817-parameter learned parent-
  error predictors. An alias emits its parent label or a unique out-of-label
  abstention token.
- Search: all 504 parent/temperature/loss-cap/trigger-cap conditions retained;
  the top 16 were rerun on 300 calibration pools before one condition froze.
- Confirmation: 1,000 paired sealed test pools, pool 400, budget 50.
- The Stage A source contains no holdout/sealed path token. The execution lock
  binds 54 files, including all 40 learned weights and both exact-match call
  sites, before confirmatory outputs existed.

Two pre-outcome administrative amendments retain harmless mirror discrepancies:
the pinned parquet contains 9,993/3,076 rather than the original public
metadata's 10,003/3,080 rows and includes an unused `label_text` field. Both
initial attempts stopped at assertions before row iteration or output.

## Confirmatory result

- Ordered path change: 99.8%.
- Query-set change: 99.1%.
- Terminal-root change: 36.6%.
- Alias accuracy loss relative to parent: 0.098--0.358 percentage points;
  every alias is coordinate-wise non-improving.
- Cumulative regret change: +0.040945, bootstrap 95% interval
  [+0.012905,+0.068910].
- Terminal root-regret change: +0.032 percentage points, bootstrap 95%
  interval [-0.055,+0.122], one-sided sign-flip p=.23594.
- Active-minus-fixed terminal effect: +0.032 points with the same interval.
- Fixed-query terminal, cumulative, and root-history differences: exactly zero.

Thus the learned no-lookup same-s construction changes acquisition and raises
the primary anytime/cumulative cost on this task, but it does not meet the
preregistered 0.50-point terminal or inference gates. It must not be described
as a source-faithful learned terminal success or as satisfying the reviewer's
score-changing condition.

## Numeric adjudication

The first machine result printed `INVALID_STEP93_IMPLEMENTATION` only because a
supplemental dense-versus-sparse equality check used absolute tolerance
`2.0e-16`; equivalent reduction orders differed by
`2.220446049250313e-16`, one binary64 ULP at one. The source hashes match,
reference feedback is bitwise exact, and all trajectories and effect gates are
unchanged.

An independent implementation reconstructs every clean/refined active and
fixed trajectory, inference, bootstrap/sign-flip result, coordinate constraint,
same-s call graph, and 128 raw-input endpoints in 82 checks. Its adjudication is
`PASS_STEP93_INDEPENDENT_RECONSTRUCTION` and the valid scientific decision is
the retained negative above. No task, threshold, gate, or outcome was replaced.

## Manuscript disposition

- Abstract, introduction, protocol, results, limitations, appendix, tables,
  reproducibility statement, and bibliography were revised.
- Step 92 remains a genuine learned raw-input positive result but is now
  accurately scoped to its decoupled-view selector.
- Step 93's positive cumulative and null terminal results are both in the main
  paper and Appendix I.
- The new PDF has 23 pages; all main scientific sections and conclusion end on
  page 9, with statements/references beginning on page 10.
- Paper PDF SHA-256:
  `a9586265a752edc94483359b8599507c9fb5122ebed487b83f456d1700170841`.

## Core bindings

- Preregistration:
  `5b91bbac3a3eac126dba44b13959c85d25a018fde1db78386662d976708f4bf0`
- Stage A complete ledger:
  `646674f795a0b9ab9597b725e8a6e7f512434e89214ac2fad2607f4e278781f0`
- Frozen configuration:
  `8de9e66fa32ac0c3e2b7fe4af9b0ea514d41d55cb7f7cbd496bcf72a6783963a`
- Execution lock:
  `f389d4be70abc4cf57f10fb7a4685ca4ee3a67a9ed9e3c7a75e0e8272d66ef62`
- Raw confirmatory arrays:
  `4b0ad4c7e321924c618f23429d84201d3787d57f7fb478f35624a9066030ab8d`
- Literal machine result:
  `bf6fe6dd3d07be47c9aafa534d0bfa1efd8a7bdbef9a702b61f3f9246d229698`
- Independent adjudication:
  `5ae09c1e3f6801940183e1e17ef06da4f683b3f2cdd8fed080dbca573b3be1e0`

## Stopping rule

Step 93 is the final score-targeted experiment for this review cycle. The
negative is retained and no replacement task is opened. Further experiments
require a newly verified factual error, direct novelty collision, or broken
core causal inference—not an unbounded request for another positive dataset.
