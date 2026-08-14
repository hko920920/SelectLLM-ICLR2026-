# Virtual pre-submission review B

Manuscript state: 28-page manuscript including the held-out IMDB result  
Supplied date: 2026-08-14  
ICLR score: **6 / marginally above acceptance**  
Project-specific score: **4 / borderline accept**  
Confidence: **4/5**  
Soundness: **4/4**  
Presentation: **3/4**  
Contribution: **3/4**  
Final recommendation: **Borderline accept**  
Reviewer discussion position: **advocate for acceptance**

## Core claim recorded by the reviewer

In active model selection without authenticated roots and root-level mass,
adding behaviorally redundant or quality-equivalent aliases from one root can
change the order and composition of acquired labels and thereby worsen final
root selection and cumulative regret.

## Principal evidence credited

- The candidate evidence frame distinguishes submitted entries, accountable
  roots, and root mass.
- Theorem 1 establishes frame-blind budget-linear loss; Theorem 2 gives a
  four-alias Select-LLM amplification; Proposition 1 gives sufficient
  refinement-invariance conditions.
- A natural exact alias changes all 1,500 paired paths but does not establish
  natural terminal harm.
- T2-A raises cumulative regret on MedQA, GSM8K, and OpenBookQA and terminal
  regret on the first two; fixed-query equality establishes mediation.
- T1 passes 3/8 and official LLM Selector passes 2/6, supporting selective but
  not broad transfer.
- The held-out IMDB learned same-s study reports +0.702 points terminal and
  active-minus-fixed harm, positive inference, 99.93% path change, 60.67%
  final-root change, and exact fixed-query zero.

## Causal-chain judgment

The review marks the full chain as supported for targeted T2 and held-out
IMDB:

`refinement -> acquisition path -> acquired evidence -> root-level cost`.

It marks natural aliases, untargeted additions, arbitrary tasks/selectors, and
production compromise as partial or unsupported.  It explicitly states that
the manuscript does not overclaim these stronger conclusions.

## Novelty judgment

Classification: **B, meaningful sequential extension**.

The review finds no direct collision after comparing active evaluation clones,
strategic-replication bandits, duplicate handling in active model selection,
Unique Rashomon Sets, strategic candidacy, clone-robust weighting, false-name
data attribution, and Select-LLM itself.  It distinguishes this paper by the
combination of latent same-root refinement, a shared label that updates all
candidates, irreversible acquisition-transcript change, fixed-query mediation,
and terminal root-level selection cost.

## Decision-changing weakness 1: operational realism

The strongest MedQA/GSM8K effects use public or independently acquired
references.  Runtime wrappers and learned adapters improve feasibility, but
the paper does not demonstrate live registry admission, hidden-label attack
success, independent full-base checkpoints, provider incentives, or
production prevalence.

The review says the manuscript already acknowledges this.  Therefore the
criticism is a limitation on practical significance, not a causal-validity
error or an overclaim.

Bounded repair proposed by the review: one admission-style audit using no
evaluation labels or item lookup, fixed independently executable endpoints,
an explicit identity/admission policy, an unseen evaluation, material
terminal active-minus-fixed harm, and fixed-query equality.

## Decision-changing weakness 2: prospective cross-task generality

The review notes that source-faithful learned studies on Banking77, 20
Newsgroups, MultiNLI, and the locked SNLI primary did not cleanly pass the
terminal gate, while IMDB used a configuration selected on disjoint rows from
the same benchmark.  Thus IMDB establishes outcome-held-out within-task
existence but not a frozen pipeline transferring to a completely new task.

Bounded repair proposed by the review: select an untouched task and registry,
freeze the roster, training recipe, threshold, budget, temperature, and gates,
then run one sealed outcome without task-internal configuration search.

## Score-changing condition

The single condition proposed to move project score 4 to 5 was:

> A completely untouched task and registry where a fully frozen,
> holdout-reference-free, item-lookup-free, source-faithful same-s pipeline
> produces material terminal and active-minus-fixed harm with exact
> fixed-query zero in one sealed outcome.

## Final assessment

The review supports acceptance based on the distinct sequential object and
strong causal identification, not on production compromise.  It places the
paper at borderline rather than confident accept because external generality
and operational transfer remain selective.

## Subsequent disposition

- Step 105 Yelp: prospective cross-benchmark `NO_GO`, missing only the frozen
  +0.5-point materiality thresholds.
- Step 113 BoolQ: strong effect but `NO_GO` because the alias-quality cap
  failed.
- Step 114 Amazon: all alias-quality checks passed, but the experiment stopped
  before outcome because the designated parent tied a challenger.

Accordingly, the score-changing condition remains open.  The attempts and
their stops must be disclosed in any future claim of prospective transfer.
