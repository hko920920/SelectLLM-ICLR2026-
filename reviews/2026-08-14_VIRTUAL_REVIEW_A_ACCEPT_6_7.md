# Virtual pre-submission review A

Manuscript: **Candidate Lists Are Priors: Evidence-Frame Dependence in Active
Model Selection**  
Supplied date: 2026-08-14  
Estimated score: **6.7/10**  
Recommendation: **acceptance**

## Summary

The review states that the submitted candidate list acts as an experimental
prior over accountable roots because active selectors allocate mass per listed
entry.  It credits the paper with formalizing the missing candidate evidence
frame, proving a budget-linear lower bound for frame-blind selectors, giving a
constant-alias Select-LLM amplification, and empirically isolating the path by
which quality-equivalent aliases alter acquired evidence and downstream root
regret.

The review treats the natural exact version alias as an anchor for transcript
dependence, the T2 interventions as the clean causal test, and the disjoint
held-out IMDB study as the strongest outcome-free learned confirmation.

## Strengths recorded by the reviewer

1. **Technical novelty.** The roots, mapping, and root mass in the candidate
   evidence frame make the implicit prior precise.  Theorem 1 supplies a tight
   budget-linear frame-blind loss; Theorem 2 supplies mechanism-specific
   constant-registry amplification; Proposition 1 provides an invariance
   design target.
2. **Experimental rigor.** Paired locks, fixed-query controls, registry
   permutations, tie policies, path audits, and multiple-testing corrections
   are unusually extensive.  The T2 aliases preserve per-example correctness,
   and fixed-query equality isolates acquisition mediation.
3. **Held-out evidence.** The IMDB outcome uses disjoint development and
   held-out rows, pre-outcome locks, strict quality/directionality gates, and
   3,000 paired runs.
4. **Selective transfer and honest negatives.** The T1 and official LLM
   Selector results show that transfer exists but is not universal.  Retained
   negative outcomes make the scope credible.
5. **Significance.** The work identifies a failure mode at the active
   evaluation/registry boundary and gives actionable root-authentication and
   refinement-invariance requirements.

## Weaknesses recorded by the reviewer

### Technical limitations

- Theorem 2 is tied to Select-LLM with exact-match evidence, limiting formal
  generality.
- Exact-answer similarity may not represent free-form or judge-based
  acquisition mechanisms.
- The threat model assumes pre-acquisition registry control without strong
  provenance or deduplication.

### Experimental limitations

- Harm is selective: several scenarios are null or beneficial; CODA and many
  learned same-s tasks do not pass terminal gates.
- The IMDB confirmation is development-informed and uses adapters from a
  shared base; independent bases would improve external validity.
- Robustness to mild departures from exact quality equivalence could be
  characterized further.

### Presentation and comparison limitations

- Dense writing and protocol detail make the central mechanism difficult to
  follow.
- The experiment breadth obscures the main takeaways.
- Connections to active-learning invariance, duplicate Bayesian hypotheses,
  and dependency-model acquisition could be expanded.

## Questions preserved

1. How sensitive are the results to small parent/alias utility deviations?
2. Which non-exact acquisition objectives were tested for frame dependence?
3. How should a real registry authenticate root mapping and mass?
4. Can response-only soft deduplication improve the robustness/utility
   trade-off?
5. Within IMDB, how much of the effect comes from adapter choice versus
   threshold and temperature?

## Overall assessment

The reviewer concludes that the central claim is convincingly supported, even
though harm is not universal and the strongest construction is
mechanism-specific.  The exact conclusion was an acceptance recommendation at
an estimated 6.7/10.

## Subsequent disposition

The writing concern was addressed by reorganizing the abstract, introduction,
evidence ladder, result ladder, and limitations.  The remaining issue was
prospective cross-task generality, which motivated Steps 105, 113, and 114.
Their literal outcomes are retained in `docs/CURRENT_STATE.md`.
