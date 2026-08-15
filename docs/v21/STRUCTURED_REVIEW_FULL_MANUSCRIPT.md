# V21 structured review panel — full manuscript

Date: 2026-08-16 (Asia/Seoul)  
Material reviewed: complete 28-page manuscript, including proofs, complete grids, prospective attempt register, SRSA-CIFAR closure, and reproducibility statements.  
Methodological warning: these are role-separated model-generated reviews, not external human reviews and not statistically independent. They provide a disciplined stress test, not an acceptance forecast.

## Panel summary

| Reviewer role | Score | Confidence | Recommendation |
|---|---:|---:|---|
| Active model selection / experimental design | 7 | 4/5 | Accept |
| Clone robustness / strategic replication | 6 | 4/5 | Weak accept |
| Skeptical practical LLM evaluation | 5 | 4/5 | Borderline |
| General ICLR machine learning | 6 | 4/5 | Weak accept |

**Median score:** 6  
**Central contribution recovered independently:** 4/4  
**Repeated fatal prior-work collision:** none  
**Repeated theorem or causal-validity error:** none  
**Panel decision under the preregistered V21 rule:** proceed to submission polishing; do not run another benchmark search.

---

## Reviewer A — Active model selection and experimental design

**Score:** 7 — accept  
**Confidence:** 4/5

### Overall assessment

The appendix substantially strengthens the main paper. It verifies that the primary T2 effects survive registry permutations and tie policies, records complete grids including negative cells, and separates source-faithful from decoupled-view audits. The learned sequence from AG News through Banking77, 20 Newsgroups, MultiNLI, SNLI, IMDB, and Yelp makes clear exactly which condition is established and which remains selective. The post-V20 register further demonstrates that failed quality, roster, and materiality gates were not silently relaxed.

### Closest prior work

The combination remains distinct from standard active model-selection methods and known-owner strategic-replication bandits. Active evaluation of perturbed agents is the closest sequential neighbor, but the root-level shared-label causal chain and fixed-query intervention are additional.

### Strengths

- The proof obligations and executable checks make the formal claims unusually auditable.
- The main causal intervention is exactly utility-preserving rather than approximately matched.
- The complete negative record changes the interpretation from cherry-picked universality to a mapped boundary.
- The root-aware remedy is both theoretically and empirically exact under authenticated mapping.

### Decision-changing criticisms

1. The complete appendix is too large to expect every reviewer to digest, so the main-text accounting is essential.
2. The practical unit “root” still needs a domain-specific authentication policy outside the paper.
3. The paper should avoid allowing the rich experiment history to distract from the single unit-of-analysis contribution.

### Recommendation rationale

The full manuscript contains a complete, technically sound, and genuinely useful evaluation-methodology contribution. Selective external magnitude limits the score, but does not outweigh the causal rigor and design implication.

---

## Reviewer B — Clone robustness, strategic replication, and false-name mechanisms

**Score:** 6 — marginally above the acceptance threshold  
**Confidence:** 4/5

### Overall assessment

The full proofs and comparison boundaries support the claim that no cited neighbor already studies the exact conjunction of latent roots, shared labels, adaptive acquisition, and terminal root selection. The contribution should still be described as a meaningful sequential extension of clone/false-name robustness, which the manuscript now does.

### Closest prior work

Strategic-replication bandits, Clone-Robust AI Alignment, clone-robust metric weighting, active evaluation, and UNREAL. The appendix confirms that these are treated as serious ancestors and baselines rather than dismissed.

### Strengths

- The product-frame theorem makes the information deficit precise.
- The constant-registry theorem demonstrates a concrete sequential amplification rather than only a static multiplicity effect.
- The approximate transcript-stability proposition usefully connects exact factorization to bounded implementation error.
- Defense experiments expose the false-merge/clean-cost limitations of response-only clone handling.

### Decision-changing criticisms

1. The root-factorization remedy assumes authenticated information whose acquisition and failure modes are not modeled.
2. The strongest practical construction is still targeted, and the broad theory is partly separated from realistic registry size.
3. The approximate-stability proposition is standard coupling machinery; it improves scope clarity but is not a major standalone theoretical contribution.

### Recommendation rationale

This is not the first clone or false-name paper, but it identifies a nontrivial sequential consequence at an important evaluation boundary and validates it carefully. I would weakly support acceptance.

---

## Reviewer C — Skeptical practical LLM evaluation

**Score:** 5 — borderline  
**Confidence:** 4/5

### Overall assessment

The complete record increases trust but confirms the practical limitation. Multiple source-faithful prospective families are cumulative-positive, terminal-null, quality-invalid, pre-outcome stopped, or submaterial. The strongest positive IMDB result is development-informed within-task, and the exact-quality T2 intervention uses references during construction. Thus the paper establishes a real controlled integrity issue, not a demonstrated common failure of deployed LLM evaluation.

### Closest prior work

Arena candidacy, leaderboard perturbation, micro-benchmark reliability, and active-evaluation methods. These works have more direct operational stories; this paper has stronger causal identification.

### Strengths

- The attempt register and literal status labels are unusually transparent.
- The authors do not reclassify BoolQ, Amazon, PRAA, or SRSA failures as successes.
- SRSA is especially informative: semantic utility is exactly fixed, paths change almost universally, controls pass, but terminal harm remains small.
- The artifact and AI-use statements are candid.

### Decision-changing criticisms

1. The production relevance remains unmeasured and may be modest.
2. Targeted construction and extensive development history limit claims about prevalence or easy exploitability.
3. A live admission demonstration or independent provider incentive model would materially strengthen the paper, but adding a selectively chosen benchmark now would not.

### Recommendation rationale

I remain borderline because practical significance is central to my evaluation. I would not argue that the paper is unsound or misleading; my score reflects the gap between an elegant integrity phenomenon and demonstrated real-world impact.

---

## Reviewer D — General ICLR machine learning

**Score:** 6 — marginally above the acceptance threshold  
**Confidence:** 4/5

### Overall assessment

The appendix confirms that the paper is not relying on a single fragile experiment. The theory is checked, the primary causal result is reconstructed independently, and the broad empirical program reports where the phenomenon fails. The contribution is understandable as a new experimental-unit requirement for adaptive evaluation.

### Closest prior work

Active model selection and clone robustness. The paper's distinction is sufficiently clear after reading both the introduction and complete related-work boundary.

### Strengths

- Coherent theory, causal experiment, transfer audit, and remedy.
- Strong reproducibility and explicit AI-use disclosure.
- Negative evidence improves rather than weakens the final scoped claim.

### Decision-changing criticisms

1. The appendix could be made easier to navigate with a one-page artifact/attempt index.
2. Several historical implementation incidents are scientifically harmless but editorially distracting.
3. The work may be better appreciated as evaluation methodology than as a new general-purpose ML algorithm.

### Recommendation rationale

The paper meets the bar for novel and well-supported knowledge. I would weakly accept, with the expectation that not every reviewer will value the evaluation-integrity framing equally.

---

## Full-manuscript decision

The panel's median is 6, all four roles recover the same central contribution, and no repeated fatal prior-work or validity defect appears. The remaining disagreement is about significance, not truth. Under the V21 decision rule this supports **submission after final artifact/anonymity polishing**, not another experimental family.
