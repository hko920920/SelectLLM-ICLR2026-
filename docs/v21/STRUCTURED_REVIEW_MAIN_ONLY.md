# V21 structured review panel — main text only

Date: 2026-08-16 (Asia/Seoul)  
Material reviewed: title through Section 7, excluding references and appendix.  
Methodological warning: these are role-separated model-generated reviews, not external human reviews and not statistically independent. They are used to identify interpretation risk, not to estimate acceptance probability.

## Panel summary

| Reviewer role | Score | Confidence | Recommendation |
|---|---:|---:|---|
| Active model selection / experimental design | 6 | 4/5 | Weak accept |
| Clone robustness / strategic replication | 5 | 4/5 | Borderline |
| Skeptical practical LLM evaluation | 5 | 4/5 | Borderline |
| General ICLR machine learning | 6 | 3/5 | Weak accept |

**Median score:** 5.5  
**Central contribution recovered independently:** 4/4  
**Repeated fatal prior-work collision:** none  
**Repeated causal-validity error:** none

---

## Reviewer A — Active model selection and experimental design

**Score:** 6 — marginally above the acceptance threshold  
**Confidence:** 4/5

### Summary

The paper argues that candidate-indexed active model-selection procedures silently treat submitted entries as experimental units, even when several entries correspond to one accountable model root. It formalizes a candidate evidence frame, proves a missing-frame lower bound and a Select-LLM amplification construction, and uses exact-quality refinements plus fixed-query counterfactuals to show that entry mass changes acquisition, reusable evidence, and final root-level regret. Learned IMDB endpoints and retained negative transfers delimit the scope; authenticated root factorization gives the design remedy.

### Closest prior work

Strategic-replication bandits and active evaluation of perturbed agents are closest sequentially; MODEL SELECTOR, CODA, LLM Selector, and Select-LLM supply the base problem. The paper's distinct element is the shared-label, latent-root experimental unit and the fixed-query mediation to selected-root cost.

### Strengths

- The fixed-query counterfactual is unusually clean and directly identifies acquisition as the mediator.
- T2 preserves per-example correctness exactly, removing a common candidate-quality confound.
- The paper distinguishes path change, evidence replacement, cumulative cost, and terminal cost instead of treating them as interchangeable.
- The negative Yelp and CIFAR results make the external boundary credible.

### Decision-changing criticisms

1. The strongest formal amplification is specific to an exact-match Select-LLM specialization, so the theory does not by itself establish broad active-selection fragility.
2. Practical terminal transfer is selective; the most prospective audits remain below the stated materiality threshold.
3. The accountable-root definition is claim-relative and operational authentication is left to external provenance or governance.

### Recommendation rationale

The causal and experimental-unit contribution is sufficiently distinct and well supported for acceptance, despite limited operational generality. I would support the paper as an integrity analysis rather than a production-security demonstration.

---

## Reviewer B — Clone robustness, strategic replication, and false-name mechanisms

**Score:** 5 — borderline  
**Confidence:** 4/5

### Summary

This work imports clone/false-name concerns into active model selection and identifies an additional sequential consequence: entry multiplicity alters which shared labels are acquired, so later quotienting cannot reconstruct the counterfactual transcript. The formal frame and root-factorization requirement are sensible, and the empirical mediation evidence is strong.

### Closest prior work

Clone-Robust AI Alignment, strategic-replication bandits, clone-robust metric weighting, and UNREAL. None appears to contain the complete shared-label/root-selection object, but the paper is a synthesis and sequential extension rather than a wholly new clone principle.

### Strengths

- The paper now states the novelty obligation directly rather than claiming priority over clones or Sybils.
- Exact-quality response-distinct variants are stronger than literal duplicate examples.
- Root-aware invariance is the correct conceptual endpoint and is tested exactly.

### Decision-changing criticisms

1. Proposition 1 is a sufficient factorization condition that is close to definitional; the deeper novelty rests more on the problem formulation and causal evidence than on this proposition.
2. The missing-frame theorem uses a deliberately constructed exponential family, making its practical interpretation limited.
3. A reviewer may still view the paper as applying established false-name robustness principles to a new evaluation pipeline unless the shared-label irreversibility remains visually central.

### Recommendation rationale

I do not find a direct subsuming prior, but the conceptual distance from clone-robustness and strategic-replication work is moderate rather than large. The paper is technically strong enough for a borderline-to-weak-accept judgment; novelty interpretation will determine the final score.

---

## Reviewer C — Skeptical practical LLM evaluation

**Score:** 5 — borderline  
**Confidence:** 4/5

### Summary

The paper demonstrates that candidate representation can drastically change active acquisition and sometimes worsen root selection. Its causal controls are convincing. The practical threat claim is intentionally narrow: the strongest exact-quality construction uses reference access, IMDB is development-informed within-task, and cleaner prospective transfers are submaterial.

### Closest prior work

Leaderboard and arena robustness work, micro-benchmark reliability, strategic candidacy, and active LLM evaluation. Those works often provide more immediate real-system consequences, whereas this paper provides a deeper offline causal explanation.

### Strengths

- The paper no longer equates path divergence with practical harm.
- It reports prospective failures and beneficial/null outcomes instead of presenting only successful tasks.
- Raw-input endpoints establish feasibility beyond cached response edits.
- Fixed-query and root-aware controls are persuasive.

### Decision-changing criticisms

1. No live registry admission, provider incentive, prevalence estimate, or hidden-task success is demonstrated.
2. The headline T2 construction has strong reference access; the cleaner prospective studies do not reach the material terminal threshold.
3. The large research program still creates a selection-risk perception, although the new prospective-accounting paragraph substantially reduces it.

### Recommendation rationale

I would not accept this as evidence of an important prevalent attack. I could accept it as an evaluation-integrity paper if the venue values carefully identified failure modes and negative boundaries. The current main text is honest enough for a borderline score, but practical significance remains the limiting factor.

---

## Reviewer D — General ICLR machine learning

**Score:** 6 — marginally above the acceptance threshold  
**Confidence:** 3/5

### Summary

The manuscript presents a coherent theory–experiment–remedy story around the fact that a candidate list may not coincide with the model units an evaluation intends to compare. It proves lower-bound and amplification results, establishes a controlled causal chain with paired interventions, and gives executable and cross-method evidence with transparent limitations.

### Closest prior work

Active model selection, active testing, clone robustness, and strategic replication. I rely on the paper's comparison for some of the fine-grained distinctions.

### Strengths

- Clear first-page figure and a recoverable central thesis.
- Strong experimental controls and unusually complete reporting.
- The theory and empirical mechanism point to the same design requirement.
- Negative results prevent the claim from becoming implausibly broad.

### Decision-changing criticisms

1. The terminology and evidence ladder remain dense for a nine-page paper.
2. The contribution is more about evaluation methodology and experimental units than a new learning algorithm; significance depends on reviewer taste.
3. The main text cannot fully establish all implementation and preregistration details without appendix reliance.

### Recommendation rationale

The paper provides new and useful knowledge with sound evidence. I would lean accept while expecting debate about practical reach and novelty magnitude.

---

## Main-text-only decision

The panel does **not** justify a confident-accept claim. It does justify continuing to full-manuscript review rather than abandoning the topic. The median 5.5 is driven by significance and prior-work interpretation, not by a repeated correctness defect. The only warranted intervention is framing/transparency refinement; no new task search is indicated.
