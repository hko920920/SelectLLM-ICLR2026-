# Human-reviewer calibration for SelectLLM V20/V21

Date: 2026-08-16 (Asia/Seoul)

This is not a probability model derived from historical acceptance data. It is a structured forecast based on the full manuscript, the closest-work collision audit, the literal experiment register, and the ICLR review dimensions of soundness, presentation, contribution, and relevance.

## Bottom line before V21 rewriting

**Current V20-like manuscript:** scientifically strong, editorially overfull, and reviewer-assignment sensitive.

Most likely independent score pattern if submitted without further restructuring:

```text
6 / 6 / 5
```

Plausible adverse pattern:

```text
6 / 5 / 3
```

Plausible favorable pattern:

```text
8 / 6 / 6
```

The central risk is not a fatal correctness flaw. It is that one reviewer interprets the paper as a selective clone-attack extension with weak operational significance rather than as a new evaluation-unit and invariance result.

## Axis-level assessment

| Axis | Current assessment | Why |
|---|---:|---|
| Soundness | **strong (7.5–8.5/10)** | Formal results, exact-quality interventions, paired controls, fixed-query mediation, complete grids, independent reconstruction. |
| Novelty | **meaningful extension (5.5–7/10)** | No single prior work has the full latent-root/shared-label/adaptive-transcript/root-cost object, but every component has close neighbors. |
| Empirical significance | **mixed (5–6.5/10)** | Large controlled and IMDB effects; prospective material transfer repeatedly selective or negative. |
| Operational realism | **limited (4.5–6/10)** | Executable endpoints exist, but no live admission, provider incentive, prevalence, or independent production registry. |
| Presentation | **currently fair-to-good (5.5–6.5/10)** | Clear local writing, but too many evidence layers compete and related work is under-expanded. |
| Reproducibility | **very strong (8.5–9.5/10)** | Locks, ledgers, validators, raw arrays, negative retention. The volume itself can overwhelm reviewers. |

## Four reviewer personas

### Reviewer 1 — Active model selection / Bayesian experimental design

**Main-text-only forecast:** 6  
**Full-paper forecast:** 6–7  
**Confidence:** 4/5

Likely positive reading:

- recognizes that the candidate roster implicitly defines a prior in candidate-indexed mechanisms;
- values the exact source specialization and fixed-query mediation;
- sees the root map as a missing experimental unit;
- regards T2 and IMDB as convincing causal evidence.

Likely concerns:

1. The Select-LLM amplification theorem is mechanism-specific.
2. Practical terminal transfer is selective.
3. Main text contains too many audit families and numerical details.

Decision-changing condition: none if the novelty boundary is explicit. This reviewer is likely to support acceptance after a clarity rewrite.

### Reviewer 2 — Clone robustness / strategic bandits / social choice

**Main-text-only forecast:** 5  
**Full-paper forecast:** 5–6  
**Confidence:** 4/5

Likely positive reading:

- appreciates the latent-root formalization and factorization remedy;
- recognizes that shared labels and irreversible acquisition differ from static clone robustness;
- values exact-quality variants and negative controls.

Likely concerns:

1. Strategic replication and clone-robustness literatures may appear to contain the core idea already.
2. Proposition 1 is only a sufficient factorization statement unless strengthened.
3. The paper may read as an application of known false-name/clone principles.

Decision-changing condition: the first two pages and closest-work table must make the four-part distinction unavoidable. A stronger characterization or approximate-stability result would help, but is not mandatory if positioning is excellent.

### Reviewer 3 — Skeptical practical LLM evaluation researcher

**Main-text-only forecast:** 3–5  
**Full-paper forecast:** 4–5  
**Confidence:** 3–4/5

Likely positive reading:

- admits that path dependence and fixed-query mediation are carefully shown;
- likes the complete failure disclosure and executable endpoints.

Likely concerns:

1. The strongest controlled construction uses references during alias construction.
2. IMDB is development-informed within one benchmark.
3. Yelp, PRAA, and SRSA do not provide material untouched-task success.
4. No production platform, provider incentive, or prevalence estimate is shown.
5. Many attempted tasks can look like search for a positive result.

Decision-changing condition: the paper must explicitly sell evaluation integrity and experimental-unit validity, not a prevalent production security exploit. It must foreground selective magnitude as a finding rather than an apology.

### Reviewer 4 — General ICLR machine learning reviewer

**Main-text-only forecast:** 5–6  
**Full-paper forecast:** 6  
**Confidence:** 3/5

Likely positive reading:

- sees theory + causal experiment + remedy as a complete paper;
- appreciates broad empirical coverage.

Likely concerns:

1. Hard to identify the single main contribution.
2. Terminology burden: entry, root, frame, refinement, T0/T1/T2, same-$s$, terminal/cumulative regret.
3. Related-work section is too compressed to independently verify novelty.
4. Dense result inventory obscures the story.

Decision-changing condition: compression and a simple claim/evidence ladder.

## Main-text-only versus full-manuscript gap

A reviewer is not guaranteed to read the appendix. Current main text already discloses important negatives, but the cumulative attempt history and distinctions among development-informed, held-out, prospective, stopped, and post-confirmatory evidence remain too dependent on appendix reading.

Expected score loss if the reviewer reads only nine pages:

- novelty expert: **0–1 point** if the closest-work boundary is not expanded;
- practical evaluator: **0–1 point** if failed prospective attempts are not summarized;
- general reviewer: **1 point** from cognitive overload.

V21 must therefore place a compact attempt/accounting row and the exact novelty boundary in main text.

## Fatal versus nonfatal criticisms

### Potentially fatal

1. A specific prior work is shown to already combine latent same-root multiplicity, shared-label adaptive acquisition, and terminal root selection with equivalent mediation analysis.
2. The T2 exact-quality claim is false or confounded.
3. Fixed-query equality is not actually implemented using identical acquired references/root scoring.
4. The main result depends on undisclosed outcome-based selection.
5. The theorem statement or proof has a material error.

The current audit has not identified any of these.

### Serious but nonfatal

1. Practical prevalence is unknown.
2. Prospective terminal transfer is selective.
3. Reference access in T2 is strong.
4. Similarity defenses have utility trade-offs.
5. The paper is a meaningful extension rather than a wholly new paradigm.

These should lower confidence or score, but do not invalidate the contribution if scope is precise.

### Presentation-only but score-relevant

1. Evidence inventory in the abstract.
2. Two-paragraph related work.
3. Too many acronyms and stage labels.
4. Appendix volume and historical workflow detail.
5. The title can sound more universal than the actual candidate-indexed scope.

## Calibrated acceptance assessment

The paper is not accurately described as “obviously reject” or “confident accept.” It is a **real borderline-to-weak-accept paper with high variance**.

A defensible qualitative forecast after V21 reconstruction is:

- central score mass: **6**;
- meaningful chance of a **5** from significance/novelty caution;
- nontrivial risk of a **3–4** from a reviewer who reads it as task shopping or an unrealistic attack;
- meaningful chance of a **7–8** from a reviewer who values the experimental-unit insight and causal rigor.

The highest expected-value intervention is not another benchmark. It is reducing the probability of the adverse interpretation.

## Objective go/no-go rule after V21

Run four genuinely blind reviews on the final PDF.

### Submit

- median ICLR score at least 6;
- at least three of four reviewers independently state the same central contribution;
- no two reviewers identify the same prior work as subsuming the paper;
- no causal-validity or theorem error.

### One rewrite, no new experiment

- median near 5;
- criticisms concentrate on novelty communication, evidence hierarchy, or presentation;
- soundness remains accepted.

### Pivot or split

- median at most 4;
- at least two reviewers independently identify the same fatal prior-work collision;
- at least two reviewers conclude that the controlled causal setting has no publishable significance even under the narrowed integrity framing;
- main-text readers cannot recover the central claim after the V21 rewrite.

## Current decision

Proceed with V21 reconstruction and blind review. Do not resume positive-task search. The current evidence is sufficient for a serious submission; the remaining uncertainty is reviewer interpretation and literature positioning.
