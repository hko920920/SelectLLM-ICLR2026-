# Closest-work and OpenReview audit

Date: 2026-08-16 (Asia/Seoul)

## Audit boundary

The fixed closest-work set was audited at three levels:

1. paper title, abstract, contribution, and method from the public paper or proceedings page;
2. public venue status from OpenReview/proceedings indexes;
3. official review, rebuttal, and decision bodies where the public forum API permits retrieval.

The third level could not be completed from this environment. Two reproducible GitHub Actions attempts queried both `api2.openreview.net/notes` and `api.openreview.net/notes` by exact title. Every request returned `403 ChallengeRequiredError`. The second run was `31900579404`; its artifact `9250965782` contains the exact failure index. We therefore do **not** claim to have read or calibrated against official review bodies that were not retrieved. Venue decisions and paper content remain independently public.

## Public venue status verified

| Work | Public status used in this audit | Public page |
|---|---|---|
| Clone-Robust AI Alignment | ICML 2025 poster | https://openreview.net/forum?id=vbw9hSlOaG |
| Dropping Just a Handful of Preferences Can Change Top Large Language Model Rankings | ICLR 2026 poster | https://openreview.net/forum?id=jNiEMDsRgc |
| How Reliable is Language Model Micro-Benchmarking? | ICLR 2026 oral | https://arxiv.org/abs/2510.08730 |
| The Leaderboard Illusion | NeurIPS 2025 Datasets and Benchmarks Track poster | https://papers.nips.cc/paper_files/paper/2025/hash/70a93f260a51123b3c0e33ecd1b4de97-Abstract-Datasets_and_Benchmarks_Track.html |
| All Models Are Wrong, Some Are Useful: Model Selection with Limited Labels | AISTATS 2025 | https://proceedings.mlr.press/v258/okanovic25a.html |
| Strategic Candidacy in Generative AI Arenas | ICML 2026 / arXiv public paper | https://arxiv.org/abs/2603.26891 |
| Active Evaluation of General Agents | AAMAS 2026 / arXiv public paper | https://arxiv.org/abs/2601.07651 |

## Collision conclusions supported by paper content

### 1. Clone-Robust AI Alignment is the strongest static collision

It defines robustness to approximate clones for RLHF reward learning and proposes similarity-weighted MLE. This directly overlaps our motivation that alternative multiplicity should not change a learned conclusion. It does not, however, choose future labels adaptively, model an accountable root map, or measure terminal root selection after an irreversible acquisition transcript.

**Verdict:** high conceptual overlap; does not subsume the complete sequential object.

### 2. Strategic-replication bandits are the strongest mechanism-design ancestor

They formalize strategic agents registering duplicate arms and design mechanisms that neutralize replication. Their owner map is supplied and rewards are arm-local. In our setting the owner/root map can be latent, one acquired label updates all entries, and the final claim concerns which accountable model root is selected.

**Verdict:** strong antecedent; our paper is a meaningful shared-label model-selection extension, not the first replication problem.

### 3. UNREAL and active evaluation are the strongest sequential collisions

UNREAL filters redundant predictors in active learning, and active-evaluation work studies sequential assessment of similar or perturbed agents. These works make it unsafe to claim the first sequential clone effect. The remaining distinction is the conjunction of latent same-root refinement, one reusable label for all entries, entry-indexed candidate mass, fixed-query acquisition mediation, and final root-level regret.

**Verdict:** closest sequential boundary; must be explained before the contribution list.

### 4. Strategic candidacy and leaderboard robustness dominate practical immediacy

Arena candidacy, preference deletion, leaderboard reliability, and micro-benchmarking offer direct real-system ranking consequences. Our paper has deeper causal and experimental-unit structure but a less immediate prevalence story. Reviewers can reasonably value those papers more highly on practical significance even if they do not subsume our method.

**Verdict:** significance comparator, not novelty collision.

### 5. False-name quotients and lineage define the remedy obligation

False-name-resistant quotients, dilution priors, and lineage attestation support the principle that a robust counting unit requires information beyond output similarity. They do not supply our active selector, but they make broad claims about a novel remedy inappropriate.

**Verdict:** cite as enabling identity evidence; claim only the transcript-level consequence and factorization condition.

## What actual OpenReview review bodies could still change

Because official review texts were unavailable, the following remain uncertain:

- which novelty distinction reviewers explicitly accepted for the closest papers;
- which practical limitations were tolerated versus considered decision-changing;
- whether rebuttals changed reviewer scores;
- whether a rejected near-neighbor failed for reasons directly relevant to this manuscript.

This uncertainty is carried into `HUMAN_REVIEWER_CALIBRATION.md`; it is not replaced by invented review summaries.

## Binding novelty position for V21

> Prior work studies duplicate arms with known owners, approximate clones in static preference learning, redundant committees in active learning, or clones in fixed rankings. This paper studies the remaining shared-label sequential object: submitted entries may hide a smaller set of accountable roots, entry mass changes which reusable labels are acquired, and the resulting irreversible transcript changes final root selection; a fixed-query counterfactual isolates this mediation.

## Resulting manuscript obligation

- Do not claim first clone, Sybil, replication, or leaderboard-manipulation effect.
- Put the shared-label/latent-root/irreversible-transcript distinction in the introduction, not only the appendix.
- Treat clone-robust weighting and quotient methods as serious baselines.
- Frame the contribution as an evaluation-unit and invariance result.
- Treat production prevalence and material prospective transfer as limitations.
