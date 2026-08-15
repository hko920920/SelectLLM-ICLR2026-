# V21 reconstruction plan and execution ledger

Date: 2026-08-16 (Asia/Seoul)

## Purpose

Reduce reviewer variance without selecting a new positive task. The paper is treated as an evaluation-unit and causal-invariance contribution, not as a universal production attack.

## Frozen inputs

- Research-history branch: `research/srsa-cifar-audit`
- Frozen source HEAD: `d96df211016fc6fd6c3f6a41137d19f3ede794de`
- V21 branch: `paper/v21-calibrated`
- V20 positive and negative experiment artifacts remain unchanged.
- No new benchmark, threshold, alias recipe, selector setting, or materiality gate is introduced.

## Completed audit documents

- `CLAIM_EVIDENCE_MATRIX.md`
- `NOVELTY_COLLISION_MATRIX.md`
- `HUMAN_REVIEWER_CALIBRATION.md`
- `CLOSEST_WORK_OPENREVIEW_AUDIT.md`

## Manuscript changes executed

1. **Conditional title**  
   `When Candidate Lists Become Priors` narrows the scope to candidate-indexed mechanisms rather than implying a universal property of every candidate list.

2. **Abstract compression**  
   Replaced the experiment inventory with one chain: missing experimental unit → adaptive transcript → fixed-query mediation → selective transfer → root-aware remedy.

3. **Introduction collision defense**  
   Added the explicit distinction from strategic replication, static clone robustness, active-learning redundancy, active evaluation, and leaderboard attacks before the contribution list.

4. **Theory generalization**  
   Added approximate transcript stability: uniform per-step total-variation error accumulates by at most `1 - product(1-epsilon_t)` and bounds any finite-range terminal cost.

5. **Result hierarchy**  
   Rebuilt the result ladder into occurrence, exact-quality causality, executable held-out confirmation, prospective magnitude boundaries, and second-family/untargeted scope.

6. **Post-V20 transparency**  
   Added a compact appendix register for BoolQ, Amazon, PRAA, PRAA-E, and SRSA-CIFAR, including SRSA's literal `NO_GO` and the duplicate-execution incident.

7. **Related-work expansion**  
   Replaced two compressed paragraphs with four obligation-boundary paragraphs.

8. **Limitations and conclusion**  
   Explicitly reject universal terminal harm, production compromise, live-registry admission, and path-change-as-harm interpretations.

## Build gate

The V21 GitHub workflow must establish all of the following before blind review:

- clean LaTeX build;
- main-text marker at most 9 pages;
- no unresolved references or citations;
- no multiply defined labels;
- required scope statements present in PDF text;
- forbidden overclaim patterns absent.

## Blind review gate

The built PDF is reviewed in two modes:

1. main-text-only, approximating a reviewer who does not rely on the appendix;
2. full manuscript, including proofs and the complete attempt register.

Four roles are used:

- active model selection / experimental design;
- clone robustness / strategic replication;
- skeptical practical LLM evaluation;
- general ICLR machine learning.

Each review must state a score, confidence, closest prior work, one acceptance/rejection reason, and at most three decision-changing criticisms.

## Final decision rule

### Submit

- median blind score at least 6;
- at least 3/4 reviewers recover the same central contribution;
- no repeated fatal prior-work collision;
- no theorem or causal-validity error.

### One rewrite, no new experiment

- median near 5;
- soundness accepted;
- criticisms concentrate on framing, density, or related-work communication.

### Pivot or split

- median at most 4;
- two reviewers identify the same subsuming prior work or fatal validity defect;
- the narrowed evaluation-integrity contribution is judged insufficient even after reconstruction.

## Prohibited rescue actions

- new task search;
- materiality relaxation;
- post-outcome threshold or alias-count changes;
- substituting a secondary condition for a failed primary;
- opening sealed Amazon/PRAA outcomes;
- introducing a third SRSA task.
