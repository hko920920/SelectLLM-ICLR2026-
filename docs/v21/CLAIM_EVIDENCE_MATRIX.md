# V21 claim–evidence matrix

Date: 2026-08-16 (Asia/Seoul)  
Scope: current V20 manuscript plus the complete post-V20 attempt register.  
Decision rule: a claim is promoted only when its strongest evidence directly establishes the claimed scope. A stronger interpretation is not inherited from a weaker experiment.

## Status vocabulary

- **ESTABLISHED** — directly supported by theory or a locked causal experiment.
- **ESTABLISHED IN A CONTROLLED SETTING** — directly supported, but only under an explicit information or mechanism boundary.
- **PARTIALLY ESTABLISHED** — direction or mechanism is supported, but generality, materiality, or operational scope is incomplete.
- **UNSUPPORTED / REMOVE** — no current evidence at the stated scope.

## Core claims

| ID | Candidate claim | Strongest evidence | Exact status | Decision-changing caveat | V21 action |
|---|---|---|---|---|---|
| C1 | A candidate-indexed active selector can depend on how the same underlying alternatives are represented in the submitted roster. | Natural exact version alias changes all 1,500 paired acquisition paths while adding no new response or utility vector; random-query control is identical. | **ESTABLISHED** | Natural terminal harm is absent and unique-label replacement is small. | Keep as the occurrence anchor; never describe it as a natural attack success. |
| C2 | Submitted entries and accountable model roots are distinct experimental units. | Candidate evidence frame $\Phi=(R,\rho,w)$; response-compatible worlds with different latent frames. | **ESTABLISHED** | The root is claim-relative and may require governance or provenance evidence. | Put this distinction in the first paragraph and Figure 1. |
| C3 | Labels alone cannot generally recover a missing candidate evidence frame. | Tight budget-linear missing-frame theorem and response-compatible construction. | **ESTABLISHED** | The construction is an information-necessity result, not a claim about typical registry size. | Keep theorem; state the scope before the bound. |
| C4 | A constant-size registry refinement can redirect an entire Select-LLM acquisition transcript. | Five-behavior, four-alias full-path amplification theorem. | **ESTABLISHED IN A CONTROLLED SETTING** | Exact aliases are blocked by exact hashing; theorem is Select-LLM/exact-match specific. | Keep as a mechanism theorem, not a practical attack theorem. |
| C5 | Response-distinct refinements with exactly unchanged correctness can worsen active model selection. | T2-A on MedQA/GSM8K/OpenBookQA: coordinate-wise utility identity; cumulative harm on 3/3 and terminal harm on 2/3. | **ESTABLISHED IN A CONTROLLED SETTING** | Construction uses public or independently acquired references. | Keep as the primary causal intervention. |
| C6 | Acquisition-path change mediates the root-level cost. | Fixed-query clean/refined root histories and regret are exactly equal while active runs differ. | **ESTABLISHED** | Applies to the tested selector/registry pairs; it does not show that every path change causes harm. | Make this the empirical centerpiece. |
| C7 | Results are not caused by candidate order or tie-breaking. | Canonical ordering, 16 registry permutations, and the full query/root tie-policy cross. | **ESTABLISHED for the headline T2 setting** | Does not validate every appendix experiment. | Retain one sentence in main; details remain in appendix. |
| C8 | The mechanism can operate through executable raw-input endpoints without holdout-reference lookup. | IMDB learned adapters accept raw text, are bound before the 13,000-row outcome, and produce +0.7024 pp terminal harm with fixed-query zero. | **ESTABLISHED IN A DEVELOPMENT-INFORMED HELD-OUT SETTING** | Task, roster, and condition were selected using 12,000 disjoint IMDB development rows. | Keep, but label it “development-informed held-out confirmation.” |
| C9 | A fully untouched cross-benchmark pipeline produces material terminal harm. | Yelp is +0.3527 pp with positive inference but fails the frozen +0.5 pp gate; later prospective families also stop or fail materiality/quality. | **NOT ESTABLISHED** | Statistical direction is not the preregistered practical success criterion. | Do not imply broad prospective confirmation. Use Yelp as a transparent near-miss. |
| C10 | The phenomenon transfers beyond the primary Select-LLM specialization. | Official LLM Selector audit passes on 2/6 datasets; CODA is mixed. | **PARTIALLY ESTABLISHED** | Released score-matrix interventions are not raw live endpoints; transfer is selective. | Keep one compact boundary statement; move task details to appendix. |
| C11 | More candidate entries are generally harmful. | Untargeted stress test: only 8/420 materially harmful versus 27 beneficial; most additions benign. | **CONTRADICTED AS A GENERAL CLAIM** | Targeted support selection is essential to the positive constructions. | Explicitly reject the generic multiplicity-harm interpretation. |
| C12 | Prediction-only similarity is a universal solution. | Exact quotient fails on response-distinct variants; soft weighting mitigates many cells but no fixed setting is non-worsening across all clean/attack conditions. | **NOT ESTABLISHED** | Similarity does not authenticate ownership or lineage. | Present response-only defenses as trade-offs, not complete remedies. |
| C13 | Authenticated root mass and refinement-invariant root summaries remove the channel. | Root-factorization proposition; exact root-aware controls; SRSA root-aware histories bitwise identical. | **ESTABLISHED UNDER AUTHENTICATED MAPPING** | Authentication itself is an external evidence/governance problem. | Promote as the design requirement and clean theoretical remedy. |
| C14 | Semantic utility equality is insufficient for transcript invariance. | T2 exact-quality aliases and SRSA structured refinements preserve semantic predictions while changing most paths. | **ESTABLISHED** | Terminal magnitude varies substantially by task geometry. | Add as a concise cross-setting conclusion. |
| C15 | Large path divergence implies practically material terminal harm. | Natural anchor and SRSA change most paths but have no or submaterial terminal harm. | **CONTRADICTED** | Path, evidence replacement, terminal-root transition, and root-quality gap are separate quantities. | Add an explicit “path is not harm” statement. |
| C16 | The effect is universally negative. | T1, official LLM Selector, CODA, untargeted tests, OpenBookQA grids, and SRSA include nulls and improvements. | **CONTRADICTED** | The paper studies selectable targeted harm and integrity, not monotone damage. | Preserve all negative directions and narrow the title/abstract language. |
| C17 | The paper demonstrates production compromise or live-registry attack success. | No live admission study; no provider incentive or prevalence measurement. | **UNSUPPORTED / REMOVE** | Executable offline endpoints are feasibility evidence only. | State as a non-claim in abstract and limitations. |
| C18 | The paper is the first work on clones, duplicate candidates, strategic replication, or ranking manipulation. | Strategic replication, clone robustness, active-evaluation clones, arena candidacy, and leaderboard robustness predate or coincide. | **UNSUPPORTED / REMOVE** | Novelty lies in the specific shared-label latent-root adaptive-selection object. | Replace broad priority language with an obligation-boundary comparison. |
| C19 | The paper contributes a new sequential evaluation object not subsumed by static clone robustness. | One acquired label updates all entries; entry mass changes which future labels are acquired; outcome is root-level selection cost; fixed-query mediation identifies the adaptive channel. | **ESTABLISHED AS A DISTINCT COMBINATION** | Reviewers may still call it an extension unless this combination is made explicit. | Put the four-part distinction before the contribution list. |
| C20 | The complete evidence supports universal external generality. | T1 3/8, LLM Selector 2/6, Yelp/SRSA submaterial, multiple learned same-$s$ terminal negatives/stops. | **NOT ESTABLISHED** | Selective transfer is itself the honest boundary. | Replace breadth language with “cross-family but selective.” |

## Post-V20 attempts and what they may support

| Family | Literal outcome | May support | May not support |
|---|---|---|---|
| Step 113 BoolQ | Strong effect; primary `NO_GO` because alias quality loss exceeded 1 pp | Effect/quality trade-off diagnosis | Confirmatory transfer success |
| Step 114 Amazon | Pre-outcome `STOP`; quality passed, designated parent tied | Roster-gate brittleness and preserved outcome seal | Any outcome claim |
| PRAA | Emotion Stage-3 pass; Language-ID Stage-3 stop | Quantile-calibration transfer limitation | Two-task prospective success |
| PRAA-E | Pre-outcome deployment-trigger stop | Strict gate behavior and preserved seal | Emotion primary outcome |
| SRSA-CIFAR | `NO_GO_SRSA_PRIMARY`; 95–98% path change, exact controls, max CIFAR-100 harm +0.181 pp | Zero-utility-confound cross-modality mechanism and magnitude boundary | Material prospective success |

## Claim hierarchy for V21

V21 should make only the following headline chain:

1. **Experimental unit:** submitted entries need not equal accountable roots.
2. **Mechanism:** entry-level mass can change adaptive acquisition under the same underlying roots.
3. **Causality:** exact-quality refinement changes acquired evidence and root cost; fixed queries remove the effect.
4. **Boundary:** transfer and terminal magnitude are selective; path divergence alone is not practical harm.
5. **Remedy:** authenticated root mass plus refinement-invariant root summaries restores invariance.

Everything else is supporting evidence, a boundary result, or a limitation.

## Required text removals or qualifications

- Remove any sentence that can be read as “candidate multiplicity generally degrades evaluation.”
- Do not call IMDB a completely untouched-task preregistration.
- Do not call Yelp, PRAA, PRAA-E, or SRSA a positive prospective confirmation.
- Do not infer ownership from response similarity.
- Do not describe executable offline wrappers as live registry admission.
- Do not use path-change rate as a surrogate for terminal harm.
- Do not hide or overwrite literal `NO_GO` and `STOP` outcomes.
