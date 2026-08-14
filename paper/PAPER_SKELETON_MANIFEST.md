# Paper skeleton manifest

Date: 2026-08-08  
Target: ICLR 2027  
Status: Step 47 integrated working manuscript, not a submission draft

## Main-text page contract

| Section | Target pages | Primary job | Locked evidence |
|---|---:|---|---|
| Abstract | 0.25 | Assumption, counterexample, evidence, implication | Steps 27, 29--31, 35 |
| 1. Introduction | 1.25 | Common belief, Figure 1, four contributions | Steps 33, 36 |
| 2. Setting and threat model | 0.90 | Roots versus entries; T0/T1/T2 boundary | Steps 28, 37 |
| 3. Theory | 1.35 | Necessity and constant-registry amplification | Steps 27, 37 |
| 4. Protocol | 1.10 | Locked data, interventions, estimands, defenses | Steps 28--31, 34--35, 46 |
| 5. Results | 2.15 | Primary effects, robustness, CODA, defense frontier | Steps 30, 31, 35, 46 |
| 6. Related work | 0.80 | Selection literature and exact novelty boundary | Steps 32, 45--46 plus primary sources |
| 7. Limitations | 0.65 | Realism and scope restrictions | Step 36 |
| 8. Conclusion | 0.20 | One implication | Step 36 |
| **Rendered main text** | **9.00** | Conclusion ends on page 9; references begin page 10 | ICLR nine-page limit |

References, the required AI-use statement, optional ethics/reproducibility statements, and appendices are outside this main-text allocation under the official instructions.

## Figure/table contract

1. `figures/figure1_registry_refinement.tex`: page-one causal schematic. It distinguishes submitted entries from accountable roots and makes the missing lineage map visible.
2. `tables/primary_llm_results.tex`: exact locked cumulative-regret effects and confidence intervals on three public-response tasks.
3. `tables/coda_results.tex`: all five official CODA pools, including the WNLI path-only boundary and negative task effects.
4. `tables/defense_frontier.tex`: the cheap 5% negative result and the 10% clean-cost trade-off.
5. `tables/soft_weight_frontier.tex`: explanatory best-case rows from the complete preregistered 63-clean/105-attack-cell soft-weight grid.

## Prose status

- The title, central claim order, theorem statements, contribution list, and numerical tables are locked for the current pass.
- Step 46 is integrated as a scoped defense audit: mitigation is acknowledged, while the 0/105 joint transcript gate and 0/63 clean-preservation gate are reported without universalizing beyond the tested rules and metric.
- Full proof exposition, complete related-work synthesis, statistical detail, Figures 2--3, and appendix tables are intentionally deferred to the next authorized step.
- Forbidden claims remain forbidden: universal selector vulnerability, monotonic harm, private-label feasibility for T2, and a claim that external lineage alone is a complete deployed defense.
