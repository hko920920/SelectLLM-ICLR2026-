# Step 80 — Main matrix-free T2-A transfer protocol

## Objective

Determine whether the current paper's primary Step 54 matrix-free public-reference
refinement retains (i) clean active value, (ii) refinement harm, (iii) exact
candidate-permutation invariance, and (iv) cumulative/terminal conclusions across
the full V2 exact-tie grid.

## Frozen scientific object

- Tasks: MedQA, GSM8K, OpenBookQA.
- Clean registry: the same 31 HELM-Lite candidates.
- Refined registry: Step 54 `matrix_free_public_reference_5pct` only.
- Frozen targets: GPT-4-1106-preview, GPT-4-0613, text-davinci-002.
- Four aliases; exactly 50/1000, 50/1000, and 25/500 changed responses.
- Every alias preserves its parent's complete per-example oracle vector.
- Constructor reads zero peer-response rows and is completed before pool creation.
- Pools: seeds 91000--91499, exactly as in Step 54.
- Budgets: 50, 30, 30; temperatures: 3, 5, 5.
- No new model/API calls.

## Methods

The primary comparison has exactly three methods:

1. clean active under V2 canonical acquisition;
2. matrix-free refined active under the same V2 acquisition;
3. clean uniform-random querying under the same V2 root-selection policy.

The fixed-query mechanism control reuses Step 54's deterministic query matrices
for clean and refined registries. It separately measures any static root-selection
effect after holding the acquired evidence fixed.

## Required outcomes

### Active-value and main-effect table

For every method and budget step, store exact-best and near-best identification,
terminal root regret, cumulative root regret, and selected roots. Report paired
10,000-repetition bootstrap intervals and sign-flip tests for:

- clean random minus clean active cumulative regret;
- refined active minus clean active cumulative regret;
- refined active minus clean random cumulative regret;
- clean active minus clean random exact-identification AUC;
- refined fixed-query minus clean fixed-query cumulative root regret;
- sequential excess harm: active refinement penalty minus fixed-query penalty.

Benjamini--Hochberg correction is across the three tasks within each contrast.

### Candidate-permutation audit

Use 16 deterministic permutations with seeds 803000--803015 on clean and refined
registries, all 500 pools. Require exact equality of queried IDs, selected roots,
root-regret paths, canonical scenario digests, and selected acquisition values.

### Exact-tie grid

Cross the same three V2 query policies with the same three V2 root policies:

- minimum global query ID;
- maximum global query ID;
- manifest-seeded SHA-256 query choice;
- minimum canonical root ID;
- maximum canonical root ID;
- manifest-seeded SHA-256 root choice.

All comparisons use exact float64 ties after canonical fixed-order reduction; no
tolerance, rounding, jitter, or outcome-selected seed is allowed.

## Frozen adjudication

`STRONG_TRANSFER` requires all of the following:

1. all Step 54 construction/data/pool digests and integrity invariants reproduce;
2. clean active significantly beats clean random in cumulative regret on at least
   two tasks;
3. refined active is significantly worse than clean active on all three tasks and
   significantly worse than clean random on at least two;
4. sequential excess harm has a strictly positive 95% lower bound on all three;
5. every candidate-permutation check is exact;
6. every tie-policy cell changes at least 90% of active paths and has positive
   mean cumulative harm, with its 95% lower bound above zero on all three tasks;
7. for MedQA and GSM8K, each query policy has positive terminal harm under at
   least two of the three root policies.

`CUMULATIVE_TRANSFER` applies if items 1--6 pass but item 7 fails. The paper may
retain the active-trajectory and cumulative-regret claim but must narrow terminal
language to the policies that pass.

`AUDIT_ONLY_TRANSFER` applies if construction/permutation integrity passes and
all tasks retain positive mean cumulative harm, but clean-versus-random,
sequential-excess, or all-cell confidence requirements fail. The title and
abstract must shift toward registry-dependence auditing and report the failed
counterfactual.

`BLOCKED` applies if the main condition loses cumulative direction on any task,
candidate permutation remains non-invariant under the V2 backend, or a frozen
input/construction digest changes. Manuscript promotion is then prohibited.

## Interpretation discipline

- Existing Step 54 outcomes are historical and must remain reported.
- This transfer run may replace headline numbers only if it passes independent
  validation; old and new backends must be distinguished in the appendix.
- A nonzero fixed-query effect is not hidden. It decomposes static candidate-mass
  influence from sequential excess harm and removes any claim that *all* harm is
  acquisition-mediated unless the sequential contrast supports it.
- OpenBookQA terminal harm is not required; cumulative/path harm is the headline.
- No failed gate may be rescued by changing task, target, seed, distance, alias
  count, budget, temperature, answer generator, or tie policy.
