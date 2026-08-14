# PRAA admission policies: entry-indexed versus authenticated root-aware

Date: 2026-08-14 (Asia/Seoul)

## Purpose

PRAA evaluates whether the missing candidate evidence frame has an operational
registry realization. The comparison is not between a weak and an
outcome-tuned selector. It uses the same candidate endpoints and active
selection code under two admission policies fixed before endpoint outputs or
selector effects are inspected.

## Common inputs

Every submitted endpoint must provide a manifest valid under
`PRAA_ADMISSION_MANIFEST_SCHEMA_V1.json`. The registry snapshot binds:

- exact endpoint artifact and dependency digests;
- provider and declared root identity;
- upstream lineage and root relation;
- deterministic runtime and preprocessing;
- raw-input and output schemas;
- native-to-target label mapping;
- manifest digest and attestation.

Response similarity, prediction correlation, or observed accuracy never creates
or changes a root identity. Those quantities are behavioral evidence, not
ownership evidence.

## Policy E: entry-indexed admission

This policy represents the vulnerable baseline audited by the paper.

1. Admit every schema-valid endpoint as a separate candidate entry.
2. Assign a uniform prior over admitted entries.
3. Use every entry response independently in acquisition grouping and posterior
   evidence.
4. Map deployment decisions back to accountable roots only for evaluation.

If one root submits its canonical endpoint plus four schema-valid derived
endpoints, that root receives five entry masses. This policy is deterministic
and may be reasonable when every endpoint is intended to count independently;
PRAA tests the failure that arises when the evaluation claim instead concerns
one accountable root.

## Policy R: authenticated root-aware canonical admission

This policy supplies the evidence frame that predictions alone cannot infer.

1. Verify every manifest and its artifact/dependency bindings.
2. Group entries by an authenticated `declared_root_id`.
3. Require exactly one entry per root to declare
   `lineage.root_relation = canonical_root_endpoint`.
4. Derived endpoints may be admitted for audit and execution, but they receive
   no independent candidate mass.
5. Assign uniform mass over authenticated roots, not entries.
6. For acquisition, posterior evidence, interim selection, and terminal
   selection, represent each root by its canonical endpoint only.
7. Break any manifest-level tie by the lexicographic SHA-256 digest of the
   canonical manifest; never by task output or accuracy.

Adding or removing a derived endpoint leaves the root set, root masses,
canonical responses, acquisition state, and deployment rule unchanged.
Therefore Policy R is refinement-invariant by construction for the PRAA alias
family.

## Why canonical admission is used

A uniform average over all entries inside a root would retain fixed root mass
but could still change the root summary when an alias is added. A response-only
cover could merge independent roots or fail to merge response-distinct aliases.
The canonical policy instead uses declared identity and a fixed within-root
rule, matching the sufficient conditions of the manuscript's root-factorization
proposition.

This is an auditable baseline, not a universal production recommendation.
Registries may prefer a separately frozen within-root mixture, but its members
and weights must be bound before refinement if invariance is claimed.

## Alias admission

The four aliases of the selected parent root must each:

- declare `root_relation = derived_endpoint`;
- bind the canonical parent manifest digest as `wrapper_of`;
- use a distinct `entry_id`, runtime entrypoint, learned-head artifact digest,
  and private abstention code;
- receive the same raw task input as all clean roots;
- have no access to item IDs, labels, peer outputs, pool seeds, selector state,
  or deployment outcomes.

Policy E counts all four aliases as entries. Policy R verifies and executes
them for audit but assigns them zero independent mass and continues to use the
canonical parent response for root decisions.

## Primary policy contrasts

For the frozen primary and replication tasks, later protocols report:

1. **E-clean versus E-refined:** the causal vulnerability contrast.
2. **E-refined versus R-refined:** whether authenticated root-aware admission
   removes the acquisition and decision effect.
3. **R-clean versus R-refined:** an exact invariance control; query, root,
   regret, and acquisition-value histories must be bitwise equal.
4. **E-clean versus R-clean:** the clean-policy utility cost, reported without
   treating zero cost as guaranteed.

The score-changing empirical gate remains the preregistered E-clean versus
E-refined terminal and active-minus-fixed effect. Policy R is a mechanistic and
operational remedy audit; it cannot rescue a failed primary effect.

## Failure rules

- Invalid identity, lineage, artifact, runtime, or label-map evidence rejects the
  endpoint before any selector run.
- More than one canonical endpoint for a root rejects the registry snapshot.
- No canonical endpoint for a root rejects the registry snapshot.
- A failed signature or digest cannot be bypassed because endpoints appear
  behaviorally similar.
- Admission-policy code, manifest schema, canonical tie rules, and registry
  snapshot hashes must be fixed before any effect outcome is opened.
