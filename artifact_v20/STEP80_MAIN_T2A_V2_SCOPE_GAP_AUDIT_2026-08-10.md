# Step 80 — Main-T2A V2 scope-gap audit

Date: 2026-08-10 (Asia/Seoul)

## Finding

Step 79 validly closes the preregistered V2 gates for the scenario built by
`run_step29_locked_phase_a.py::build_wrapper`. That constructor is the legacy
matrix-aware exact-quality stress wrapper: it inspects public peer-response rows
and response frequencies when choosing alternative wrong outputs.

The current paper's headline empirical condition is different. It is Step 54's
`matrix_free_public_reference_5pct`, which uses the frozen target response,
public answer space/reference, and deterministic hashes, while reading zero
peer-response rows. Its old-backend effects are the abstract's
`+3.2657/+3.7851/+0.9552` cumulative regret values.

Therefore:

- Step 79 Gate A clean-active versus clean-random is transferable evidence about
  the clean selector, but it uses seed block 50000--50499 rather than Step 54's
  headline block 91000--91499.
- Step 79 proves that the repaired acquisition implementation is candidate-order
  invariant on all audited legacy scenarios and synthetic boundaries.
- Step 79 does **not** directly establish that the paper's headline matrix-free
  condition retains its harm under every V2 tie policy.
- Replacing the abstract's Step 54 numbers with Step 79's legacy-wrapper numbers
  would conflate two threat tiers and is forbidden.

## Decision

Manuscript integration is paused. The only authorized next experiment is a
bounded scope-transfer confirmation that applies the already locked V2 semantics
to the already locked Step 54 primary matrix-free condition and fresh pools.
There is no new target, attack construction, task, hyperparameter, or outcome
search.

This is a protocol-compliance repair to Step 72's requirement that Gates A and B
be evaluated on the exact primary setup. It is not a fifth realism branch.
