# SRSA-CIFAR final one-shot runbook — 2026-08-15

## Authoritative branch

- Branch: `research/srsa-final-clean`
- Scientific source commit: `be5bbdd7e438c8eae9f2a22a3c18ff668f759c70`
- Active workflow: `.github/workflows/srsa-one-shot.yml`
- No other workflow file exists on this branch.

The prior branches remain unchanged as the audit history. Their technical failures and PRAA/PRAA-E scientific stops are not reclassified or deleted.

## Why this path replaces the prior orchestration

The prior one-shot workflow mixed scientific execution with cross-run artifact import, intermediate branch commits, workflow self-modification, and repeated pushes. Those operations created technical failure points unrelated to the experiment.

The final path removes all of them:

1. one workflow;
2. one trigger file;
3. one GitHub Actions run;
4. no cross-run artifact download;
5. no `git commit` or `git push` inside Actions;
6. no workflow-file modification inside Actions;
7. no repository write permission;
8. all intermediate state passed only through artifacts from the same run.

## Fixed scientific configuration

The following remain exactly as preregistered in `SRSA_CONFIG_2026-08-14.json`:

- CIFAR-100 as the only decision-changing primary task;
- CIFAR-10 as a non-substitutable replication;
- four fixed public checkpoint roots per task;
- four structured wrappers for every attacked root;
- 5% deterministic structured-response variant rate;
- pool size 500, budget 2, temperature 0.05;
- 3,000 paired runs per task;
- all four roots in the familywise test;
- terminal harm and active-minus-fixed materiality threshold of 0.5 percentage points;
- fixed-query exact-zero control;
- authenticated root-aware exact invariance;
- no third task and no replication rescue of the primary.

No task, checkpoint, wrapper, canary rate, threshold, seed, materiality criterion, or inference rule may change after authorization.

## Execution stages

### Preflight

Verifies the frozen config, compact Stage-0 binding, workflow identity, source tests, and absence of a prior Stage-4 result.

### Stage 0 — public metadata lock

Materializes the verified compact binding derived from the successful public-metadata artifact. No CIFAR row, label, or prediction is accessed.

### Stage 1 — data seal

Downloads and verifies the official CIFAR archives. It writes:

- a public raw-input artifact with no labels; and
- a physically separate sealed-label artifact.

### Stage 2 — clean endpoints

Eight matrix jobs execute the four frozen public checkpoints on CIFAR-100 and CIFAR-10. These jobs receive public inputs only and cannot access labels.

### Stage 3 — pre-outcome lock

Checks wrapper semantic equality, root-aware canonicalization, deterministic variant rates, fixed pools, and exact code/data/model bindings. It receives no sealed labels and executes no selector outcome.

### Stage 4 — single label opening

Only this job downloads the sealed-label artifact. It executes all four attacks, fixed-query controls, root-aware controls, bootstrap inference, sign-flip inference, and deployment secondary metrics. It writes one literal `GO_SRSA_PRIMARY` or `NO_GO_SRSA_PRIMARY` artifact.

## Completion rule

The study is complete only when all three conditions hold:

1. the `stage4-outcome` job is completed;
2. the `srsa-final-stage4-result` artifact exists; and
3. `SRSA_STAGE4_RESULT.json` contains a valid literal decision whose bindings match the run.

A green preflight, Stage-2 endpoint success, or an uploaded placeholder is not a final scientific result.

## Technical failure and rerun rule

- A failure before the Stage-4 sealed-label download may be repaired only if the repair changes no scientific setting and the exact failure boundary is recorded.
- Once Stage 4 downloads or opens labels, the run is outcome-accessing. It must not be rerun as confirmatory evidence unless an explicit protocol deviation is documented and the result is treated accordingly.
- Scientific `NO_GO` is a valid completed result, not a CI error.
- Technical exceptions are `INVALID`, not negative scientific evidence.

## Permission model

The workflow requests only:

```yaml
permissions:
  contents: read
  actions: read
```

It does not modify the repository. No GitHub workflow-write permission, PAT, deploy key, or Actions-side push is required.
