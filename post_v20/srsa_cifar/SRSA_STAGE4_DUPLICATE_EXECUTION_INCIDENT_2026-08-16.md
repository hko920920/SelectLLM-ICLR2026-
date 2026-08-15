# SRSA Stage 4 duplicate-execution incident — 2026-08-16

## Classification

`DUPLICATE_POST_OUTCOME_EXECUTION_DUE_TO_CONCURRENT_ORCHESTRATION`

The first valid Stage 4 execution had already completed before a second path-corrected continuation was triggered.

## Authoritative first execution

- Run: `31895293285`
- Job: `95037579202`
- Artifact: `9249629903` (`srsa-stage4-result-technical-retry`)
- Artifact digest: `sha256:458eb623eb08090a44f24783d9a475c289095808561ceea92770811ea0ed5492`
- Literal decision: `NO_GO_SRSA_PRIMARY`

## Unplanned duplicate

- Run: `31896042798`
- Job: `95039399217`
- Artifact: `9249820403` (`srsa-stage4-result`)
- Artifact digest: `sha256:66860bdf1b05b6d237971193b995fe299acf67407443d52ab52fb7ce1389ab48`
- Literal decision: `NO_GO_SRSA_PRIMARY`

The two result JSON files have the same SHA-256:

```text
62b186dc075d8c18aec121b66cf22621c5b7b934bd8c3d093e9678bd7bfa3db9
```

The paired-run arrays also have the same SHA-256:

```text
bfea462f0025dec3f276e4655caf9b0ad81049bd50f9c3701b45c01d4a12422a
```

Thus the complete result and paired trajectories were reproduced bitwise. No task, model, wrapper, seed, selector, estimand, inference rule, or gate changed between executions.

## Consequence

The first run remains the only authoritative confirmatory outcome. The second run is retained only as an unplanned exact reproducibility duplicate and is not used for task selection, parameter selection, inference, or claim strengthening.

The strict claim that labels were globally opened in only one job across every execution must not be made. Across the two runs, there were two label-opening jobs. The scientific `NO_GO_SRSA_PRIMARY` decision and all numerical estimates remain unchanged.

No further SRSA Stage 4 execution is permitted.
