# PRAA Stage 4: input-only primary/deployment execution and pre-outcome lock

Date: 2026-08-14 (Asia/Seoul)

## Authorization

Stage 4 is authorized only by a Stage-3 family PASS. It may execute the eight
frozen clean roots and the four learned derived endpoints per task on the
already sealed `primary_outcome` and `deployment` **input packages**. It must
not download, reconstruct, inspect, or receive any primary/deployment label,
label statistic, selector trajectory, regret, root quality, or final decision.

The sole purpose is to bind every future endpoint output and certify, without
labels, that the learned endpoints remain inside the structural one-point
quality envelope before any outcome is opened.

## Frozen execution

Every clean root uses exactly the Stage-2 runtime, preprocessing, label map,
and artifact digest:

- exact raw `input.text` string;
- Transformer native tokenizer, truncation, `max_length=128`, deterministic CPU
  inference;
- fastText and `langid` exact raw string inference;
- task-native mapped hard label;
- mapped target probability vector where the runtime exposes scores.

The selected parent additionally exposes the already frozen Stage-3 feature
interface. Each of the four saved Stage-3 heads is replayed on those features;
its threshold is copied bitwise from the Stage-3 receipt. Triggered endpoints
emit their distinct out-of-task private codes and otherwise copy the parent
hard prediction.

No threshold, head, feature, private code, batch size, or runtime may be changed
because a primary/deployment trigger count appears inconvenient.

## Input-only trigger certificate

For each task, partition, and derived endpoint:

- trigger count must be at least one;
- trigger count must not exceed `floor(0.01 * N_partition)`;
- the derived response vector must differ from the parent and from the other
  three derived vectors;
- replay from the Stage-3 checkpoint, threshold, feature vector, and parent
  output must be exact.

Because a derived endpoint outputs only its parent label or an out-of-task code,
these count bounds guarantee outcome accuracy loss at most one percentage point
for every possible unseen label vector. They do not guarantee a favorable
selector effect.

The frozen maxima are:

| Task | Partition | Rows | Maximum triggers per endpoint |
|---|---|---:|---:|
| Emotion | primary | 1,500 | 15 |
| Emotion | deployment | 500 | 5 |
| Language ID | primary | 8,000 | 80 |
| Language ID | deployment | 2,000 | 20 |

A scientific trigger failure stops before labels. No task, threshold, endpoint,
or outcome may be replaced.

## Clean endpoint output contract

For every root and partition, Stage 4 stores in UID order:

- mapped hard prediction;
- mapped probability/score vector;
- top confidence or native score;
- endpoint receipt and exact runtime/artifact/environment bindings.

Probability vectors are not used by the exact-match primary selector. They are
bound now for the prespecified deployment NLL secondary and may not be
recomputed with a different runtime after outcome opening.

## Admission manifests and registry snapshots

Stage 4 materializes one manifest per clean and derived endpoint under
`PRAA_ADMISSION_MANIFEST_SCHEMA_V1.json`.

Clean roots declare one canonical endpoint per authenticated root. Derived
endpoints declare the selected parent root, `root_relation=derived_endpoint`,
and bind the canonical parent plus the exact head checkpoint, threshold,
private code, wrapper code, preprocessing, and dependency artifacts.

Two immutable registry snapshots are generated per task:

- `entry_indexed`: four clean entries or four clean plus four derived entries,
  with uniform entry mass;
- `root_aware_canonical`: the same admitted manifests, grouped by authenticated
  root ID with uniform root mass and exactly one canonical endpoint per root.

Manifest and registry validation uses identity/lineage evidence only. It does
not use response similarity, task labels, model quality, or selector effects.

## Pre-outcome lock

A Stage-4 PASS binds:

- Stage-1, Stage-2, and Stage-3 receipt hashes;
- all primary/deployment input and UID hashes;
- every clean hard-prediction and score-vector hash;
- every selected-parent feature, head-score, trigger, and derived-response hash;
- all endpoint manifest and registry snapshot hashes;
- exact selector code, seeds, budget, temperature, tie rules, bootstrap, sign
  flip, Holm, estimand, and deployment rules already frozen in the primary
  protocol;
- an explicit assertion that no primary/deployment label or selector outcome
  was accessed.

## Decision

`PASS_PRAA_STAGE4_INPUT_ONLY_PREOUTCOME` requires all bindings, endpoint
replays, trigger certificates, response distinctness, manifest validation, and
registry-snapshot rules on both tasks and both partitions.

`STOP_PRAA_STAGE4_INPUT_ONLY_PREOUTCOME` is final for a scientific trigger or
admission gate. A technical artifact or wrapper defect may be corrected only
under the same scientific settings and input packages.

Only a Stage-4 final lock authorizes a separate outcome job to reconstruct the
exact primary/deployment labels, verify their Stage-1 seals, and execute the
already frozen policy contrasts. Step 114 remains unopened and unrelated.
