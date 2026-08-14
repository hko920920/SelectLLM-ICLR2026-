# PRAA Stage 3: learned endpoint recipe and safety certification

Date: 2026-08-14 (Asia/Seoul)

This protocol is fixed before Stage-2 accuracies and parent identities are read.
It applies mechanically to the parent selected by the frozen Stage-2 rule.

## Information boundary

Stage 3 may read `error_head_train` inputs and labels, `safety` inputs and
labels, and label-free `target_threshold` inputs. It may execute only the
selected parent on those partitions. It may not access any `primary_outcome`
or `deployment` input, label, prediction, selector path, or effect.

## Parent features

The feature interface is determined solely by the selected parent's declared
runtime family.

- Transformer classifier: final hidden state at token position zero,
  concatenated with logits mapped into the frozen task-label order.
- fastText classifier: sentence vector, top-eight mapped label probabilities,
  top-one probability, top-one/top-two margin, entropy, and frozen raw-text
  length/script fractions.
- `langid`: full normalized native score vector mapped into the target label
  space, the same confidence summaries, and the same raw-text fractions.

No feature interface is selected by observed AUROC, downstream effect, or task
outcome.

## Four fixed linear heads

Seeds are `315100..315103` for Emotion and `315200..315203` for Language ID.
For each seed:

1. standardize features on the full `error_head_train` partition;
2. target one iff the selected parent prediction is wrong;
3. require at least 25 error and 25 correct rows;
4. bootstrap all error rows and an equal number of correct rows;
5. train one linear layer for 30 epochs with BCE-with-logits, batch size 256,
   AdamW, learning rate `1e-3`, and weight decay `1e-3`;
6. store weight, bias, feature mean, and feature scale in safetensors.

All four checkpoints must be nonempty, hash-distinct, and exactly replayable.
Error-prediction metrics are diagnostics only.

## Frozen threshold and endpoint output

Each head is scored on the complete label-free `target_threshold` partition.
The threshold is the `0.993` quantile using NumPy `method="higher"`. There is no
quantile grid or task-specific tuning.

A triggered derived endpoint emits its own code outside the task label space;
otherwise it emits the parent hard label. The four codes are distinct. Under
exact-match utility this construction cannot improve any coordinate.

## Safety gates

For each derived endpoint on the safety partition:

- trigger count is at least one and at most `floor(0.01 * N_safety)`;
- its response vector differs from the parent and the other derived endpoints;
- coordinate-wise improvement count is zero;
- accuracy loss relative to the parent is at most one percentage point;
- all features, scores, thresholds, checkpoints, and responses replay exactly.

## Decision

Both tasks must pass every binding and safety gate for
`PASS_PRAA_STAGE3_ERROR_HEAD_AND_THRESHOLD`. A scientific gate failure stops
this study family. A technical implementation defect may be corrected only
under the same tasks, parents, inputs, features, seeds, recipe, quantile, and
gates.

A PASS authorizes a later input-only stage to execute clean and derived
endpoints on the already sealed primary and deployment inputs. It does not
authorize opening their labels or running active selection.
