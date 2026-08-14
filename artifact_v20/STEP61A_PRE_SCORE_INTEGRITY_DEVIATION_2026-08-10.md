# Step 61A — Pre-Score Integrity Deviation

**Time:** after the validated Stage-0 lock and before blind-package creation, ERFA
scoring, risk decisions, or any registry-effect computation.

## Event

While checking the format of MODEL SELECTOR candidate-name metadata, a shell command
printed the first eight lines of
`imagenet_pytorch_models/models.txt` and the first three lines of
`imagenet_v2_top-images/models.txt`.  Those lines contain a model name followed by an
aggregate accuracy percentage.  This is therefore a limited accidental aggregate
outcome exposure and must not be described as fully outcome-unseen.

No `oracle.npy` was deserialized.  No per-instance correctness vector, HELM metric,
candidate partition, stress root, mutation, ERFA score, risk flag, query path, regret
effect, materiality label, correlation, or AUROC was observed.

## Why the prospective decisions remain fixed

The exposure occurred only after the following protocol was already SHA-256 locked:

`48166a731cd7e160cdb9c192dc6b8ed2eed0dfee46e06819082c1cde808cb205`.

That protocol fixes all task IDs, candidate splitting, response-only root quantiles,
mutations, seeds, budgets, score, baselines, thresholds, statistics, and gates.  None
of those rules uses candidate accuracy or model metadata.  Therefore the exposed
numbers cannot legally change any experimental choice.

## Binding corrective action

1. Do not read or parse any `models.txt` file in Stage A or Stage B.
2. Use stable original column IDs (`candidate_000`, `candidate_001`, ...) for every
   MODEL SELECTOR task.
3. Retain both affected tasks and all their cells; no substitution, removal,
   retargeting, or special analysis is allowed.
4. Keep the affected tasks in every primary statistic and disclose this event in any
   eventual paper use of Step 61.
5. The accurate status wording is now: **protocol-locked before limited aggregate
   accuracy exposure; fully unseen with respect to per-instance outcomes and all
   registry-effect endpoints**.

This clarification narrows the blindness claim; it does not modify the scientific
protocol.
