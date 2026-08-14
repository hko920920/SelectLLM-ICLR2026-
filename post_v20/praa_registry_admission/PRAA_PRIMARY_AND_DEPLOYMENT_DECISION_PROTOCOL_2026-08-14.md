# PRAA primary, remedy, and deployment decision protocol

Date: 2026-08-14 (Asia/Seoul)

This protocol is fixed before Stage-2 safety accuracies, parent identities,
learned heads, primary predictions, or any primary/deployment label is read.
No selector parameter may differ by task.

## Frozen selector

The vulnerable policy is the equation-faithful literal exact-match
specialization used in the prior source-faithful chain. The same function

`s(a,b) = 1[a = b]`

is used for candidate-output grouping during acquisition and for
reference-output posterior feedback after a label is acquired.

Common settings:

- clean entries: four;
- derived entries: four, all mapped to the selected parent root;
- clean roots: four;
- pool size: 500;
- annotation budget: 2;
- temperature: 0.05;
- paired runs per task: 3,000;
- Emotion seeds: 1,320,000 through 1,322,999;
- Language-ID seeds: 1,323,000 through 1,325,999;
- bootstrap replicates: 10,000;
- one-sided paired sign flips: 100,000.

Every seed samples a pool without replacement. Clean and refined runs share the
same pool, references, runtime predictions, budget, temperature, and tie rules.
Entries are canonicalized by full response SHA-256, declared root ID, and entry
ID. Query ties are resolved by the smallest frozen item UID digest. Root ties
are resolved by the smallest frozen root-ID digest. Registry input order is not
a decision variable.

## Four frozen policy conditions

1. `E-clean`: entry-indexed admission with four clean entries.
2. `E-refined`: entry-indexed admission with the same clean entries plus four
   selected-parent derived entries.
3. `R-clean`: authenticated root-aware canonical admission on the clean
   registry.
4. `R-refined`: authenticated root-aware canonical admission with the same four
   derived entries admitted for audit but assigned no independent mass.

The primary vulnerability contrast is `E-refined - E-clean`. The exact remedy
control is `R-refined - R-clean`. The latter must be bitwise zero but cannot
rescue a failed primary contrast.

## Estimands

Root utility is exact-match accuracy on the complete task primary-outcome
block. Let the oracle best-root set contain every clean root attaining maximum
complete-block accuracy. Regret of a selected root is its loss minus the
minimum clean-root loss; every member of a tied best set has zero regret.

Per paired run we report:

- ordered query-path change;
- final-root change;
- terminal-regret difference;
- cumulative-regret difference through budget two;
- fixed-query terminal and cumulative differences;
- active-minus-fixed terminal difference;
- root-aware clean/refined differences.

The fixed-query counterfactual runs the refined entry-indexed registry on the
exact `E-clean` query matrix. It isolates acquisition mediation from a static
registry/deployment effect.

## Primary task decision

Emotion is the prespecified primary task. `GO_PRAA_PRIMARY` requires all of:

1. all data, code, runtime, model, head, threshold, manifest, admission, seed,
   and prediction bindings pass;
2. all four derived entries are response-distinct and coordinate-wise
   non-improving, each losing at most one primary accuracy point;
3. at least 50% of paired acquisition paths change;
4. fixed-query root histories are exactly equal and both fixed terminal and
   fixed cumulative effects are exactly zero;
5. mean terminal harm is at least +0.5 percentage points;
6. mean active-minus-fixed terminal harm is at least +0.5 percentage points;
7. paired-bootstrap 95% lower bounds for both terminal quantities exceed zero;
8. their one-sided paired sign-flip p-values pass the frozen two-task Holm
   family correction;
9. the cumulative-regret mean and 95% lower bound are positive;
10. `R-clean` and `R-refined` query, root, acquisition-value, terminal, and
    cumulative histories are bitwise equal.

Language ID is a simultaneous replication under identical gates and is always
reported. It is not a replacement if Emotion fails. No third task is allowed.

## Multiplicity

For each decision-changing terminal quantity, compute the two raw task-level
one-sided p-values and apply Holm correction across the frozen Emotion and
Language-ID family. All raw and adjusted values are reported. Prior Steps 105,
113, and 114 remain listed in the separately frozen attempt register; PRAA does
not erase them.

## Untouched deployment endpoint

The deployment partition is never used to train heads, set thresholds, select
the parent, form acquisition pools, or evaluate a primary gate.

After the primary and deployment labels are authorized for opening, each clean
root's deployment accuracy and negative log-likelihood (where a calibrated
probability is natively available) are computed from fresh raw-input inference
bound to the same endpoint manifest. For every paired active-selection run, map
its final selected root to that root's deployment loss. Report the paired
`E-refined - E-clean` deployment-loss difference and confidence interval.

Deployment is a prespecified secondary practical endpoint. It cannot turn a
primary NO-GO into GO.

## Outcome access order

Before labels are opened, an input-only stage must bind all clean and derived
primary/deployment predictions, certify one to `floor(0.01*N)` triggers per
derived entry on each block, validate manifests, and write a pre-outcome lock.
Only then may a separate outcome job reconstruct exact labels and execute the
four policy conditions.

## Literal closure

- `GO`: promote the primary result and report the simultaneous replication.
- `NO_GO`: retain both task outcomes and stop adding tasks.
- `STOP`: a scientific pre-outcome gate failed; labels remain unopened.
- `INVALID`: a post-opening binding or execution failure invalidated the run;
  the same outcome may be rerun only after a narrowly documented technical fix,
  never under changed scientific settings.
