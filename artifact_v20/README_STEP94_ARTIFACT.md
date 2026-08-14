# Anonymous Step 94 / V16 review artifact

V16 adds the preregistered fresh-task learned single-similarity audit to the
sealed V15 review artifact. It retains the result as a negative terminal
confirmation rather than replacing it or searching the two unselected tests.

Before any test outcome was used, Step 94 fixed a three-task menu (20
Newsgroups, MASSIVE en-US, and SST-5), deterministic development partitions,
16 learned candidate roots per task, four learned error gates per eligible
parent, a complete search grid, a development-only selection rule, and one
authorized confirmatory execution. The complete Stage-A ledger contains 648
search conditions and 24 verification conditions per task. The frozen rule
selected 20 Newsgroups after a +0.7268 percentage-point verification terminal
effect.

Each selected endpoint is an integrated 205,245-parameter checkpoint that
contains its parent classifier and one independently trained gate. It accepts
raw text only and emits either the parent class or a unique never-correct
abstention symbol. One literal exact-match similarity drives both acquisition
and posterior evidence; test references, peer responses, item lookup, and a
separate feedback view are unavailable to the endpoint.

Across 2,000 sealed test pools, every acquisition path and query set changes,
48.2% of final roots change, and cumulative regret rises by 0.082157
[0.038544, 0.126554] (p=.00020). The terminal change is only +0.1198
[-0.0444, +0.2895] percentage points (p=.0813), below the preregistered
+0.5-point materiality threshold and with an interval crossing zero. Fixed-
query terminal and cumulative effects are exactly zero, and fixed root
histories are bitwise identical. The retained decision is
`NO_GO_RETAIN_FRESH_TASK_NEGATIVE`.

This result strengthens the paper by independently replicating source-faithful
learned anytime harm on a second fresh task with integrated raw-input variants.
It does not establish material terminal harm, a full independently fine-tuned
base model, live registry admission, or production compromise.

## Validation

From the extracted artifact root:

```powershell
python validate_step94_anonymous_release.py --integrity-only
```

checks every manifest byte, the inherited V15 binding, the current anonymous
23-page PDF and nine-page scientific-text boundary, all 128 Step 94 learned
files, authority/result bindings, and workstation-path anonymity.

With Python, NumPy, SciPy, PyTorch, Transformers, safetensors, pypdf, and LaTeX
installed:

```powershell
python validate_step94_anonymous_release.py --full
```

also independently reconstructs the Step 94 raw deltas, quality constraints,
bootstrap intervals, sign-flip tests, path/query/root changes, fixed-query
controls, gates, and retained decision; checks every manuscript number; and
verifies the paper build and page boundary.

The top-level command verifies inherited V15 manifest and receipt bindings but
does not transitively rerun all earlier-stage experiments. Their stage-specific
entry points remain bundled.

## Evidence layout

- `STEP94_FRESH_TASK_LEARNED_VARIANT_PREREGISTRATION_2026-08-13.md` and
  Amendments A--D are the outcome-blind authority and transparent corrections.
- `STEP94_STAGE0_MANIFEST_2026-08-13.json` binds all three task revisions,
  partitions, inputs, and sealed outcomes.
- `STEP94_STAGEA_COMPLETE_LEDGER_2026-08-13.json` retains the complete
  development search and verification record.
- `STEP94_STAGEA_FROZEN_CONFIG_2026-08-13.json` and
  `STEP94_CONFIRMATORY_EXECUTION_LOCK_2026-08-13.json` bind selection, code,
  all 128 learned files, the one selected test, and 2,000 seeds before outcome
  opening.
- `STEP94_CONFIRMATORY_RAW_2026-08-13.npz` and its JSON retain the sealed run.
- `STEP94_INDEPENDENT_VALIDATION_2026-08-13.json` records the first independent
  reconstruction; `validate_step94_artifact_reconstruction.py` repeats it
  read-only inside the review package.
- `STEP94_REVIEW_CLOSURE_REPORT_2026-08-13.md` records exactly what this audit
  closes and what remains open.

Public encoder weights, source-package caches, and third-party repositories are
not embedded. The two unselected test outcomes are retained as sealed provenance
but were not opened for model selection or post-hoc replacement-task analysis.
