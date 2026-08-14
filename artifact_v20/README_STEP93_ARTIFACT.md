# Anonymous Step 93 / V14 review artifact

V14 closes the Select-LLM method-fidelity audit. The paper and artifact now
distinguish the exact-answer primary single-similarity specialization from the
reference-free F1/BLEU and AG News decoupled-view audits.

The new Banking77 condition was preregistered before any row was downloaded.
Twelve learned intent roots and 28 learned parent-error predictors are bundled.
One calibration-selected parent contributes four distinct 50,817-parameter
aliases. Each endpoint reads raw text and frozen weights only, and emits either
the parent label or a unique never-correct abstention token. One literal
`s(a,b)=1[a=b]` drives both candidate agreement and reference feedback; the
runner cannot accept a separate feedback matrix.

Across 1,000 sealed pools, 99.8% of paths change and cumulative regret rises by
0.040945 [0.012905, 0.068910]. Terminal harm is only 0.032
[-0.055, 0.122] percentage points (p=.236), so the preregistered terminal gate
fails. The retained decision is `NO_GO_RETAIN_SOURCE_FAITHFUL_NEGATIVE`.
Fixed-query effects are exactly zero. No replacement task was opened.

The literal machine result initially used the label `INVALID` because a
supplemental dense/sparse check imposed tolerance `2.0e-16` and observed one
binary64 ULP (`2.220446e-16`). An independent 82-check reconstruction verifies
the implementation, all trajectories and statistics, and 128 raw-input
endpoints, then adjudicates the valid retained negative without changing a
scientific gate.

## Validation

From the extracted artifact root:

```powershell
python validate_step93_anonymous_release.py --integrity-only
```

checks every manifest byte, inherited V13 binding, the current PDF, all 40
Step 93 learned-weight files, result/receipt consistency, and anonymity.

With Python, PyTorch, Transformers, SciPy, safetensors, and LaTeX installed:

```powershell
python validate_step93_anonymous_release.py --full
```

also independently reconstructs all Step 93 active/fixed trajectories,
bootstrap and sign-flip inference, source-fidelity assertions, 128 raw-input
endpoint samples, manuscript numbers, and the 23-page PDF with a nine-page main
scientific-text boundary.

This top-level command verifies inherited manifest and receipt bindings but does
not transitively rerun the Step 80/87/88/90/92 validators. Those stage-specific
entry points remain separately invocable when their public dependencies are
available.

## Evidence layout

- `STEP93_SOURCE_FAITHFUL_LEARNED_ABSTENTION_PREREGISTRATION_2026-08-13.md`
  and amendments A/B are the outcome-blind authority.
- `STEP93_STAGEA_COMPLETE_SEARCH_LEDGER_2026-08-13.json` retains all 504
  conditions and top-16 verification.
- `STEP93_CONFIRMATORY_EXECUTION_LOCK_2026-08-13.json` binds 54 pre-outcome
  files, including 40 weights.
- `STEP93_SOURCE_FAITHFUL_CONFIRMATORY_RAW_2026-08-13.npz` and its JSON retain
  the literal sealed run.
- `STEP93_INDEPENDENT_VALIDATION_AND_ADJUDICATION_2026-08-13.json` records the
  valid negative adjudication.
- `STEP93_METHOD_FIDELITY_CLOSURE_REPORT_2026-08-13.md` explains the reviewer
  issue, correction, result, manuscript scope, and stopping rule.

Public encoder weights and third-party repositories are not embedded.
