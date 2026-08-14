# Anonymous Step 95 / V17 review artifact

V17 adds one prospectively locked MultiNLI learned-adapter study to the sealed
V16 review artifact.  The study stops at its development gate; it does not open
or analyze the matched-validation outcome and does not substitute the
mismatched split or another task.

Eight pinned public NLI checkpoints form the candidate bank.  A deterministic
development rule selects one accountable DistilBERT parent and a six-root
roster.  Four independent checkpoints each train the final two DistilBERT
blocks plus a binary parent-error head (14,767,874 learned parameters).  The
endpoint reads premise/hypothesis text only and emits the parent class or a
unique out-of-class abstention token.  One literal exact-match similarity drives
both acquisition and reference evidence.

On 1,500 development-verification pools, 99.33% of paths and 99.20% of query
sets change.  Cumulative regret rises by 0.050557 [0.042609, 0.058891], and
fixed-query effects are exactly zero.  Terminal and active-minus-fixed harm are
only +0.04987 [0.02640, 0.07307] percentage points, one tenth of the frozen
+0.5-point materiality requirement; three aliases also exceed the one-point
quality cap on search.  The literal decision is
`NO_GO_DEVELOPMENT_GATE_STOP`.

## Validation

From the extracted artifact root:

```powershell
python validate_step95_anonymous_release.py --integrity-only
```

checks every manifest byte, the inherited V16 binding, the anonymous 24-page
paper and nine-page scientific-text boundary, all four Step 95 checkpoints,
the complete development ledger, the unopened-test binding, the absence of
confirmatory outputs, and workstation-path anonymity.

With Python, NumPy, SciPy, PyTorch, Transformers, safetensors, pypdf, and LaTeX
installed:

```powershell
python validate_step95_anonymous_release.py --full
```

also reruns the selected 400 search and 1,500 verification trajectories from
raw development arrays; reconstructs parent/roster selection, inference,
fixed-query mediation, every gate, and the no-go decision; checks manuscript
numbers; and rebuilds the paper.

## Evidence layout

- `STEP95_MNLI_FULL_ADAPTER_PREREGISTRATION_2026-08-13.md` is the prospective
  authority; Amendment A records a tokenizer-only Windows compatibility fix,
  and incident B records the detached monitor and failed duplicate technical
  retry before any effect was inspected.
- `STEP95_STAGE0_MANIFEST_2026-08-13.json` binds dataset/model revisions,
  partitions, test input, and sealed outcome.
- `STEP95_STAGEA_COMPLETE_LEDGER_2026-08-13.json` retains all 48 search cells,
  the top-eight verification, gates, and literal stop.
- `STEP95_STAGEA_INDEPENDENT_VALIDATION_2026-08-13.json` and
  `STEP95_DEVELOPMENT_STOP_RECEIPT_2026-08-13.json` reconstruct and bind the
  stop without opening the test.
- `step95_models/` contains only the four learned adapters.  Public base weights
  and third-party repositories are not embedded.

The prewritten confirmatory runner is bundled as unexecuted protocol code.  No
Step 95 execution lock, confirmatory raw array, result, or confirmatory
validation receipt exists.
