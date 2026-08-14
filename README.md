# Candidate Lists Are Priors

Reproducibility and research-handoff repository for the anonymous ICLR 2027
manuscript **Candidate Lists Are Priors: Evidence-Frame Dependence in Active
Model Selection**.

This repository is intentionally organized around scientific status rather
than experiment chronology.  It contains the complete validated V20 review
artifact, the current manuscript, the last external-review records, and the
post-V20 experiments that were retained as `NO_GO` or pre-outcome `STOP`.

## Current scientific status

- The main causal result and the disjoint held-out IMDB learned same-similarity
  confirmation are supported.
- Step 104 IMDB is the literal promoted decision:
  `GO_STEP104_IMDB_HELDOUT_LEARNED_SAME_S_TERMINAL_PRIMARY`.
- Step 105 Yelp is a retained prospective negative:
  `NO_GO_RETAIN_STEP105_PROSPECTIVE_NEGATIVE`.
- Step 113 BoolQ produced a strong terminal effect but failed its locked
  one-point alias-quality cap and remains
  `NO_GO_STEP113_PRIMARY_CONFIRMATION`.
- Step 114 Amazon Polarity stopped before any outcome inference or opening
  because the designated parent tied a challenger on the locked safety set:
  `STOP_STEP114_DEVELOPMENT_NOT_CERTIFIED`.

The manuscript establishes targeted evidence-frame dependence.  It does not
establish universal harm, a live-registry compromise, hidden-label attack
success, or broad prospective cross-task transfer.

## Repository map

| Path | Purpose |
|---|---|
| `paper/` | Current anonymous manuscript source and PDF |
| `artifact_v20/` | Byte-preserved validated review artifact through Step 105 |
| `post_v20/step113_boolq/` | Complete BoolQ `NO_GO`, arrays, heads, and independent validation |
| `post_v20/step114_amazon/` | Complete pre-outcome Amazon `STOP` record; outcome is not included or authorized |
| `reviews/` | Last review records and adjudication notes |
| `docs/CURRENT_STATE.md` | Exact handoff point and integrity boundary |
| `docs/REVIEW_TO_EXPERIMENT_TRACE.md` | Reviewer criticism to experiment and disposition map |
| `docs/REPRODUCIBILITY.md` | Validation and rebuild commands |
| `docs/DATA_AND_MODELS.md` | Pinned public dependencies and non-redistribution policy |

## Fast verification

From the repository root:

```bash
cd artifact_v20
python validate_step105_anonymous_release.py --integrity-only
python validate_step105_anonymous_release.py --full
```

The full validator checks the release manifest, rebuilds the paper, validates
rendered table values, and independently reconstructs the Step 104 and Step
105 decisions from bound arrays.  Optional raw-input inference requires the
pinned public Hugging Face models described in the artifact README:

```bash
python validate_step105_anonymous_release.py --raw-input
```

See `docs/REPRODUCIBILITY.md` before running any post-V20 code.

## Critical continuation rule

Do **not** open or reconstruct the Step 114 Amazon outcome as a confirmatory
result.  Its locked protocol says the family stops when development is not
certified, and its sole failed gate was already observed.  A future
cross-task study must be a genuinely new protocol and must not be represented
as a continuation or rescue of Step 114.

## Large files

Learned adapter weights are tracked with Git LFS.  Install Git LFS before
cloning if exact raw-input reconstruction is required:

```bash
git lfs install
git clone https://github.com/hko920920/SelectLLM-ICLR2026-.git
```

Public datasets and public base-model weights are deliberately not
redistributed.  Fetch them from the pinned repository revisions and verify the
recorded hashes.

## Anonymity warning

This is a public repository under an identifiable GitHub account.  Linking it
from an anonymous submission may reveal author identity.  Use only in a way
consistent with the applicable conference anonymity policy.

## License

No project-wide reuse license has yet been selected.  Third-party code and
data retain their original licenses.  A maintainer must choose an appropriate
license before inviting downstream redistribution or modification.
