# Reproducibility guide

## Supported environment

The latest post-V20 work was executed on Windows with Python 3, PyTorch
2.6.0+cu124, Transformers 4.56.2, and the package versions in
`requirements-current.txt`.  The paper was built with Latexmk 4.88.

The V20 artifact contains its own dependency inventory and portable
validators.  Treat that artifact as authoritative for submission claims.

## Validate V20

```bash
cd artifact_v20
python validate_step105_anonymous_release.py --integrity-only
python validate_step105_anonymous_release.py --full
```

Expected markers:

- `PASS_STEP105_ANONYMOUS_RELEASE`
- successful paper rebuild;
- successful independent reconstruction of Step 104 and Step 105.

Raw-input re-inference is optional because public base weights are not
bundled:

```bash
python validate_step105_anonymous_release.py --raw-input
```

## Build the current paper

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
python validate_paper_tables.py
```

Do not edit the bundled ICLR style files.

## Audit Step 113

The primary outcome has already been opened and adjudicated.  It may be
replayed but not relabeled as a success:

```bash
cd post_v20/step113_boolq
python step113_validate_independent.py
```

Expected literal decision: `NO_GO_STEP113_PRIMARY_CONFIRMATION`.

`diagnose_step113_rank_cap.py` is explicitly post-outcome and is diagnostic
only.

## Audit Step 114

Step 114 stopped at development.  Review:

- `STEP114_AMAZON_POLARITY_FINAL_PROSPECTIVE_PROTOCOL_2026-08-14.md`;
- `STEP114_PREOUTCOME_COMPATIBILITY_LOCK_2026-08-14.json`;
- `STEP114_DEVELOPMENT_LEDGER_2026-08-14.json`;
- `STEP114_STOP_CLOSURE_2026-08-14.md`.

Do not run `predict`, `preoutcome-lock`, or `confirm`.  Those commands are
scientifically unauthorized after the recorded development stop.

## Integrity checks before editing

1. Run the V20 integrity validator.
2. Verify the repository SHA-256 manifest.
3. Make source changes in a new commit.
4. Rebuild the paper and run `validate_paper_tables.py`.
5. Never overwrite a historical lock, ledger, raw array, or validator receipt.

Historical files are append-only scientific records.
