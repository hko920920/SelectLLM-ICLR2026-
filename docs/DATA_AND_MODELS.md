# Data and model dependency policy

## What is bundled

- learned adapter/head checkpoints needed for exact replay;
- machine-readable result arrays and ledgers;
- preregistrations, execution locks, manifests, and validators;
- the manuscript source and rendered PDF.

## What is not bundled

- Hugging Face public base-model caches;
- complete third-party public datasets when they can be reacquired;
- local package caches, extracted validation scratch, and obsolete artifact
  archives;
- the unopened Step 114 primary outcome.

This avoids redistributing third-party data and keeps the Git history below
platform limits.  GitHub blocks regular Git objects above 100 MiB; learned
weights are therefore stored through Git LFS.

## Post-V20 public pins

### Step 113 BoolQ

- dataset: `google/boolq`;
- revision: `35b264d03638db9f4ce671b711558bf7ff0f80d5`;
- model repository IDs and exact revisions are bound in
  `post_v20/step113_boolq/STEP113_PUBLIC_METADATA_SNAPSHOT_2026-08-14.json`.

### Step 114 Amazon Polarity

- dataset: `fancyzhx/amazon_polarity`;
- revision: `9d9c45c18f8c3cf1b23a3c27917b60cbf28f3289`;
- exact blob: `amazon_polarity/test-00000-of-00001.parquet`;
- expected bytes: 117,422,360;
- expected SHA-256:
  `65613cc6ec1ab30e19c4dcb8d8fa5612159d77bfa493704b4a9a9c167424992e`;
- model repository IDs and revisions are bound in
  `post_v20/step114_amazon/STEP114_PUBLIC_METADATA_SNAPSHOT_2026-08-14.json`.

The dataset may be reacquired for auditing the development stop, but the
protocol forbids proceeding to its primary outcome.

## Third-party code

The V20 artifact records exact external dependency provenance.  Do not assume
that this repository's future project license overrides the licenses of ICLR
styles, public datasets, models, or third-party selector implementations.
