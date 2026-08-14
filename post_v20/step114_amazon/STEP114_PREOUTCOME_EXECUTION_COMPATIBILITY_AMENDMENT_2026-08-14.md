# Step 114 pre-outcome execution compatibility amendment

Date: 2026-08-14 (Asia/Seoul)

Status: recorded after a development-only API exception and before any
primary-outcome model inference, label opening, statistic, or prediction
inspection.

## Trigger

The locked command `python step114_pipeline.py develop` completed safety
inference for three of four candidate roots and then stopped before writing a
development ledger.  The fourth candidate uses a DistilBERT sequence
classifier, while its repository tokenizer emits `token_type_ids`.  The
installed Transformers version forwarded that tokenizer field to
`DistilBertForSequenceClassification.forward`, which rejects it:

`TypeError: DistilBertForSequenceClassification.forward() got an unexpected keyword argument 'token_type_ids'`

## Bounded compatibility action

The original locked protocol and all four locked Step 114 source files remain
unchanged.  The separate runner `step114_compat_runner.py` removes only the
unsupported `token_type_ids` keyword for DistilBERT and delegates every other
argument and computation to the original model forward method.  DistilBERT
does not use segment/type embeddings, so this is an API compatibility action,
not a change to tokenization, logits, labels, models, thresholds, seeds,
indices, gates, or analysis.

All remaining Step 114 pipeline stages will be invoked through this same
runner.  The independent validator remains the originally locked validator.

## Integrity boundary

At this amendment:

- the exact Amazon Polarity test blob had been hashed and the disjoint blocks
  had been written;
- the primary-outcome labels remained sealed;
- no primary-outcome model inference had run;
- no primary-outcome prediction or statistic had been inspected;
- no development ledger, error head, threshold, or gate result existed;
- no scientific criterion was changed.

The accompanying compatibility lock binds this amendment, the runner, the
original predata lock, the Stage 0 manifest, and all originally locked source
files before execution resumes.
