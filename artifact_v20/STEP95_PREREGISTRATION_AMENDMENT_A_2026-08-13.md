# Step 95 preregistration amendment A: DeBERTa tokenizer compatibility

This amendment is locked after the first Stage-A process stopped during public
root preflight and before the development JSON was read. Roots 1--7 loaded.
Constructing the pinned root-8 `cross-encoder/nli-deberta-v3-small` fast
tokenizer terminated the Windows Python process with native exit code
`-1073741819`; no Python exception, prediction, model accuracy, learned
adapter, selector result, or test outcome was produced. The Stage-A output,
model directory, ledger, and frozen configuration remain absent.

The same root-8 repository, revision, files, label map, bank position, and all
scientific rules are retained. Root 8 alone will use the Transformers slow
tokenizer (`use_fast=False`) for both preflight and prediction. All other roots
retain their default tokenizers. Slow and fast tokenizers implement the same
pinned SentencePiece vocabulary; this is a platform-compatibility choice, not
an outcome-dependent model substitution.

The rerun must again load all eight roots before reading development labels.
This amendment changes no task, split, row, label, checkpoint, revision,
roster rule, adapter training rule, threshold, temperature, budget, seed,
statistical gate, or stopping rule.
