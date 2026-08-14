# PRAA Stage 2: clean safety inference and deterministic parent selection

Date: 2026-08-14 (Asia/Seoul)

## Authorization and purpose

Stage 2 is authorized by the Stage-1 artifact/data-seal PASS. It performs the
first model execution in PRAA, but only on each task's frozen `safety` inputs.
Its sole scientific purpose is to certify the clean registry and choose one
alias parent per task without selector outcomes or primary/deployment data.

Stage 2 must not run an alias, train an error head, inspect target-threshold,
primary-outcome, or deployment inputs, execute active selection, or calculate
any clean/refined effect.

## Frozen runtimes

Common runtime:

- Python 3.11;
- NumPy 1.26.4;
- PyTorch 2.6.0 for Transformer endpoints;
- Transformers 4.56.2;
- Tokenizers 0.22.2;
- Safetensors 0.8.0;
- SentencePiece 0.2.0;
- `fasttext-wheel==0.9.2` for fastText endpoints;
- `langid==1.1.6` for the package endpoint.

CPU execution is fixed to evaluation mode, four intra-op threads, one inter-op
thread, seed zero, disabled gradient tracking, and deterministic PyTorch
algorithms. No mixed precision or stochastic sampling is used.

## Frozen input preprocessing

- input is the exact Stage-1 `input.text` string;
- no lowercasing, trimming, language filtering, or text normalization is added;
- Transformer endpoints use their native locked tokenizer;
- truncation is enabled with `max_length=128` and native padding;
- Transformer batch size is 64 for emotion and 32 for XLM-R language ID;
- fastText and `langid` receive the exact string as one inference unit.

The 128-token limit is fixed from the public short-text task definitions before
any endpoint output is inspected. It is shared by all Transformer endpoints.

## Frozen label mappings

### Emotion

Each native output index is mapped through the exact semantic `id2label` bound
in Stage 1 to:

`sadness, joy, love, anger, fear, surprise`.

### Language identification

- XLM-R uses its exact 20-code semantic `id2label`;
- Meta fastText and GlotLID strip `__label__` and map frozen ISO-639-3/script
  codes into the 20 target ISO-639-1 codes;
- `langid` uses its native ISO-639-1 output;
- every out-of-target native label maps to integer `20`, a private
  always-incorrect code.

The mapping is deterministic and does not use task labels.

## Exact runtime artifacts

Every endpoint runner must download the Stage-1-selected weight/package at its
exact revision, verify its byte size and SHA-256 before loading, and bind the
wrapper code hash and installed dependency versions. Any mismatch stops that
endpoint.

## Safety-only endpoint receipts

For each of eight clean roots, Stage 2 writes:

- ordered safety UIDs;
- mapped hard predictions;
- top native confidence/score where the endpoint exposes one;
- input file SHA-256 and UID digest;
- exact artifact and environment bindings;
- explicit assertions that no non-safety input or selector was executed.

Endpoint jobs do not receive labels.

## Safety-label reconstruction

The parent-selection job reacquires only the exact public dataset revision and
uses the Stage-1 safety input package's `source_index` fields to reconstruct
labels in that exact UID order. It must reproduce the Stage-1 sealed safety
label SHA-256 before any clean accuracy is computed. It does not download or
open the Stage-1 artifact containing target-threshold, primary, or deployment
labels.

## Deterministic clean-registry certification

For each task:

1. all four endpoint receipts and bindings must pass;
2. all prediction UIDs must exactly match the Stage-1 safety UIDs;
3. every mapped prediction must be in the target label space or the frozen
   out-of-target code;
4. all four complete safety response vectors must be pairwise distinct;
5. exact correct counts are computed against safety labels;
6. the `best_root_set` is every root attaining the maximum exact count;
7. the selected alias parent is the member of the best-root set minimizing
   `SHA256("PRAA-parent-tie-v1|" + root_id)`.

A tied best set is valid. Unlike Step 114, no unique-best gate is imposed.
Future regret is defined against the oracle best-root set, while the alias
parent remains the deterministic selected member above.

## Stage-2 decision

`PASS_PRAA_STAGE2_CLEAN_SAFETY_AND_PARENT` requires all endpoint and clean
registry checks above on both frozen tasks.

`STOP_PRAA_STAGE2_CLEAN_SAFETY_AND_PARENT` is issued for an artifact,
preprocessing, label-map, UID, response-distinctness, or execution failure.
Technical wrapper defects may be corrected on the same frozen inputs,
artifacts, and rules. The task, registry, parent rule, and preprocessing may not
change.

A Stage-2 PASS authorizes a later protocol to:

- run the selected parent on `error_head_train`, `safety`, and label-free
  `target_threshold` inputs with native features;
- train the frozen four error heads on `error_head_train` labels;
- run all clean roots and parent aliases on input-only primary/deployment
  blocks before outcome labels are opened.

It does not itself authorize opening primary or deployment labels or executing
the active selector.
