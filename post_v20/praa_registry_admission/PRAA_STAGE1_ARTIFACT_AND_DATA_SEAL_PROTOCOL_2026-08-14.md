# PRAA Stage 1: exact artifact materialization and data sealing

Date: 2026-08-14 (Asia/Seoul)

## Authorization

Stage 1 is authorized by `PASS_PRAA_STAGE0_METADATA`. It may materialize the
exact public datasets and endpoint artifacts fixed in the Stage-0 lock, verify
schemas and label mappings, and create deterministic development, outcome, and
deployment packages. It may not run any model, inspect predictions or logits,
train an error head, execute a selector, or compute any clean/refined effect.

## Frozen information boundary

Allowed:

- exact dataset files and task rows at the locked revisions;
- task labels solely to validate the documented label space and write separate
  sealed-label packages;
- model/config/tokenizer/fastText/package artifacts solely for hashing and
  schema inspection;
- deterministic input-only partitioning.

Forbidden:

- model inference, representations, logits, error scores, alias triggers;
- candidate accuracy, agreement, diversity, ranking, or parent selection;
- selector trajectories, terminal choices, regret, or deployment metrics;
- changing either task or clean registry.

## Deterministic row identity and partition rule

For every source row, define an opaque UID as SHA-256 of:

`task_key | dataset_revision | official_split | original_row_index | canonical_raw_input`

Rows are ordered by SHA-256 of:

`PRAA-stage1-partition-v1 | opaque_uid`

No label enters the UID or ordering key.

### Primary emotion task

Dataset: `dair-ai/emotion@cab853a1dbdf4c42c2b3ef2173804746df8825fe`
using the documented split configuration.

- official train, 16,000 rows:
  - first 12,000 by frozen hash order: `error_head_train`;
  - remaining 4,000: `safety`.
- official validation, 2,000 rows:
  - all rows: label-free-at-use `target_threshold`.
- official test, 2,000 rows:
  - first 1,500 by frozen hash order: `primary_outcome`;
  - remaining 500: `deployment`.

### Language-identification replication

Dataset: `papluca/language-identification@aa56583bf2bc52b0565770607d6fc3faebecf9e2`.

- official train, 70,000 rows:
  - first 50,000 by frozen hash order: `error_head_train`;
  - next 10,000: `safety`;
  - remaining 10,000: `development_reserve`, frozen unused unless a later
    protocol explicitly authorizes a result-blind integrity check.
- official validation, 10,000 rows:
  - all rows: label-free-at-use `target_threshold`.
- official test, 10,000 rows:
  - first 8,000 by frozen hash order: `primary_outcome`;
  - remaining 2,000: `deployment`.

The deployment block is never used for acquisition, thresholding, error-head
training, parent selection, or any GO/NO-GO decision. It is a prespecified
secondary check of the actual selected root on untouched rows.

## Input and label separation

For every partition, Stage 1 writes an input JSONL containing only:

- opaque UID;
- canonical raw input fields;
- official source split and original row index;
- frozen partition name.

Labels are written separately in UID order. Later stages may read labels for
`error_head_train` and `safety`. `target_threshold`, `primary_outcome`, and
`deployment` labels remain prohibited until the stage that explicitly
unseals them. Hashes and cardinalities may be read at any stage.

## Exact endpoint materialization

Stage 1 downloads only the Stage-0 retained endpoints at their exact revisions
or package version. It records SHA-256 and byte size for every materialized
configuration, tokenizer, model-weight, fastText, or package distribution file.
It does not load model tensors or call inference.

### Emotion clean registry

1. `dk409/emotion-roberta@66fcef009eac311aab4387b49b408bd9a4c6b1a3`
2. `nateraw/bert-base-uncased-emotion@064d252021b51d95cd0547c89c6489100da0dc4c`
3. `bhadresh-savani/distilbert-base-uncased-emotion@ce6f4ffcde7642ca2cac02381a16da38e5498ff7`
4. `bhadresh-savani/albert-base-v2-emotion@4812613b3c07c549e13f09bb266dbf0e59f48de7`

Every config must expose six output indices. Semantic label maps may be exact
names or generic numeric labels explicitly bound to the dataset order
`sadness, joy, love, anger, fear, surprise`.

### Language-identification clean registry

1. `papluca/xlm-roberta-base-language-detection@9865598389ca9d95637462f743f683b51d75b87b`
2. `facebook/fasttext-language-identification@3af127d4124fc58b75666f3594bb5143b9757e78`
3. `cis-lmu/glotlid@85cd6716494360367b75f642b5bc78667605d0b4`
4. `langid==1.1.6`, with PyPI distribution SHA-256 recorded.

The frozen target labels are:

`ar, bg, de, el, en, es, fr, hi, it, ja, nl, pl, pt, ru, sw, th, tr, ur, vi, zh`.

Later endpoint wrappers must map native labels deterministically into this set;
all out-of-target labels map to one private always-incorrect code. Stage 1
records the mapping specification but does not run the endpoints.

## Stage-1 PASS gates

Stage 1 returns `PASS_PRAA_STAGE1_ARTIFACT_AND_DATA_SEAL` only if:

1. both exact dataset revisions materialize;
2. source split cardinalities match the frozen protocol;
3. raw input fields and target label sets match exactly;
4. all partition cardinalities and UID uniqueness checks pass;
5. every input package has a separately ordered label package with identical
   UID digest and cardinality;
6. every retained endpoint/package materializes at the exact locked revision;
7. every required artifact is nonempty and receives a SHA-256 digest;
8. emotion configs expose a deterministic six-class map;
9. the receipt confirms zero model inference and zero selector execution.

Any failure produces `STOP_PRAA_STAGE1_ARTIFACT_AND_DATA_SEAL`. A parser,
network, or library-compatibility defect may be corrected and rerun on the same
locked data, task, registry, and partition rule. No scientific setting may
change.
