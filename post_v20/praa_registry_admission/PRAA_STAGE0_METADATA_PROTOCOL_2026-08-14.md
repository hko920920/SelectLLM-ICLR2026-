# PRAA Stage 0: outcome-free task and registry metadata protocol

Date: 2026-08-14 (Asia/Seoul)

## Scientific purpose

The Prospective Registry Admission Audit (PRAA) is a new post-review study
family. It is not a retroactive rescue of Steps 105, 113, or 114. Its purpose is
to address two remaining review boundaries in one auditable experiment:

1. one-shot transfer to task/registry families untouched by the prior chain;
2. an admission-style comparison between entry-indexed and authenticated
   root-aware registry policies using independently executable raw-input
   endpoints.

A favorable result is useful only if it survives rules fixed before any task
row, model output, safety statistic, selector trajectory, or outcome is viewed.
This protocol therefore optimizes evidential power through task size,
multiclass structure, endpoint diversity, and deterministic controls—not by
searching outcomes.

## Authority and retained prior outcomes

- Step 104 IMDB remains the promoted within-task held-out confirmation.
- Step 105 Yelp remains a retained submaterial prospective `NO_GO`.
- Step 113 BoolQ remains a strong-effect but quality-invalid `NO_GO`.
- Step 114 Amazon remains a pre-outcome roster-certification `STOP`; its
  primary outcome must remain unopened.
- No PRAA result may relabel or erase those dispositions.

## Stage-0 information boundary

Stage 0 may access only public repository metadata:

- dataset/model repository identifiers and immutable revisions;
- model cards, configs, label maps, licenses, file inventories, and provider
  identities;
- package index metadata and distribution hashes;
- public task descriptions and declared split cardinalities.

Stage 0 must not download or inspect dataset rows, labels, predictions,
representations, logits, checkpoint tensors, selector trajectories, or any
sealed outcome. Public model-card metrics may be recorded only as metadata and
may not be used to tune thresholds, budgets, aliases, or selector settings.

## Frozen task suite

The study freezes two tasks simultaneously rather than adding tasks until one
passes.

### Primary: `dair-ai/emotion`

- task family: six-class short-text emotion classification;
- intended official split: train/validation/test = 16,000/2,000/2,000;
- scientific rationale: untouched affect classification, multiclass, raw text,
  multiple public Transformer architectures, and a clean departure from the
  prior sentiment/NLI/topic/intent chain.

### Replication: `papluca/language-identification`

- task family: balanced 20-class multilingual language identification;
- intended official split: train/validation/test = 70,000/10,000/10,000;
- scientific rationale: untouched multilingual identification with a large
  balanced outcome and operationally heterogeneous endpoints spanning a
  Transformer, fastText systems, and a statistical package classifier.

No third task is authorized inside this family. If either task is metadata-
ineligible, PRAA stops before row access and requires a separately justified
future protocol; it is not replaced here.

## Frozen candidate endpoint rosters

Stage 0 tests metadata eligibility only. It does not select endpoints by
observed task predictions.

### Emotion candidate roster

1. `ragunath-ravi/deberta-v3-emotion-classifier`
2. `dk409/emotion-roberta`
3. `nateraw/bert-base-uncased-emotion`
4. `bhadresh-savani/distilbert-base-uncased-emotion`
5. backup only if a listed endpoint is metadata-ineligible:
   `bhadresh-savani/albert-base-v2-emotion`

The retained clean registry is the first four metadata-eligible endpoints in
this fixed order, subject to exact six-label compatibility. A backup may replace
only an endpoint that is unavailable, gated, lacks an immutable revision,
fails to expose loadable weights/config, or has an incompatible label space.
No task row or model output may be used for replacement.

### Language-identification candidate roster

1. `papluca/xlm-roberta-base-language-detection`
2. `facebook/fasttext-language-identification`
3. `cis-lmu/glotlid`
4. PyPI package endpoint `langid==1.1.6`
5. backup only if a listed endpoint is metadata-ineligible:
   `HPLT/OpenLID-v3`

Every endpoint must deterministically map its native labels to the frozen 20
ISO-style target codes in the dataset card. Predictions outside the target set
map to a private out-of-task code and are always incorrect. A backup may be
used only for metadata ineligibility, never for observed accuracy or selector
effect.

## Metadata eligibility gates

A task is Stage-0 eligible only if all apply:

1. its repository is public and has an immutable revision;
2. the documented label space is single-label multiclass with at least four
   classes;
3. the intended outcome split contains at least 2,000 rows;
4. the raw input schema and target label map can be frozen unambiguously;
5. four independently executable candidate endpoints pass the endpoint gates.

An endpoint is Stage-0 eligible only if all apply:

1. public, non-gated access;
2. immutable repository revision or immutable package/version distribution;
3. a loadable model/package artifact with recorded digest metadata;
4. a deterministic raw-input inference path;
5. a deterministic mapping into the target label space;
6. declared provider/root identity and implementation family;
7. license recorded for audit;
8. no dependency on target labels, item IDs, peer outputs, pool seeds, or
   selector state at inference.

## Deterministic registry retention

- Emotion: retain the first four eligible candidates in the frozen order.
- Language ID: retain the first four eligible candidates in the frozen order.
- No quality ranking occurs in Stage 0.
- Development/safety certification in a later locked stage may choose the
  alias parent through a deterministic best-root-set rule with a frozen
  tie-break; it may not replace the task or clean registry.

## Stage-0 outputs

The metadata screen must emit:

- exact dataset and endpoint revisions;
- file inventories and public artifact metadata;
- target label maps;
- provider and architecture families;
- licenses and gated/private status;
- explicit pass/fail reasons;
- a SHA-256 digest of the complete report.

Only after an independently reproducible Stage-0 `PASS` may a later protocol
access task rows, create splits, train error heads, or generate predictions.

## Decision rule

- `PASS_PRAA_STAGE0_METADATA`: both frozen tasks and four endpoints per task
  satisfy every metadata gate.
- `STOP_PRAA_STAGE0_METADATA`: any required task or roster cannot be certified.

A Stage-0 stop permits correction of a demonstrable metadata parser bug on the
same frozen candidate list. It does not permit adding a new task or candidate
because the available roster appears scientifically inconvenient.
