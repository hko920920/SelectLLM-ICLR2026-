# PRAA Stage-0 public metadata screen

Decision: `PASS_PRAA_STAGE0_METADATA`

This receipt is metadata-only: no dataset rows, labels, model tensors,
predictions, logits, trajectories, or sealed outcomes were accessed.

- workflow run: [31775008752](https://github.com/hko920920/SelectLLM-ICLR2026-/actions/runs/31775008752)
- workflow artifact: `praa-stage0-public-metadata-screen`
- artifact SHA-256: `2393d97e1f01c2fd1d06a54e11bc2fde3eeb5b755c2f07ea603730a1d952fd2c`
- canonical report SHA-256: `7db16a2458326d25fb858405437d3c81fd8e1c8618dffc1995158c87d5ecd9b0`

## Primary: six-class emotion classification

- dataset: `dair-ai/emotion`
- immutable revision: `cab853a1dbdf4c42c2b3ef2173804746df8825fe`
- documented split sizes: 16,000 train / 2,000 validation / 2,000 test
- decision: `PASS`

Retained clean registry, in frozen order:

1. `dk409/emotion-roberta@66fcef009eac311aab4387b49b408bd9a4c6b1a3`
2. `nateraw/bert-base-uncased-emotion@064d252021b51d95cd0547c89c6489100da0dc4c`
3. `bhadresh-savani/distilbert-base-uncased-emotion@ce6f4ffcde7642ca2cac02381a16da38e5498ff7`
4. `bhadresh-savani/albert-base-v2-emotion@4812613b3c07c549e13f09bb266dbf0e59f48de7`

The roster spans three providers and four implementation families: RoBERTa,
BERT, DistilBERT, and ALBERT. The initially listed DeBERTa endpoint returned
HTTP 401 at the public metadata endpoint and was excluded under the frozen
metadata-only backup rule before any task row or model weight was accessed.

## Replication: balanced 20-class language identification

- dataset: `papluca/language-identification`
- immutable revision: `aa56583bf2bc52b0565770607d6fc3faebecf9e2`
- documented split sizes: 70,000 train / 10,000 validation / 10,000 test
- decision: `PASS`

Retained clean registry, in frozen order:

1. `papluca/xlm-roberta-base-language-detection@9865598389ca9d95637462f743f683b51d75b87b`
2. `facebook/fasttext-language-identification@3af127d4124fc58b75666f3594bb5143b9757e78`
3. `cis-lmu/glotlid@85cd6716494360367b75f642b5bc78667605d0b4`
4. `langid==1.1.6`

The roster spans four providers and four endpoint families: XLM-RoBERTa,
Meta fastText LID, GlotLID fastText, and the `langid` statistical classifier.
Every endpoint will map predictions into the frozen 20-code target space;
out-of-target native labels will map to a private always-incorrect code.

## Scientific interpretation

Stage 0 closes only task/registry metadata eligibility. It does not establish
model quality, alias quality, path change, causal harm, or deployment impact.
Those quantities remain unseen.

The study is now bound to both tasks simultaneously. There is no third-task
replacement path. The next authorized stage is limited to exact artifact
materialization, label-map validation, deterministic data partitioning, and
creation of separate input and sealed-label packages. Selector-effect analysis
and primary-outcome opening remain prohibited until later locks pass.
