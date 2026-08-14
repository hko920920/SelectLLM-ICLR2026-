# Step 102 IMDB Wrmurray learned-adapter development protocol

## Timing and provenance

This protocol is written after the outcome-blind Step 100 and Step 101 geometry
stops and before any Wrmurray-based error adapter is initialized or trained.  The
13,000 IMDB outcome labels remain sealed and unopened.  Steps 100 and 101 retain
their NO-GO labels.

Step 101 showed that only `wrmurray/roberta-base-finetuned-imdb` had oracle
terminal harm above +1 point on both development partitions (+1.503/+1.586
points).  It missed the self-imposed 2.5-point parent-gap screen because the
roster included the unusually strong ALBERT challenger.  Step 102 is explicitly
development-informed: it fixes that parent and restores the three other public
legacy challengers used before the ALBERT result was known.  This is model
development, not a held-out primary claim.

## Frozen roster and geometry confirmation

1. parent: `wrmurray/roberta-base-finetuned-imdb`, revision
   `7aa8ca3fae56a1860d8b4c6bf727b91370821ad5`;
2. `lvwerra/distilbert-imdb`, revision
   `0fc02cd68445b599a9cb2da2368050e7fb31d29a`;
3. `textattack/distilbert-base-uncased-imdb`, revision
   `5b0f46c2fc4b86bf21f0ec0409bed77ee142b332`;
4. `Harsha901/tinybert-imdb-sentiment-analysis-model`, revision
   `0cd5d1ac6c06eb0f5f81b7022f95691528c981fa`.

All predictions are already outcome-blind development vectors, use identity
labels and max length 256, and are hash-bound.  Before training, the exact
reference-aware oracle screen is rerun with pool 500, budget 10, temperature
.025, four aliases, and Step 100's search/verification seed ranges.  Both
partitions must satisfy the Step 101 gates: parent gap >=2.5 points, parent error
>=3%, challenger complementarity >=15%, terminal harm >=1 point, path change
>=50%, directional transitions, and exact fixed-query zero.  Failure stops.

## Four independently learned raw-text adapters

Four adapters independently initialize from the pinned RoBERTa parent.  Token
embeddings and encoder layers 0--9 remain frozen; encoder layers 10--11 and the
classification head train to predict parent error from review text.

- `adapter_train`: the same fixed 6,000 development rows;
- seeds `102100+a`, `a in {0,1,2,3}`;
- two epochs and class-balanced bootstrap;
- max length 256, batch 8, gradient accumulation 4;
- AdamW `2e-5`, weight decay `.01`, 10% warmup then decay;
- gradient cap 1.0, deterministic CUDA, CUDA-only mixed precision.

Every saved adapter must be hash-distinct and contain at least ten million
learned parameters.  Runtime takes review text and the frozen parent hard label
only; it emits the parent label or unique `ABSTAIN_a`.  It cannot receive any
reference, item ID/lookup, peer output, or selector state.

## Complete development search and verification

Threshold construction, all 81 cells, partitions, seeds, selection rule, and
gates are exactly those frozen in Step 100:

- calibration loss cap `.0025/.005/.0075`;
- trigger cap `.025/.05/.10`;
- budget `5/10/20`;
- temperature `.01/.025/.05`;
- same literal exact-match similarity for acquisition and posterior evidence;
- safety realized loss <=.5% and exact 99% CP upper bound <=1%;
- search seeds `1000000:1000399`;
- selected-cell verification seeds `1001000:1002499`;
- selection maximizes terminal, cumulative, path, then smaller loss, trigger,
  budget, and temperature;
- all quality, >=50% path, directionality, >=+.5 terminal and active-minus-fixed,
  positive cumulative, inference, and exact fixed-query gates remain mandatory.

No held-out outcome is opened by Step 102.  A development pass is labeled
`GO_STEP102_TO_HELDOUT_LOCK`; failure is
`NO_GO_STEP102_LEARNED_DEVELOPMENT_STOP`.  Only a pass permits a separately
written, pre-outcome held-out protocol and lock.
