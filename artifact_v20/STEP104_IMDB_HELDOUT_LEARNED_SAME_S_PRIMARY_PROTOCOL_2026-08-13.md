# Step 104 IMDB held-out learned same-s terminal primary protocol

## Status and provenance

This held-out protocol is written after development Steps 100--103 and before
any label in the 13,000-row IMDB outcome is opened or analyzed.  It is a
development-informed held-out confirmation, not a before-all-data
preregistration.  Steps 100--102 retain their NO-GO labels.  Step 103 is a
development GO, not an outcome claim.

The 25,000 official IMDB test rows were divided by a fixed within-label SHA-256
rule at Step 100: 12,000 development rows and 13,000 held-out rows.  The latter's
label-free input file and separately sealed label file were created and hashed
before any IMDB root prediction.  No held-out label, class-specific prediction,
quality value, effect, or trajectory has been inspected through Step 103.

## Immutable held-out object

- Dataset: `stanfordnlp/imdb`, config `plain_text`, revision
  `e6281661ce1c48d982bc483cf8a173c1bbeb5d31`.
- Label-free input:
  `step100_test_inputs/imdb_test_heldout_inputs.json`, 13,000 rows, SHA-256
  `6ac998dbe34e88c01fd940d551bb9b01deb63b15ad8855f5428835f96c480de9`.
- Sealed outcome:
  `external_data/step100_sealed/imdb_test_heldout_sealed_outcomes.npz`, SHA-256
  `1ee1c6005330792de2a375064efa47a51b616a8145f72a022946d0ded9a4b2db`.
- Roster, in order:
  1. parent `wrmurray/roberta-base-finetuned-imdb`, revision
     `7aa8ca3fae56a1860d8b4c6bf727b91370821ad5`;
  2. `lvwerra/distilbert-imdb`, revision
     `0fc02cd68445b599a9cb2da2368050e7fb31d29a`;
  3. `textattack/distilbert-base-uncased-imdb`, revision
     `5b0f46c2fc4b86bf21f0ec0409bed77ee142b332`;
  4. `Harsha901/tinybert-imdb-sentiment-analysis-model`, revision
     `0cd5d1ac6c06eb0f5f81b7022f95691528c981fa`.
- Four learned RoBERTa adapters and their SHA-256 digests: exactly those in
  `STEP102_STAGEA_FROZEN_CONFIG_2026-08-13.json`; each must be hash-distinct and
  contain 14,767,874 trained parameters.
- Thresholds, in alias order:
  `[0.9515677094459534, 0.9398181438446045,
    0.9488589763641357, 0.9394304752349854]`.
- Selected development cell:
  `l0.0075_t0.025_b5_tau0.050`.
- Pool size 500, budget 5, temperature .05, four aliases mapped to the parent
  root, max input length 256.
- Acquisition and posterior evidence call the same literal
  `s(a,b)=1[a=b]`.
- Exactly 3,000 paired seeds `1003000:1005999`.
- Paired bootstrap repetitions 10,000 and paired sign-flip repetitions 100,000.
  Inference seeds are `100800/100801`, `100801/100802`, and
  `100802/100803` for terminal, active-minus-fixed terminal, and cumulative
  effects.

No model, revision, roster, adapter, threshold, condition, task, split, seed, or
similarity can be replaced.  A failed outcome ends Step 104.

## Pre-outcome lock

All four root prediction vectors, four error-score vectors, trigger vectors, and
four alias response vectors are generated from the label-free input and hashed
before outcome access.  Runtime accepts review text and the already computed
parent hard label only.  It cannot receive a reference, label, item ID, lookup,
peer output, pool, posterior, trajectory, selector state, split, or dataset ID.

A lock binds this protocol; all Step 100--103 authority and decisions; complete
development grid and selected-cell raw vectors; label-free input and prediction
packages; sealed-outcome byte hash; all source code; all adapter weights/audits;
all root revisions; thresholds; selector parameters; and seeds.  The sealed
outcome may be opened once only after the lock is written.

## Mandatory primary gates

### Implementation and independence

1. Every bound protocol, data, model, prediction, code, grid, ledger, outcome,
   and lock hash matches.
2. Steps 100--102 remain NO-GO and Step 103's robust selection is exactly
   reconstructed.
3. Raw-input re-inference reproduces every locked root and alias prediction
   bitwise.
4. Four adapter hashes are distinct and each has 14,767,874 learned parameters.
5. Endpoint signature/source contains no forbidden outcome or selector-state
   input.
6. Acquisition and posterior evidence use the same literal exact-match
   implementation.

### Quality and mechanism

7. Every alias is coordinate-wise non-improving relative to its parent.
8. Each alias has exact quality-loss count at most
   `floor(.01*13000)=130`.
9. At least 50% of ordered acquisition paths change.
10. Parent-to-challenger terminal transitions exceed challenger-to-parent
    transitions.
11. Fixed-query root histories are bitwise identical to clean histories and all
    fixed terminal/cumulative effects are exactly zero.

### Outcome

12. Mean terminal harm is at least +.5 percentage point.
13. Mean active-minus-fixed terminal harm is at least +.5 percentage point.
14. Terminal and active-minus-fixed 95% paired-bootstrap lower bounds exceed
    zero and one-sided paired sign-flip p-values are at most .05.
15. Cumulative regret mean and paired-bootstrap lower bound exceed zero.

The only success label is
`GO_STEP104_IMDB_HELDOUT_LEARNED_SAME_S_TERMINAL_PRIMARY`.
Any failed gate yields `NO_GO_RETAIN_STEP104_NEGATIVE`.

## Interpretation boundary

A pass is a clean held-out-outcome success for four independently learned
14.77M-parameter raw-text adapters on a new task family, with a source-faithful
single similarity, material terminal harm, active-minus-fixed mediation, and
exact fixed-query zero.  The roster and condition were developed on the disjoint
12,000-row development package, so the result is reported as held-out rather
than before-all-data preregistered.  It does not establish live registry
admission, secret-label compromise, a separately trained full base checkpoint,
average harm, or production attack.
