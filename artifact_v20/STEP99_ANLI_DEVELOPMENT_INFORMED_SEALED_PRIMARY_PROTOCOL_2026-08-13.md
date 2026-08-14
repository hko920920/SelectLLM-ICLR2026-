# Step 99 ANLI development-informed sealed primary protocol

## Status, provenance, and non-relabeling

This protocol is written after the train-only Step 98 development decision and
before the selected ANLI `dev_r3` outcome is opened or analyzed.  Step 98 remains
`NO_GO_STEP98_DEVELOPMENT_STOP`: it failed its preregistered auxiliary condition
that the clean selector choose the parent root in at least 60% of runs on both
development partitions.  Nothing in Step 99 relabels that decision.

The independent reconstruction
`STEP98_STAGEA_INDEPENDENT_VALIDATION_2026-08-13.json` establishes that this was
the only failed Step 98 development gate.  On the search partition the clean
parent-selection rate was 53.125%, while terminal harm was +3.7665 points,
parent-to-challenger transitions were 46.5% versus 1.0% in the reverse
direction, and fixed-query effects were exactly zero.  The 60% frequency is
therefore neither necessary for the causal mechanism nor one of the external
review's score-changing requirements.  It is dropped prospectively rather than
changed after seeing a held-out outcome.

Step 99 is a development-informed, prospective held-out confirmation.  It is
not represented as a before-all-data preregistration.  Its selected model,
adapters, thresholds, roster, similarity, task, input rows, seeds, and sealed
outcome were already fixed under Step 98 and cannot be changed here.

## Immutable experimental object

- Dataset: `facebook/anli`, config `plain_text`, selected split `dev_r3`.
- Dataset revision:
  `8e4813d81f46d313dac7892e1c28076917cfcdf9`.
- Label-free input authority:
  `step98_test_inputs/anli_r3_dev_inputs.json`, 1,200 rows, SHA-256 bound in
  `STEP98_SELECTED_DEV_SEAL_MANIFEST_2026-08-13.json`.
- Sealed outcome authority:
  `external_data/step98_sealed/anli_r3_dev_sealed_outcomes.npz`, whose SHA-256
  was bound before Step 98 learned development.
- Four clean roots, mappings, revisions, parent, and four learned adapter files:
  exactly those in `STEP98_STAGEA_FROZEN_CONFIG_2026-08-13.json`.
- Four thresholds:
  exactly those in that frozen config; no refit, grid, backoff, or test-quality
  adaptation is allowed.
- Runtime alias endpoint: premise, hypothesis, and the parent's response only.
  It cannot receive labels, references, item IDs, lookups, peer outputs, pool,
  posterior, trajectory, selector state, split, or round.
- Alias response: the parent label unless the adapter score exceeds its frozen
  threshold; otherwise a unique out-of-label `ABSTAIN_a` response.
- Acquisition and posterior evidence use the same literal exact-match
  similarity `s(a,b)=1[a=b]`.
- Pool size 500, annotation budget 10, temperature .025, four aliases mapped to
  the parent root.
- Exactly 3,000 paired seeds `985000:987999`.
- Paired-bootstrap repetitions 10,000 and paired sign-flip repetitions 100,000.
  Test inference seeds are `98900/98901`, `98901/98902`, and `98902/98903` for
  terminal, active-minus-fixed terminal, and cumulative effects.

The label-free root and alias prediction package must be generated and hashed,
and a lock must bind this protocol, all upstream authority, all code, all model
weights, all predictions, the sealed-outcome hash, and all seeds before the
outcome is opened.  A failed outcome ends Step 99; there is no replacement
split, round, task, roster, model, adapter, threshold, similarity, or seed range.

## Mandatory primary gates

### Implementation and independence

1. Every protocol, upstream ledger, input, code, model, prediction, sealed
   outcome, and lock hash matches.
2. Step 98 remains recorded as NO-GO and its independent reconstruction passes.
3. The four adapters are hash-distinct and each contains at least ten million
   learned parameters.
4. Endpoint signature and source audit admit only raw text, parent predictions,
   locked adapter paths, and locked thresholds; no held-out outcome is available
   during prediction construction.
5. Root and alias predictions reconstructed from the label-free inputs are
   bitwise equal to the locked pre-outcome package.
6. Acquisition and posterior evidence call the same literal exact-match
   implementation.

### Quality and mechanism

7. Each alias is coordinate-wise non-improving relative to its parent.
8. Each alias loses at most one percentage point of utility, using the exact
   integer gate `loss_count <= floor(.01*1200)`.
9. At least 50% of paired acquisition paths change.
10. Parent-to-challenger final-root transitions exceed challenger-to-parent
    transitions.
11. Fixed-query root histories are bitwise identical to clean histories and all
    fixed terminal and cumulative effects are exactly zero.

### Outcome

12. Mean terminal harm is at least +0.5 percentage point.
13. Mean active-minus-fixed terminal harm is at least +0.5 percentage point.
14. Terminal and active-minus-fixed paired-bootstrap 95% lower bounds exceed
    zero and their one-sided paired sign-flip p-values are at most .05.
15. Cumulative regret mean and paired-bootstrap lower bound exceed zero.

The only success label is
`GO_STEP99_ANLI_DEVELOPMENT_INFORMED_SEALED_PRIMARY`.
Any failed gate yields `NO_GO_RETAIN_STEP99_NEGATIVE`.

## Interpretation boundary

A pass is a held-out-outcome success for a frozen learned raw-input endpoint on
a new dataset family, using one source-faithful similarity and showing terminal,
active-minus-fixed, and fixed-query mediation gates.  Because the protocol is
development-informed and follows prior negative attempts, it is reported with
that provenance and is not called an untouched before-all-data preregistration.
It still does not establish live registry admission, secret-label compromise,
a separately trained full base checkpoint, average harm, or production attack.
