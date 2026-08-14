# Step 104 review closure: learned raw-input same-similarity held-out primary

## Decision

`GO_STEP104_IMDB_HELDOUT_LEARNED_SAME_S_TERMINAL_PRIMARY`

The previously identified score-changing condition is now met on a disjoint
held-out population. The result uses four independently seeded learned
raw-text adapters, no held-out reference or item lookup at construction or
runtime, the same literal exact-match similarity for acquisition and reference
evidence, material terminal and active-minus-fixed harm, and exact zero effect
under a fixed query matrix.

This is a development-informed held-out confirmation, not a before-all-data
preregistration, independently trained full base model, live registry
admission, or production compromise.

## Selection and outcome separation

- Step 100 split the official 25,000-row IMDB test population within label by
  SHA-256 into 12,000 development rows and a separately stored 13,000-row
  sealed outcome before candidate outcomes were analyzed.
- Steps 100 and 101 retained roster-geometry stops.
- Step 102 retained a failed development selection: the search-maximizing cell
  did not replicate on disjoint development verification.
- Step 103, before evaluating all other cells on verification, froze a robust
  rule that maximized the smaller terminal effect across development search and
  verification among dual-eligible cells. All 27 unique conditions (81 nominal
  cells) were retained; six were dual-eligible.
- The selected cell produced +1.5605 terminal points on search and +0.5332
  [0.4185, 0.6484] on verification. Step 103 independently reconstructed the
  complete selection while the 13,000 outcomes remained unopened.
- Step 104 generated all root and alias predictions without labels and bound
  those predictions, models, revisions, adapters, thresholds, code, seeds,
  input, and sealed-outcome hash before the outcome opened once.

## Locked primary result

Across 3,000 paired pools:

- terminal regret: +0.7024 percentage points, 95% paired-bootstrap interval
  [+0.6155, +0.7882], one-sided sign-flip p = 9.9999e-6;
- active-minus-fixed terminal regret: +0.7024 points, interval
  [+0.6124, +0.7891], p = 9.9999e-6;
- cumulative regret: +0.0279467, interval [+0.0246687, +0.0312927];
- ordered-path and query-set change: 99.9333%;
- final-root change: 60.6667%;
- parent-to-challenger versus reverse transitions: 35.4333% versus 12.1333%;
- fixed-query terminal and cumulative effects: exactly zero, with bitwise-equal
  root histories;
- alias loss counts: 60/68/57/68 of 13,000, below the allowed 130, with no
  coordinate-wise improvement;
- all 16 frozen gates: pass.

## Independent validation

`PASS_STEP104_PRIMARY_INDEPENDENT_RECONSTRUCTION` re-inferred all four roots and
all four adapters from the 13,000 raw texts, reconstructed every primary array,
paired inference quantity, and gate, and found no mismatch.

`validate_step104_artifact_reconstruction.py` is a second implementation that
does not import the experimental selector runner. It independently reconstructs
exact-group acquisition, randomized canonical tie rules, fixed-query replay,
root regret, bootstrap intervals, sign-flip tests, primary arrays, and all
gates from the bound pre-outcome responses and sealed outcome.

## Retained negative trail

The Step 96 primary quality near-miss and secondary sensitivity, Step 97 SciTail
geometry stop, Step 98 development stop, Step 99 ANLI held-out negative, and
Steps 100--102 IMDB development failures remain in the artifact and manuscript.
They are not relabeled or pooled with Step 104.

## Manuscript effect

The main text now leads with the Step 104 held-out primary rather than the SNLI
secondary result. The abstract, introduction, protocol, results, limitations,
reproducibility statement, appendix, table-source validator, and artifact map
all state the development-informed provenance and the remaining operational
boundary. The scientific text remains within nine pages.
