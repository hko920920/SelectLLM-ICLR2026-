# Step 103 IMDB robust dual-development selection protocol

## Timing and status

This protocol is written after Step 102's single search-maximizing cell failed
on selector verification and before any of the other Step 102 cells is evaluated
on that verification partition.  The 13,000 IMDB outcome labels remain sealed
and unopened.  Step 102 remains `NO_GO_STEP102_LEARNED_DEVELOPMENT_STOP`.

Step 102 established all of the following without outcome access: both roster
geometry partitions pass; four independently trained 14,767,874-parameter
RoBERTa error adapters exist; all 81 search cells pass every search gate; and
the search-maximizing `B=5,tau=.05` cell does not transfer to verification.  The
remaining problem is development-cell robustness, not task or oracle geometry.

## Complete outcome-blind verification map

Every Step 102 cell is now evaluated on `selector_verify` using its already
frozen threshold, aliases, budget, temperature, and seeds
`1001000:1002499`.  No threshold, model, prediction, partition, seed, or cell is
changed.  The three trigger-cap labels within each loss cap produced bitwise
identical thresholds and alias matrices under the frozen safety backoff.  The
implementation may therefore compute the 27 unique
`loss-cap x budget x temperature` conditions once and expand them back to all 81
protocol cells, recording the exact equivalence classes.

For both search and verification, a cell must retain the Step 102 gates:

1. coordinate-wise non-improvement and <=1-point loss on every development
   partition;
2. path-change rate >=50%;
3. parent-to-challenger transitions exceed the reverse;
4. terminal and active-minus-fixed terminal mean harm >=+.5 point;
5. cumulative regret mean positive; and
6. bitwise fixed-query root-history equality with terminal/cumulative zero.

Among dual-eligible cells, select the cell maximizing, in order:

1. the smaller search/verification terminal harm;
2. the smaller search/verification cumulative harm;
3. the smaller search/verification path-change rate;
4. smaller calibration loss cap;
5. smaller trigger-cap label;
6. smaller budget;
7. smaller temperature; and
8. lexical cell ID.

The selected verification vectors alone receive 10,000 paired bootstrap and
100,000 one-sided paired sign-flip repetitions, using seeds
`103700/103701`, `103701/103702`, and `103702/103703`.  Terminal and
active-minus-fixed lower bounds must exceed zero with `p<=.05`; cumulative mean
and lower bound must exceed zero.  Any failure stops without trying another cell
or opening the outcome.

The development success label is `GO_STEP103_TO_HELDOUT_LOCK`; failure is
`NO_GO_STEP103_ROBUST_DEVELOPMENT_STOP`.  A GO is model development only.  It
permits one separately written held-out protocol binding the selected condition,
models, predictions, outcome hash, and seeds before the sealed outcome is opened.
