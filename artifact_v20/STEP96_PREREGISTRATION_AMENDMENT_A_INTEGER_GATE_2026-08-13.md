# Step 96 preregistration Amendment A: integer quality-gate repair

## Timing and scope

This amendment is written after Stage A completed and before any Step 96 test
label was opened. The selected cell, thresholds, adapters, root predictions,
search/validation trajectories, effect vectors, inference seeds, materiality
threshold, test seed range, and every other gate remain unchanged.

## Defect

The protocol requires each alias to lose **at most one percentage point** on
threshold, search, and validation. The implementation compared floating means
with the literal expression `value <= 0.0100`. For the selected cell, every
threshold-partition loss count is exactly `45/4500 = 0.01`, but NumPy subtraction
of two separately computed means serialized this as `0.010000000000000009`.
The code consequently emitted a false gate failure despite the integer count
meeting the locked boundary exactly.

The selected cell's directly reconstructed loss counts are:

- threshold: `[45,45,45,45]` of 4,500; allowed count 45;
- selector search: `[57,53,42,54]` of 6,000; allowed count 60;
- validation: `[86,79,72,68]` of 9,842; allowed count 98.

## Frozen repair

Quality compliance is reconstructed from integer counts:

`utility_loss_count <= floor(0.01 * partition_size + 1e-12)`.

Equivalently, a `1e-12` comparison tolerance may be used when replaying the
stored mean. This implements the already locked phrase “at most one percentage
point”; it does not relax the cap. A repair receipt must bind the original
ledger/config hashes, arrays, counts, amendment, patched source, and absence of
any Step 96 confirmatory output. Only if all other original gates were already
true may the repaired decision become
`GO_TO_STEP96_ONE_TIME_CONFIRMATORY_LOCK`.

No alternate cell may be selected and no development effect may be recomputed
or changed by this repair.
