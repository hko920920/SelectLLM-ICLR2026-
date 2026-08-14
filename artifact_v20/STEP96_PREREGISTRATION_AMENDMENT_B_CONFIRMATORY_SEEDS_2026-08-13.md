# Step 96 preregistration Amendment B: confirmatory inference seeds

## Timing and scope

This amendment is written after the development decision was repaired under
Amendment A and before the SNLI test outcome is opened. The main protocol fixes
the 3,000 trajectory seeds and all confirmatory gates but omitted explicit
bootstrap/sign-flip seeds for the sealed analysis.

The missing implementation constants are now fixed as follows:

- terminal bootstrap/sign-flip seeds: `96910/96911`;
- active-minus-fixed terminal bootstrap/sign-flip seeds: `96911/96912`;
- cumulative bootstrap/sign-flip seeds: `96912/96913`.

Repetition counts remain 10,000 bootstrap and 100,000 sign flips, inherited
from the protocol. No cell, threshold, model, adapter, test trajectory seed,
gate, or decision threshold changes.
