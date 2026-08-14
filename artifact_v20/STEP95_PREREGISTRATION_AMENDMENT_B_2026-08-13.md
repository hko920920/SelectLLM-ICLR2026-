# Step 95 execution incident B: detached monitor and failed duplicate retry

The external monitor for the first Stage-A execution was terminated after it
returned no streamed stdout for more than seven minutes.  The Python child was
not terminated and continued the original Stage-A computation.  Inspection
showed that it had passed public-root inference and the deterministic
parent-selection rule and had written adapter checkpoints.  Their filenames
revealed only the rule-selected parent, `typeform_distilbert`.  No
search-condition effect, verification effect, development gate, confirmatory
result, or sealed-test outcome had been written or inspected.  The sealed
matched-test outcome remained unopened.

Before the surviving child was discovered, one duplicate deterministic retry
was launched.  That retry repeated the same frozen preflight, root inference,
parent selection, and adapter training, but stopped with an operating-system
disk-space error while saving its first adapter.  It wrote no search row,
verification result, ledger, frozen configuration, or test result.  Its empty
model directory was retained as
`STEP95_FAILED_RETRY_EMPTY_2026-08-13`.  The four completed checkpoints of the
still-running original process were restored to their expected
`step95_models` directory before any Stage-A ledger or configuration was
written.

The surviving original process remains the sole authoritative Stage-A run.
This incident note changes no scientific choice, threshold, gate, task, root,
roster rule, seed, model revision, or selection rule and is not an input to the
running process.  If that original run fails a development gate, the study
stops before the sealed test exactly as preregistered.
