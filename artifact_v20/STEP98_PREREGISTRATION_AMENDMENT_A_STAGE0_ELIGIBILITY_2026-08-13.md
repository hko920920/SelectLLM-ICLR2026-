# Step 98 preregistration amendment A: Stage-0 eligibility control flow

This amendment is written after the first Stage-0 invocation stopped and before
any root model was loaded for ANLI inference, any candidate prediction or
effect was produced, any adapter was trained, or any ANLI dev label was
inspected or exported.

The preregistration states that a round with fewer than 4,800 train rows in any
label is ineligible.  The first implementation correctly detected that R1 label
2 has 4,523 rows, but raised an exception rather than recording R1 as
ineligible and continuing to R2/R3.  No Stage-0 output or manifest was written.
The implementation is repaired to record such a round and continue; the quota,
round order, selection rule, models, gates, and every downstream condition are
unchanged.

The installed `datasets` builder also materialized all ANLI split caches while
the code requested only `split=train_r1`.  The experiment code iterated only
the requested train object; it did not index, summarize, print, export, or use a
dev/test example or label.  Console output exposed only public split sizes.
This cache-materialization behavior is recorded in the manifest.  It supplies
no round-selection or outcome information and does not authorize dev access.
