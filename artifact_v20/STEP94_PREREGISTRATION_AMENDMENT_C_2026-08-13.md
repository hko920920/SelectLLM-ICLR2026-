# Step 94 preregistration Amendment C: failed-output preservation

Locked: `2026-08-13T12:15:49.4101096+09:00`

Amendment B said the failed partial Stage-0 files would be deleted before
regeneration.  The workspace safety policy rejected deletion.  Instead, both
partial directories were atomically moved to the recoverable audit quarantine
`STEP94_STAGE0_FAILED_PARTIAL_2026-08-13/`.  They contain only the 20 Newsgroups
input-only test JSON and its sealed label NPZ created before the MASSIVE
coverage assertion stopped the run.  Stage A and confirmation are forbidden
from reading the quarantine.

This preservation-only substitution changes no scientific rule or data.  The
corrected Stage 0 creates fresh canonical directories and binds their hashes;
the quarantine is retained solely as an audit record of the failed run.
