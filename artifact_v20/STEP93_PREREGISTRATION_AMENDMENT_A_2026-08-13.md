# Step 93 preregistration amendment A: executable-mirror row counts

Locked: `2026-08-13T01:37:37.4455445+09:00`

This is a pre-outcome administrative correction to the Step 93 protocol. The
original Banking77 public metadata reports 10,003 training and 3,080 test
examples. The already-preregistered, pinned executable mirror revision
`mteb/banking77@18072d2685ea682290f7b8924d94c62acc19c0b2` materializes 9,993
training and 3,076 test examples.

The first Stage 0 attempt stopped at its immutable row-count assertion
immediately after dataset materialization. At this amendment lock:

- no dataset row text or label had been printed or manually inspected;
- no train partition had been iterated;
- no model had been trained or evaluated;
- no selector path, root, regret, or test statistic existed; and
- no Stage 0 workspace output file had been created.

Stage 0 will therefore bind the executable counts 9,993/3,076 and the raw
parquet hashes. Every scientific choice remains unchanged: dataset revision,
hash split, model and adapter architectures, eligible parents, 504-condition
Stage A grid, Stage B seeds, source-faithful single exact-match similarity, and
all confirmatory gates. This amendment authorizes no outcome-dependent change.
