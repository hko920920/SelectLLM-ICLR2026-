# Step 95 post-hoc terminal-bottleneck diagnosis

This report uses only the frozen `selector_verify` development partition. It does not open or inspect the sealed MultiNLI outcome, cannot relabel Step 95, and is not confirmatory evidence.

## Reproduction

- Locked 1,500-run terminal effect: +0.04987 pp.
- Locked 1,500-run final-root change: 21.13%.
- The fixed diagnostic trajectory sample is seeds 954000--954399 (400 runs).

## Geometry

- Mean pool best-minus-second gap: 1.038 pp; tied in 5.75% of pools.
- Mean pool best-minus-worst gap: 25.218 pp.
- Mean terminal delta conditional on a changed root: +0.218 pp.
- At the observed 19.00% root-change rate, the +0.5 pp gate requires a net conditional drop of 2.632 pp.

## Query shift

- Parent-error fraction among clean/refined queries: 8.31% / 11.74%.
- Mean root-disagreement categories among clean/refined queries: 1.677 / 1.685.
- Mean unique labels replaced per run: 5.447 of 20.

## Budget diagnostic

| Budget | Terminal pp | Root change | Cumulative delta |
|---:|---:|---:|---:|
| 5 | +0.1500 | 29.25% | +0.05239 |
| 10 | +0.0805 | 24.00% | +0.05787 |
| 15 | +0.0555 | 19.50% | +0.06056 |
| 20 | +0.0415 | 19.00% | +0.06309 |
| 30 | +0.0265 | 18.25% | +0.06661 |
| 40 | +0.0290 | 15.25% | +0.06941 |
| 60 | +0.0350 | 13.75% | +0.07732 |
| 80 | +0.0255 | 11.75% | +0.08356 |
| 100 | +0.0185 | 10.75% | +0.08782 |

## Interpretation boundary

These post-hoc diagnostics identify mechanism bottlenecks only. Any follow-up must use a newly locked task/endpoint and a design justified without consulting that task's outcomes.
