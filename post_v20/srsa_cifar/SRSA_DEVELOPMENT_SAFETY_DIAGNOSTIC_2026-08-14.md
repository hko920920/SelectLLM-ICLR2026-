# SRSA development-only safety-block diagnostic

Date: 2026-08-14 (Asia/Seoul)
Status: development evidence only; never confirmatory.

## Boundary

This diagnostic was run after PRAA had already opened the Emotion and
Language-ID **safety** blocks. It used only the four clean safety-prediction
vectors, their already-open safety labels, and their existing input UIDs. It did
not access any PRAA primary/deployment input, label, prediction, selector path,
or effect. It did not access any CIFAR row, label, prediction, or outcome.

The purpose was solely to choose between two deterministic structured-variant
rates before the CIFAR family was executed. This diagnostic must be disclosed
whenever SRSA-CIFAR is described.

## Frozen diagnostic settings

- tasks: PRAA Emotion safety (4,000 rows) and Language-ID safety (10,000 rows);
- clean roots: the four already executed endpoints per task;
- attacked roots: all four, one at a time;
- structured wrappers: four around the attacked root;
- variant rates compared: 1% and 5%;
- variant membership: SHA-256 of the existing safety input UID, wrapper index,
  and fixed salt;
- pool size: 500;
- budget: 2;
- temperature: 0.05;
- paired seeds: 1,000 through 1,499 (500 runs);
- root regret: complete safety-block accuracy regret;
- the same typed compatibility was used for grouping and label feedback;
- fixed-query controls used the exact clean query matrix.

## Root accuracies

Emotion roots in registry order: `0.96625`, `0.97175`, `0.99050`, `0.95125`.
Language-ID roots in registry order: `0.99790`, `0.97170`, `0.98000`, `0.95380`.

## Realized wrapper rates

| Task | Nominal | Wrapper 1 | Wrapper 2 | Wrapper 3 | Wrapper 4 |
|---|---:|---:|---:|---:|---:|
| Emotion | 1% | 1.175% | 1.275% | 0.900% | 0.800% |
| Emotion | 5% | 5.025% | 5.200% | 5.650% | 5.250% |
| Language-ID | 1% | 0.860% | 0.950% | 1.070% | 1.000% |
| Language-ID | 5% | 5.030% | 4.570% | 5.060% | 5.100% |

## Paired results

Effects below are attacked minus clean and are reported as fractions; `0.005`
would equal 0.5 percentage points.

| Task | Rate | Attacked root | Path change | Final-root change | Mean terminal effect | Mean cumulative effect | Fixed terminal | Fixed root history |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Emotion | 1% | 0 | 0.748 | 0.328 | +0.003651 | +0.006339 | 0 | exact |
| Emotion | 1% | 1 | 0.944 | 0.292 | +0.001683 | +0.007259 | 0 | exact |
| Emotion | 1% | 2 | 0.994 | 0.260 | -0.001601 | -0.009783 | 0 | exact |
| Emotion | 1% | 3 | 0.936 | 0.384 | +0.003114 | +0.004821 | 0 | exact |
| Emotion | 5% | 0 | 0.942 | 0.382 | +0.004355 | +0.007506 | 0 | exact |
| Emotion | 5% | 1 | 0.990 | 0.360 | +0.002841 | +0.005482 | 0 | exact |
| Emotion | 5% | 2 | 1.000 | 0.300 | -0.000714 | -0.008705 | 0 | exact |
| Emotion | 5% | 3 | 0.966 | 0.410 | +0.003104 | +0.004417 | 0 | exact |
| Language-ID | 1% | 0 | 0.926 | 0.120 | -0.000028 | -0.002305 | 0 | exact |
| Language-ID | 1% | 1 | 0.880 | 0.184 | +0.001391 | +0.009521 | 0 | exact |
| Language-ID | 1% | 2 | 0.734 | 0.124 | +0.001117 | +0.006321 | 0 | exact |
| Language-ID | 1% | 3 | 0.754 | 0.180 | +0.002230 | +0.004829 | 0 | exact |
| Language-ID | 5% | 0 | 0.984 | 0.238 | +0.000672 | -0.000555 | 0 | exact |
| Language-ID | 5% | 1 | 0.960 | 0.202 | +0.001317 | +0.009027 | 0 | exact |
| Language-ID | 5% | 2 | 0.902 | 0.190 | +0.001930 | +0.007059 | 0 | exact |
| Language-ID | 5% | 3 | 0.934 | 0.210 | +0.001894 | +0.004951 | 0 | exact |

## Development choice

The 5% rate was selected before CIFAR execution because it produced consistently
larger path-change rates across both safety tasks while preserving exact
fixed-query equality. It was **not** selected because every terminal effect was
positive or maximal; one Emotion root remained beneficial and the largest
Language-ID terminal mean occurred at 1%. All four CIFAR root attacks are
therefore prespecified and familywise corrected rather than choosing a root
from these diagnostics.

CIFAR uses a raw-image digest rather than a prior NLP UID. No value in this
file is a CIFAR result or evidence for the SRSA primary claim.
