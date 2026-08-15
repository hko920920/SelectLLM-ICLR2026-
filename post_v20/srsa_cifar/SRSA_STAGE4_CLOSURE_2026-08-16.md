# SRSA-CIFAR Stage 4 literal closure — 2026-08-16

**Decision:** `NO_GO_SRSA_PRIMARY`

The cleaned pipeline passed Stage 0, Stage 1, all eight real endpoint executions, Stage 2 closure, Stage 3 pre-outcome locking, and the corrected Stage 4 preflight. The first authorized path-only technical retry then opened the sealed labels and executed every frozen condition. A later unplanned duplicate execution reproduced the complete result and paired-run arrays bitwise; it is not used for inference or claim strengthening.

## Primary result

- CIFAR-100: `NO_GO_SRSA_PRIMARY`
- Passing frozen roots: none
- Maximum mean terminal harm: 0.181 pp on `cifar100_vgg11_bn`
- Frozen materiality gate: +0.500 pp terminal harm and +0.500 pp active-minus-fixed harm

| CIFAR-100 attacked root | Path change | Terminal harm | Simultaneous LCB | Holm p | Failed gates |
|---|---:|---:|---:|---:|---|
| `cifar100_resnet20` | 95.00% | 0.071 pp | 0.011 pp | 0.0095999 | terminal_harm_at_least_half_point, active_minus_fixed_at_least_half_point |
| `cifar100_vgg11_bn` | 95.87% | 0.181 pp | 0.120 pp | 3.99996e-05 | terminal_harm_at_least_half_point, active_minus_fixed_at_least_half_point |
| `cifar100_mobilenetv2_x0_5` | 98.47% | -0.144 pp | -0.206 pp | 1 | terminal_harm_at_least_half_point, active_minus_fixed_at_least_half_point, terminal_simultaneous_lower_above_zero, holm_adjusted_signflip_at_most_005, cumulative_mean_above_zero, cumulative_simultaneous_lower_above_zero |
| `cifar100_shufflenetv2_x0_5` | 95.77% | 0.134 pp | 0.072 pp | 3.99996e-05 | terminal_harm_at_least_half_point, active_minus_fixed_at_least_half_point |

## Replication

- CIFAR-10: `NEGATIVE_SRSA_REPLICATION`
- Passing frozen roots: none
- Maximum mean terminal harm: 0.282 pp on `cifar10_resnet20`

## What succeeded scientifically

- Ordered acquisition paths changed for roughly 95–98% of paired runs.
- Positive roots generally had simultaneous lower bounds above zero and Holm-significant effects.
- Active-minus-fixed was numerically identical to the active terminal effect.
- Fixed-query terminal and cumulative effects were exactly zero.
- Authenticated root-aware histories were bitwise identical to clean histories.
- Semantic predictions of wrappers remained identical to their parent endpoints.

## Why the literal decision is NO_GO

The effects were statistically detectable but too small for the preregistered practical materiality threshold. The best CIFAR-100 terminal harm was +0.181 pp, below the required +0.500 pp. CIFAR-10 also remained below the threshold at +0.282 pp. Therefore no root passed all gates.

The protocol forbids a third task and forbids changing the wrapper rate, seed, selector, or materiality threshold after outcome access. SRSA prospective task search is closed.

## Authoritative provenance

- Stage 0–3 source run: `31888204323`
- **Authoritative first Stage 4 run:** `31895293285`
- Authoritative artifact ID: `9249629903`
- Authoritative artifact digest: `sha256:458eb623eb08090a44f24783d9a475c289095808561ceea92770811ea0ed5492`
- Frozen scientific commit: `23dedb414875be0f9531d1bef0634ed546ffafdd`
- Result canonical SHA-256: `879ad921090645464dc32ad39fc5154fa2c12cf8a6d363e1c5deff2eb92344be`
- Result JSON file SHA-256: `62b186dc075d8c18aec121b66cf22621c5b7b934bd8c3d093e9678bd7bfa3db9`
- Paired-run file SHA-256: `bfea462f0025dec3f276e4655caf9b0ad81049bd50f9c3701b45c01d4a12422a`

## Duplicate-execution disclosure

Run `31896042798` was an unplanned later duplicate caused by concurrent orchestration. Its result JSON and paired-run file are bitwise identical to the authoritative first run. It is retained only as a reproducibility duplicate. Globally, labels were opened by two jobs across the two executions; therefore a global single-opening claim must not be made. No further Stage 4 run is permitted.
