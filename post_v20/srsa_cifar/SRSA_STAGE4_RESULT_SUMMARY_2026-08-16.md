# SRSA-CIFAR Stage 4 closure — 2026-08-16

## Literal decision

`NO_GO_SRSA_PRIMARY`

The clean same-run execution completed Stage 0 through Stage 3, and the outcome-blind path-only technical retry completed the single authorized Stage 4 label opening and every frozen primary/replication condition.

## Authoritative execution

- source run: `31888204323`
- source head: `23dedb414875be0f9531d1bef0634ed546ffafdd`
- Stage 4 technical retry run: `31895293285`
- Stage 4 job: `95037579202` — `success`
- result artifact: `9249629903`, `srsa-stage4-result-technical-retry`
- artifact digest: `sha256:458eb623eb08090a44f24783d9a475c289095808561ceea92770811ea0ed5492`
- declared result hash: `879ad921090645464dc32ad39fc5154fa2c12cf8a6d363e1c5deff2eb92344be`
- result-file SHA-256: `62b186dc075d8c18aec121b66cf22621c5b7b934bd8c3d093e9678bd7bfa3db9`
- paired-run SHA-256: `bfea462f0025dec3f276e4655caf9b0ad81049bd50f9c3701b45c01d4a12422a`

## Information boundary

- Stage 3 lock was verified before label download.
- The sealed-label artifact was downloaded only by the authorized Stage 4 job.
- Labels were opened once.
- CIFAR-100 primary and CIFAR-10 replication were both executed.
- All four frozen root attacks were reported on both tasks.
- No third task was introduced.

## CIFAR-100 primary

No root passed every preregistered gate.

| Attacked root | Path change | Terminal harm | Active-minus-fixed | Fixed-query | Root-aware | Primary gate |
|---|---:|---:|---:|---|---|---|
| ResNet20 | 95.00% | +0.0712 pp | +0.0712 pp | exact zero | exact zero | FAIL materiality |
| VGG11-BN | 95.87% | **+0.1806 pp** | **+0.1806 pp** | exact zero | exact zero | FAIL materiality |
| MobileNetV2-x0.5 | 98.47% | -0.1438 pp | -0.1438 pp | exact zero | exact zero | FAIL direction and materiality |
| ShuffleNetV2-x0.5 | 95.77% | +0.1342 pp | +0.1342 pp | exact zero | exact zero | FAIL materiality |

The adversarial maximum was VGG11-BN at `+0.180583` percentage points. Its simultaneous lower bound was positive, its Holm-adjusted one-sided p-value was approximately `4.0e-5`, and cumulative harm was `+0.470239` percentage points. Nevertheless, the frozen terminal and active-minus-fixed materiality gates each required `+0.5` percentage points. Therefore the result is a literal `NO_GO_SRSA_PRIMARY`.

## CIFAR-10 non-substitutable replication

The replication decision was `NEGATIVE_SRSA_REPLICATION`. No root reached every gate. Its largest terminal harm was `+0.282361` percentage points, still below the frozen `+0.5`-point threshold. The replication cannot replace or rescue the CIFAR-100 primary.

## Scientific interpretation

The experiment gives a clean causal diagnostic but not the preregistered score-changing success:

1. structured refinements changed ordered acquisition paths in roughly 95–98% of paired runs;
2. fixed-query effects were exactly zero;
3. authenticated root-aware histories were bitwise identical;
4. several active effects were positive and statistically separated from zero;
5. the effects were not large enough to satisfy the frozen materiality threshold.

This result must not be reported as a confirmatory GO or as evidence that the internal project score necessarily rises from 4 to 5. It can be reported as transparent negative prospective evidence supporting the mechanism/control story and delimiting its effect size on these CIFAR registries.
