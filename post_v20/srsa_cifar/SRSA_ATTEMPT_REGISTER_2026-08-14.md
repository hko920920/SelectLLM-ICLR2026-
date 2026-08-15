# Prospective transfer and admission attempt register

Date: 2026-08-14 (Asia/Seoul)  
Last updated: 2026-08-16 (Asia/Seoul)

This register is cumulative. A later study does not erase, reinterpret, or replace an earlier literal decision.

| Family | Task / registry | Literal status | Outcome access | Decision-relevant reason |
|---|---|---|---|---|
| Step 105 | Yelp | `NO_GO` | opened under its lock | direction and mediation passed, but terminal materiality remained below the frozen +0.5-point gate |
| Step 113 | BoolQ | `NO_GO` | opened under its lock | strong effect, but learned alias utility loss exceeded the frozen one-point quality cap |
| Step 114 | Amazon | `STOP` | primary outcome never opened | alias quality passed; designated parent tied a challenger under the old unique-best gate |
| PRAA | Emotion | `PASS_PRAA_STAGE3_TASK` | safety only | all learned-head and one-point safety gates passed; no primary/deployment outcome opened |
| PRAA | Language ID | `STOP_PRAA_STAGE3_TASK` | safety only | transferred 0.993-quantile thresholds triggered 144, 107, 80, and 123 of 10,000 safety rows; two aliases lost more than one point |
| PRAA | two-task family | `STOP_PRAA_STAGE3_ERROR_HEAD_AND_THRESHOLD` | no primary/deployment outcome opened | the frozen protocol required both tasks to pass Stage 3 |
| SRSA development | prior NLP safety blocks | development only | already-open safety blocks only | raw-input-hash structured-variant rates 1% and 5% were examined; 5% was selected before any CIFAR execution |
| SRSA-CIFAR | CIFAR-100 primary + CIFAR-10 replication | `NO_GO_SRSA_PRIMARY` | opened once after Stage-3 PASS | acquisition paths changed in 95–98% of runs and exact controls passed, but the largest CIFAR-100 terminal harm was +0.1806 points, below the frozen +0.5-point materiality gate; CIFAR-10 replication was also negative |

## Binding interpretation

1. SRSA-CIFAR is not an exact rerun or repair of PRAA.
2. PRAA primary and deployment outcomes remain sealed permanently under its stopped family.
3. The 5% structured-variant rate is development-informed and must never be described as selected without prior diagnostics.
4. CIFAR-100 is the only decision-changing primary task.
5. CIFAR-10 cannot replace a failed CIFAR-100 result.
6. No third task may be introduced after a CIFAR outcome is opened.
7. All four fixed roots were attacked and reported; the adversarial maximum used familywise inference rather than post-outcome attacker selection.
8. The valid `NO_GO_SRSA_PRIMARY` terminates prospective task search under this register.

## SRSA-CIFAR closure retained

The source one-shot run `31888204323` completed Stage 0, Stage 1, all eight clean endpoint jobs, Stage 2 closure, and Stage 3 pre-outcome. Its first Stage-4 job stopped before label download because the downloaded Stage-3 artifact retained a `stage3_out/` directory prefix. An outcome-blind technical retry, run `31895293285`, used the exact immutable source artifacts and corrected only that extraction path. It verified Stage-3 `PASS_SRSA_STAGE3_PREOUTCOME`, downloaded the sealed labels only afterward, opened labels once, and executed every frozen condition.

The literal primary decision is `NO_GO_SRSA_PRIMARY`. The strongest CIFAR-100 root attack was VGG11-BN with +0.180583 percentage points terminal harm. The effect had a positive simultaneous lower bound and Holm-adjusted one-sided p-value approximately `4.0e-5`, while fixed-query effects were exactly zero and authenticated root-aware histories were bitwise identical. It nevertheless failed the frozen +0.5-point terminal and active-minus-fixed materiality gates. CIFAR-10 had no passing root and cannot rescue the primary.

## PRAA failure diagnosis retained

The Language-ID failure was not an execution error. The selected parent had safety accuracy 0.9979, while the four derived endpoints had accuracies 0.9853, 0.9891, 0.9904, and 0.9869. The failed gates were `safety_trigger_at_most_one_percent` and `safety_loss_within_one_point`. This demonstrates that a label-free quantile can rank errors well yet fail to transfer an absolute trigger rate under split shift.

SRSA removes this failure mode structurally: every wrapper returns the parent's semantic label on every row. Its utility difference is identically zero rather than estimated from a calibration split.
