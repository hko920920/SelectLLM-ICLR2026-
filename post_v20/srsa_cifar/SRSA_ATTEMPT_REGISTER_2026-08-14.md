# Prospective transfer and admission attempt register

Date: 2026-08-14 (Asia/Seoul)

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
| SRSA-CIFAR | CIFAR-100 primary + CIFAR-10 replication | `PENDING` | none | new modality, new tasks, real public checkpoints, no learned alias or transferred threshold |

## Binding interpretation

1. SRSA-CIFAR is not an exact rerun or repair of PRAA.
2. PRAA primary and deployment outcomes remain sealed permanently under its stopped family.
3. The 5% structured-variant rate is development-informed and must never be described as selected without prior diagnostics.
4. CIFAR-100 is the only decision-changing primary task.
5. CIFAR-10 cannot replace a failed CIFAR-100 result.
6. No third task may be introduced after a CIFAR outcome is opened.
7. All four fixed roots are attacked and reported; the adversarial maximum uses familywise inference rather than post-outcome attacker selection.
8. A valid `NO_GO` is publishable negative evidence and terminates prospective task search.

## PRAA failure diagnosis retained

The Language-ID failure was not an execution error. The selected parent had safety accuracy 0.9979, while the four derived endpoints had accuracies 0.9853, 0.9891, 0.9904, and 0.9869. The failed gates were `safety_trigger_at_most_one_percent` and `safety_loss_within_one_point`. This demonstrates that a label-free quantile can rank errors well yet fail to transfer an absolute trigger rate under split shift.

SRSA removes this failure mode structurally: every wrapper returns the parent's semantic label on every row. Its utility difference is identically zero rather than estimated from a calibration split.
