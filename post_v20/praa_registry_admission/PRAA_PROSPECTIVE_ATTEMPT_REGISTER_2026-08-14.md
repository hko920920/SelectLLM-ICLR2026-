# Prospective attempt register for the external-generality question

Date frozen: 2026-08-14 (Asia/Seoul)

This register is fixed before any PRAA primary or deployment outcome is opened.
It prevents the PRAA study from being represented as the first or only attempt
at prospective task transfer.

| Study | Task / family | Literal disposition | Decision-changing observation | Replacement allowed? |
|---|---|---|---|---|
| Step 105 | Yelp Polarity / binary sentiment | `NO_GO_RETAIN_STEP105_PROSPECTIVE_NEGATIVE` | Directional terminal and active-minus-fixed harm of +0.3527 pp with exact fixed-query zero, below the frozen +0.5 pp materiality gates | No |
| Step 113 | BoolQ / binary question answering | `NO_GO_STEP113_PRIMARY_CONFIRMATION` | Strong +3.9619 pp terminal and active-minus-fixed harm with exact fixed-query zero, but each alias exceeded the frozen one-point quality-loss cap | No |
| Step 114 | Amazon Polarity / binary sentiment | `STOP_STEP114_DEVELOPMENT_NOT_CERTIFIED` | Alias quality and structural checks passed, but the designated parent tied a challenger on safety; primary predictions and labels remained unopened | No; outcome remains sealed |
| PRAA | Emotion primary + Language-ID replication / simultaneous two-task family | Pending | Both tasks, registries, data partitions, artifacts, admission policy, and attempt accounting are fixed before model-quality and outcome inspection | No third task |

## Family-level interpretation rule

PRAA is a new independently motivated registry-admission study, not a
continuation that erases the three earlier dispositions. Its scientific report
must cite this table and present all four rows, regardless of whether PRAA is
positive, negative, invalid, or stopped before outcome.

Within PRAA:

- the Emotion primary and Language-ID replication were selected and sealed
  simultaneously;
- neither may be replaced because its clean accuracy, alias behavior, selector
  path, or outcome is inconvenient;
- no third task is authorized;
- a technical implementation defect may be corrected only under the same task,
  registry, data, model, endpoint, admission, and decision rules;
- after a scientific gate is evaluated, the literal GO/NO_GO/STOP disposition
  is final.

## Multiplicity accounting

The final inferential protocol must treat the two PRAA tasks as one frozen
family. The primary decision uses the Emotion task's preregistered gates, while
all task-level p-values and intervals are reported for both tasks and a frozen
family correction is applied. The Language-ID task is a simultaneous
replication, not an optional replacement outcome.
