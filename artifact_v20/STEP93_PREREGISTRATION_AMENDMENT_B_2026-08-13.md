# Step 93 preregistration amendment B: unused mirror metadata column

Locked: `2026-08-13T01:38:18.2206013+09:00`

After amendment A, the second Stage 0 attempt stopped at the immutable schema
assertion. The pinned executable mirror contains columns `text`, `label`, and
`label_text`, rather than only `text` and `label`. No row was iterated, emitted,
partitioned, modeled, or scored; all Stage 0 workspace outputs remained absent.

Step 93 will require exactly these three columns and will ignore `label_text`.
Only `text` is an endpoint input and only integer `label` is a reference. No
scientific design, search choice, outcome, or confirmatory gate changes.
