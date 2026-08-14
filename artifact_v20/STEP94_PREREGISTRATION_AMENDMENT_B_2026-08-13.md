# Step 94 preregistration Amendment B: test class-coverage validator correction

Locked: `2026-08-13T12:13:41+09:00`

The first Stage-0 materialization stopped before completion because the code
required the set of labels present in every official test split to equal the
entire training label alphabet.  The MASSIVE en-US official test split did not
satisfy that auxiliary assertion.  Full test-class coverage was never a
scientific eligibility criterion, selection rule, or success gate in the
preregistration; exact-match multiclass evaluation remains well-defined when a
class has no test example.

At failure, no root or variant had been trained, no test prediction or selector
trajectory existed, and no accuracy, effect, root, regret, class identity, or
class-frequency statistic was printed or inspected.  The only revealed fact
was the Boolean failure of the unnecessary coverage assertion.  Partial
Stage-0 input/sealed-output directories are deleted and regenerated from
scratch.

The correction is limited to replacing exact test-alphabet equality with the
already enforced checks that the test split is nonempty and every label is an
integer in `[0,C)`.  Development partitions must still cover all `C` classes.
The task menu, revisions, partitions, training, grid, selection rule, one-test
stopping rule, and all confirmatory gates remain unchanged.
