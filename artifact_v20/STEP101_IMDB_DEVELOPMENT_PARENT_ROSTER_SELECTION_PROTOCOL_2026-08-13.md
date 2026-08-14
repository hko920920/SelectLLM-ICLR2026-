# Step 101 IMDB development-only parent/roster selection protocol

## Timing and status

This development protocol is written after Step 100 stopped at its preregistered
geometry gate and before any additional root prediction is computed.  The 13,000
IMDB outcome labels remain sealed and unopened.  Step 100 remains
`NO_GO_STEP100_GEOMETRY_STOP`; it is not relabeled.

Step 101 is explicitly outcome-blind model development, not a primary result.
Its purpose is to avoid further blind roster guesses before one held-out
confirmation.  Candidate identities are selected solely from public model-card
IMDB accuracy above 94%, architecture support for a learned >=10M-parameter
raw-input adapter, and pinned public revisions.

## Frozen candidate-parent menu

1. `aychang/roberta-base-imdb`, revision
   `cb6bcadd0540b61c9623bd6295d51ac445ceb135`, card accuracy 94.668%;
2. `wrmurray/roberta-base-finetuned-imdb`, revision
   `7aa8ca3fae56a1860d8b4c6bf727b91370821ad5`, card accuracy 95.52%;
3. `dfurman/deberta-v3-base-imdb`, revision
   `cbfcf7b54b2fb47d13b75f3c3a517f95808a2285`, card accuracy 95.77%.

No additional candidate can be added.  All use identity labels, max length 256,
and hard-label argmax.

## Frozen challengers and geometry

For each candidate parent, the other three roots are the top three legacy roots
by the mean of the already disclosed Step 100 search/verification accuracy:

1. `textattack/albert-base-v2-imdb`;
2. `lvwerra/distilbert-imdb`;
3. `textattack/distilbert-base-uncased-imdb`.

This deterministic rule excludes only the lowest legacy TinyBERT.  Existing
Step 100 prediction arrays are reused; no sealed outcome is accessed.

On both `selector_search` and `selector_verify`, four reference-aware oracle
aliases abstain on every parent-wrong coordinate.  Pool size is 500, budget 10,
temperature .025, with Step 100's respective 400 and 1,500 seed ranges.  A
candidate is eligible only if on both partitions:

1. parent accuracy exceeds every challenger by at least 2.5 points;
2. parent error prevalence is at least 3%;
3. some challenger is correct on at least 15% of parent errors;
4. oracle terminal harm is at least +1 point;
5. path-change rate is at least 50%;
6. parent-to-challenger transitions exceed the reverse; and
7. fixed-query terminal/cumulative effects are exactly zero and root histories
   are bitwise equal.

Select the eligible parent maximizing the smaller search/verification oracle
terminal harm; ties use larger verification harm, larger smaller parent gap,
then the menu order above.  If no parent is eligible, stop without adapter
training or outcome access.  Every candidate result remains reported.

Downloaded candidate weights may be removed after their prediction vector and
revision/hash audit are saved because local disk lacks symlink support.  This is
cache cleanup, not model replacement; the selected revision must be downloaded
again and hash-bound for adapter training and held-out prediction.
