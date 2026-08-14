# Anonymous validation-closure artifact (V11 / Step 88)

This archive supports **Candidate Lists Are Priors: Registry-Refinement Attacks
on Active Model Selection**.

V11 adds a result-blind audit of the official LLM Selector implementation at
commit `15faa47dab102d7a92124b13c1838b55524fc2bf`. Before any outcome was
printed, the protocol retained all six released datasets and mechanically fixed
one parent per dataset. Four aliases copy that parent's complete strong-judge
utility row coordinate-wise, while a public hash rule changes only the parent's
own weak-judge score view. The intervention reads no peer candidate, strong
outcome, sampled pool, acquisition path, or selector state.

AlpacaEval and Bingo pass the frozen six-part terminal-harm gate, yielding
`GO_STRONG_CROSS_SELECTOR` (2/6). Arena-Hard improves, Flickr30k misses the
corrected gate, and MEDIQA/MT-Bench are terminal-null; every outcome is retained.
All 6,000 ordered paths change, every fixed-query terminal and cumulative effect
is exactly zero, and every alias remains exactly strong-utility-equivalent to its
parent. This is a controlled pairwise decision-view refinement over released
score matrices, not an executable raw-response endpoint or production-registry
attack.

Offline byte-integrity and anonymity check:

```text
python validate_step88_anonymous_release.py --integrity-only
```

Full replay (network access is needed for the locked public dependencies):

```text
python fetch_step68_public_dependencies.py
python validate_step88_anonymous_release.py --full
```

Full mode first runs the inherited V10 closure, then independently reconstructs
the Step 88 aliases, all stored trajectories and statistics, the six-dataset BH
adjustment and promotion decision. It also replays 300 result-independent paths
using the literal official entropy equation, runs 85,934 invariant checks, checks
the manuscript's numbers and negative-result retention, rebuilds the paper, and
enforces the nine-page main-text boundary. The validator imports no code from the
Step 88 runner.

The official LLM Selector repository and its six source score matrices are not
embedded. The inherited dependency fetcher obtains the exact public commit; the
Step 88 validator binds every source matrix by SHA-256. The historical manifest
filename `STEP82_RELEASE_MANIFEST.json` is retained for backward compatibility;
its V11 refresh block and `step88` bindings identify this release.
