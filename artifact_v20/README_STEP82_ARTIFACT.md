# Anonymous validation-closure artifact (V8)

This artifact supports the manuscript **Candidate Lists Are Priors:
Registry-Refinement Attacks on Active Model Selection**.  It is a validation
closure over frozen evidence, not a claim to regenerate every exploratory run.

## What is new in V8

- the exact-condition V2 transfer preregistration and execution locks;
- the frozen Step 80 result and raw paired arrays;
- the runner-independent Step 80 validator (2,882 checks);
- the canonical-order, 16-permutation, and complete 3-by-3 tie-policy audits;
- the current nine-page manuscript and evidence-to-text validator.

The historical Step 54 backend remains bundled and labeled as historical.  The
paper's headline numbers come from Step 80.

## Validation modes

Offline integrity and anonymity:

```text
python validate_step82_anonymous_release.py --integrity-only
```

Full validation (network is needed once for locked public dependencies):

```text
python fetch_step68_public_dependencies.py
python validate_step82_anonymous_release.py --full
```

Full mode verifies the fetched HELM files and upstream repositories, reruns the
ten prior result validators, rebuilds the paper, runs the current manuscript
integration checker, and independently reconstructs the Step 80 aliases,
acquisition paths, statistics, and adjudication.  The independent Step 80 replay
takes several minutes on a typical CPU.

## Scope

Public benchmark data and third-party repositories are fetch-only and are not
stored in the upload archive.  One historical V2 source file had a workstation
root replaced by `Path(__file__).resolve().parent`; the old and portable hashes
and the exact one-line transformation are recorded in
`STEP82_PATH_REDACTION_MAP.json`.  Six earlier path-only rewrites remain bound by
the inherited Step 68 redaction map.

The artifact verifies frozen results and manuscript claims.  It does not claim
clinical deployment evidence, universal vulnerability across active selectors,
or operational generation of the exact-quality public-reference wrappers.
