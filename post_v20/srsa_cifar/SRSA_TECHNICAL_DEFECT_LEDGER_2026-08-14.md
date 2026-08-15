# SRSA technical defect ledger

Initial date: 2026-08-14 (Asia/Seoul)
Last updated: 2026-08-15 (Asia/Seoul)

Every item below was identified before the single Stage-4 outcome job opened any CIFAR label for selection, inference, primary evaluation, or deployment evaluation. Repairs are restricted to execution fidelity and information-boundary enforcement. They do not change a task, checkpoint roster, alias count, canary rate, pool size, budget, temperature, seed range, estimand, multiplicity rule, materiality threshold, or closure rule.

## Defect 1 — Stage 0 missing NumPy dependency

- workflow: `SRSA Stage 0 public metadata binding`
- run: `31792412480`
- head: `2d95021685ce5e4847fa2d066e768f132f53cf91`
- failing step: `Bind dataset archives, checkpoints, source, and zero-input runtimes`
- exception: `ModuleNotFoundError: No module named 'numpy'`
- elapsed within scientific executable: approximately two seconds

### Information boundary at failure

- CIFAR archives were not downloaded by the executable;
- no archive was extracted;
- no dataset row was decoded;
- no label was accessed;
- no checkpoint was downloaded or evaluated on CIFAR input;
- no CIFAR prediction, selector path, primary outcome, or deployment outcome was accessed.

### Authorized repair

Add the already-standard project runtime dependency `numpy==1.26.4` to the Stage 0 installation step. No task, source commit, archive URL, checkpoint URL, roster, response schema, canary rate, selector setting, seed, estimand, gate, or information boundary changes.

The exact technical rerun completed successfully as run `31792763410` at head `10e3b28a639db648cc9c66afa1c8dd8913778447`. Its immutable artifact is `srsa-stage0-public-metadata-receipt`, artifact ID `9217245752`, with archive digest `sha256:cc1215e8172002085a607819cc6787b5dfc6ab669261a8e13d3b0703921866a0`.

## Defect 2 — Successful Stage 0 receipt was not materialized as the Stage 1 lock

The successful Stage 0 execution uploaded its receipt as a GitHub Actions artifact, while the prewritten Stage 1 executable expected the repository path `post_v20/srsa_cifar/SRSA_STAGE0_FINAL_LOCK_2026-08-14.json`. Consequently, Stage 1 could not start from the successful receipt without an explicit artifact-to-lock handoff.

### Information boundary at discovery

- Stage 0 had downloaded public archives and checkpoints and had performed zero-input model construction checks only;
- no CIFAR archive had been extracted by Stage 0;
- no CIFAR row had been decoded by Stage 0;
- no label, CIFAR prediction, selector path, primary outcome, or deployment outcome had been accessed;
- Stage 1, Stage 2, Stage 3, and Stage 4 had not executed.

### Authorized repair

The one-shot workflow imports exactly run `31792763410`, artifact ID `9217245752`, and the frozen artifact digest above. It adds only authoritative execution provenance, writes the final Stage 0 lock, and commits that lock before Stage 1. It does not rerun Stage 0, select among multiple Stage 0 outcomes, or modify any scientific setting.

Each later stage similarly writes and commits a hash-bound final lock before the next stage. Stage 3 is committed before the sealed-label artifact is downloaded by the single Stage-4 job.

## Defect 3 — Development selector tie implementation did not match the frozen protocol wording

The initial outcome-agnostic core used seed-derived hashes to break exact query and root ties. The frozen SRSA protocol and pre-decode clarification instead specify:

- query tie: smallest frozen unique item UID;
- root tie: smallest frozen root-manifest digest.

The difference is decision-relevant in principle even though both rules are deterministic.

### Information boundary at discovery

- no SRSA selector had been executed on CIFAR inputs;
- no CIFAR candidate-quality statistic had been computed;
- no sealed label had been used for selector execution, primary inference, or deployment evaluation;
- no primary or deployment outcome had been opened;
- no tie rule was chosen after seeing a CIFAR selection path or effect.

### Authorized repair

A separate `srsa_selector_frozen.py` implements the literal preregistered rules. It preserves the frozen posterior, typed compatibility, acquisition objective, feedback, root mapping, budget, temperature, pools, and seeds. The implementation is checked against a literal reference loop and has explicit tests for both tie rules, fixed-query zero effect, and vectorized equivalence.

Stage 3 binds the SHA-256 of this selector and of the Stage-4 executable before outcome access. Stage 4 refuses to run if either file changes after the pre-outcome lock.

## Defect 4 — Raw-input canary key required an explicit implementation binding

The mathematical protocol defines each structured-response canary from canonical raw RGB bytes. The reusable helper accepts a generic string key, whose variable name alone does not establish whether the caller supplied a unique item ID or the raw-image identity.

### Information boundary at discovery

- no CIFAR structured wrapper had been constructed;
- no realized CIFAR canary fraction had been inspected;
- no selector or outcome had been executed.

### Authorized repair

Stage 3 supplies `image_sha256 = SHA256(canonical raw RGB bytes)` as the canary key and separately retains the archive-position-aware unique UID only for item identity and query tie resolution. Stage 3 materializes and binds the full alias-tag array before labels or selector outcomes are available. No salt, numerator, denominator, alias count, or canary gate changes.

## One-shot execution rule after these repairs

The repaired path is a single ordered workflow:

1. import and commit the exact successful Stage 0 receipt;
2. create public inputs and the physically separate label seal;
3. execute all eight real clean endpoints without labels;
4. bind all wrappers, pools, code hashes, and final gates in Stage 3;
5. only if Stage 3 passes, download the sealed labels in one Stage-4 job;
6. execute and report all four attacks on both fixed tasks;
7. commit the literal `GO_SRSA_PRIMARY` or `NO_GO_SRSA_PRIMARY` result;
8. introduce no third task and perform no result-dependent repair.

A pre-outcome scientific `STOP` and a valid primary `NO_GO` are successful workflow completions. GitHub Actions failure is reserved for a technical `INVALID` condition.
