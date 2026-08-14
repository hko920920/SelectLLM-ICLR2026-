# SRSA technical defect ledger

Date: 2026-08-14 (Asia/Seoul)

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

The next execution is an exact technical rerun under the same scientific protocol.
