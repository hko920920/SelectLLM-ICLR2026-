# SelectLLM research continuation worklog

Date: 2026-08-14 (Asia/Seoul)

## Authority and scope

- Repository baseline: `7fb093f9ffc31611f963c68211f2cc5ea073a5dc`.
- Current staging branch: `research/step115-direct-deployment`.
- This branch is an infrastructure and design staging area. Its name does **not** by itself authorize or preregister a Step 115 scientific outcome.
- `docs/CURRENT_STATE.md` remains the scientific handoff authority.
- The Step 114 Amazon outcome remains unopened and scientifically unauthorized after the recorded development stop.
- Historical locks, ledgers, arrays, receipts, and V20 artifact bytes are append-only and must not be overwritten.

## Notion synchronization status

The designated project page is:

`https://app.notion.com/p/Select-LLM-ICLR-2026-3bce7b0a7d38808286e5c9f94225568f`

Direct connector access currently returns `object_not_found`. No unrelated Notion page is an authorized substitute. This GitHub worklog preserves the continuation record until the designated page is shared with the connected Notion integration; once accessible, the same record should be copied beneath that page.

## Runtime investigation

A GitHub Actions probe was added at:

`.github/workflows/step115-runtime-probe.yml`

Observed results:

1. A clean GitHub-hosted Ubuntu runner can check out the repository and materialize Git LFS objects. The probed Step 92 root checkpoint is 402,696 bytes after materialization, rather than a 131-byte LFS pointer.
2. Outbound PyPI access works.
3. DNS resolution and download from `huggingface.co` work.
4. The pinned DistilBERT safetensors file at revision `12040accade4e8a0f71eabdb258fecc2e7e948be` downloads successfully, has 267,954,768 bytes, and matches the recorded SHA-256 `5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063`.

Therefore the assistant's restricted local container is not a project-level blocker. Networked reproducible execution can proceed through GitHub Actions without asking the maintainer to manually download public model weights.

## Clean-checkout V20 audit observation

The authoritative Step 105 integrity validator fails on a clean Git checkout at one manifest entry:

`paper_draft/__pycache__/validate_paper_tables.cpython-311.pyc`

The V20 manifest records this file as 75,786 bytes with SHA-256 `616b9d17fc37afbcbb190472f30c356f9a58b46daf1b531bb7d8fedef263faf5`, while `.gitignore` excludes both `__pycache__/` and `*.pyc`. The original validated V20 ZIP is not duplicated in Git, so the byte is not recoverable from the public repository alone.

This is classified as a Git-extraction/reproducibility gap, not as evidence that the original local V20 ZIP failed its recorded validation. The historical manifest and artifact are not modified. It does not block LFS, public dependency retrieval, or new experiment execution.

## Scientific decision boundary

The remaining score-changing uncertainty is still the one stated in the handoff and review records: a genuinely untouched task and registry, with a fully frozen source-faithful pipeline, must produce material terminal and active-minus-fixed harm with exact fixed-query zero in a single sealed outcome. A future study must be scientifically independent of the stopped Step 114 family and must account for the accumulated Step 105, Step 113, and Step 114 outcomes.

No new primary experiment has been authorized or opened in this continuation session yet. The next step is to complete an outcome-free design audit before creating a new lock.
