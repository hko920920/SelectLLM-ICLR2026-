# PRAA Stage-2 clean safety and parent-selection receipt

Decision: `PASS_PRAA_STAGE2_CLEAN_SAFETY_AND_PARENT`

## Execution and independent validation

- authoritative workflow: [Run 31776356942](https://github.com/hko920920/SelectLLM-ICLR2026-/actions/runs/31776356942)
- authoritative ledger artifact ID: `9210581881`
- artifact SHA-256: `899fcfe0b17795ab9076d7b6c9eeab75d6f13f83f1858ecd734e0de89ca2ba88`
- ledger JSON file SHA-256: `46a6733c6f4aed396fb3b91d2e79f27af857bf280acbec401453bbb6348f128a`
- canonical ledger SHA-256 recorded inside the ledger: `bee163bbaa1bef46e1c0c1d6a88a473cf2d27be22361c8975d036b299bb57d86`
- independent reconstruction workflow: [Run 31778119196](https://github.com/hko920920/SelectLLM-ICLR2026-/actions/runs/31778119196)
- independent artifact ID: `9210656926`
- independent artifact SHA-256: `479ec837fc7976ca67ab860933b1b0b4bfea369f74478cb5afeda152518518fb`
- independent JSON file SHA-256: `5489d599f7fcb16b4ccd8629419e1e204c85fc21b9ec8c9cd0265a194a6e33d1`
- independent result SHA-256: `649d507bbc5ef097c9d45aa017dbbb27b5712d8f30bca143789ce88d709669a9`

The independent program did not import either Stage-2 execution program. It
reconstructed the exact safety labels from the pinned public datasets,
reproduced the Stage-1 label seals, loaded all eight endpoint vectors,
recomputed pairwise distinctness and exact correct counts, and re-applied the
frozen best-set/tie rule.

## Emotion primary task

Safety rows: `4,000`

| Root | Correct | Accuracy |
|---|---:|---:|
| `emotion-roberta-dk409` | 3,865 | 96.625% |
| `emotion-bert-nateraw` | 3,887 | 97.175% |
| `emotion-distilbert-bhadresh` | 3,962 | **99.050%** |
| `emotion-albert-bhadresh` | 3,805 | 95.125% |

- best-root set: `{emotion-distilbert-bhadresh}`
- selected alias parent: `emotion-distilbert-bhadresh`
- selected-parent prediction SHA-256:
  `d7c4f3f57c4004037c414e1447ec82b02e67f308b21219aad45d1eb14bff64bf`
- every clean safety response vector is pairwise distinct.

## Language-identification replication

Safety rows: `10,000`

| Root | Correct | Accuracy |
|---|---:|---:|
| `lid-xlmroberta-papluca` | 9,979 | **99.790%** |
| `lid-fasttext-facebook` | 9,717 | 97.170% |
| `lid-fasttext-glotlid` | 9,800 | 98.000% |
| `lid-langid-package` | 9,538 | 95.380% |

- best-root set: `{lid-xlmroberta-papluca}`
- selected alias parent: `lid-xlmroberta-papluca`
- selected-parent prediction SHA-256:
  `4a5a3e154a7b5be03b471e61b0d55d9ce70645441e858f034dbeba25132f0dfa`
- every clean safety response vector is pairwise distinct.

## Information boundary

Stage 2 opened only the two frozen safety-label blocks. It did not execute an
alias or error head, run a selector, open a non-safety label, or access any
primary/deployment input, label, prediction, path, or effect.

The parent-selection rule did not require a unique best root. In fact both
frozen tasks produced a singleton best set, so the predeclared SHA-256 tie rule
was not needed but was independently reconstructed.

The next authorized stage is the already preregistered four-head construction
on `error_head_train`, `safety`, and label-free `target_threshold` only.
