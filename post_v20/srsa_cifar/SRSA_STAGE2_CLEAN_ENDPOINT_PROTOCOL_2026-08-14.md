# SRSA Stage 2: clean endpoint execution and registry binding

Date: 2026-08-14 (Asia/Seoul)
Status: frozen before any CIFAR endpoint is evaluated on a decoded CIFAR row.

## Information boundary

Stage 2 may read only the Stage-1 public raw-input artifact and the Stage-0
source/checkpoint lock. It may not download the Stage-1 sealed-label artifact,
read any label, construct a structured wrapper, execute a selector, or compute
candidate accuracy, regret, or deployment loss.

## Fixed roots

For each task, execute the four roots in the frozen configuration order:

1. `resnet20`
2. `vgg11_bn`
3. `mobilenetv2_x0_5`
4. `shufflenetv2_x0_5`

The exact source commit, checkpoint URL and SHA-256, parameter count,
normalization, model class, and runtime package versions must replay Stage 0.
No root is selected, removed, calibrated, or ranked.

## Input and preprocessing

Every endpoint consumes the same ordered Stage-1 public `uint8` RGB array.
Input is converted to float32 in `[0,1]`, transposed to NCHW, and normalized by
the task-specific frozen channel mean and standard deviation. No augmentation,
resize, crop, stochastic transform, test-time adaptation, or calibration is
allowed.

## Outputs

Each endpoint writes, in public UID order:

- hard prediction in the frozen class index space;
- raw float32 logits;
- partition marker;
- exact UID vector;
- source, checkpoint, code, package, and array hashes.

The complete logits are retained for the prespecified deployment NLL endpoint.
They are generated before labels are available.

## Stage decision

`PASS_SRSA_STAGE2_CLEAN_ENDPOINTS` requires:

1. all eight endpoint jobs replay every Stage-0 and Stage-1 binding;
2. every checkpoint loads without a missing or unexpected tensor;
3. every endpoint emits finite logits of the exact expected shape;
4. UID and partition vectors are bitwise identical across all endpoints of a
   task;
5. all four checkpoint hashes are distinct within each task;
6. all four hard-prediction vectors are pairwise distinct within each task;
7. no label or selector quantity is accessed.

A pairwise-identical response vector is a scientific pre-outcome `STOP`, not a
technical failure. Other binding or runtime defects are `INVALID` and may be
repaired only without changing a scientific setting.
