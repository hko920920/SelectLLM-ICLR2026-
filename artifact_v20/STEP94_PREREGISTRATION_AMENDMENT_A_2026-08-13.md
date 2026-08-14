# Step 94 preregistration Amendment A: optimizer and replay details

Locked: `2026-08-13T12:10:48.7564501+09:00`

Status at amendment lock: the Step-94 test splits and selector outcomes remain
unmaterialized; no Stage-0, Stage-A, or confirmatory output exists.  This
amendment fills deterministic training details omitted from the authority
document.  It does not change the task menu, partitions, search space,
selection rule, endpoint, or success gates.

## Root training

- AdamW, learning rate `1e-3`, weight decay `1e-4`.
- 20 epochs, batch size 256, global gradient norm clipped to 1.0.
- Roots 0--12 use the deterministically hash-ordered prefix specified by their
  fraction.
- Full-data roots 13--15 use an independently salted size-preserving bootstrap
  of the root-training partition.
- Epoch orders are deterministic SHA-256-derived permutations.
- GELU is the exact PyTorch functional GELU with default approximation.

## Error-gate training

- AdamW, learning rate `2e-3`, weight decay `1e-4`.
- 30 epochs, batch size 256, global gradient norm clipped to 1.0.
- Each gate uses an independently salted size-preserving bootstrap of the
  gate-training partition.
- Binary cross-entropy with logits uses positive weight
  `n_parent_correct / n_parent_wrong` computed within that bootstrap.
- Parent logits concatenated to CLS are float32; saved inference scores are
  sigmoid probabilities evaluated in float64 downstream.

## Numerical and deterministic details

- PyTorch, NumPy, and Python RNGs are seeded; cuDNN benchmark is disabled and
  deterministic mode is enabled.
- Root and gate checkpoint tensors are stored with safetensors.
- Root inference logits are stored as float64; response codes, references, and
  parent maps are int64.
- Exact ties in root-bank accuracy, threshold scores, acquisition, and root
  choice follow the deterministic rules in the authority document and code;
  code-source hashes are bound before confirmation.
- The same frozen encoder representation may be cached within a stage, but no
  test representation is computed during Stage A.
