# SRSA pre-decode clarification 1: input identity and serialization

Date: 2026-08-14 (Asia/Seoul)
Status: frozen before any CIFAR archive is extracted or any CIFAR row is decoded.

This clarification changes no task, roster, endpoint, alias count, canary rate,
selector parameter, seed, estimand, gate, or outcome-access rule. It makes the
raw-input identity convention explicit before Stage 1.

## Canonical raw image

An official CIFAR test image is serialized as a C-contiguous `uint8` array of
shape `(32, 32, 3)` in RGB channel order. `raw_RGB_bytes` denotes exactly the
3,072 bytes of this canonical array.

## Digests and unique input identifier

For archive position `i` in task `d`:

- `image_sha256 = SHA256(raw_RGB_bytes)`;
- `partition_digest = SHA256("srsa-cifar-partition-v1" || raw_RGB_bytes)`;
- `uid = SHA256("srsa-cifar-uid-v1" || d || image_sha256 || uint64_be(i))`.

Rows are ordered by `(partition_digest, archive_position)`. The archive
position is a deterministic tie-break only. This preserves the preregistered
input-only partition rule even if an official test archive contains exact
image duplicates.

## Canary and selector tie separation

The structured-response canary remains a function of the raw input only:

`canary_j(x) = 1[uint64(SHA256("srsa-canary-v1" || j || raw_RGB_bytes)) mod 20 = 0]`.

Its implementation may equivalently hash the already bound
`image_sha256`. It must not use `uid`, because `uid` contains the archive
position used only for uniqueness.

Query ties use the smallest unique `uid`. Root ties use the smallest frozen
root-manifest digest. Neither tie rule uses a label, prediction quality,
selector outcome, or registry input order.

## Public and sealed artifacts

The public Stage-1 artifact contains canonical images, `image_sha256`, unique
`uid`, archive position, and partition marker. It contains no task label.

A physically separate sealed artifact contains only ordered `uid`, partition
marker, and task label. Stages 2 and 3 may download only the public artifact.
The sealed artifact is authorized only for the single final outcome stage.
