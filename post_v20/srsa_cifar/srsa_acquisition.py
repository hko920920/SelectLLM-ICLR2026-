#!/usr/bin/env python3
"""Numerically stable acquisition arithmetic for SRSA selectors.

The acquisition objective is the sum of squared posterior masses of response
equivalence classes. Algebraically identical response partitions must receive
bitwise identical acquisition values so that the frozen UID tie rule is not
bypassed by BLAS or summation-order noise. This module canonicalizes every
partition by membership bitmasks, uses ``math.fsum`` within and across groups,
and evaluates each unique partition once per posterior.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


class StableAcquisitionError(RuntimeError):
    pass


@dataclass(frozen=True)
class PartitionTable:
    entries: int
    rows: int
    unique_signatures: tuple[tuple[int, ...], ...]
    inverse: np.ndarray


def _validate_posterior(posterior: np.ndarray, entries: int) -> np.ndarray:
    value = np.asarray(posterior, dtype=np.float64)
    if value.shape != (entries,):
        raise StableAcquisitionError((value.shape, entries))
    if not np.all(np.isfinite(value)) or np.any(value < 0.0):
        raise StableAcquisitionError("posterior is non-finite or negative")
    total = math.fsum(float(item) for item in value)
    if not math.isfinite(total) or total <= 0.0:
        raise StableAcquisitionError(total)
    return value


def canonical_partition_signatures(equality: np.ndarray) -> list[tuple[int, ...]]:
    """Return one canonical tuple of group bitmasks per row."""

    relation = np.asarray(equality, dtype=bool)
    if relation.ndim != 3 or relation.shape[1] != relation.shape[2]:
        raise StableAcquisitionError(relation.shape)
    rows, entries, _ = relation.shape
    if entries <= 0 or entries > 63:
        raise StableAcquisitionError(entries)
    if not np.all(np.diagonal(relation, axis1=1, axis2=2)):
        raise StableAcquisitionError("response relation is not reflexive")
    if not np.array_equal(relation, np.swapaxes(relation, 1, 2)):
        raise StableAcquisitionError("response relation is not symmetric")

    bit_weights = np.left_shift(
        np.uint64(1),
        np.arange(entries, dtype=np.uint64),
    )
    row_masks = np.sum(
        relation.astype(np.uint64) * bit_weights[None, None, :],
        axis=2,
        dtype=np.uint64,
    )
    signatures: list[tuple[int, ...]] = []
    full_mask = (1 << entries) - 1
    for row in range(rows):
        groups = tuple(sorted({int(mask) for mask in row_masks[row]}))
        union = 0
        for mask in groups:
            if mask <= 0 or union & mask:
                raise StableAcquisitionError(
                    {"row": row, "overlapping_or_empty_group": mask}
                )
            members = [entry for entry in range(entries) if mask & (1 << entry)]
            for member in members:
                if int(row_masks[row, member]) != mask:
                    raise StableAcquisitionError(
                        {"row": row, "non_transitive_group": mask}
                    )
            union |= mask
        if union != full_mask:
            raise StableAcquisitionError(
                {"row": row, "partition_union": union, "expected": full_mask}
            )
        signatures.append(groups)
    return signatures


def partition_table_from_equality(equality: np.ndarray) -> PartitionTable:
    relation = np.asarray(equality, dtype=bool)
    if relation.ndim != 3 or relation.shape[1] != relation.shape[2]:
        raise StableAcquisitionError(relation.shape)
    signatures = canonical_partition_signatures(relation)
    unique_signatures = tuple(sorted(set(signatures)))
    mapping = {
        signature: index for index, signature in enumerate(unique_signatures)
    }
    inverse = np.asarray(
        [mapping[signature] for signature in signatures],
        dtype=np.int32,
    )
    inverse.setflags(write=False)
    return PartitionTable(
        entries=relation.shape[1],
        rows=relation.shape[0],
        unique_signatures=unique_signatures,
        inverse=inverse,
    )


def stable_acquisition_from_table(
    table: PartitionTable,
    posterior: np.ndarray,
) -> np.ndarray:
    """Evaluate one cached partition table under one posterior."""

    posterior_value = _validate_posterior(posterior, table.entries)
    unique_values = np.empty(len(table.unique_signatures), dtype=np.float64)
    for index, signature in enumerate(table.unique_signatures):
        squared_masses: list[float] = []
        for mask in signature:
            mass = math.fsum(
                float(posterior_value[entry])
                for entry in range(table.entries)
                if mask & (1 << entry)
            )
            squared_masses.append(mass * mass)
        unique_values[index] = math.fsum(sorted(squared_masses))
    values = unique_values[table.inverse]
    if values.shape != (table.rows,):
        raise StableAcquisitionError((values.shape, table.rows))
    return values


def stable_acquisition_from_equality(
    equality: np.ndarray,
    posterior: np.ndarray,
) -> np.ndarray:
    """Evaluate sum-of-squared group mass with canonical stable arithmetic."""

    return stable_acquisition_from_table(
        partition_table_from_equality(equality),
        posterior,
    )


def stable_acquisition_from_keys(
    keys: np.ndarray,
    posterior: np.ndarray,
) -> np.ndarray:
    """Key-based wrapper using exactly the same canonical relation path."""

    key_array = np.asarray(keys)
    if key_array.ndim != 2:
        raise StableAcquisitionError(key_array.shape)
    equality = np.equal(key_array[:, :, None], key_array[:, None, :])
    return stable_acquisition_from_equality(equality, posterior)
