#!/usr/bin/env python3
"""Preregistered-tie, vectorized selector for the SRSA outcome stage.

The original development selector randomized exact ties with a seed-derived
hash. The frozen SRSA protocol instead requires the smallest bound item UID
and the smallest root-manifest digest. This module implements those literal
tie rules and a budget-two vectorization without changing the selector's
posterior, acquisition objective, feedback, or root decision semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from srsa_core import (
    Registry,
    SRSAError,
    complete_root_regret,
    reference_feedback,
    response_keys,
)


@dataclass(frozen=True)
class FrozenRunSummary:
    terminal: np.ndarray
    cumulative: np.ndarray
    queries: np.ndarray
    roots: np.ndarray
    attacker_selected: np.ndarray
    posterior_roots: np.ndarray


def response_equality_tensor(registry: Registry) -> np.ndarray:
    registry.validate()
    keys = response_keys(registry.labels, registry.tags)
    return np.equal(keys[:, :, None], keys[:, None, :])


def acquisition_from_equality(
    equality: np.ndarray,
    posterior: np.ndarray,
) -> np.ndarray:
    equality = np.asarray(equality, dtype=bool)
    posterior = np.asarray(posterior, dtype=np.float64)
    if equality.ndim != 3 or equality.shape[1:] != (
        len(posterior),
        len(posterior),
    ):
        raise SRSAError((equality.shape, posterior.shape))
    return np.einsum(
        "i,rij,j->r",
        posterior,
        equality,
        posterior,
        optimize=True,
        dtype=np.float64,
    )


def choose_query_frozen(
    acquisition: np.ndarray,
    active: np.ndarray,
    pool: np.ndarray,
    uids: Sequence[str],
) -> int:
    """Choose minimum acquisition, then the smallest frozen unique UID."""

    values = np.asarray(acquisition, dtype=np.float64).copy()
    active_array = np.asarray(active, dtype=bool)
    pool_array = np.asarray(pool, dtype=np.int64)
    if values.shape != active_array.shape or values.shape != pool_array.shape:
        raise SRSAError((values.shape, active_array.shape, pool_array.shape))
    values[~active_array] = np.inf
    minimum = float(np.min(values))
    tied = np.flatnonzero(values == minimum)
    if not len(tied):
        raise SRSAError("no active query candidate")
    return min(
        (int(position) for position in tied),
        key=lambda position: (
            str(uids[int(pool_array[position])]),
            int(pool_array[position]),
        ),
    )


def choose_root_frozen(
    tied_entries: Sequence[int] | np.ndarray,
    entry_roots: np.ndarray,
    root_digests: Sequence[str],
) -> int:
    """Choose the smallest root-manifest digest among tied entries."""

    roots = sorted({int(entry_roots[int(entry)]) for entry in tied_entries})
    if not roots:
        raise SRSAError("empty root tie")
    if any(root < 0 or root >= len(root_digests) for root in roots):
        raise SRSAError((roots, len(root_digests)))
    return min(roots, key=lambda root: (str(root_digests[root]), root))


def _softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    shifted = np.asarray(scores, dtype=np.float64) / float(temperature)
    shifted = shifted - float(np.max(shifted))
    posterior = np.exp(shifted)
    posterior /= float(np.sum(posterior))
    return posterior


def _root_mass(
    posterior: np.ndarray,
    entry_roots: np.ndarray,
    root_count: int,
) -> np.ndarray:
    return np.bincount(
        np.asarray(entry_roots, dtype=np.int64),
        weights=np.asarray(posterior, dtype=np.float64),
        minlength=root_count,
    ).astype(np.float64)


def _uid_ranks(uids: Sequence[str]) -> np.ndarray:
    values = np.asarray([str(uid) for uid in uids], dtype="<U128")
    if len(set(values.tolist())) != len(values):
        raise SRSAError("query tie UIDs must be unique")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.int32)
    ranks[order] = np.arange(len(values), dtype=np.int32)
    return ranks


def _choose_positions_batched(
    values: np.ndarray,
    pools: np.ndarray,
    uid_ranks: np.ndarray,
    *,
    excluded_positions: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    pools = np.asarray(pools, dtype=np.int32)
    if values.shape != pools.shape:
        raise SRSAError((values.shape, pools.shape))
    working = values.copy()
    if excluded_positions is not None:
        excluded = np.asarray(excluded_positions, dtype=np.int64)
        if excluded.shape != (len(pools),):
            raise SRSAError((excluded.shape, len(pools)))
        working[np.arange(len(pools)), excluded] = np.inf
    minima = np.min(working, axis=1)
    tied = working == minima[:, None]
    ranks = uid_ranks[pools]
    sentinel = np.iinfo(np.int32).max
    tied_ranks = np.where(tied, ranks, sentinel)
    positions = np.argmin(tied_ranks, axis=1).astype(np.int32)
    if np.any(tied_ranks[np.arange(len(pools)), positions] == sentinel):
        raise SRSAError("batched query selection found no candidate")
    return positions


def _roots_for_score_patterns(
    score_patterns: np.ndarray,
    entry_roots: np.ndarray,
    root_digests: Sequence[str],
) -> np.ndarray:
    patterns = np.asarray(score_patterns, dtype=np.float64)
    if patterns.ndim != 2 or patterns.shape[1] != len(entry_roots):
        raise SRSAError((patterns.shape, np.asarray(entry_roots).shape))
    roots = np.empty(len(patterns), dtype=np.int16)
    for index, scores in enumerate(patterns):
        tied = np.flatnonzero(scores == float(np.max(scores)))
        roots[index] = choose_root_frozen(tied, entry_roots, root_digests)
    return roots


def run_active_frozen(
    registry: Registry,
    clean: Registry,
    references: np.ndarray,
    uids: Sequence[str],
    pools: np.ndarray,
    seeds: Sequence[int],
    root_digests: Sequence[str],
    *,
    budget: int,
    temperature: float,
    attack_root: int | None = None,
) -> FrozenRunSummary:
    """Execute the frozen budget-two selector with exact preregistered ties."""

    registry.validate(len(references))
    clean.validate(len(references))
    references_array = np.asarray(references, dtype=np.int64)
    pools_array = np.asarray(pools, dtype=np.int32)
    if int(budget) != 2:
        raise SRSAError("SRSA frozen vectorization requires budget == 2")
    if temperature <= 0:
        raise SRSAError(temperature)
    if pools_array.ndim != 2 or len(pools_array) != len(seeds):
        raise SRSAError((pools_array.shape, len(seeds)))
    if pools_array.shape[1] < budget:
        raise SRSAError((pools_array.shape, budget))
    if len(uids) != len(references_array):
        raise SRSAError((len(uids), len(references_array)))
    if len(root_digests) != clean.labels.shape[1]:
        raise SRSAError((len(root_digests), clean.labels.shape))
    if len(set(str(value) for value in root_digests)) != len(root_digests):
        raise SRSAError("root-manifest digests must be unique")
    if np.any(pools_array < 0) or np.any(pools_array >= len(references_array)):
        raise SRSAError("pool index out of range")
    if any(len(np.unique(pool)) != len(pool) for pool in pools_array):
        raise SRSAError("pool contains duplicate items")

    feedback = reference_feedback(registry.labels, registry.tags, references_array)
    equality = response_equality_tensor(registry)
    _, root_regrets = complete_root_regret(clean.labels, references_array)
    root_count = clean.labels.shape[1]
    uid_ranks = _uid_ranks(uids)
    runs = len(pools_array)

    posterior0 = np.full(
        registry.labels.shape[1],
        1.0 / registry.labels.shape[1],
        dtype=np.float64,
    )
    acquisition0 = acquisition_from_equality(equality, posterior0)
    position0 = _choose_positions_batched(
        acquisition0[pools_array],
        pools_array,
        uid_ranks,
    )
    query0 = pools_array[np.arange(runs), position0]

    feedback_patterns, feedback_inverse = np.unique(
        feedback,
        axis=0,
        return_inverse=True,
    )
    pattern_acquisition = np.empty(
        (len(feedback_patterns), len(references_array)),
        dtype=np.float64,
    )
    pattern_root_mass = np.empty(
        (len(feedback_patterns), root_count),
        dtype=np.float64,
    )
    for pattern_index, pattern in enumerate(feedback_patterns):
        posterior = _softmax(pattern, temperature)
        pattern_acquisition[pattern_index] = acquisition_from_equality(
            equality,
            posterior,
        )
        pattern_root_mass[pattern_index] = _root_mass(
            posterior,
            registry.roots,
            root_count,
        )

    first_pattern = feedback_inverse[query0]
    acquisition1 = pattern_acquisition[first_pattern[:, None], pools_array]
    position1 = _choose_positions_batched(
        acquisition1,
        pools_array,
        uid_ranks,
        excluded_positions=position0,
    )
    query1 = pools_array[np.arange(runs), position1]

    roots_after_first = _roots_for_score_patterns(
        feedback_patterns,
        registry.roots,
        root_digests,
    )[first_pattern]
    second_scores = feedback[query0] + feedback[query1]
    second_patterns, second_inverse = np.unique(
        second_scores,
        axis=0,
        return_inverse=True,
    )
    roots_after_second = _roots_for_score_patterns(
        second_patterns,
        registry.roots,
        root_digests,
    )[second_inverse]

    queries = np.column_stack([query0, query1]).astype(np.int32)
    roots = np.column_stack([roots_after_first, roots_after_second]).astype(np.int16)
    terminal = root_regrets[roots_after_second].astype(np.float64)
    cumulative = (
        root_regrets[roots_after_first] + root_regrets[roots_after_second]
    ).astype(np.float64)
    posterior_roots = np.empty((runs, 2, root_count), dtype=np.float64)
    posterior_roots[:, 0, :] = _root_mass(
        posterior0,
        registry.roots,
        root_count,
    )
    posterior_roots[:, 1, :] = pattern_root_mass[first_pattern]
    attacker_selected = (
        roots_after_second == int(attack_root)
        if attack_root is not None
        else np.zeros(runs, dtype=bool)
    )
    return FrozenRunSummary(
        terminal=terminal,
        cumulative=cumulative,
        queries=queries,
        roots=roots,
        attacker_selected=attacker_selected,
        posterior_roots=posterior_roots,
    )


def run_fixed_frozen(
    registry: Registry,
    clean: Registry,
    references: np.ndarray,
    fixed_queries: np.ndarray,
    root_digests: Sequence[str],
    *,
    attack_root: int | None = None,
) -> FrozenRunSummary:
    """Execute the frozen root decision on a fixed two-query acquisition path."""

    registry.validate(len(references))
    clean.validate(len(references))
    references_array = np.asarray(references, dtype=np.int64)
    queries = np.asarray(fixed_queries, dtype=np.int32)
    if queries.ndim != 2 or queries.shape[1] != 2:
        raise SRSAError((queries.shape, "budget must equal two"))
    if np.any(queries < 0) or np.any(queries >= len(references_array)):
        raise SRSAError("fixed query index out of range")
    feedback = reference_feedback(registry.labels, registry.tags, references_array)
    _, root_regrets = complete_root_regret(clean.labels, references_array)

    first_scores = feedback[queries[:, 0]]
    first_patterns, first_inverse = np.unique(
        first_scores,
        axis=0,
        return_inverse=True,
    )
    first_roots = _roots_for_score_patterns(
        first_patterns,
        registry.roots,
        root_digests,
    )[first_inverse]
    second_scores = first_scores + feedback[queries[:, 1]]
    second_patterns, second_inverse = np.unique(
        second_scores,
        axis=0,
        return_inverse=True,
    )
    second_roots = _roots_for_score_patterns(
        second_patterns,
        registry.roots,
        root_digests,
    )[second_inverse]

    roots = np.column_stack([first_roots, second_roots]).astype(np.int16)
    terminal = root_regrets[second_roots].astype(np.float64)
    cumulative = (root_regrets[first_roots] + root_regrets[second_roots]).astype(
        np.float64
    )
    root_count = clean.labels.shape[1]
    posterior_roots = np.full(
        (len(queries), 2, root_count),
        np.nan,
        dtype=np.float64,
    )
    attacker_selected = (
        second_roots == int(attack_root)
        if attack_root is not None
        else np.zeros(len(queries), dtype=bool)
    )
    return FrozenRunSummary(
        terminal=terminal,
        cumulative=cumulative,
        queries=queries.copy(),
        roots=roots,
        attacker_selected=attacker_selected,
        posterior_roots=posterior_roots,
    )
