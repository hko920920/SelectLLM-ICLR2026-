#!/usr/bin/env python3
"""Vectorized SRSA selector execution for the final paired outcome stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from srsa_core import (
    Registry,
    SRSAError,
    choose_query,
    choose_root,
    complete_root_regret,
    reference_feedback,
    response_keys,
)


@dataclass(frozen=True)
class FastRunSummary:
    terminal: np.ndarray
    cumulative: np.ndarray
    queries: np.ndarray
    roots: np.ndarray
    attacker_selected: np.ndarray


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


def run_active_fast(
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
) -> FastRunSummary:
    registry.validate(len(references))
    clean.validate(len(references))
    pools = np.asarray(pools, dtype=np.int32)
    references = np.asarray(references, dtype=np.int64)
    if pools.ndim != 2 or len(pools) != len(seeds):
        raise SRSAError((pools.shape, len(seeds)))
    if pools.shape[1] < budget:
        raise SRSAError((pools.shape, budget))
    if len(root_digests) != clean.labels.shape[1]:
        raise SRSAError((len(root_digests), clean.labels.shape))
    if budget <= 0 or temperature <= 0:
        raise SRSAError((budget, temperature))

    feedback = reference_feedback(registry.labels, registry.tags, references)
    _, root_regrets = complete_root_regret(clean.labels, references)
    equality = response_equality_tensor(registry)
    runs = len(seeds)
    terminal = np.zeros(runs, dtype=np.float64)
    cumulative = np.zeros(runs, dtype=np.float64)
    queries = np.full((runs, budget), -1, dtype=np.int32)
    roots = np.full((runs, budget), -1, dtype=np.int16)

    for run_index, seed in enumerate(seeds):
        pool = pools[run_index]
        if np.any(pool < 0) or np.any(pool >= len(references)):
            raise SRSAError(f"pool index out of range at run {run_index}")
        if len(np.unique(pool)) != len(pool):
            raise SRSAError(f"pool contains duplicates at run {run_index}")
        active = np.ones(len(pool), dtype=bool)
        scores = np.zeros(registry.labels.shape[1], dtype=np.float64)
        pool_equality = equality[pool]
        for step in range(budget):
            shifted = scores / float(temperature)
            shifted -= float(np.max(shifted))
            posterior = np.exp(shifted)
            posterior /= float(np.sum(posterior))
            acquisition = acquisition_from_equality(pool_equality, posterior)
            position = choose_query(
                acquisition,
                active,
                pool,
                uids,
                int(seed),
                step,
            )
            active[position] = False
            query = int(pool[position])
            queries[run_index, step] = query
            scores += feedback[query]
            tied_entries = np.flatnonzero(scores == float(np.max(scores)))
            root = choose_root(
                tied_entries,
                registry.roots,
                root_digests,
                int(seed),
                step,
            )
            roots[run_index, step] = root
            regret = float(root_regrets[root])
            cumulative[run_index] += regret
            if step == budget - 1:
                terminal[run_index] = regret

    attacker_selected = (
        roots[:, -1] == int(attack_root)
        if attack_root is not None
        else np.zeros(runs, dtype=bool)
    )
    return FastRunSummary(
        terminal=terminal,
        cumulative=cumulative,
        queries=queries,
        roots=roots,
        attacker_selected=attacker_selected,
    )


def run_fixed_fast(
    registry: Registry,
    clean: Registry,
    references: np.ndarray,
    fixed_queries: np.ndarray,
    seeds: Sequence[int],
    root_digests: Sequence[str],
    *,
    attack_root: int | None = None,
) -> FastRunSummary:
    registry.validate(len(references))
    clean.validate(len(references))
    references = np.asarray(references, dtype=np.int64)
    queries = np.asarray(fixed_queries, dtype=np.int32)
    if queries.ndim != 2 or len(queries) != len(seeds):
        raise SRSAError((queries.shape, len(seeds)))
    feedback = reference_feedback(registry.labels, registry.tags, references)
    _, root_regrets = complete_root_regret(clean.labels, references)
    runs, budget = queries.shape
    terminal = np.zeros(runs, dtype=np.float64)
    cumulative = np.zeros(runs, dtype=np.float64)
    roots = np.full((runs, budget), -1, dtype=np.int16)

    for run_index, seed in enumerate(seeds):
        scores = np.zeros(registry.labels.shape[1], dtype=np.float64)
        for step, query in enumerate(queries[run_index]):
            if not (0 <= int(query) < len(references)):
                raise SRSAError((run_index, step, int(query)))
            scores += feedback[int(query)]
            tied_entries = np.flatnonzero(scores == float(np.max(scores)))
            root = choose_root(
                tied_entries,
                registry.roots,
                root_digests,
                int(seed),
                step,
            )
            roots[run_index, step] = root
            regret = float(root_regrets[root])
            cumulative[run_index] += regret
            if step == budget - 1:
                terminal[run_index] = regret

    attacker_selected = (
        roots[:, -1] == int(attack_root)
        if attack_root is not None
        else np.zeros(runs, dtype=bool)
    )
    return FastRunSummary(
        terminal=terminal,
        cumulative=cumulative,
        queries=queries.copy(),
        roots=roots,
        attacker_selected=attacker_selected,
    )
