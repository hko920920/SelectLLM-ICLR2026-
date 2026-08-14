#!/usr/bin/env python3
"""Core mechanics for the Structured-Response Sybil Admission Audit (SRSA).

This module is deliberately outcome-agnostic. It implements the frozen typed
response compatibility, deterministic raw-input canaries, entry-indexed
Select-LLM specialization, fixed-query mediation control, root-aware
canonicalization, and familywise summaries.

Candidate responses never use the wildcard tag. Human references always do.
Therefore candidate grouping remains an equivalence relation while the exact
same compatibility function gives every semantic clone the parent's feedback.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

WILDCARD_TAG = -1
BASE_TAG = 0
DEFAULT_ALIAS_COUNT = 4
DEFAULT_CANARY_DENOMINATOR = 20
DEFAULT_CANARY_NUMERATOR = 1
DEFAULT_CANARY_SALT = "srsa-canary-v1"


class SRSAError(RuntimeError):
    """Raised for an invalid SRSA binding or execution."""


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_u64(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def raw_image_uid(raw_rgb_bytes: bytes) -> str:
    if not isinstance(raw_rgb_bytes, (bytes, bytearray, memoryview)):
        raise TypeError(type(raw_rgb_bytes))
    return hashlib.sha256(bytes(raw_rgb_bytes)).hexdigest()


def partition_order_key(raw_rgb_bytes: bytes, archive_position: int) -> tuple[str, int]:
    digest = hashlib.sha256(
        b"srsa-cifar-partition-v1" + bytes(raw_rgb_bytes)
    ).hexdigest()
    return digest, int(archive_position)


def canary_membership(
    uid: str,
    alias_index: int,
    *,
    numerator: int = DEFAULT_CANARY_NUMERATOR,
    denominator: int = DEFAULT_CANARY_DENOMINATOR,
    salt: str = DEFAULT_CANARY_SALT,
) -> bool:
    if not (0 < numerator <= denominator):
        raise SRSAError((numerator, denominator))
    if alias_index < 0:
        raise SRSAError(alias_index)
    digest = hashlib.sha256(
        f"{salt}|{alias_index}|{uid}".encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:8], "big")
    return (value % denominator) < numerator


def make_alias_tags(
    uids: Sequence[str],
    *,
    alias_count: int = DEFAULT_ALIAS_COUNT,
    numerator: int = DEFAULT_CANARY_NUMERATOR,
    denominator: int = DEFAULT_CANARY_DENOMINATOR,
    salt: str = DEFAULT_CANARY_SALT,
) -> np.ndarray:
    if alias_count <= 0:
        raise SRSAError(alias_count)
    tags = np.zeros((len(uids), alias_count), dtype=np.int16)
    for row, uid in enumerate(uids):
        for alias in range(alias_count):
            if canary_membership(
                str(uid),
                alias,
                numerator=numerator,
                denominator=denominator,
                salt=salt,
            ):
                tags[row, alias] = alias + 1
    return tags


def compatible(
    left_labels: np.ndarray,
    left_tags: np.ndarray,
    right_labels: np.ndarray,
    right_tags: np.ndarray,
) -> np.ndarray:
    """The one frozen typed similarity used by grouping and feedback."""

    left_labels = np.asarray(left_labels)
    left_tags = np.asarray(left_tags)
    right_labels = np.asarray(right_labels)
    right_tags = np.asarray(right_tags)
    semantic = np.equal(left_labels, right_labels)
    variant = (
        np.equal(left_tags, right_tags)
        | np.equal(left_tags, WILDCARD_TAG)
        | np.equal(right_tags, WILDCARD_TAG)
    )
    return semantic & variant


def reference_feedback(
    candidate_labels: np.ndarray,
    candidate_tags: np.ndarray,
    references: np.ndarray,
) -> np.ndarray:
    candidate_labels = np.asarray(candidate_labels, dtype=np.int64)
    candidate_tags = np.asarray(candidate_tags, dtype=np.int16)
    references = np.asarray(references, dtype=np.int64)
    if candidate_labels.shape != candidate_tags.shape:
        raise SRSAError((candidate_labels.shape, candidate_tags.shape))
    if candidate_labels.ndim != 2 or len(references) != len(candidate_labels):
        raise SRSAError((candidate_labels.shape, references.shape))
    reference_labels = references[:, None]
    reference_tags = np.full((len(references), 1), WILDCARD_TAG, dtype=np.int16)
    return compatible(
        candidate_labels,
        candidate_tags,
        reference_labels,
        reference_tags,
    ).astype(np.float64)


def response_keys(labels: np.ndarray, tags: np.ndarray) -> np.ndarray:
    """Canonical candidate grouping keys.

    Candidate tags must be non-wildcard. On that restricted domain the frozen
    compatibility relation is exactly equality of (label, tag), so these keys
    are an efficient representation of the same relation.
    """

    labels = np.asarray(labels, dtype=np.int64)
    tags = np.asarray(tags, dtype=np.int16)
    if labels.shape != tags.shape or labels.ndim != 2:
        raise SRSAError((labels.shape, tags.shape))
    if np.any(tags == WILDCARD_TAG) or np.any(tags < BASE_TAG):
        raise SRSAError("candidate response contains a wildcard or invalid tag")
    stride = int(np.max(tags, initial=0)) + 1
    return labels * stride + tags.astype(np.int64)


@dataclass(frozen=True)
class Registry:
    labels: np.ndarray
    tags: np.ndarray
    roots: np.ndarray
    entry_ids: tuple[str, ...]

    def validate(self, expected_rows: int | None = None) -> None:
        labels = np.asarray(self.labels)
        tags = np.asarray(self.tags)
        roots = np.asarray(self.roots)
        if labels.shape != tags.shape or labels.ndim != 2:
            raise SRSAError((labels.shape, tags.shape))
        if expected_rows is not None and len(labels) != int(expected_rows):
            raise SRSAError((len(labels), expected_rows))
        if labels.shape[1] != len(roots) or labels.shape[1] != len(self.entry_ids):
            raise SRSAError((labels.shape, roots.shape, len(self.entry_ids)))
        if np.any(roots < 0):
            raise SRSAError("negative root id")
        if len(set(self.entry_ids)) != len(self.entry_ids):
            raise SRSAError("duplicate entry id")
        response_keys(labels, tags)


def clean_registry(core_predictions: np.ndarray, root_ids: Sequence[str]) -> Registry:
    predictions = np.asarray(core_predictions, dtype=np.int64)
    if predictions.ndim != 2:
        raise SRSAError(predictions.shape)
    if predictions.shape[1] != len(root_ids):
        raise SRSAError((predictions.shape, len(root_ids)))
    registry = Registry(
        labels=predictions.copy(),
        tags=np.zeros_like(predictions, dtype=np.int16),
        roots=np.arange(predictions.shape[1], dtype=np.int16),
        entry_ids=tuple(f"clean::{root}" for root in root_ids),
    )
    registry.validate()
    return registry


def structured_refinement(
    clean: Registry,
    uids: Sequence[str],
    attack_root: int,
    root_ids: Sequence[str],
    *,
    alias_count: int = DEFAULT_ALIAS_COUNT,
    numerator: int = DEFAULT_CANARY_NUMERATOR,
    denominator: int = DEFAULT_CANARY_DENOMINATOR,
    salt: str = DEFAULT_CANARY_SALT,
) -> Registry:
    clean.validate(len(uids))
    if not (0 <= attack_root < clean.labels.shape[1]):
        raise SRSAError(attack_root)
    if len(root_ids) != clean.labels.shape[1]:
        raise SRSAError((len(root_ids), clean.labels.shape))
    alias_labels = np.repeat(
        clean.labels[:, attack_root : attack_root + 1],
        alias_count,
        axis=1,
    )
    alias_tags = make_alias_tags(
        uids,
        alias_count=alias_count,
        numerator=numerator,
        denominator=denominator,
        salt=salt,
    )
    registry = Registry(
        labels=np.concatenate([clean.labels, alias_labels], axis=1),
        tags=np.concatenate([clean.tags, alias_tags], axis=1),
        roots=np.concatenate(
            [
                clean.roots,
                np.full(alias_count, attack_root, dtype=np.int16),
            ]
        ),
        entry_ids=clean.entry_ids
        + tuple(
            f"structured::{root_ids[attack_root]}::v{index + 1}"
            for index in range(alias_count)
        ),
    )
    registry.validate(len(uids))
    return registry


def exact_refinement(
    clean: Registry,
    attack_root: int,
    root_ids: Sequence[str],
    *,
    alias_count: int = DEFAULT_ALIAS_COUNT,
) -> Registry:
    clean.validate()
    if not (0 <= attack_root < clean.labels.shape[1]):
        raise SRSAError(attack_root)
    alias_labels = np.repeat(
        clean.labels[:, attack_root : attack_root + 1],
        alias_count,
        axis=1,
    )
    alias_tags = np.zeros_like(alias_labels, dtype=np.int16)
    registry = Registry(
        labels=np.concatenate([clean.labels, alias_labels], axis=1),
        tags=np.concatenate([clean.tags, alias_tags], axis=1),
        roots=np.concatenate(
            [clean.roots, np.full(alias_count, attack_root, dtype=np.int16)]
        ),
        entry_ids=clean.entry_ids
        + tuple(
            f"exact::{root_ids[attack_root]}::{index + 1}"
            for index in range(alias_count)
        ),
    )
    registry.validate()
    return registry


def semantic_alias_equality(clean: Registry, refined: Registry) -> bool:
    clean.validate(len(refined.labels))
    refined.validate(len(clean.labels))
    for entry in range(clean.labels.shape[1], refined.labels.shape[1]):
        root = int(refined.roots[entry])
        if not np.array_equal(refined.labels[:, entry], clean.labels[:, root]):
            return False
    return True


def sample_pools(
    population_size: int,
    seeds: Sequence[int],
    pool_size: int,
) -> np.ndarray:
    if not (0 < pool_size <= population_size):
        raise SRSAError((pool_size, population_size))
    return np.asarray(
        [
            sorted(
                random.Random(int(seed)).sample(
                    range(population_size),
                    pool_size,
                )
            )
            for seed in seeds
        ],
        dtype=np.int32,
    )


def _query_tie_key(uid: str, seed: int, step: int) -> tuple[int, str]:
    return stable_u64("srsa", "query", seed, step, uid), uid


def _root_tie_key(root_digest: str, seed: int, step: int) -> tuple[int, str]:
    return stable_u64("srsa", "root", seed, step, root_digest), root_digest


def choose_query(
    acquisition: np.ndarray,
    active: np.ndarray,
    pool: np.ndarray,
    uids: Sequence[str],
    seed: int,
    step: int,
) -> int:
    value = np.asarray(acquisition, dtype=np.float64).copy()
    value[~np.asarray(active, dtype=bool)] = np.inf
    minimum = float(np.min(value))
    tied = np.flatnonzero(value == minimum)
    if not len(tied):
        raise SRSAError("no query tie candidate")
    return min(
        (int(position) for position in tied),
        key=lambda position: _query_tie_key(
            str(uids[int(pool[position])]),
            int(seed),
            int(step),
        ),
    )


def choose_root(
    tied_entries: Iterable[int],
    entry_roots: np.ndarray,
    root_digests: Sequence[str],
    seed: int,
    step: int,
) -> int:
    roots = sorted({int(entry_roots[int(entry)]) for entry in tied_entries})
    if not roots:
        raise SRSAError("empty root tie")
    return min(
        roots,
        key=lambda root: _root_tie_key(
            str(root_digests[root]),
            int(seed),
            int(step),
        ),
    )


def acquisition_values(keys: np.ndarray, posterior: np.ndarray) -> np.ndarray:
    keys = np.asarray(keys, dtype=np.int64)
    posterior = np.asarray(posterior, dtype=np.float64)
    if keys.ndim != 2 or keys.shape[1] != len(posterior):
        raise SRSAError((keys.shape, posterior.shape))
    values = np.zeros(len(keys), dtype=np.float64)
    for row, row_keys in enumerate(keys):
        _, inverse = np.unique(row_keys, return_inverse=True)
        mass = np.bincount(inverse, weights=posterior)
        values[row] = float(np.sum(mass * mass))
    return values


def complete_root_regret(
    core_predictions: np.ndarray,
    references: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.asarray(core_predictions, dtype=np.int64)
    references = np.asarray(references, dtype=np.int64)
    if predictions.ndim != 2 or len(predictions) != len(references):
        raise SRSAError((predictions.shape, references.shape))
    accuracies = np.mean(predictions == references[:, None], axis=0)
    losses = 1.0 - accuracies
    regrets = losses - float(np.min(losses))
    return accuracies.astype(np.float64), regrets.astype(np.float64)


@dataclass(frozen=True)
class RunSummary:
    terminal: np.ndarray
    cumulative: np.ndarray
    queries: np.ndarray
    roots: np.ndarray
    attacker_selected: np.ndarray


def run_active(
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
) -> RunSummary:
    registry.validate(len(references))
    clean.validate(len(references))
    if len(root_digests) != clean.labels.shape[1]:
        raise SRSAError((len(root_digests), clean.labels.shape))
    if len(pools) != len(seeds):
        raise SRSAError((pools.shape, len(seeds)))
    if budget <= 0 or temperature <= 0:
        raise SRSAError((budget, temperature))

    feedback = reference_feedback(registry.labels, registry.tags, references)
    _, root_regrets = complete_root_regret(clean.labels, references)
    all_keys = response_keys(registry.labels, registry.tags)
    runs = len(seeds)
    terminal = np.zeros(runs, dtype=np.float64)
    cumulative = np.zeros(runs, dtype=np.float64)
    queries = np.full((runs, budget), -1, dtype=np.int32)
    roots = np.full((runs, budget), -1, dtype=np.int16)

    for run_index, seed in enumerate(seeds):
        pool = np.asarray(pools[run_index], dtype=np.int32)
        active = np.ones(len(pool), dtype=bool)
        scores = np.zeros(registry.labels.shape[1], dtype=np.float64)
        pool_keys = all_keys[pool]
        for step in range(budget):
            shifted = scores / float(temperature)
            shifted -= float(np.max(shifted))
            posterior = np.exp(shifted)
            posterior /= float(np.sum(posterior))
            acquisition = acquisition_values(pool_keys, posterior)
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
    return RunSummary(
        terminal=terminal,
        cumulative=cumulative,
        queries=queries,
        roots=roots,
        attacker_selected=attacker_selected,
    )


def run_fixed(
    registry: Registry,
    clean: Registry,
    references: np.ndarray,
    fixed_queries: np.ndarray,
    seeds: Sequence[int],
    root_digests: Sequence[str],
    *,
    attack_root: int | None = None,
) -> RunSummary:
    registry.validate(len(references))
    clean.validate(len(references))
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
    return RunSummary(
        terminal=terminal,
        cumulative=cumulative,
        queries=queries.copy(),
        roots=roots,
        attacker_selected=attacker_selected,
    )


def canonicalize_root_aware(
    refined: Registry,
    clean: Registry,
) -> Registry:
    """Authenticated fixed-root-mass admission.

    Every admitted wrapper maps to an existing clean root and contributes no
    independent mass. The canonical selector input is therefore exactly the
    clean registry.
    """

    refined.validate(len(clean.labels))
    clean.validate(len(refined.labels))
    if not semantic_alias_equality(clean, refined):
        raise SRSAError("wrapper semantic prediction differs from parent")
    if set(int(root) for root in refined.roots) != set(range(clean.labels.shape[1])):
        raise SRSAError("root mapping is incomplete")
    return clean_registry(clean.labels, [entry.split("::", 1)[-1] for entry in clean.entry_ids])


def bootstrap_means(
    values: np.ndarray,
    *,
    seed: int,
    repetitions: int,
    batch: int = 500,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise SRSAError(values.shape)
    rng = np.random.default_rng(seed)
    output: list[np.ndarray] = []
    remaining = int(repetitions)
    while remaining > 0:
        count = min(batch, remaining)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        output.append(np.mean(values[indices], axis=1))
        remaining -= count
    return np.concatenate(output)


def bonferroni_lower_bound(
    values: np.ndarray,
    *,
    seed: int,
    repetitions: int,
    family_size: int,
    alpha: float = 0.05,
) -> float:
    if family_size <= 0 or not (0 < alpha < 1):
        raise SRSAError((family_size, alpha))
    means = bootstrap_means(values, seed=seed, repetitions=repetitions)
    return float(np.quantile(means, alpha / family_size))


def signflip_pvalue(
    values: np.ndarray,
    *,
    seed: int,
    repetitions: int,
    batch: int = 500,
) -> float:
    values = np.asarray(values, dtype=np.float64)
    observed = float(np.mean(values))
    rng = np.random.default_rng(seed)
    exceed = 0
    remaining = int(repetitions)
    while remaining > 0:
        count = min(batch, remaining)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(count, len(values)))
        exceed += int(np.sum(np.mean(signs * values[None, :], axis=1) >= observed))
        remaining -= count
    return float((exceed + 1) / (repetitions + 1))


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    items = sorted(
        ((str(key), float(value)) for key, value in pvalues.items()),
        key=lambda item: (item[1], item[0]),
    )
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (key, value) in enumerate(items):
        candidate = min(1.0, (m - rank) * value)
        running = max(running, candidate)
        adjusted[key] = running
    return adjusted


def digest_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()
