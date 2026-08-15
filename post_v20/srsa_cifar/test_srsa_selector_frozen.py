from __future__ import annotations

import numpy as np

import srsa_core as core
import srsa_selector_frozen as frozen
from test_srsa_core import synthetic_fixture


def test_query_tie_uses_smallest_bound_uid():
    acquisition = np.asarray([0.2, 0.2, 0.2], dtype=np.float64)
    active = np.asarray([True, True, True])
    pool = np.asarray([2, 0, 1], dtype=np.int32)
    uids = ["b", "c", "a"]
    position = frozen.choose_query_frozen(acquisition, active, pool, uids)
    assert position == 0


def test_root_tie_uses_smallest_manifest_digest():
    entry_roots = np.asarray([0, 1, 2, 1], dtype=np.int16)
    digests = ["ff", "01", "80"]
    root = frozen.choose_root_frozen(
        np.asarray([0, 1, 2, 3]),
        entry_roots,
        digests,
    )
    assert root == 1


def reference_active(
    registry: core.Registry,
    clean: core.Registry,
    references: np.ndarray,
    uids: list[str],
    pools: np.ndarray,
    root_digests: list[str],
    temperature: float,
):
    feedback = core.reference_feedback(registry.labels, registry.tags, references)
    keys = core.response_keys(registry.labels, registry.tags)
    _, regrets = core.complete_root_regret(clean.labels, references)
    queries = np.full((len(pools), 2), -1, dtype=np.int32)
    roots = np.full((len(pools), 2), -1, dtype=np.int16)
    terminal = np.zeros(len(pools), dtype=np.float64)
    cumulative = np.zeros(len(pools), dtype=np.float64)
    posterior_roots = np.zeros(
        (len(pools), 2, clean.labels.shape[1]),
        dtype=np.float64,
    )
    for run, pool in enumerate(pools):
        active = np.ones(len(pool), dtype=bool)
        scores = np.zeros(registry.labels.shape[1], dtype=np.float64)
        for step in range(2):
            shifted = scores / temperature
            shifted -= np.max(shifted)
            posterior = np.exp(shifted)
            posterior /= np.sum(posterior)
            posterior_roots[run, step] = np.bincount(
                registry.roots.astype(np.int64),
                weights=posterior,
                minlength=clean.labels.shape[1],
            )
            acquisition = core.acquisition_values(keys[pool], posterior)
            position = frozen.choose_query_frozen(
                acquisition,
                active,
                pool,
                uids,
            )
            active[position] = False
            query = int(pool[position])
            queries[run, step] = query
            scores += feedback[query]
            tied = np.flatnonzero(scores == np.max(scores))
            root = frozen.choose_root_frozen(
                tied,
                registry.roots,
                root_digests,
            )
            roots[run, step] = root
            cumulative[run] += regrets[root]
            if step == 1:
                terminal[run] = regrets[root]
    return queries, roots, terminal, cumulative, posterior_roots


def test_vectorized_budget_two_matches_literal_reference():
    predictions, references, image_keys, root_ids, root_digests = synthetic_fixture(500)
    clean = core.clean_registry(predictions, root_ids)
    refined = core.structured_refinement(clean, image_keys, 2, root_ids)
    seeds = tuple(range(9010, 9070))
    pools = core.sample_pools(len(references), seeds, 200)
    expected = reference_active(
        refined,
        clean,
        references,
        image_keys,
        pools,
        root_digests,
        0.05,
    )
    observed = frozen.run_active_frozen(
        refined,
        clean,
        references,
        image_keys,
        pools,
        seeds,
        root_digests,
        budget=2,
        temperature=0.05,
        attack_root=2,
    )
    assert np.array_equal(observed.queries, expected[0])
    assert np.array_equal(observed.roots, expected[1])
    assert np.array_equal(observed.terminal, expected[2])
    assert np.array_equal(observed.cumulative, expected[3])
    assert np.array_equal(observed.posterior_roots, expected[4])


def test_fixed_query_structured_effect_is_exactly_zero():
    predictions, references, image_keys, root_ids, root_digests = synthetic_fixture(600)
    clean = core.clean_registry(predictions, root_ids)
    refined = core.structured_refinement(clean, image_keys, 3, root_ids)
    seeds = tuple(range(9200, 9270))
    pools = core.sample_pools(len(references), seeds, 200)
    clean_active = frozen.run_active_frozen(
        clean,
        clean,
        references,
        image_keys,
        pools,
        seeds,
        root_digests,
        budget=2,
        temperature=0.05,
    )
    clean_fixed = frozen.run_fixed_frozen(
        clean,
        clean,
        references,
        clean_active.queries,
        root_digests,
    )
    refined_fixed = frozen.run_fixed_frozen(
        refined,
        clean,
        references,
        clean_active.queries,
        root_digests,
        attack_root=3,
    )
    assert np.array_equal(clean_fixed.roots, refined_fixed.roots)
    assert np.array_equal(clean_fixed.terminal, refined_fixed.terminal)
    assert np.array_equal(clean_fixed.cumulative, refined_fixed.cumulative)
