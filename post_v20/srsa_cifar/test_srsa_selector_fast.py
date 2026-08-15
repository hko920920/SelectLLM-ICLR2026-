from __future__ import annotations

import numpy as np

import srsa_core as core
import srsa_selector_fast as fast
from srsa_acquisition import stable_acquisition_from_keys
from test_srsa_core import synthetic_fixture


def test_vectorized_acquisition_matches_stable_group_enumeration():
    predictions, _, uids, root_ids, _ = synthetic_fixture(300)
    clean = core.clean_registry(predictions, root_ids)
    refined = core.structured_refinement(clean, uids, 2, root_ids)
    rng = np.random.default_rng(17)
    posterior = rng.random(refined.labels.shape[1])
    posterior /= np.sum(posterior)
    equality = fast.response_equality_tensor(refined)
    observed = fast.acquisition_from_equality(equality[:100], posterior)
    expected = stable_acquisition_from_keys(
        core.response_keys(refined.labels[:100], refined.tags[:100]),
        posterior,
    )
    assert np.array_equal(observed, expected)


def literal_active(
    registry: core.Registry,
    clean: core.Registry,
    references: np.ndarray,
    uids: list[str],
    pools: np.ndarray,
    seeds: tuple[int, ...],
    root_digests: list[str],
    *,
    budget: int,
    temperature: float,
    attack_root: int,
):
    feedback = core.reference_feedback(registry.labels, registry.tags, references)
    keys = core.response_keys(registry.labels, registry.tags)
    _, regrets = core.complete_root_regret(clean.labels, references)
    terminal = np.zeros(len(seeds), dtype=np.float64)
    cumulative = np.zeros(len(seeds), dtype=np.float64)
    queries = np.full((len(seeds), budget), -1, dtype=np.int32)
    roots = np.full((len(seeds), budget), -1, dtype=np.int16)
    for run, seed in enumerate(seeds):
        pool = pools[run]
        active = np.ones(len(pool), dtype=bool)
        scores = np.zeros(registry.labels.shape[1], dtype=np.float64)
        for step in range(budget):
            shifted = scores / temperature
            shifted -= np.max(shifted)
            posterior = np.exp(shifted)
            posterior /= np.sum(posterior)
            acquisition = stable_acquisition_from_keys(keys[pool], posterior)
            position = core.choose_query(
                acquisition,
                active,
                pool,
                uids,
                seed,
                step,
            )
            active[position] = False
            query = int(pool[position])
            queries[run, step] = query
            scores += feedback[query]
            tied = np.flatnonzero(scores == np.max(scores))
            root = core.choose_root(
                tied,
                registry.roots,
                root_digests,
                seed,
                step,
            )
            roots[run, step] = root
            cumulative[run] += regrets[root]
            if step == budget - 1:
                terminal[run] = regrets[root]
    return {
        "terminal": terminal,
        "cumulative": cumulative,
        "queries": queries,
        "roots": roots,
        "attacker_selected": roots[:, -1] == attack_root,
    }


def literal_fixed(
    registry: core.Registry,
    clean: core.Registry,
    references: np.ndarray,
    queries: np.ndarray,
    seeds: tuple[int, ...],
    root_digests: list[str],
    *,
    attack_root: int,
):
    feedback = core.reference_feedback(registry.labels, registry.tags, references)
    _, regrets = core.complete_root_regret(clean.labels, references)
    roots = np.full(queries.shape, -1, dtype=np.int16)
    terminal = np.zeros(len(seeds), dtype=np.float64)
    cumulative = np.zeros(len(seeds), dtype=np.float64)
    for run, seed in enumerate(seeds):
        scores = np.zeros(registry.labels.shape[1], dtype=np.float64)
        for step, query in enumerate(queries[run]):
            scores += feedback[int(query)]
            tied = np.flatnonzero(scores == np.max(scores))
            root = core.choose_root(
                tied,
                registry.roots,
                root_digests,
                seed,
                step,
            )
            roots[run, step] = root
            cumulative[run] += regrets[root]
            if step == queries.shape[1] - 1:
                terminal[run] = regrets[root]
    return {
        "terminal": terminal,
        "cumulative": cumulative,
        "queries": queries.copy(),
        "roots": roots,
        "attacker_selected": roots[:, -1] == attack_root,
    }


def test_fast_active_and_fixed_match_stable_literal_implementation():
    predictions, references, uids, root_ids, root_digests = synthetic_fixture(500)
    clean = core.clean_registry(predictions, root_ids)
    refined = core.structured_refinement(clean, uids, 1, root_ids)
    seeds = tuple(range(8100, 8160))
    pools = core.sample_pools(len(references), seeds, 200)

    expected = literal_active(
        refined,
        clean,
        references,
        uids,
        pools,
        seeds,
        root_digests,
        budget=2,
        temperature=0.05,
        attack_root=1,
    )
    observed = fast.run_active_fast(
        refined,
        clean,
        references,
        uids,
        pools,
        seeds,
        root_digests,
        budget=2,
        temperature=0.05,
        attack_root=1,
    )
    for field in expected:
        assert np.array_equal(getattr(observed, field), expected[field])

    expected_fixed = literal_fixed(
        refined,
        clean,
        references,
        expected["queries"],
        seeds,
        root_digests,
        attack_root=1,
    )
    observed_fixed = fast.run_fixed_fast(
        refined,
        clean,
        references,
        expected["queries"],
        seeds,
        root_digests,
        attack_root=1,
    )
    for field in expected_fixed:
        assert np.array_equal(getattr(observed_fixed, field), expected_fixed[field])
