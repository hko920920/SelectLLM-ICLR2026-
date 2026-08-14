from __future__ import annotations

import numpy as np

import srsa_core as core
import srsa_selector_fast as fast
from test_srsa_core import synthetic_fixture


def test_vectorized_acquisition_matches_group_enumeration():
    predictions, _, uids, root_ids, _ = synthetic_fixture(300)
    clean = core.clean_registry(predictions, root_ids)
    refined = core.structured_refinement(clean, uids, 2, root_ids)
    rng = np.random.default_rng(17)
    posterior = rng.random(refined.labels.shape[1])
    posterior /= np.sum(posterior)
    equality = fast.response_equality_tensor(refined)
    observed = fast.acquisition_from_equality(equality[:100], posterior)
    expected = core.acquisition_values(
        core.response_keys(refined.labels[:100], refined.tags[:100]),
        posterior,
    )
    assert np.allclose(observed, expected, rtol=0, atol=1e-15)


def test_fast_active_and_fixed_match_reference_implementation():
    predictions, references, uids, root_ids, root_digests = synthetic_fixture(500)
    clean = core.clean_registry(predictions, root_ids)
    refined = core.structured_refinement(clean, uids, 1, root_ids)
    seeds = tuple(range(8100, 8160))
    pools = core.sample_pools(len(references), seeds, 200)

    slow = core.run_active(
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
    quick = fast.run_active_fast(
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
    assert np.array_equal(quick.queries, slow.queries)
    assert np.array_equal(quick.roots, slow.roots)
    assert np.array_equal(quick.terminal, slow.terminal)
    assert np.array_equal(quick.cumulative, slow.cumulative)
    assert np.array_equal(quick.attacker_selected, slow.attacker_selected)

    slow_fixed = core.run_fixed(
        refined,
        clean,
        references,
        slow.queries,
        seeds,
        root_digests,
        attack_root=1,
    )
    quick_fixed = fast.run_fixed_fast(
        refined,
        clean,
        references,
        slow.queries,
        seeds,
        root_digests,
        attack_root=1,
    )
    assert np.array_equal(quick_fixed.queries, slow_fixed.queries)
    assert np.array_equal(quick_fixed.roots, slow_fixed.roots)
    assert np.array_equal(quick_fixed.terminal, slow_fixed.terminal)
    assert np.array_equal(quick_fixed.cumulative, slow_fixed.cumulative)
    assert np.array_equal(quick_fixed.attacker_selected, slow_fixed.attacker_selected)
