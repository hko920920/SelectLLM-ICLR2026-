from __future__ import annotations

import numpy as np

import srsa_core as core


def synthetic_fixture(rows: int = 600):
    rng = np.random.default_rng(20260814)
    references = rng.integers(0, 5, size=rows, dtype=np.int64)
    predictions = np.empty((rows, 4), dtype=np.int64)
    predictions[:, 0] = references
    for root, error_rate in enumerate((0.12, 0.16, 0.20, 0.24)):
        predictions[:, root] = references
        mask = rng.random(rows) < error_rate
        predictions[mask, root] = (
            predictions[mask, root] + root + 1
        ) % 5
    raw = [
        int(index).to_bytes(8, "big")
        + bytes([index % 251, (index * 7) % 251])
        for index in range(rows)
    ]
    uids = [core.raw_image_uid(value) for value in raw]
    root_ids = ["r0", "r1", "r2", "r3"]
    root_digests = [core.canonical_sha256({"root": root}) for root in root_ids]
    return predictions, references, uids, root_ids, root_digests


def test_canary_is_deterministic_and_rate_bounded():
    _, _, uids, _, _ = synthetic_fixture(4000)
    first = core.make_alias_tags(uids)
    second = core.make_alias_tags(uids)
    assert np.array_equal(first, second)
    rates = np.mean(first > 0, axis=0)
    assert np.all(rates > 0.035)
    assert np.all(rates < 0.065)
    for alias in range(first.shape[1]):
        assert set(np.unique(first[:, alias])).issubset({0, alias + 1})


def test_structured_wrappers_are_semantically_identical():
    predictions, _, uids, root_ids, _ = synthetic_fixture()
    clean = core.clean_registry(predictions, root_ids)
    for attack_root in range(4):
        refined = core.structured_refinement(
            clean, uids, attack_root, root_ids
        )
        assert core.semantic_alias_equality(clean, refined)
        for entry in range(4, 8):
            assert np.array_equal(
                refined.labels[:, entry],
                clean.labels[:, attack_root],
            )


def test_single_compatibility_gives_identical_feedback():
    predictions, references, uids, root_ids, _ = synthetic_fixture()
    clean = core.clean_registry(predictions, root_ids)
    refined = core.structured_refinement(clean, uids, 2, root_ids)
    feedback = core.reference_feedback(
        refined.labels, refined.tags, references
    )
    for alias in range(4):
        assert np.array_equal(feedback[:, 4 + alias], feedback[:, 2])


def test_candidate_grouping_changes_only_on_canaries():
    predictions, _, uids, root_ids, _ = synthetic_fixture()
    clean = core.clean_registry(predictions, root_ids)
    refined = core.structured_refinement(clean, uids, 1, root_ids)
    keys = core.response_keys(refined.labels, refined.tags)
    canary = np.any(refined.tags[:, 4:] > 0, axis=1)
    assert np.any(canary)
    assert np.any(~canary)
    for row in np.flatnonzero(~canary)[:20]:
        assert len(set(keys[row, [1, 4, 5, 6, 7]].tolist())) == 1
    for row in np.flatnonzero(canary)[:20]:
        assert len(set(keys[row, [1, 4, 5, 6, 7]].tolist())) > 1


def test_fixed_query_effect_is_exactly_zero():
    predictions, references, uids, root_ids, root_digests = synthetic_fixture()
    clean = core.clean_registry(predictions, root_ids)
    refined = core.structured_refinement(clean, uids, 3, root_ids)
    seeds = tuple(range(5000, 5100))
    pools = core.sample_pools(len(references), seeds, 200)
    clean_active = core.run_active(
        clean,
        clean,
        references,
        uids,
        pools,
        seeds,
        root_digests,
        budget=2,
        temperature=0.05,
    )
    fixed = core.run_fixed(
        refined,
        clean,
        references,
        clean_active.queries,
        seeds,
        root_digests,
        attack_root=3,
    )
    assert np.array_equal(fixed.roots, clean_active.roots)
    assert np.array_equal(fixed.terminal, clean_active.terminal)
    assert np.array_equal(fixed.cumulative, clean_active.cumulative)


def test_root_aware_canonicalization_is_bitwise_clean():
    predictions, references, uids, root_ids, root_digests = synthetic_fixture()
    clean = core.clean_registry(predictions, root_ids)
    refined = core.structured_refinement(clean, uids, 0, root_ids)
    canonical = core.canonicalize_root_aware(refined, clean)
    assert np.array_equal(canonical.labels, clean.labels)
    assert np.array_equal(canonical.tags, clean.tags)
    assert np.array_equal(canonical.roots, clean.roots)

    seeds = tuple(range(6000, 6050))
    pools = core.sample_pools(len(references), seeds, 200)
    left = core.run_active(
        clean,
        clean,
        references,
        uids,
        pools,
        seeds,
        root_digests,
        budget=2,
        temperature=0.05,
    )
    right = core.run_active(
        canonical,
        clean,
        references,
        uids,
        pools,
        seeds,
        root_digests,
        budget=2,
        temperature=0.05,
    )
    assert core.digest_array(left.queries) == core.digest_array(right.queries)
    assert core.digest_array(left.roots) == core.digest_array(right.roots)
    assert core.digest_array(left.terminal) == core.digest_array(right.terminal)
    assert core.digest_array(left.cumulative) == core.digest_array(right.cumulative)


def test_structured_refinement_can_change_active_path():
    predictions, references, uids, root_ids, root_digests = synthetic_fixture()
    clean = core.clean_registry(predictions, root_ids)
    refined = core.structured_refinement(clean, uids, 3, root_ids)
    seeds = tuple(range(7000, 7200))
    pools = core.sample_pools(len(references), seeds, 200)
    clean_active = core.run_active(
        clean,
        clean,
        references,
        uids,
        pools,
        seeds,
        root_digests,
        budget=2,
        temperature=0.05,
    )
    attacked = core.run_active(
        refined,
        clean,
        references,
        uids,
        pools,
        seeds,
        root_digests,
        budget=2,
        temperature=0.05,
        attack_root=3,
    )
    assert np.mean(np.any(clean_active.queries != attacked.queries, axis=1)) > 0.10


def test_holm_adjust_is_monotone_and_bounded():
    adjusted = core.holm_adjust({"a": 0.001, "b": 0.02, "c": 0.03, "d": 0.8})
    assert set(adjusted) == {"a", "b", "c", "d"}
    assert all(0.0 <= value <= 1.0 for value in adjusted.values())
    values = [adjusted[key] for key in ("a", "b", "c", "d")]
    assert values == sorted(values)
