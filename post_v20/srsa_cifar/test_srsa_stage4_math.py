from __future__ import annotations

import numpy as np

import srsa_stage4_outcome as outcome


def test_discrete_bootstrap_preserves_constant_sample():
    values = np.full(3000, 0.0125, dtype=np.float64)
    means = outcome.bootstrap_means_discrete(
        values,
        seed=17,
        repetitions=200,
    )
    assert means.shape == (200,)
    assert np.array_equal(means, np.full(200, 0.0125))


def test_discrete_signflip_all_zero_is_one():
    values = np.zeros(3000, dtype=np.float64)
    assert outcome.signflip_pvalue_discrete(
        values,
        seed=19,
        repetitions=1000,
    ) == 1.0


def test_discrete_signflip_detects_strictly_positive_constant():
    values = np.full(3000, 0.01, dtype=np.float64)
    pvalue = outcome.signflip_pvalue_discrete(
        values,
        seed=23,
        repetitions=10000,
    )
    assert 0.0 < pvalue <= 1.0 / 10001.0


def test_nll_per_root_matches_manual_softmax_loss():
    logits = np.asarray(
        [
            [[2.0, 0.0], [0.0, 2.0]],
            [[0.0, 2.0], [2.0, 0.0]],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 1], dtype=np.int64)
    observed = outcome.nll_per_root(logits, labels)
    good = np.log1p(np.exp(-2.0))
    bad = np.log1p(np.exp(2.0))
    assert np.allclose(observed, np.asarray([good, bad]), rtol=0, atol=1e-15)


def test_bootstrap_interval_is_ordered_and_contains_mean():
    values = np.asarray([0.0] * 2400 + [0.02] * 600, dtype=np.float64)
    interval = outcome.bootstrap_interval_discrete(
        values,
        seed=29,
        repetitions=2000,
    )
    mean = float(np.mean(values))
    assert interval[0] <= mean <= interval[1]
