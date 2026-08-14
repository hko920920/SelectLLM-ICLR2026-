#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from stage3_learned_common import (
    array_sha256,
    head_scores,
    make_aliases,
    quality_audit,
    save_head,
    threshold_higher,
    train_head,
)


def main() -> None:
    rng = np.random.default_rng(81234)
    features = rng.normal(size=(1200, 17)).astype(np.float32)
    target = (features[:, 0] + 0.4 * features[:, 1] > 1.0).astype(np.int16)
    assert 25 <= int(target.sum()) <= len(target) - 25

    with tempfile.TemporaryDirectory(prefix="praa-stage3-synthetic-") as tmp:
        path = Path(tmp) / "head.safetensors"
        state, audit = train_head(features, target, seed=315100)
        record = save_head(state, path, audit)
        scores_a = head_scores(features, path)
        scores_b = head_scores(features, path)
        assert np.array_equal(scores_a, scores_b)
        assert record["sha256"]

        target_scores = scores_a[:400]
        threshold = threshold_higher(target_scores, 0.993)
        score_matrix = np.repeat(scores_a[:, None], 4, axis=1)
        thresholds = np.asarray(
            [threshold, threshold + 1e-8, threshold + 2e-8, threshold + 3e-8]
        )
        parent = rng.integers(0, 6, size=len(features), dtype=np.int16)
        references = rng.integers(0, 6, size=len(features), dtype=np.int16)
        aliases, trigger = make_aliases(parent, score_matrix, thresholds, 6)
        audit_quality = quality_audit(parent, aliases, references)
        assert trigger.shape == (len(features), 4)
        assert audit_quality["coordinate_wise_nonimproving"]
        assert len({array_sha256(aliases[:, index]) for index in range(4)}) >= 1

    print("PASS_PRAA_STAGE3_SYNTHETIC")


if __name__ == "__main__":
    main()
