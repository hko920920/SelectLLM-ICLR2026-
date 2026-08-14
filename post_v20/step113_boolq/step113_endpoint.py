from __future__ import annotations

from pathlib import Path

import numpy as np

import step113_common as common


def execute_aliases_from_raw_pairs(
    questions: list[str],
    passages: list[str],
    head_paths: tuple[str, ...] | list[str],
    thresholds: np.ndarray,
    *,
    local_files_only: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Execute the frozen parent and four error heads from raw BoolQ pairs only."""
    if len(head_paths) != common.N_ALIASES:
        raise ValueError("four learned error heads are required")
    parent, logits, features, audit = common.infer_candidate_pairs(
        common.PARENT,
        questions,
        passages,
        return_features=True,
        local_files_only=local_files_only,
    )
    if features is None:
        raise AssertionError("parent feature extraction failed")
    scores = np.column_stack(
        [common.error_head_scores(features, common.ROOT / Path(path)) for path in head_paths]
    )
    aliases, triggers = common.make_aliases(parent, scores, np.asarray(thresholds))
    return parent, aliases, triggers, scores, {
        "parent": audit,
        "logits_sha256": common.array_sha256(logits),
        "features_sha256": common.array_sha256(features),
        "scores_sha256": common.array_sha256(scores),
        "aliases_sha256": common.array_sha256(aliases),
    }
