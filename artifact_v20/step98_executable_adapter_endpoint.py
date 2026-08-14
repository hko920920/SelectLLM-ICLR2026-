from __future__ import annotations

from pathlib import Path

import numpy as np

import step98_common as common


def execute_aliases_from_raw_pairs(
    premises: list[str],
    hypotheses: list[str],
    parent_predictions: np.ndarray,
    adapter_paths: list[str],
    thresholds: list[float],
) -> np.ndarray:
    if len(premises) != len(hypotheses) or len(premises) != len(parent_predictions):
        raise ValueError("raw pair and parent-prediction lengths differ")
    if len(adapter_paths) != common.N_ALIASES or len(thresholds) != common.N_ALIASES:
        raise ValueError("Step 98 requires four adapters and thresholds")
    parent = common.as_learned_spec(common.ROOT_SPECS[0])
    scores: list[np.ndarray] = []
    for relative in adapter_paths:
        path = common.ROOT / Path(relative)
        model, tokenizer = common.load_adapter_model(parent, path)
        scores.append(common.infer_error_scores(model, tokenizer, premises, hypotheses))
        model.cpu()
    matrix = np.stack(scores, axis=1)
    aliases, _ = common.make_alias_codes(
        np.asarray(parent_predictions, dtype=np.int64), matrix, thresholds
    )
    return aliases.astype(np.int64)
