from __future__ import annotations

from pathlib import Path

import numpy as np

import step100_common as common


def execute_aliases_from_raw_text(
    texts: list[str],
    parent_predictions: np.ndarray,
    adapter_paths: list[str],
    thresholds: list[float],
) -> np.ndarray:
    if len(texts) != len(parent_predictions):
        raise ValueError("text and parent-prediction lengths differ")
    if len(adapter_paths) != common.N_ALIASES or len(thresholds) != common.N_ALIASES:
        raise ValueError("Step 100 requires four adapters and thresholds")
    scores: list[np.ndarray] = []
    for relative in adapter_paths:
        model, tokenizer = common.load_adapter_model(
            common.ROOT_SPECS[0], common.ROOT / Path(relative)
        )
        scores.append(common.infer_error_scores(model, tokenizer, texts))
        model.cpu()
    aliases, _ = common.make_alias_codes(
        np.asarray(parent_predictions, dtype=np.int64),
        np.stack(scores, axis=1),
        thresholds,
    )
    return aliases.astype(np.int64)
