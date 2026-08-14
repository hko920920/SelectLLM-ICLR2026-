from __future__ import annotations

from pathlib import Path

import numpy as np

import step102_common as common


def execute_aliases_from_raw_text(
    texts: list[str],
    parent_predictions: np.ndarray,
    adapter_paths: list[str],
    thresholds: list[float],
) -> np.ndarray:
    if len(texts) != len(parent_predictions):
        raise ValueError("text and parent-prediction lengths differ")
    if len(adapter_paths) != common.N_ALIASES or len(thresholds) != common.N_ALIASES:
        raise ValueError("Step 102 requires four adapters and thresholds")
    score_columns: list[np.ndarray] = []
    for relative in adapter_paths:
        model, tokenizer = common.load_adapter_model(common.ROOT / Path(relative))
        score_columns.append(common.infer_error_scores(model, tokenizer, texts))
        model.cpu()
    aliases, _ = common.make_alias_codes(
        np.asarray(parent_predictions, dtype=np.int64),
        np.stack(score_columns, axis=1), thresholds,
    )
    return aliases.astype(np.int64)
