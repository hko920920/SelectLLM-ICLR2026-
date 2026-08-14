from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import step95_common as common


def execute_aliases_from_raw_pairs(
    premises: list[str],
    hypotheses: list[str],
    parent_key: str,
    adapter_paths: list[str | Path],
    thresholds: list[float],
) -> dict[str, Any]:
    """Execute a fixed parent plus learned adapters from raw text pairs only."""
    if len(premises) != len(hypotheses):
        raise ValueError("premise/hypothesis length mismatch")
    if len(adapter_paths) != common.N_ALIASES or len(thresholds) != common.N_ALIASES:
        raise ValueError("Step 95 requires exactly four aliases")
    parent = common.root_by_key(parent_key)
    if not parent.eligible_parent:
        raise ValueError("Step 95 adapters require a frozen eligible DistilBERT parent")
    parent_predictions, parent_audit = common.infer_root_predictions(
        parent, premises, hypotheses
    )
    scores: list[np.ndarray] = []
    adapter_hashes: list[str] = []
    for relative in adapter_paths:
        path = Path(relative)
        if not path.is_absolute():
            path = common.ROOT / path
        model, tokenizer = common.load_adapter_model(parent, path)
        scores.append(common.infer_error_scores(model, tokenizer, premises, hypotheses))
        adapter_hashes.append(common.sha256_path(path))
    score_matrix = np.stack(scores, axis=1)
    alias_codes, triggers = common.make_alias_codes(
        parent_predictions, score_matrix, thresholds
    )
    return {
        "parent_predictions": parent_predictions,
        "adapter_error_scores": score_matrix,
        "alias_codes": alias_codes,
        "triggers": triggers,
        "parent_audit": parent_audit,
        "adapter_sha256": adapter_hashes,
        "runtime_inputs": "premise_hypothesis_text_only",
    }

