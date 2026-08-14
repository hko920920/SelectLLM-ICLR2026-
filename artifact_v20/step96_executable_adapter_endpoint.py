from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import step96_common as common


def execute_aliases_from_raw_pairs(
    premises: list[str],
    hypotheses: list[str],
    adapter_paths: list[str | Path],
    thresholds: list[float],
) -> dict[str, Any]:
    """Run the fixed public parent and four learned gates from raw text only."""
    if len(premises) != len(hypotheses):
        raise ValueError("premise/hypothesis length mismatch")
    if len(adapter_paths) != common.N_ALIASES or len(thresholds) != common.N_ALIASES:
        raise ValueError("Step 96 requires exactly four aliases")
    parent = common.ROOT_SPECS[0]
    parent_predictions, parent_audit = common.infer_root_predictions(
        parent, premises, hypotheses
    )
    score_columns: list[np.ndarray] = []
    adapter_hashes: list[str] = []
    for relative in adapter_paths:
        path = Path(relative)
        if not path.is_absolute():
            path = common.ROOT / path
        model, tokenizer = common.load_adapter_model(parent, path)
        score_columns.append(
            common.infer_error_scores(model, tokenizer, premises, hypotheses)
        )
        adapter_hashes.append(common.sha256_path(path))
    scores = np.stack(score_columns, axis=1)
    alias_codes, triggers = common.make_alias_codes(parent_predictions, scores, thresholds)
    return {
        "parent_predictions": parent_predictions,
        "adapter_error_scores": scores,
        "alias_codes": alias_codes,
        "triggers": triggers,
        "parent_audit": parent_audit,
        "adapter_sha256": adapter_hashes,
        "runtime_inputs": "premise_hypothesis_text_only",
    }
