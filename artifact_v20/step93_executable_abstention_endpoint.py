from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

import step93_common as common


RUNTIME_INPUT_CONTRACT = (
    "raw_text",
    "frozen_encoder_weights",
    "selected_parent_weights",
    "selected_abstention_adapter_weights",
    "frozen_scalar_thresholds",
)

FORBIDDEN_RUNTIME_INPUTS = (
    "reference_label",
    "item_index_or_id",
    "exact_prompt_lookup",
    "peer_output",
    "pool_seed",
    "trajectory",
    "acquisition_score",
    "posterior",
    "selector_state",
)


def execute_raw_text_endpoint(
    texts: list[str],
    parent_checkpoint: Path,
    abstention_checkpoints: list[Path],
    thresholds: list[float],
) -> dict[str, Any]:
    """Execute four learned aliases using only raw text and frozen weights.

    Integer codes 0..76 are the Banking77 response labels. Codes 77..80 are a
    one-to-one serialization of literal tokens ABSTAIN_0..ABSTAIN_3. Equality
    of these codes is exactly equality of the emitted response tokens.
    """
    if len(abstention_checkpoints) != common.N_ALIASES:
        raise ValueError(len(abstention_checkpoints))
    if len(thresholds) != common.N_ALIASES:
        raise ValueError(len(thresholds))
    tokenizer, encoder, device = common.load_encoder()
    hidden = common.encode_texts([str(text) for text in texts], tokenizer, encoder, device)
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    parent_model = common.load_classifier(parent_checkpoint, torch.device("cpu"))
    parent_logits = common.infer_classifier(parent_model, hidden)
    parent_predictions = np.argmax(parent_logits, axis=1).astype(np.int64)
    learned_scores = []
    for path in abstention_checkpoints:
        adapter = common.load_abstention(path, torch.device("cpu"))
        learned_scores.append(common.infer_error_score(adapter, hidden))
    error_scores = np.stack(learned_scores, axis=1)
    response_codes, triggers = common.make_alias_codes(
        parent_predictions,
        error_scores,
        thresholds,
    )
    return {
        "parent_logits": parent_logits,
        "parent_predictions": parent_predictions,
        "learned_error_scores": error_scores,
        "response_codes": response_codes,
        "triggers": triggers,
        "abstention_code_to_literal": {
            common.N_CLASSES + alias: f"ABSTAIN_{alias}"
            for alias in range(common.N_ALIASES)
        },
        "runtime_input_contract": RUNTIME_INPUT_CONTRACT,
        "forbidden_runtime_inputs": FORBIDDEN_RUNTIME_INPUTS,
    }
