from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

import step94_common as common


def execute_registry_from_raw_texts(
    texts: list[str],
    roster_checkpoint_paths: list[Path],
    integrated_variant_paths: list[Path],
    thresholds: list[float],
    n_classes: int,
) -> dict[str, Any]:
    """Replay clean roots and learned variants from raw text only.

    The signature deliberately has no label, item ID, lookup table, peer
    output, pool seed, trajectory, acquisition score, posterior, or selector
    state argument.
    """
    if len(roster_checkpoint_paths) != common.N_ROSTER_ROOTS:
        raise ValueError(len(roster_checkpoint_paths))
    if len(integrated_variant_paths) != common.N_ALIASES or len(thresholds) != common.N_ALIASES:
        raise ValueError((len(integrated_variant_paths), len(thresholds)))
    tokenizer, encoder, device = common.load_encoder()
    hidden = common.encode_texts(texts, tokenizer, encoder, device)
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    root_predictions: list[np.ndarray] = []
    for path in roster_checkpoint_paths:
        model = common.load_classifier(path, n_classes, device)
        logits = common.infer_classifier(model, hidden)
        root_predictions.append(np.argmax(logits, axis=1).astype(np.int64))

    variant_codes: list[np.ndarray] = []
    variant_scores: list[np.ndarray] = []
    variant_parent_predictions: list[np.ndarray] = []
    for alias, (path, threshold) in enumerate(zip(integrated_variant_paths, thresholds)):
        model = common.load_integrated(path, n_classes, device)
        codes, scores = common.infer_integrated_codes(
            model, hidden, float(threshold), alias
        )
        parent_codes = codes.copy()
        parent_codes[codes == n_classes + alias] = -1
        with torch.inference_mode():
            logits_parts: list[np.ndarray] = []
            model = model.to(device).eval()
            for start in range(0, len(hidden), 2048):
                value = torch.from_numpy(hidden[start : start + 2048]).float().to(device)
                parent_logits, _ = model(value)
                logits_parts.append(torch.argmax(parent_logits, dim=1).cpu().numpy())
            model.cpu()
        parent_predictions = np.concatenate(logits_parts).astype(np.int64)
        nontriggered = codes < n_classes
        if not np.array_equal(codes[nontriggered], parent_predictions[nontriggered]):
            raise AssertionError({"alias": alias, "endpoint_parent_replay": False})
        variant_codes.append(codes)
        variant_scores.append(scores)
        variant_parent_predictions.append(parent_predictions)

    return {
        "clean_root_codes": np.stack(root_predictions, axis=1).astype(np.int64),
        "variant_codes": np.stack(variant_codes, axis=1).astype(np.int64),
        "variant_gate_scores": np.stack(variant_scores, axis=1).astype(np.float64),
        "variant_parent_predictions": np.stack(
            variant_parent_predictions, axis=1
        ).astype(np.int64),
        "hidden_sha256": common.array_sha256(hidden),
    }
