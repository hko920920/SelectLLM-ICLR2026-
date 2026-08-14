from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import stage2_clean_safety_endpoint as stage2
from stage3_learned_common import Stage3Error, array_sha256


def extract_partitions(
    task: str,
    runtime_spec: Mapping[str, Any],
    texts_by_partition: Mapping[str, list[str]],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Run one exact frozen Transformer parent and return hidden-plus-logit features."""

    torch.manual_seed(0)
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)

    with tempfile.TemporaryDirectory(prefix="praa-stage3-transformer-") as tmp:
        model_dir, binding = stage2.materialize_transformer(
            runtime_spec, Path(tmp) / "model"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        use_safetensors = str(runtime_spec["weight"]["path"]).endswith(
            ".safetensors"
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            local_files_only=True,
            use_safetensors=use_safetensors,
        )
        model.eval()
        mapping = stage2.transformer_mapping(task, model.config)
        n_classes = 6 if task == "emotion" else 20
        if set(mapping.values()) != set(range(n_classes)):
            raise Stage3Error(f"incomplete target label mapping: {mapping}")
        batch_size = 64 if task == "emotion" else 32

        outputs: dict[str, dict[str, np.ndarray]] = {}
        with torch.inference_mode():
            for partition, texts in texts_by_partition.items():
                predictions: list[np.ndarray] = []
                logits_parts: list[np.ndarray] = []
                feature_parts: list[np.ndarray] = []
                for start in range(0, len(texts), batch_size):
                    encoded = tokenizer(
                        texts[start : start + batch_size],
                        padding=True,
                        truncation=True,
                        max_length=128,
                        return_tensors="pt",
                    )
                    result = model(**encoded, output_hidden_states=True)
                    native_logits = result.logits.float().cpu().numpy().astype(
                        np.float32
                    )
                    mapped_logits = np.empty(
                        (len(native_logits), n_classes), dtype=np.float32
                    )
                    for native_index, target_index in mapping.items():
                        mapped_logits[:, target_index] = native_logits[:, native_index]
                    hidden = (
                        result.hidden_states[-1][:, 0, :]
                        .float()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                    predictions.append(
                        np.argmax(mapped_logits, axis=1).astype(np.int16)
                    )
                    logits_parts.append(mapped_logits)
                    feature_parts.append(
                        np.concatenate([hidden, mapped_logits], axis=1).astype(
                            np.float32
                        )
                    )
                if not predictions:
                    raise Stage3Error(f"empty partition {partition}")
                partition_predictions = np.concatenate(predictions)
                partition_logits = np.concatenate(logits_parts)
                partition_features = np.concatenate(feature_parts)
                if not (
                    len(partition_predictions)
                    == len(partition_logits)
                    == len(partition_features)
                    == len(texts)
                ):
                    raise Stage3Error(f"feature row mismatch for {partition}")
                if not np.all(np.isfinite(partition_features)):
                    raise Stage3Error(f"non-finite features for {partition}")
                outputs[partition] = {
                    "predictions": partition_predictions,
                    "logits": partition_logits,
                    "features": partition_features,
                }

        runtime = {
            **binding,
            "runtime_kind": "transformer_hidden_plus_mapped_logits",
            "model_class": model.__class__.__name__,
            "tokenizer_class": tokenizer.__class__.__name__,
            "model_type": str(model.config.model_type),
            "batch_size": batch_size,
            "max_length": 128,
            "native_index_to_target_index": {
                str(key): int(value) for key, value in mapping.items()
            },
            "partitions": {
                partition: {
                    "rows": len(value["predictions"]),
                    "prediction_sha256": array_sha256(value["predictions"]),
                    "logits_sha256": array_sha256(value["logits"]),
                    "features_sha256": array_sha256(value["features"]),
                    "feature_dimension": int(value["features"].shape[1]),
                }
                for partition, value in outputs.items()
            },
        }
        return outputs, runtime
