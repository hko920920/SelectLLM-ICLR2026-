from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import step93_common as selector_core
import step95_common as adapter_core
import step96_common as training_core


ROOT = Path(__file__).resolve().parent
DATE = "2026-08-13"
PROTOCOL = ROOT / f"STEP97_SCITAIL_CONSERVATIVE_TERMINAL_BRIDGE_PREREGISTRATION_{DATE}.md"
DATASET_ID = "allenai/scitail"
DATASET_CONFIG = "tsv_format"
DATASET_REVISION = "0cc4353235b289165dfde1c7c5d1be983f99ce44"
SPLIT_SALT = "step97-scitail-conservative-v1"
N_CLASSES = 2
N_ALIASES = 4
ROSTER_SIZE = 4
POOL_SIZE = 500
BUDGET = 10
PARTITION_QUOTAS = {
    "adapter_train": 2500,
    "roster_gate": 500,
    "threshold": 750,
    "selector_search": 1000,
}
TAU = 0.025
LOSS_CAP = 0.005
TRIGGER_CAP = 0.05


@dataclass(frozen=True)
class RootSpec:
    key: str
    repo_id: str
    revision: str
    raw_to_dataset: tuple[int, int, int]
    parent: bool = False
    slow_tokenizer: bool = False


ROOT_SPECS: tuple[RootSpec, ...] = (
    RootSpec(
        "crossencoder_deberta_small",
        "cross-encoder/nli-deberta-v3-small",
        "fa2804872c3b4bd748f38c0185cc85775361e735",
        (1, 0, 1),
        parent=True,
        slow_tokenizer=True,
    ),
    RootSpec(
        "moritz_minilm",
        "MoritzLaurer/MiniLM-L6-mnli",
        "6e0917f1a395b7a6c0f054a56b91c45d8e3af92f",
        (0, 1, 1),
    ),
    RootSpec(
        "mfac_bert_mini",
        "M-FAC/bert-mini-finetuned-mnli",
        "780061727f47254ff763de653920bb8b7e2fd5f2",
        (0, 1, 1),
    ),
    RootSpec(
        "mfac_bert_tiny",
        "M-FAC/bert-tiny-finetuned-mnli",
        "618f766f89b50853abc1bea92fd38e1973818f0b",
        (0, 1, 1),
    ),
)


sha256_path = selector_core.sha256_path
array_sha256 = selector_core.array_sha256
stable_u64 = selector_core.stable_u64
exact_match_similarity = selector_core.exact_match_similarity
exact_match_feedback = selector_core.exact_match_feedback
sample_pools = selector_core.sample_pools
run_active = selector_core.run_active
run_fixed = selector_core.run_fixed
effect_summary = selector_core.effect_summary
RunSummary = selector_core.RunSummary
threshold_from_caps = adapter_core.threshold_from_caps
make_alias_codes = adapter_core.make_alias_codes
infer_error_scores = adapter_core.infer_error_scores
save_adapter = adapter_core.save_adapter
seed_everything = adapter_core.seed_everything


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def row_order_key(row: dict[str, Any]) -> tuple[int, int]:
    return (
        stable_u64(
            SPLIT_SALT,
            DATASET_REVISION,
            str(row["premise"]),
            str(row["hypothesis"]),
            int(row["label"]),
            int(row["source_index"]),
        ),
        int(row["source_index"]),
    )


def tokenizer_for(spec: RootSpec) -> Any:
    return AutoTokenizer.from_pretrained(
        spec.repo_id, revision=spec.revision, use_fast=not spec.slow_tokenizer
    )


def infer_root_predictions(
    spec: RootSpec,
    premises: list[str],
    hypotheses: list[str],
    batch_size: int = 64,
) -> tuple[np.ndarray, dict[str, Any]]:
    tokenizer = tokenizer_for(spec)
    model = AutoModelForSequenceClassification.from_pretrained(
        spec.repo_id, revision=spec.revision, use_safetensors=None
    )
    if int(model.config.num_labels) != 3:
        raise AssertionError({"root": spec.key, "num_labels": model.config.num_labels})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    raw_columns: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(premises), batch_size):
            batch = tokenizer(
                premises[start : start + batch_size],
                hypotheses[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            raw_columns.append(torch.argmax(model(**batch).logits, dim=1).cpu().numpy())
    raw = np.concatenate(raw_columns).astype(np.int64)
    predictions = np.asarray(spec.raw_to_dataset, dtype=np.int64)[raw]
    audit = {
        "root_key": spec.key,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "model_type": str(model.config.model_type),
        "raw_to_dataset": list(spec.raw_to_dataset),
        "prediction_sha256": array_sha256(predictions),
        "samples": len(predictions),
        "tokenizer_class": tokenizer.__class__.__name__,
    }
    model.cpu()
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions.astype(np.int64), audit


_trainable_parameter_names = training_core._trainable_parameter_names
encode_pairs = training_core.encode_pairs


def train_error_adapter(
    parent: RootSpec,
    premises: list[str],
    hypotheses: list[str],
    parent_predictions: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> tuple[nn.Module, Any, dict[str, Any]]:
    # The Step-96 routine is architecture-generic after its parent metadata check.
    proxy = training_core.RootSpec(
        parent.key,
        parent.repo_id,
        parent.revision,
        (2, 0, 1),
        parent=True,
        slow_tokenizer=True,
    )
    return training_core.train_error_adapter(
        proxy,
        premises,
        hypotheses,
        parent_predictions,
        labels,
        seed,
        epochs=2,
        batch_size=8,
        gradient_accumulation=4,
    )


def load_adapter_model(parent: RootSpec, path: Path) -> tuple[nn.Module, Any]:
    tokenizer = tokenizer_for(parent)
    model = AutoModelForSequenceClassification.from_pretrained(
        parent.repo_id,
        revision=parent.revision,
        num_labels=2,
        ignore_mismatched_sizes=True,
        use_safetensors=None,
    )
    _trainable_parameter_names(model)
    state = load_file(str(path))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise AssertionError({"unexpected_adapter_tensors": unexpected})
    expected_missing = {name for name, _ in model.named_parameters() if name not in state}
    if set(missing) != expected_missing:
        raise AssertionError("adapter missing-key reconstruction drift")
    return model.eval(), tokenizer
