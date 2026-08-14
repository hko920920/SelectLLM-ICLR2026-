from __future__ import annotations

import hashlib
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


ROOT = Path(__file__).resolve().parent
DATE = "2026-08-13"
PROTOCOL = ROOT / f"STEP96_SNLI_DIRECTIONAL_TERMINAL_BRIDGE_PREREGISTRATION_{DATE}.md"
DATASET_ID = "stanfordnlp/snli"
DATASET_CONFIG = "plain_text"
DATASET_REVISION = "cdb5c3d5eed6ead6e5a341c8e56e669bb666725b"
SPLIT_SALT = "step96-snli-directional-v1"
N_CLASSES = 3
N_ALIASES = 4
ROSTER_SIZE = 4
POOL_SIZE = 500
BUDGET = 10
PARTITION_QUOTAS = {
    "adapter_train": 3000,
    "roster_gate": 1000,
    "threshold": 1500,
    "selector_search": 2000,
}
TAUS = (0.025, 0.05, 0.10)
LOSS_CAPS = (0.0025, 0.0050, 0.0075, 0.0100)
TRIGGER_CAPS = (0.05, 0.10, 0.20, 0.30)


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
        (2, 0, 1),
        parent=True,
        slow_tokenizer=True,
    ),
    RootSpec(
        "moritz_minilm",
        "MoritzLaurer/MiniLM-L6-mnli",
        "6e0917f1a395b7a6c0f054a56b91c45d8e3af92f",
        (0, 1, 2),
    ),
    RootSpec(
        "mfac_bert_mini",
        "M-FAC/bert-mini-finetuned-mnli",
        "780061727f47254ff763de653920bb8b7e2fd5f2",
        (0, 1, 2),
    ),
    RootSpec(
        "mfac_bert_tiny",
        "M-FAC/bert-tiny-finetuned-mnli",
        "618f766f89b50853abc1bea92fd38e1973818f0b",
        (0, 1, 2),
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
        spec.repo_id,
        revision=spec.revision,
        use_fast=not spec.slow_tokenizer,
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
    if int(model.config.num_labels) != N_CLASSES:
        raise AssertionError({"root": spec.key, "num_labels": model.config.num_labels})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    output: list[np.ndarray] = []
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
            output.append(torch.argmax(model(**batch).logits, dim=1).cpu().numpy())
    raw = np.concatenate(output).astype(np.int64)
    mapping = np.asarray(spec.raw_to_dataset, dtype=np.int64)
    predictions = mapping[raw]
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


def _trainable_parameter_names(model: nn.Module) -> list[str]:
    names: list[str] = []
    for name, parameter in model.named_parameters():
        trainable = (
            name.startswith("deberta.encoder.layer.4.")
            or name.startswith("deberta.encoder.layer.5.")
            or name.startswith("pooler.")
            or name.startswith("classifier.")
        )
        parameter.requires_grad_(trainable)
        if trainable:
            names.append(name)
    if not names:
        raise AssertionError("parent does not expose the frozen DeBERTa adapter layout")
    return names


def encode_pairs(
    tokenizer: Any, premises: list[str], hypotheses: list[str]
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        premises,
        hypotheses,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    return {key: value.cpu() for key, value in encoded.items()}


def train_error_adapter(
    parent: RootSpec,
    premises: list[str],
    hypotheses: list[str],
    parent_predictions: np.ndarray,
    labels: np.ndarray,
    seed: int,
    epochs: int = 2,
    batch_size: int = 8,
    gradient_accumulation: int = 4,
) -> tuple[nn.Module, Any, dict[str, Any]]:
    if not parent.parent:
        raise AssertionError("Step 96 parent binding changed")
    seed_everything(seed)
    tokenizer = tokenizer_for(parent)
    model = AutoModelForSequenceClassification.from_pretrained(
        parent.repo_id,
        revision=parent.revision,
        num_labels=2,
        ignore_mismatched_sizes=True,
        use_safetensors=None,
    )
    names = _trainable_parameter_names(model)
    targets = (np.asarray(parent_predictions) != np.asarray(labels)).astype(np.int64)
    positives = np.flatnonzero(targets == 1)
    negatives = np.flatnonzero(targets == 0)
    if not len(positives) or not len(negatives):
        raise AssertionError({"errors": len(positives), "correct": len(negatives)})
    rng = np.random.default_rng(stable_u64("step96-balanced-bootstrap", seed))
    half = max(len(positives), len(negatives))
    bootstrap = np.concatenate(
        [
            rng.choice(positives, size=half, replace=True),
            rng.choice(negatives, size=half, replace=True),
        ]
    )
    rng.shuffle(bootstrap)
    encoded = encode_pairs(tokenizer, premises, hypotheses)
    target_tensor = torch.from_numpy(targets).long()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=2e-5, weight_decay=0.01)
    batches_per_epoch = int(np.ceil(len(bootstrap) / batch_size))
    optimizer_steps = int(np.ceil(batches_per_epoch / gradient_accumulation)) * epochs
    warmup_steps = int(np.floor(0.10 * optimizer_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (
            step / max(1, warmup_steps)
            if step < warmup_steps
            else max(0.0, (optimizer_steps - step) / max(1, optimizer_steps - warmup_steps))
        ),
    )
    use_amp = bool(torch.cuda.is_available())
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    losses: list[float] = []
    completed_steps = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        order_rng = np.random.default_rng(stable_u64("step96-adapter-epoch", seed, epoch))
        order = bootstrap[order_rng.permutation(len(bootstrap))]
        for batch_index, start in enumerate(range(0, len(order), batch_size)):
            indices = torch.from_numpy(order[start : start + batch_size]).long()
            batch = {key: value[indices].to(device) for key, value in encoded.items()}
            target = target_tensor[indices].to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = model(**batch, labels=target).loss / gradient_accumulation
            scaler.scale(loss).backward()
            losses.append(float(loss.detach().cpu()) * gradient_accumulation)
            if (batch_index + 1) % gradient_accumulation == 0 or start + batch_size >= len(order):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                completed_steps += 1
    model.cpu().eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, tokenizer, {
        "seed": seed,
        "train_rows": len(labels),
        "parent_error_prevalence": float(np.mean(targets)),
        "balanced_bootstrap_rows": len(bootstrap),
        "positive_bootstrap_fraction": float(np.mean(targets[bootstrap])),
        "epochs": epochs,
        "optimizer_steps": completed_steps,
        "warmup_steps": warmup_steps,
        "mean_training_loss": float(np.mean(losses)),
        "trainable_parameter_count": int(sum(p.numel() for p in parameters)),
        "trainable_tensor_names": names,
    }


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


def protocol_sha256() -> str:
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
