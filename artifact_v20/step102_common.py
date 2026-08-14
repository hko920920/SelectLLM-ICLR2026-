from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import step100_common as base


ROOT = Path(__file__).resolve().parent
DATE = base.DATE
PROTOCOL = ROOT / f"STEP102_IMDB_WRMURRAY_LEARNED_DEVELOPMENT_PROTOCOL_{DATE}.md"
N_CLASSES = base.N_CLASSES
N_ALIASES = base.N_ALIASES
ROSTER_SIZE = base.ROSTER_SIZE
POOL_SIZE = base.POOL_SIZE
MAX_LENGTH = base.MAX_LENGTH
LOSS_CAPS = base.LOSS_CAPS
TRIGGER_CAPS = base.TRIGGER_CAPS
BUDGETS = base.BUDGETS
TAUS = base.TAUS
SEARCH_SEEDS = base.SEARCH_SEEDS
VERIFY_SEEDS = base.VERIFY_SEEDS
TEST_SEEDS = base.TEST_SEEDS
DEV_QUOTAS_PER_LABEL = base.DEV_QUOTAS_PER_LABEL

ROOT_SPECS: tuple[base.RootSpec, ...] = (
    base.RootSpec(
        "wrmurray_roberta_parent",
        "wrmurray/roberta-base-finetuned-imdb",
        "7aa8ca3fae56a1860d8b4c6bf727b91370821ad5",
        parent=True,
    ),
    base.ROOT_SPECS[0],
    base.ROOT_SPECS[1],
    base.ROOT_SPECS[3],
)

sha256_path = base.sha256_path
array_sha256 = base.array_sha256
stable_u64 = base.stable_u64
sample_pools = base.sample_pools
run_active = base.run_active
run_fixed = base.run_fixed
effect_summary = base.effect_summary
exact_match_feedback = base.exact_match_feedback
json_dump = base.json_dump
infer_root_predictions = base.infer_root_predictions
threshold_from_caps = base.threshold_from_caps
make_alias_codes = base.make_alias_codes
seed_everything = base.seed_everything


def _trainable_parameter_names(model: nn.Module) -> list[str]:
    names: list[str] = []
    for name, parameter in model.named_parameters():
        trainable = (
            name.startswith("roberta.encoder.layer.10.")
            or name.startswith("roberta.encoder.layer.11.")
            or name.startswith("classifier.")
        )
        parameter.requires_grad_(trainable)
        if trainable:
            names.append(name)
    if not names:
        raise AssertionError("parent does not expose the frozen RoBERTa adapter layout")
    return names


def encode_texts(tokenizer: Any, texts: list[str]) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        texts, padding=True, truncation=True, max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    return {key: value.cpu() for key, value in encoded.items()}


def train_error_adapter(
    texts: list[str], parent_predictions: np.ndarray, labels: np.ndarray,
    seed: int, epochs: int = 2, batch_size: int = 8,
    gradient_accumulation: int = 4,
) -> tuple[nn.Module, Any, dict[str, Any]]:
    parent = ROOT_SPECS[0]
    seed_everything(seed)
    tokenizer = AutoTokenizer.from_pretrained(parent.repo_id, revision=parent.revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        parent.repo_id, revision=parent.revision, num_labels=2,
        ignore_mismatched_sizes=True, use_safetensors=None,
    )
    names = _trainable_parameter_names(model)
    targets = (np.asarray(parent_predictions) != np.asarray(labels)).astype(np.int64)
    positives = np.flatnonzero(targets == 1)
    negatives = np.flatnonzero(targets == 0)
    if not len(positives) or not len(negatives):
        raise AssertionError({"errors": len(positives), "correct": len(negatives)})
    rng = np.random.default_rng(stable_u64("step102-balanced-bootstrap", seed))
    half = max(len(positives), len(negatives))
    bootstrap = np.concatenate([
        rng.choice(positives, size=half, replace=True),
        rng.choice(negatives, size=half, replace=True),
    ])
    rng.shuffle(bootstrap)
    encoded = encode_texts(tokenizer, texts)
    targets_tensor = torch.from_numpy(targets).long()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=2e-5, weight_decay=0.01)
    batches_per_epoch = math.ceil(len(bootstrap) / batch_size)
    optimizer_steps = math.ceil(batches_per_epoch / gradient_accumulation) * epochs
    warmup_steps = math.floor(0.10 * optimizer_steps)
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
    completed = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        epoch_rng = np.random.default_rng(stable_u64("step102-adapter-epoch", seed, epoch))
        order = bootstrap[epoch_rng.permutation(len(bootstrap))]
        for batch_index, start in enumerate(range(0, len(order), batch_size)):
            indices = torch.from_numpy(order[start : start + batch_size]).long()
            batch = {key: value[indices].to(device) for key, value in encoded.items()}
            target = targets_tensor[indices].to(device)
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
                completed += 1
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
        "optimizer_steps": completed,
        "warmup_steps": warmup_steps,
        "mean_training_loss": float(np.mean(losses)),
        "trainable_parameter_count": int(sum(parameter.numel() for parameter in parameters)),
        "trainable_tensor_names": names,
    }


def save_adapter(model: nn.Module, path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    tensors = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata={str(k): str(v) for k, v in metadata.items()})
    flattened = torch.cat([value.float().reshape(-1) for value in tensors.values()])
    return {
        "sha256": sha256_path(path),
        "bytes": path.stat().st_size,
        "tensor_names": sorted(tensors),
        "parameter_count": int(flattened.numel()),
        "nonzero_count": int(torch.count_nonzero(flattened).item()),
        "variance": float(torch.var(flattened, unbiased=False).item()),
    }


def load_adapter_model(path: Path) -> tuple[nn.Module, Any]:
    parent = ROOT_SPECS[0]
    tokenizer = AutoTokenizer.from_pretrained(parent.repo_id, revision=parent.revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        parent.repo_id, revision=parent.revision, num_labels=2,
        ignore_mismatched_sizes=True, use_safetensors=None,
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


def infer_error_scores(
    model: nn.Module, tokenizer: Any, texts: list[str], batch_size: int = 16,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start : start + batch_size], padding=True, truncation=True,
                max_length=MAX_LENGTH, return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            output.append(torch.softmax(model(**batch).logits.float(), dim=1)[:, 1].cpu().numpy())
    model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(output).astype(np.float64)
