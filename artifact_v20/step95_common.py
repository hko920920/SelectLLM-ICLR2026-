from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

import step93_common as selector_core


ROOT = Path(__file__).resolve().parent
DATE = "2026-08-13"
PROTOCOL = ROOT / f"STEP95_MNLI_FULL_ADAPTER_PREREGISTRATION_{DATE}.md"
DATASET_ID = "nyu-mll/multi_nli"
DATASET_REVISION = "da70db2af9d09693783c3320c4249840212ee221"
SPLIT_SALT = "step95-mnli-full-adapter-v1"
N_CLASSES = 3
N_ALIASES = 4
ROSTER_SIZE = 6
POOL_SIZE = 500
BUDGET = 20
PARTITION_QUOTAS = {
    "adapter_train": 800,
    "parent_select": 200,
    "threshold": 250,
    "selector_search": 250,
    "selector_verify": 350,
}
TAUS = (0.05, 0.10, 0.25)
LOSS_CAPS = (0.0025, 0.0050, 0.0075, 0.0100)
TRIGGER_CAPS = (0.05, 0.10, 0.15, 0.20)


@dataclass(frozen=True)
class RootSpec:
    key: str
    repo_id: str
    revision: str
    raw_to_dataset: tuple[int, int, int]
    eligible_parent: bool = False


ROOT_SPECS: tuple[RootSpec, ...] = (
    RootSpec(
        "textattack_distilbert",
        "textattack/distilbert-base-uncased-MNLI",
        "2cee56ec53fc7935042c094638345db757eece0d",
        (0, 1, 2),
        True,
    ),
    RootSpec(
        "typeform_distilbert",
        "typeform/distilbert-base-uncased-mnli",
        "cfa538a0fddbbd978fefe8966c1aeff7ad409c90",
        (0, 1, 2),
        True,
    ),
    RootSpec(
        "ishan_distilbert",
        "ishan/distilbert-base-uncased-mnli",
        "5b5436f6f59086b00ac829afecc16d1bd926cbfb",
        (0, 1, 2),
        True,
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
    RootSpec(
        "crossencoder_distilroberta",
        "cross-encoder/nli-distilroberta-base",
        "b14d131f9d32668a5e6a982729b57ff6ed5dfcbd",
        (2, 0, 1),
    ),
    RootSpec(
        "moritz_minilm",
        "MoritzLaurer/MiniLM-L6-mnli",
        "6e0917f1a395b7a6c0f054a56b91c45d8e3af92f",
        (0, 1, 2),
    ),
    RootSpec(
        "crossencoder_deberta_small",
        "cross-encoder/nli-deberta-v3-small",
        "fa2804872c3b4bd748f38c0185cc85775361e735",
        (2, 0, 1),
    ),
)


sha256_path = selector_core.sha256_path
array_sha256 = selector_core.array_sha256
stable_u64 = selector_core.stable_u64
exact_match_similarity = selector_core.exact_match_similarity
exact_match_feedback = selector_core.exact_match_feedback
build_source_faithful_groups = selector_core.build_source_faithful_groups
select_source_faithful_acquisition = selector_core.select_source_faithful_acquisition
sample_pools = selector_core.sample_pools
run_active = selector_core.run_active
run_fixed = selector_core.run_fixed
bootstrap_interval = selector_core.bootstrap_interval
signflip_pvalue = selector_core.signflip_pvalue
effect_summary = selector_core.effect_summary
RunSummary = selector_core.RunSummary


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def row_order_key(row: dict[str, Any]) -> tuple[int, int]:
    return (
        stable_u64(
            SPLIT_SALT,
            str(row["pair_id"]),
            str(row["premise"]),
            str(row["hypothesis"]),
            str(row["genre"]),
            int(row["label"]),
        ),
        int(row["source_index"]),
    )


def root_by_key(key: str) -> RootSpec:
    for spec in ROOT_SPECS:
        if spec.key == key:
            return spec
    raise KeyError(key)


def encode_pairs(tokenizer: Any, premises: list[str], hypotheses: list[str]) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        premises,
        hypotheses,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    return {key: value.cpu() for key, value in encoded.items()}


def infer_root_predictions(
    spec: RootSpec,
    premises: list[str],
    hypotheses: list[str],
    batch_size: int = 64,
) -> tuple[np.ndarray, dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(spec.repo_id, revision=spec.revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        spec.repo_id,
        revision=spec.revision,
        use_safetensors=None,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    raw_predictions: list[np.ndarray] = []
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
            logits = model(**batch).logits
            raw_predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
    raw = np.concatenate(raw_predictions).astype(np.int64)
    mapping = np.asarray(spec.raw_to_dataset, dtype=np.int64)
    if np.any(raw < 0) or np.any(raw >= len(mapping)):
        raise AssertionError({"root": spec.key, "raw_prediction_range": [int(raw.min()), int(raw.max())]})
    predictions = mapping[raw]
    audit = {
        "root_key": spec.key,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "model_type": str(model.config.model_type),
        "raw_to_dataset": list(spec.raw_to_dataset),
        "prediction_sha256": array_sha256(predictions),
        "samples": len(predictions),
    }
    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions.astype(np.int64), audit


def _trainable_parameter_names(model: nn.Module) -> list[str]:
    names: list[str] = []
    for name, parameter in model.named_parameters():
        trainable = (
            name.startswith("distilbert.transformer.layer.4.")
            or name.startswith("distilbert.transformer.layer.5.")
            or name.startswith("pre_classifier.")
            or name.startswith("classifier.")
        )
        parameter.requires_grad_(trainable)
        if trainable:
            names.append(name)
    if not names:
        raise AssertionError("selected parent does not expose the frozen DistilBERT adapter layout")
    return names


def train_error_adapter(
    parent: RootSpec,
    premises: list[str],
    hypotheses: list[str],
    parent_predictions: np.ndarray,
    labels: np.ndarray,
    seed: int,
    epochs: int = 2,
    batch_size: int = 16,
    gradient_accumulation: int = 2,
) -> tuple[nn.Module, Any, dict[str, Any]]:
    if not (len(premises) == len(hypotheses) == len(parent_predictions) == len(labels)):
        raise ValueError("adapter-training arrays have inconsistent lengths")
    seed_everything(seed)
    tokenizer = AutoTokenizer.from_pretrained(parent.repo_id, revision=parent.revision)
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
    if len(positives) == 0 or len(negatives) == 0:
        raise AssertionError({"positive_errors": len(positives), "correct": len(negatives)})
    rng = np.random.default_rng(stable_u64("step95-balanced-bootstrap", seed))
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
    batches_per_epoch = math.ceil(len(bootstrap) / batch_size)
    optimizer_steps = math.ceil(batches_per_epoch / gradient_accumulation) * epochs
    warmup_steps = int(math.floor(0.10 * optimizer_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, optimizer_steps)
    use_amp = bool(torch.cuda.is_available())
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    completed_optimizer_steps = 0
    losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        epoch_rng = np.random.default_rng(stable_u64("step95-adapter-epoch", seed, epoch))
        order = bootstrap[epoch_rng.permutation(len(bootstrap))]
        for batch_index, start in enumerate(range(0, len(order), batch_size)):
            indices = torch.from_numpy(order[start : start + batch_size]).long()
            batch = {key: value[indices].to(device) for key, value in encoded.items()}
            target = target_tensor[indices].to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = model(**batch, labels=target).loss / gradient_accumulation
            scaler.scale(loss).backward()
            losses.append(float(loss.detach().cpu()) * gradient_accumulation)
            should_step = (
                (batch_index + 1) % gradient_accumulation == 0
                or start + batch_size >= len(order)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                completed_optimizer_steps += 1
    model.cpu().eval()
    trainable_count = int(sum(parameter.numel() for parameter in parameters))
    audit = {
        "seed": seed,
        "train_rows": len(labels),
        "parent_error_prevalence": float(np.mean(targets)),
        "balanced_bootstrap_rows": len(bootstrap),
        "positive_bootstrap_fraction": float(np.mean(targets[bootstrap])),
        "epochs": epochs,
        "optimizer_steps": completed_optimizer_steps,
        "warmup_steps": warmup_steps,
        "mean_training_loss": float(np.mean(losses)),
        "trainable_parameter_count": trainable_count,
        "trainable_tensor_names": names,
    }
    return model, tokenizer, audit


def save_adapter(model: nn.Module, path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    tensors = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not tensors:
        raise AssertionError("empty learned adapter")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(path),
        metadata={str(key): str(value) for key, value in metadata.items()},
    )
    flattened = torch.cat([value.float().reshape(-1) for value in tensors.values()])
    return {
        "sha256": sha256_path(path),
        "bytes": path.stat().st_size,
        "tensor_names": sorted(tensors),
        "parameter_count": int(flattened.numel()),
        "nonzero_count": int(torch.count_nonzero(flattened).item()),
        "variance": float(torch.var(flattened, unbiased=False).item()),
    }


def load_adapter_model(parent: RootSpec, path: Path) -> tuple[nn.Module, Any]:
    tokenizer = AutoTokenizer.from_pretrained(parent.repo_id, revision=parent.revision)
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
    expected_missing = {
        name for name, _ in model.named_parameters() if name not in state
    }
    if set(missing) != expected_missing:
        raise AssertionError("adapter missing-key reconstruction drift")
    return model.eval(), tokenizer


def infer_error_scores(
    model: nn.Module,
    tokenizer: Any,
    premises: list[str],
    hypotheses: list[str],
    batch_size: int = 64,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    outputs: list[np.ndarray] = []
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
            scores = torch.softmax(model(**batch).logits.float(), dim=1)[:, 1]
            outputs.append(scores.cpu().numpy())
    model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(outputs).astype(np.float64)


def threshold_from_caps(
    scores: np.ndarray,
    parent_correct: np.ndarray,
    loss_cap: float,
    trigger_cap: float,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    scores = np.asarray(scores, dtype=np.float64)
    parent_correct = np.asarray(parent_correct, dtype=bool)
    if scores.ndim != 1 or parent_correct.shape != scores.shape or len(scores) == 0:
        raise ValueError((scores.shape, parent_correct.shape))
    if not np.all(np.isfinite(scores)):
        raise ValueError("non-finite learned error score")
    n = len(scores)
    max_trigger = int(math.floor(float(trigger_cap) * n + 1e-12))
    max_loss = int(math.floor(float(loss_cap) * n + 1e-12))
    order = np.argsort(-scores, kind="stable")
    cumulative_loss = np.cumsum(parent_correct[order].astype(np.int64))
    feasible_k = 0
    for k in range(1, min(n, max_trigger) + 1):
        if int(cumulative_loss[k - 1]) <= max_loss:
            feasible_k = k
        else:
            break
    while feasible_k > 0 and feasible_k < n and scores[order[feasible_k - 1]] == scores[order[feasible_k]]:
        feasible_k -= 1
    if feasible_k == 0:
        threshold = float(np.nextafter(np.max(scores), np.inf))
    elif feasible_k == n:
        threshold = float(np.nextafter(np.min(scores), -np.inf))
    else:
        upper = float(scores[order[feasible_k - 1]])
        lower = float(scores[order[feasible_k]])
        threshold = float(lower + (upper - lower) / 2.0)
    trigger = scores > threshold
    realized_loss = float(np.mean(trigger & parent_correct))
    realized_trigger = float(np.mean(trigger))
    if realized_loss > loss_cap + 1e-15 or realized_trigger > trigger_cap + 1e-15:
        raise AssertionError("threshold cap violation")
    return threshold, trigger, {
        "threshold": threshold,
        "trigger_count": int(np.sum(trigger)),
        "trigger_fraction": realized_trigger,
        "utility_loss_count": int(np.sum(trigger & parent_correct)),
        "utility_loss": realized_loss,
    }


def make_alias_codes(
    parent_predictions: np.ndarray,
    error_scores: np.ndarray,
    thresholds: list[float] | tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    parent_predictions = np.asarray(parent_predictions, dtype=np.int64)
    error_scores = np.asarray(error_scores, dtype=np.float64)
    if error_scores.shape != (len(parent_predictions), N_ALIASES):
        raise ValueError(error_scores.shape)
    triggers = error_scores > np.asarray(thresholds, dtype=np.float64)[None, :]
    codes = np.repeat(parent_predictions[:, None], N_ALIASES, axis=1)
    for alias in range(N_ALIASES):
        codes[triggers[:, alias], alias] = N_CLASSES + alias
    return codes.astype(np.int64), triggers

