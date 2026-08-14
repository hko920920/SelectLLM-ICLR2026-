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
from scipy.stats import beta
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import step93_common as selector


ROOT = Path(__file__).resolve().parent
DATE = "2026-08-13"
PROTOCOL = ROOT / f"STEP105_YELP_PROSPECTIVE_CROSS_TASK_ONE_SHOT_PROTOCOL_{DATE}.md"
METADATA = ROOT / f"STEP105_PUBLIC_METADATA_SNAPSHOT_{DATE}.json"
PREDATA_LOCK = ROOT / f"STEP105_PREDATA_LOCK_{DATE}.json"
DATASET_ID = "fancyzhx/yelp_polarity"
DATASET_CONFIG = "plain_text"
DATASET_REVISION = "bbf1c97a1f0cf005e5aded43839fd814654a1557"
SALT = "step105-yelp-prospective-cross-task-one-shot-v1"
MAX_LENGTH = 256
N_CLASSES = 2
N_ALIASES = 4
ROSTER_SIZE = 4
POOL_SIZE = 500
BUDGET = 5
TAU = 0.05
LOSS_CAP = 0.0075
TRIGGER_CAP = 0.025
DEV_QUOTAS_PER_LABEL = {
    "adapter_train": 3000,
    "threshold_calibration": 500,
    "threshold_safety": 1000,
}
ADAPTER_SEEDS = tuple(105100 + alias for alias in range(N_ALIASES))
TEST_SEEDS = tuple(range(1_053_000, 1_056_000))
BOOTSTRAP_SEEDS = (105800, 105801, 105802)
SIGNFLIP_SEEDS = (105801, 105802, 105803)


@dataclass(frozen=True)
class RootSpec:
    key: str
    repo_id: str
    revision: str
    parent: bool = False


ROOT_SPECS: tuple[RootSpec, ...] = (
    RootSpec(
        "victorsanh_roberta_parent",
        "VictorSanh/roberta-base-finetuned-yelp-polarity",
        "709b46dd24f08e9b310fd7630cd0728f6d52b570",
        parent=True,
    ),
    RootSpec(
        "fabrice_bert",
        "fabriceyhc/bert-base-uncased-yelp_polarity",
        "5793d97b55d24e604d9157704e9d3a76735515c1",
    ),
    RootSpec(
        "randell_distilbert",
        "randellcotta/distilbert-base-uncased-finetuned-yelp-polarity",
        "a70e629510f730cdaa8407db4928705cfedb257c",
    ),
    RootSpec(
        "jiaqilee_robust_bert",
        "JiaqiLee/robust-bert-yelp",
        "a40e8daa515ff7cd09fed21c3d56f50ab7da8f25",
    ),
)


sha256_path = selector.sha256_path
array_sha256 = selector.array_sha256
stable_u64 = selector.stable_u64
sample_pools = selector.sample_pools
run_active = selector.run_active
run_fixed = selector.run_fixed
effect_summary = selector.effect_summary
exact_match_feedback = selector.exact_match_feedback


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def row_digest(index: int, text: str, label: int) -> str:
    payload = "\x1f".join((DATASET_REVISION, str(index), text, str(label), SALT))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def uid(index: int, text: str) -> str:
    payload = f"{DATASET_REVISION}\x1f{index}\x1f{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def infer_root_predictions(
    spec: RootSpec, texts: list[str], batch_size: int = 32,
) -> tuple[np.ndarray, dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(spec.repo_id, revision=spec.revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        spec.repo_id, revision=spec.revision, use_safetensors=None,
    )
    if int(model.config.num_labels) != N_CLASSES:
        raise AssertionError({"root": spec.key, "num_labels": model.config.num_labels})
    if bool(getattr(model.config, "trust_remote_code", False)):
        raise AssertionError({"root": spec.key, "custom_code": True})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    columns: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start : start + batch_size], padding=True, truncation=True,
                max_length=MAX_LENGTH, return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            columns.append(torch.argmax(model(**batch).logits, dim=1).cpu().numpy())
    predictions = np.concatenate(columns).astype(np.int64)
    audit = {
        "root_key": spec.key,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "parent": spec.parent,
        "model_type": str(model.config.model_type),
        "architectures": list(model.config.architectures or []),
        "num_labels": int(model.config.num_labels),
        "id2label": {str(key): str(value) for key, value in model.config.id2label.items()},
        "label2id": {str(key): int(value) for key, value in model.config.label2id.items()},
        "max_length": MAX_LENGTH,
        "prediction_sha256": array_sha256(predictions),
        "samples": len(predictions),
        "tokenizer_class": tokenizer.__class__.__name__,
        "model_class": model.__class__.__name__,
    }
    model.cpu()
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions, audit


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
        raise AssertionError("frozen parent does not expose the RoBERTa top-two-layer recipe")
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
    rng = np.random.default_rng(stable_u64("step105-balanced-bootstrap", seed))
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
        epoch_rng = np.random.default_rng(stable_u64("step105-adapter-epoch", seed, epoch))
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
        "parent_error_count": int(np.sum(targets)),
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
    if not tensors:
        raise AssertionError("empty learned adapter")
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


def threshold_from_caps(
    scores: np.ndarray, parent_correct: np.ndarray,
    loss_cap: float, trigger_cap: float,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    scores = np.asarray(scores, dtype=np.float64)
    parent_correct = np.asarray(parent_correct, dtype=bool)
    n = len(scores)
    max_trigger = math.floor(trigger_cap * n + 1e-12)
    max_loss = math.floor(loss_cap * n + 1e-12)
    order = np.argsort(-scores, kind="stable")
    cumulative_loss = np.cumsum(parent_correct[order].astype(np.int64))
    feasible = 0
    for k in range(1, min(n, max_trigger) + 1):
        if int(cumulative_loss[k - 1]) <= max_loss:
            feasible = k
        else:
            break
    while feasible > 0 and feasible < n and scores[order[feasible - 1]] == scores[order[feasible]]:
        feasible -= 1
    if feasible == 0:
        threshold = float(np.nextafter(np.max(scores), np.inf))
    elif feasible == n:
        threshold = float(np.nextafter(np.min(scores), -np.inf))
    else:
        upper = float(scores[order[feasible - 1]])
        lower = float(scores[order[feasible]])
        threshold = lower + (upper - lower) / 2.0
    trigger = scores > threshold
    if np.mean(trigger & parent_correct) > loss_cap + 1e-15 or np.mean(trigger) > trigger_cap + 1e-15:
        raise AssertionError("threshold cap violation")
    return float(threshold), trigger, {
        "threshold": float(threshold),
        "trigger_count": int(np.sum(trigger)),
        "trigger_fraction": float(np.mean(trigger)),
        "utility_loss_count": int(np.sum(trigger & parent_correct)),
        "utility_loss": float(np.mean(trigger & parent_correct)),
    }


def cp_upper(loss_count: int, rows: int) -> float:
    return 1.0 if loss_count >= rows else float(beta.ppf(0.99, loss_count + 1, rows - loss_count))


def fit_frozen_thresholds(
    parent: np.ndarray, labels: np.ndarray, scores: np.ndarray,
    calibration: np.ndarray, safety: np.ndarray,
) -> tuple[list[float], list[dict[str, Any]]]:
    thresholds: list[float] = []
    audits: list[dict[str, Any]] = []
    for alias in range(N_ALIASES):
        cal_threshold, _, cal_audit = threshold_from_caps(
            scores[calibration, alias], parent[calibration] == labels[calibration],
            LOSS_CAP, TRIGGER_CAP,
        )
        safe_threshold, _, safe_audit = threshold_from_caps(
            scores[safety, alias], parent[safety] == labels[safety], 0.005, TRIGGER_CAP,
        )
        threshold = max(cal_threshold, safe_threshold)
        safe_correct = parent[safety] == labels[safety]
        candidates = np.sort(np.unique(scores[safety, alias]))
        while True:
            trigger = scores[safety, alias] > threshold
            loss_count = int(np.sum(trigger & safe_correct))
            if (
                loss_count <= math.floor(0.005 * len(safety) + 1e-12)
                and cp_upper(loss_count, len(safety)) <= 0.01
                and int(np.sum(trigger)) <= math.floor(TRIGGER_CAP * len(safety) + 1e-12)
            ):
                break
            larger = candidates[candidates > threshold]
            threshold = (
                float(larger[0]) if len(larger)
                else float(np.nextafter(np.max(scores[safety, alias]), np.inf))
            )
        thresholds.append(float(threshold))
        audits.append({
            "alias": alias,
            "calibration": cal_audit,
            "safety_initial": safe_audit,
            "final_threshold": float(threshold),
            "safety_final_trigger_count": int(np.sum(trigger)),
            "safety_final_loss_count": loss_count,
            "safety_final_cp99_upper": cp_upper(loss_count, len(safety)),
        })
    return thresholds, audits


def make_alias_codes(
    parent_predictions: np.ndarray, error_scores: np.ndarray,
    thresholds: list[float] | tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    parent_predictions = np.asarray(parent_predictions, dtype=np.int64)
    error_scores = np.asarray(error_scores, dtype=np.float64)
    if error_scores.shape != (len(parent_predictions), N_ALIASES):
        raise ValueError(error_scores.shape)
    triggers = error_scores > np.asarray(thresholds, dtype=np.float64)[None, :]
    aliases = np.repeat(parent_predictions[:, None], N_ALIASES, axis=1)
    for alias in range(N_ALIASES):
        aliases[triggers[:, alias], alias] = N_CLASSES + alias
    return aliases.astype(np.int64), triggers


def compute_effects(
    root_predictions: np.ndarray, aliases: np.ndarray, references: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    roots = np.asarray(root_predictions, dtype=np.int64)
    aliases = np.asarray(aliases, dtype=np.int64)
    references = np.asarray(references, dtype=np.int64)
    clean_parents = np.arange(ROSTER_SIZE, dtype=np.int64)
    refined_parents = np.concatenate([clean_parents, np.zeros(N_ALIASES, dtype=np.int64)])
    refined_codes = np.column_stack([roots, aliases])
    pools = sample_pools(len(references), TEST_SEEDS, POOL_SIZE)
    clean = run_active(roots, references, clean_parents, roots, pools, TEST_SEEDS, BUDGET, TAU)
    refined = run_active(
        refined_codes, references, refined_parents, roots, pools, TEST_SEEDS, BUDGET, TAU,
    )
    fixed = run_fixed(
        refined_codes, references, refined_parents, roots, pools, clean.queries, TEST_SEEDS,
    )
    terminal = refined.terminal - clean.terminal
    fixed_terminal = fixed.terminal - clean.terminal
    active_minus_fixed = refined.terminal - fixed.terminal
    cumulative = refined.cumulative - clean.cumulative
    fixed_cumulative = fixed.cumulative - clean.cumulative
    clean_final = clean.roots[:, -1]
    refined_final = refined.roots[:, -1]
    summary = {
        "terminal": effect_summary(terminal, BOOTSTRAP_SEEDS[0], SIGNFLIP_SEEDS[0]),
        "active_minus_fixed_terminal": effect_summary(
            active_minus_fixed, BOOTSTRAP_SEEDS[1], SIGNFLIP_SEEDS[1],
        ),
        "cumulative": effect_summary(cumulative, BOOTSTRAP_SEEDS[2], SIGNFLIP_SEEDS[2]),
        "path_change_rate": float(np.mean(np.any(clean.queries != refined.queries, axis=1))),
        "query_set_change_rate": float(np.mean([
            set(map(int, clean.queries[i])) != set(map(int, refined.queries[i]))
            for i in range(len(TEST_SEEDS))
        ])),
        "final_root_change_rate": float(np.mean(clean_final != refined_final)),
        "parent_to_challenger_rate": float(np.mean((clean_final == 0) & (refined_final != 0))),
        "challenger_to_parent_rate": float(np.mean((clean_final != 0) & (refined_final == 0))),
        "clean_parent_terminal_rate": float(np.mean(clean_final == 0)),
        "refined_parent_terminal_rate": float(np.mean(refined_final == 0)),
        "max_abs_fixed_terminal_delta": float(np.max(np.abs(fixed_terminal))),
        "max_abs_fixed_cumulative_delta": float(np.max(np.abs(fixed_cumulative))),
        "fixed_root_history_exact": bool(np.array_equal(fixed.roots, clean.roots)),
    }
    arrays = {
        "pools": pools,
        "clean_queries": clean.queries,
        "refined_queries": refined.queries,
        "clean_roots": clean.roots,
        "refined_roots": refined.roots,
        "fixed_roots": fixed.roots,
        "clean_terminal": clean.terminal,
        "refined_terminal": refined.terminal,
        "fixed_terminal": fixed.terminal,
        "clean_cumulative": clean.cumulative,
        "refined_cumulative": refined.cumulative,
        "fixed_cumulative": fixed.cumulative,
        "terminal_delta": terminal,
        "active_minus_fixed_terminal_delta": active_minus_fixed,
        "cumulative_delta": cumulative,
    }
    return summary, arrays
