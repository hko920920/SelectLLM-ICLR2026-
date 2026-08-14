from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoModel, AutoTokenizer

from step61a_blind_common import build_group_structure, select_llm_acquisition


ROOT = Path(__file__).resolve().parent
DATE = "2026-08-12"
DATASET_ID = "fancyzhx/ag_news"
DATASET_REVISION = "eb185aade064a813bc0b7f42de02595523103ca4"
BASE_MODEL = "distilbert/distilbert-base-uncased"
BASE_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
BASE_MODEL_SHA256 = "5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063"
SPLIT_SALT = "step92-learned-adapter-v1"
ROOT_FRACTIONS = (0.02, 0.04, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50, 0.65, 0.80, 0.90, 1.00)
ELIGIBLE_PARENTS = (4, 5, 6, 7, 8)
ROOT_SEED_BASE = 92100
CALIBRATION_SEED_BASE = 92200
N_CLASSES = 4
HIDDEN_SIZE = 768
POOL_SIZE = 400
BUDGET = 50


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_u64(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def train_partition(index: int) -> str:
    residue = stable_u64(SPLIT_SALT, "train", int(index)) % 10
    if residue <= 6:
        return "model_train"
    if residue <= 8:
        return "adapter_train"
    return "stagea_calibration"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ClassificationAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(HIDDEN_SIZE)
        self.down = nn.Linear(HIDDEN_SIZE, 128)
        self.out = nn.Linear(128, N_CLASSES)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.out(torch.nn.functional.gelu(self.down(self.norm(hidden))))


class ContextualTemperatureAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(HIDDEN_SIZE)
        self.down = nn.Linear(HIDDEN_SIZE, 64)
        self.out = nn.Linear(64, 1)

    def log_temperature(self, hidden: torch.Tensor) -> torch.Tensor:
        raw = self.out(torch.nn.functional.gelu(self.down(self.norm(hidden))))[:, 0]
        return torch.clamp(raw, -3.0, 3.0)

    def forward(self, hidden: torch.Tensor, parent_logits: torch.Tensor) -> torch.Tensor:
        scale = torch.exp(self.log_temperature(hidden))[:, None]
        return parent_logits * scale


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def save_module(module: nn.Module, path: Path, metadata: dict[str, object]) -> None:
    tensors = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in module.state_dict().items()
    }
    text_metadata = {str(key): str(value) for key, value in metadata.items()}
    text_metadata["parameter_count"] = str(trainable_parameter_count(module))
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=text_metadata)


def load_classifier(path: Path, device: torch.device) -> ClassificationAdapter:
    model = ClassificationAdapter()
    model.load_state_dict(load_file(str(path)))
    return model.to(device).eval()


def load_temperature(path: Path, device: torch.device) -> ContextualTemperatureAdapter:
    model = ContextualTemperatureAdapter()
    model.load_state_dict(load_file(str(path)))
    return model.to(device).eval()


def adapter_tensor_audit(path: Path) -> dict[str, Any]:
    tensors = load_file(str(path))
    total = int(sum(tensor.numel() for tensor in tensors.values()))
    flattened = torch.cat([tensor.detach().float().reshape(-1) for tensor in tensors.values()])
    return {
        "sha256": sha256_path(path),
        "tensor_names": sorted(tensors),
        "parameter_count": total,
        "nonzero_count": int(torch.count_nonzero(flattened).item()),
        "variance": float(torch.var(flattened, unbiased=False).item()),
        "bytes": path.stat().st_size,
    }


def load_encoder() -> tuple[Any, nn.Module, torch.device]:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION, local_files_only=True)
    encoder = AutoModel.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        local_files_only=True,
        use_safetensors=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return tokenizer, encoder, device


def encode_texts(
    texts: list[str],
    tokenizer: Any,
    encoder: nn.Module,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = encoder(**encoded).last_hidden_state[:, 0, :]
            outputs.append(hidden.float().cpu().numpy())
    if not outputs:
        return np.empty((0, HIDDEN_SIZE), dtype=np.float32)
    return np.concatenate(outputs).astype(np.float32, copy=False)


def _batch_order(size: int, seed: int, epoch: int) -> np.ndarray:
    rng = np.random.default_rng(stable_u64("step92-batch", seed, epoch))
    return rng.permutation(size)


def train_classifier(
    hidden: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    seed: int,
    epochs: int = 25,
    batch_size: int = 512,
) -> ClassificationAdapter:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClassificationAdapter().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x = torch.from_numpy(hidden[indices]).float()
    y = torch.from_numpy(labels[indices]).long()
    model.train()
    for epoch in range(epochs):
        epoch_order = _batch_order(len(indices), seed, epoch)
        for start in range(0, len(indices), batch_size):
            order = torch.from_numpy(epoch_order[start : start + batch_size]).long()
            xb = x[order].to(device)
            yb = y[order].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model.cpu().eval()


def train_temperature_adapter(
    hidden: np.ndarray,
    parent_logits: np.ndarray,
    labels: np.ndarray,
    seed: int,
    epochs: int = 30,
    batch_size: int = 512,
) -> ContextualTemperatureAdapter:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContextualTemperatureAdapter().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(stable_u64("step92-bootstrap", seed))
    bootstrap = rng.integers(0, len(labels), size=len(labels), dtype=np.int64)
    x = torch.from_numpy(hidden[bootstrap]).float()
    logits = torch.from_numpy(parent_logits[bootstrap]).float()
    y = torch.from_numpy(labels[bootstrap]).long()
    model.train()
    for epoch in range(epochs):
        epoch_order = _batch_order(len(bootstrap), seed, epoch)
        for start in range(0, len(bootstrap), batch_size):
            order = torch.from_numpy(epoch_order[start : start + batch_size]).long()
            xb = x[order].to(device)
            lb = logits[order].to(device)
            yb = y[order].to(device)
            optimizer.zero_grad(set_to_none=True)
            log_temperature = model.log_temperature(xb)
            calibrated = lb * torch.exp(log_temperature)[:, None]
            loss = torch.nn.functional.cross_entropy(calibrated, yb)
            loss = loss + 1e-3 * torch.mean(log_temperature.square())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model.cpu().eval()


def infer_classifier(module: ClassificationAdapter, hidden: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = module.to(device).eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(hidden), batch_size):
            value = torch.from_numpy(hidden[start : start + batch_size]).float().to(device)
            outputs.append(module(value).float().cpu().numpy())
    module.cpu()
    return np.concatenate(outputs).astype(np.float64)


def infer_log_temperature(
    module: ContextualTemperatureAdapter,
    hidden: np.ndarray,
    batch_size: int = 2048,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = module.to(device).eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(hidden), batch_size):
            value = torch.from_numpy(hidden[start : start + batch_size]).float().to(device)
            outputs.append(module.log_temperature(value).float().cpu().numpy())
    module.cpu()
    return np.concatenate(outputs).astype(np.float64)


def softmax_confidence(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probability = np.exp(shifted)
    probability /= np.sum(probability, axis=1, keepdims=True)
    return np.max(probability, axis=1)


def learned_response_codes(
    parent_logits: np.ndarray,
    adapter_log_temperatures: np.ndarray,
    thresholds: Iterable[float],
    bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if adapter_log_temperatures.shape != (len(parent_logits), 4):
        raise ValueError(adapter_log_temperatures.shape)
    hard = np.argmax(parent_logits, axis=1).astype(np.int64)
    codes: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    triggers: list[np.ndarray] = []
    for alias, threshold in enumerate(thresholds):
        scaled = parent_logits * np.exp(adapter_log_temperatures[:, [alias]])
        adapted_hard = np.argmax(scaled, axis=1).astype(np.int64)
        if not np.array_equal(adapted_hard, hard):
            raise AssertionError("positive contextual temperature changed a hard label")
        confidence = softmax_confidence(scaled)
        trigger = confidence >= float(threshold)
        bucket = np.minimum(bins - 1, np.floor(confidence * bins).astype(np.int64))
        code = hard.copy()
        code[trigger] = N_CLASSES + hard[trigger] * bins + bucket[trigger]
        codes.append(code)
        confidences.append(confidence)
        triggers.append(trigger)
    return np.stack(codes, axis=1), np.stack(confidences, axis=1), np.stack(triggers, axis=1)


def quantile_thresholds(confidences: np.ndarray, trigger_fraction: float) -> list[float]:
    if confidences.shape[1] != 4:
        raise ValueError(confidences.shape)
    return [
        float(np.quantile(confidences[:, alias], 1.0 - trigger_fraction, method="higher"))
        for alias in range(4)
    ]


def sample_pools(n: int, seeds: tuple[int, ...], pool_size: int = POOL_SIZE) -> np.ndarray:
    return np.asarray(
        [sorted(random.Random(int(seed)).sample(range(n), min(pool_size, n))) for seed in seeds],
        dtype=np.int64,
    )


def choose_query(tied: np.ndarray, pool: np.ndarray, seed: int, step: int) -> int:
    return min(
        (int(position) for position in tied),
        key=lambda position: (
            stable_u64("step92", "query", seed, step, int(pool[position])),
            int(pool[position]),
        ),
    )


def choose_root(tied_entries: np.ndarray, parents: np.ndarray, seed: int, step: int) -> int:
    roots = sorted(set(int(parents[int(entry)]) for entry in tied_entries))
    return min(roots, key=lambda root: (stable_u64("step92", "root", seed, step, root), root))


@dataclass(frozen=True)
class RunSummary:
    terminal: np.ndarray
    cumulative: np.ndarray
    queries: np.ndarray
    roots: np.ndarray


def run_active(
    codes: np.ndarray,
    feedback: np.ndarray,
    parents: np.ndarray,
    core_feedback: np.ndarray,
    pools: np.ndarray,
    seeds: tuple[int, ...],
    budget: int,
    tau: float,
) -> RunSummary:
    runs = len(seeds)
    terminal = np.zeros(runs, dtype=np.float64)
    cumulative = np.zeros(runs, dtype=np.float64)
    queries = np.full((runs, budget), -1, dtype=np.int64)
    roots = np.full((runs, budget), -1, dtype=np.int64)
    for run_index, seed in enumerate(seeds):
        pool = pools[run_index]
        groups = build_group_structure(codes[pool])
        active = np.ones(len(pool), dtype=bool)
        scores = np.zeros(codes.shape[1], dtype=np.float64)
        root_quality = np.mean(core_feedback[pool], axis=0)
        best_quality = float(np.max(root_quality))
        for step in range(budget):
            shifted = scores / tau
            shifted -= float(np.max(shifted))
            posterior = np.exp(shifted)
            posterior /= float(np.sum(posterior))
            acquisition = select_llm_acquisition(groups, posterior)
            acquisition[~active] = np.inf
            tied = np.flatnonzero(acquisition == float(np.min(acquisition)))
            position = choose_query(tied, pool, int(seed), step)
            active[position] = False
            query = int(pool[position])
            queries[run_index, step] = query
            scores += feedback[query]
            root = choose_root(np.flatnonzero(scores == float(np.max(scores))), parents, int(seed), step)
            roots[run_index, step] = root
            regret = best_quality - float(root_quality[root])
            cumulative[run_index] += regret
            if step == budget - 1:
                terminal[run_index] = regret
    return RunSummary(terminal, cumulative, queries, roots)


def run_fixed(
    feedback: np.ndarray,
    parents: np.ndarray,
    core_feedback: np.ndarray,
    pools: np.ndarray,
    queries: np.ndarray,
    seeds: tuple[int, ...],
) -> RunSummary:
    runs, budget = queries.shape
    terminal = np.zeros(runs, dtype=np.float64)
    cumulative = np.zeros(runs, dtype=np.float64)
    roots = np.full((runs, budget), -1, dtype=np.int64)
    for run_index, seed in enumerate(seeds):
        pool = pools[run_index]
        root_quality = np.mean(core_feedback[pool], axis=0)
        best_quality = float(np.max(root_quality))
        scores = np.zeros(feedback.shape[1], dtype=np.float64)
        for step, query in enumerate(queries[run_index]):
            scores += feedback[int(query)]
            root = choose_root(np.flatnonzero(scores == float(np.max(scores))), parents, int(seed), step)
            roots[run_index, step] = root
            regret = best_quality - float(root_quality[root])
            cumulative[run_index] += regret
            if step == budget - 1:
                terminal[run_index] = regret
    return RunSummary(terminal, cumulative, np.asarray(queries), roots)


def bootstrap_interval(values: np.ndarray, seed: int, repetitions: int = 50_000) -> list[float]:
    rng = np.random.default_rng(seed)
    blocks: list[np.ndarray] = []
    remaining = repetitions
    while remaining:
        count = min(2000, remaining)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        blocks.append(np.mean(values[indices], axis=1))
        remaining -= count
    return [float(value) for value in np.quantile(np.concatenate(blocks), [0.025, 0.975])]


def signflip_pvalue(values: np.ndarray, seed: int, repetitions: int = 100_000) -> float:
    observed = float(np.mean(values))
    rng = np.random.default_rng(seed)
    exceed = 0
    remaining = repetitions
    while remaining:
        count = min(2000, remaining)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(count, len(values)))
        exceed += int(np.sum(np.mean(signs * values[None, :], axis=1) >= observed))
        remaining -= count
    return float((exceed + 1) / (repetitions + 1))


def effect_summary(values: np.ndarray, seed: int) -> dict[str, Any]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "bootstrap_95": bootstrap_interval(values, seed),
        "one_sided_signflip_p": signflip_pvalue(values, seed + 1),
        "positive_fraction": float(np.mean(values > 0)),
        "negative_fraction": float(np.mean(values < 0)),
        "zero_fraction": float(np.mean(values == 0)),
    }
