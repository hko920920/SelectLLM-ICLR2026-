from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from scipy.stats import rankdata
from torch import nn


class Stage3Error(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(b"|")
    digest.update(json.dumps(array.shape).encode("utf-8"))
    digest.update(b"|")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def digest_lines(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(str(line).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def stable_u64(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


class LinearErrorHead(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.linear = nn.Linear(dimension, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear(values).squeeze(-1)


def train_head(
    features: np.ndarray,
    targets: np.ndarray,
    seed: int,
    *,
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    values = np.asarray(features, dtype=np.float32)
    labels = np.asarray(targets, dtype=np.int64)
    if values.ndim != 2 or labels.shape != (len(values),):
        raise Stage3Error((values.shape, labels.shape))
    if not np.all(np.isfinite(values)):
        raise Stage3Error("non-finite training features")
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    if len(positives) < 25 or len(negatives) < 25:
        raise Stage3Error(
            f"insufficient parent-error classes: positive={len(positives)}, negative={len(negatives)}"
        )

    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    normalized = ((values - mean) / scale).astype(np.float32)

    rng = np.random.default_rng(seed)
    size = len(positives)
    bootstrap = np.concatenate(
        [
            rng.choice(positives, size=size, replace=True),
            rng.choice(negatives, size=size, replace=True),
        ]
    ).astype(np.int64)
    rng.shuffle(bootstrap)

    seed_everything(seed)
    model = LinearErrorHead(normalized.shape[1]).cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = nn.BCEWithLogitsLoss()
    losses: list[float] = []
    for epoch in range(epochs):
        order_rng = np.random.default_rng(
            stable_u64("PRAA-stage3-head-epoch", seed, epoch)
        )
        order = bootstrap[order_rng.permutation(len(bootstrap))]
        model.train()
        total = 0.0
        seen = 0
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            x = torch.from_numpy(normalized[batch])
            y = torch.from_numpy(labels[batch].astype(np.float32))
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
            seen += len(batch)
        losses.append(total / seen)

    state = {
        "linear.weight": model.linear.weight.detach().cpu().contiguous(),
        "linear.bias": model.linear.bias.detach().cpu().contiguous(),
        "feature_mean": torch.from_numpy(mean).contiguous(),
        "feature_scale": torch.from_numpy(scale).contiguous(),
    }
    audit = {
        "seed": int(seed),
        "feature_dimension": int(values.shape[1]),
        "train_rows": int(len(values)),
        "parent_error_rows": int(len(positives)),
        "parent_correct_rows": int(len(negatives)),
        "balanced_bootstrap_rows": int(len(bootstrap)),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "parameter_count": int(model.linear.weight.numel() + model.linear.bias.numel()),
    }
    return state, audit


def save_head(
    state: dict[str, torch.Tensor],
    path: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        state,
        str(path),
        metadata={"praa_stage3": json.dumps(metadata, sort_keys=True)},
    )
    if not path.is_file() or path.stat().st_size <= 0:
        raise Stage3Error(f"empty head checkpoint {path}")
    return {
        "path": path.name,
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "parameter_count": int(metadata["parameter_count"]),
    }


def head_scores(features: np.ndarray, checkpoint: Path) -> np.ndarray:
    state = load_file(str(checkpoint), device="cpu")
    required = {"linear.weight", "linear.bias", "feature_mean", "feature_scale"}
    if set(state) != required:
        raise Stage3Error(f"checkpoint keys drift: {sorted(state)}")
    values = torch.from_numpy(np.asarray(features, dtype=np.float32))
    normalized = (values - state["feature_mean"]) / state["feature_scale"]
    logits = normalized @ state["linear.weight"].T + state["linear.bias"]
    scores = torch.sigmoid(logits[:, 0]).numpy().astype(np.float64)
    if not np.all(np.isfinite(scores)):
        raise Stage3Error("non-finite head scores")
    return scores


def threshold_higher(scores: np.ndarray, quantile: float = 0.993) -> float:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise Stage3Error("invalid threshold scores")
    return float(np.quantile(values, quantile, method="higher"))


def make_aliases(
    parent_predictions: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    parent = np.asarray(parent_predictions, dtype=np.int16)
    matrix = np.asarray(scores, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if matrix.shape != (len(parent), 4) or thresholds.shape != (4,):
        raise Stage3Error((parent.shape, matrix.shape, thresholds.shape))
    trigger = matrix >= thresholds[None, :]
    aliases = np.repeat(parent[:, None], 4, axis=1)
    for index in range(4):
        aliases[trigger[:, index], index] = n_classes + index
    return aliases.astype(np.int16), trigger


def quality_audit(
    parent_predictions: np.ndarray,
    aliases: np.ndarray,
    references: np.ndarray,
) -> dict[str, Any]:
    parent = np.asarray(parent_predictions, dtype=np.int16)
    variants = np.asarray(aliases, dtype=np.int16)
    labels = np.asarray(references, dtype=np.int16)
    if variants.shape != (len(parent), 4) or labels.shape != parent.shape:
        raise Stage3Error((parent.shape, variants.shape, labels.shape))
    parent_correct = parent == labels
    alias_correct = variants == labels[:, None]
    improvements = np.sum(alias_correct & ~parent_correct[:, None], axis=0)
    losses = np.sum(parent_correct[:, None] & ~alias_correct, axis=0)
    return {
        "parent_accuracy": float(np.mean(parent_correct)),
        "alias_accuracy": np.mean(alias_correct, axis=0).tolist(),
        "improvement_counts": improvements.astype(int).tolist(),
        "loss_counts": losses.astype(int).tolist(),
        "coordinate_wise_nonimproving": bool(np.all(improvements == 0)),
        "loss_within_one_point": bool(
            np.all(losses <= int(np.floor(0.01 * len(labels))))
        ),
    }


def binary_auc(targets: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(targets, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(values, method="average")
    return float(
        (np.sum(ranks[labels == 1]) - positives * (positives + 1) / 2)
        / (positives * negatives)
    )
