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
from transformers import AutoModel, AutoTokenizer

import step93_common as selector_core


ROOT = Path(__file__).resolve().parent
DATE = "2026-08-13"
PROTOCOL = ROOT / f"STEP94_FRESH_TASK_LEARNED_VARIANT_PREREGISTRATION_{DATE}.md"
AMENDMENT_A = ROOT / f"STEP94_PREREGISTRATION_AMENDMENT_A_{DATE}.md"
BASE_MODEL = "distilbert/distilbert-base-uncased"
BASE_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
BASE_MODEL_SHA256 = "5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063"
SPLIT_SALT = "step94-fresh-learned-variant-v1"
HIDDEN_SIZE = 768
N_BANK_ROOTS = 16
N_ROSTER_ROOTS = 8
N_ALIASES = 4
POOL_SIZE = 500
BUDGET = 50
ROOT_FRACTIONS = (
    0.03,
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
    1.00,
    1.00,
)
GAP_TEMPLATES: dict[str, tuple[float, ...]] = {
    "dense": (0.005, 0.010, 0.020, 0.030, 0.050, 0.080, 0.120),
    "mixed": (0.010, 0.025, 0.050, 0.080, 0.120, 0.180, 0.250),
    "wide": (0.020, 0.050, 0.100, 0.150, 0.220, 0.300, 0.400),
}


@dataclass(frozen=True)
class TaskSpec:
    key: str
    order: int
    repo_id: str
    revision: str
    development_splits: tuple[str, ...]
    test_split: str
    filenames: dict[str, str]
    n_classes: int
    text_column: str = "text"
    label_column: str = "label"
    upstream_id: str | None = None
    upstream_revision: str | None = None


TASKS: tuple[TaskSpec, ...] = (
    TaskSpec(
        key="20newsgroups",
        order=0,
        repo_id="SetFit/20_newsgroups",
        revision="f1b91292074e7cfb69be58b642d583ec262f30ed",
        development_splits=("train",),
        test_split="test",
        filenames={"train": "train.jsonl", "test": "test.jsonl"},
        n_classes=20,
    ),
    TaskSpec(
        key="massive_en_us",
        order=1,
        repo_id="SetFit/amazon_massive_intent_en-US",
        revision="f7672a018e8ceb37fc0184dcfbb7e665155ffea6",
        development_splits=("train", "validation"),
        test_split="test",
        filenames={
            "train": "train.jsonl",
            "validation": "validation.jsonl",
            "test": "test.jsonl",
        },
        n_classes=60,
        upstream_id="AmazonScience/massive",
        upstream_revision="ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740",
    ),
    TaskSpec(
        key="sst5",
        order=2,
        repo_id="SetFit/sst5",
        revision="e51bdcd8cd3a30da231967c1a249ba59361279a3",
        development_splits=("train", "dev"),
        test_split="test",
        filenames={"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"},
        n_classes=5,
    ),
)


sha256_path = selector_core.sha256_path
canonical_json_sha256 = selector_core.canonical_json_sha256
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


def task_by_key(key: str) -> TaskSpec:
    for task in TASKS:
        if task.key == key:
            return task
    raise KeyError(key)


def development_partition(task_key: str, index: int) -> str:
    residue = stable_u64(SPLIT_SALT, task_key, "development", int(index)) % 20
    if residue <= 10:
        return "root_train"
    if residue <= 13:
        return "gate_train"
    if residue <= 16:
        return "stagea_search"
    return "stagea_verification"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ClassificationAdapter(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.n_classes = int(n_classes)
        self.norm = nn.LayerNorm(HIDDEN_SIZE)
        self.down = nn.Linear(HIDDEN_SIZE, 128)
        self.out = nn.Linear(128, self.n_classes)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.out(torch.nn.functional.gelu(self.down(self.norm(hidden))))


class ErrorGate(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.n_classes = int(n_classes)
        width = HIDDEN_SIZE + self.n_classes
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, hidden: torch.Tensor, parent_logits: torch.Tensor) -> torch.Tensor:
        value = torch.cat([hidden, parent_logits], dim=1)
        return self.out(torch.nn.functional.gelu(self.down(self.norm(value))))[:, 0]


class IntegratedVariant(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.n_classes = int(n_classes)
        self.parent = ClassificationAdapter(self.n_classes)
        self.gate = ErrorGate(self.n_classes)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        parent_logits = self.parent(hidden)
        gate_logits = self.gate(hidden, parent_logits)
        return parent_logits, gate_logits


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


def load_classifier(path: Path, n_classes: int, device: torch.device) -> ClassificationAdapter:
    model = ClassificationAdapter(n_classes)
    model.load_state_dict(load_file(str(path)))
    return model.to(device).eval()


def load_gate(path: Path, n_classes: int, device: torch.device) -> ErrorGate:
    model = ErrorGate(n_classes)
    model.load_state_dict(load_file(str(path)))
    return model.to(device).eval()


def load_integrated(path: Path, n_classes: int, device: torch.device) -> IntegratedVariant:
    model = IntegratedVariant(n_classes)
    model.load_state_dict(load_file(str(path)))
    return model.to(device).eval()


def tensor_audit(path: Path) -> dict[str, Any]:
    tensors = load_file(str(path))
    total = int(sum(tensor.numel() for tensor in tensors.values()))
    flattened = torch.cat([tensor.detach().float().reshape(-1) for tensor in tensors.values()])
    nonzero = int(torch.count_nonzero(flattened).item())
    return {
        "sha256": sha256_path(path),
        "tensor_names": sorted(tensors),
        "parameter_count": total,
        "nonzero_count": nonzero,
        "nonzero_fraction": float(nonzero / max(1, total)),
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
    batch_size: int = 96,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=192,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = encoder(**encoded).last_hidden_state[:, 0, :]
            outputs.append(hidden.float().cpu().numpy())
    if not outputs:
        return np.empty((0, HIDDEN_SIZE), dtype=np.float32)
    return np.concatenate(outputs).astype(np.float32, copy=False)


def _batch_order(size: int, seed: int, epoch: int, namespace: str) -> np.ndarray:
    rng = np.random.default_rng(stable_u64(namespace, seed, epoch))
    return rng.permutation(size)


def root_subset_indices(size: int, task_key: str, root: int, fraction: float) -> np.ndarray:
    if root >= 13 and math.isclose(fraction, 1.0):
        rng = np.random.default_rng(stable_u64("step94-root-bootstrap", task_key, root))
        return rng.integers(0, size, size=size, dtype=np.int64)
    keys = np.asarray(
        [stable_u64("step94-root-order", task_key, root, index) for index in range(size)],
        dtype=np.uint64,
    )
    order = np.argsort(keys, kind="stable")
    count = max(1, int(math.ceil(float(fraction) * size)))
    return order[:count].astype(np.int64)


def train_classifier(
    hidden: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    n_classes: int,
    seed: int,
    epochs: int = 20,
    batch_size: int = 256,
) -> ClassificationAdapter:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClassificationAdapter(n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x = torch.from_numpy(hidden[indices]).float()
    y = torch.from_numpy(labels[indices]).long()
    model.train()
    for epoch in range(epochs):
        order = _batch_order(len(indices), seed, epoch, "step94-root-batch")
        for start in range(0, len(indices), batch_size):
            rows = torch.from_numpy(order[start : start + batch_size]).long()
            xb = x[rows].to(device)
            yb = y[rows].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model.cpu().eval()


def train_error_gate(
    hidden: np.ndarray,
    parent_logits: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    seed: int,
    epochs: int = 30,
    batch_size: int = 256,
) -> ErrorGate:
    if not (len(hidden) == len(parent_logits) == len(labels)):
        raise ValueError("gate-training arrays have inconsistent lengths")
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ErrorGate(n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    targets = (np.argmax(parent_logits, axis=1) != labels).astype(np.float32)
    rng = np.random.default_rng(stable_u64("step94-gate-bootstrap", seed))
    bootstrap = rng.integers(0, len(labels), size=len(labels), dtype=np.int64)
    x = torch.from_numpy(hidden[bootstrap]).float()
    logits = torch.from_numpy(parent_logits[bootstrap]).float()
    y = torch.from_numpy(targets[bootstrap]).float()
    positives = float(torch.sum(y).item())
    negatives = float(len(y) - positives)
    if positives <= 0 or negatives <= 0:
        raise AssertionError({"seed": seed, "positives": positives, "negatives": negatives})
    positive_weight = torch.tensor(negatives / positives, dtype=torch.float32, device=device)
    model.train()
    for epoch in range(epochs):
        order = _batch_order(len(bootstrap), seed, epoch, "step94-gate-batch")
        for start in range(0, len(bootstrap), batch_size):
            rows = torch.from_numpy(order[start : start + batch_size]).long()
            xb = x[rows].to(device)
            lb = logits[rows].to(device)
            yb = y[rows].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(xb, lb)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                output, yb, pos_weight=positive_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model.cpu().eval()


def infer_classifier(
    module: ClassificationAdapter, hidden: np.ndarray, batch_size: int = 2048
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = module.to(device).eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(hidden), batch_size):
            value = torch.from_numpy(hidden[start : start + batch_size]).float().to(device)
            outputs.append(module(value).float().cpu().numpy())
    module.cpu()
    return np.concatenate(outputs).astype(np.float64)


def infer_gate_scores(
    module: ErrorGate,
    hidden: np.ndarray,
    parent_logits: np.ndarray,
    batch_size: int = 2048,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = module.to(device).eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(hidden), batch_size):
            hb = torch.from_numpy(hidden[start : start + batch_size]).float().to(device)
            lb = torch.from_numpy(parent_logits[start : start + batch_size]).float().to(device)
            outputs.append(torch.sigmoid(module(hb, lb)).float().cpu().numpy())
    module.cpu()
    return np.concatenate(outputs).astype(np.float64)


def make_integrated_variant(
    parent: ClassificationAdapter,
    gate: ErrorGate,
    n_classes: int,
) -> IntegratedVariant:
    model = IntegratedVariant(n_classes)
    model.parent.load_state_dict(parent.state_dict())
    model.gate.load_state_dict(gate.state_dict())
    return model.eval()


def infer_integrated_codes(
    model: IntegratedVariant,
    hidden: np.ndarray,
    threshold: float,
    alias_id: int,
    batch_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    codes: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(hidden), batch_size):
            hb = torch.from_numpy(hidden[start : start + batch_size]).float().to(device)
            parent_logits, gate_logits = model(hb)
            prediction = torch.argmax(parent_logits, dim=1)
            score = torch.sigmoid(gate_logits)
            code = prediction.clone()
            code[score > float(threshold)] = int(model.n_classes + alias_id)
            codes.append(code.cpu().numpy())
            scores.append(score.float().cpu().numpy())
    model.cpu()
    return np.concatenate(codes).astype(np.int64), np.concatenate(scores).astype(np.float64)


def threshold_from_caps(
    scores: np.ndarray,
    parent_correct: np.ndarray,
    loss_cap: float,
    trigger_cap: float,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    return selector_core.threshold_from_caps(scores, parent_correct, loss_cap, trigger_cap)


def make_alias_codes(
    parent_predictions: np.ndarray,
    gate_scores: np.ndarray,
    thresholds: list[float] | tuple[float, ...],
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    parent_predictions = np.asarray(parent_predictions, dtype=np.int64)
    gate_scores = np.asarray(gate_scores, dtype=np.float64)
    if gate_scores.shape != (len(parent_predictions), N_ALIASES):
        raise ValueError(gate_scores.shape)
    triggers = gate_scores > np.asarray(thresholds, dtype=np.float64)[None, :]
    codes = np.repeat(parent_predictions[:, None], N_ALIASES, axis=1)
    for alias in range(N_ALIASES):
        codes[triggers[:, alias], alias] = int(n_classes + alias)
    return codes.astype(np.int64), triggers


def select_roster(
    search_accuracies: np.ndarray,
    template_name: str,
) -> tuple[int, ...]:
    values = np.asarray(search_accuracies, dtype=np.float64)
    if values.shape != (N_BANK_ROOTS,):
        raise ValueError(values.shape)
    best = min(
        range(N_BANK_ROOTS),
        key=lambda root: (-float(values[root]), int(root)),
    )
    selected = [int(best)]
    best_accuracy = float(values[best])
    for gap in GAP_TEMPLATES[template_name]:
        target = best_accuracy - float(gap)
        eligible = [root for root in range(N_BANK_ROOTS) if root not in selected]
        chosen = min(
            eligible,
            key=lambda root: (abs(float(values[root]) - target), int(root)),
        )
        selected.append(int(chosen))
    if len(set(selected)) != N_ROSTER_ROOTS:
        raise AssertionError(selected)
    return tuple(selected)


def prepare_construction(
    roster_predictions: np.ndarray,
    labels: np.ndarray,
    parent_position: int,
    gate_scores: np.ndarray,
    loss_cap: float,
    trigger_cap: float,
    n_classes: int,
) -> dict[str, Any]:
    parent_predictions = roster_predictions[:, parent_position].astype(np.int64)
    parent_correct = parent_predictions == labels
    thresholds: list[float] = []
    audits: list[dict[str, Any]] = []
    expected: list[np.ndarray] = []
    for alias in range(N_ALIASES):
        threshold, trigger, audit = threshold_from_caps(
            gate_scores[:, alias], parent_correct, loss_cap, trigger_cap
        )
        thresholds.append(float(threshold))
        expected.append(trigger)
        audits.append({"alias": alias, **audit})
    alias_codes, triggers = make_alias_codes(
        parent_predictions, gate_scores, thresholds, n_classes
    )
    if not np.array_equal(triggers, np.stack(expected, axis=1)):
        raise AssertionError("threshold replay mismatch")
    parent_feedback = exact_match_feedback(parent_predictions[:, None], labels)
    alias_feedback = exact_match_feedback(alias_codes, labels)
    losses = np.mean(parent_feedback, axis=0)[0] - np.mean(alias_feedback, axis=0)
    pairwise = [
        float(np.mean(alias_codes[:, left] != alias_codes[:, right]))
        for left in range(N_ALIASES)
        for right in range(left + 1, N_ALIASES)
    ]
    return {
        "thresholds": thresholds,
        "threshold_audits": audits,
        "alias_codes": alias_codes,
        "triggers": triggers,
        "alias_utility_losses": np.asarray(losses, dtype=np.float64).tolist(),
        "realized_trigger_fractions": np.mean(triggers, axis=0).tolist(),
        "coordinate_wise_nonimproving": bool(
            np.all(alias_feedback <= np.repeat(parent_feedback, N_ALIASES, axis=1))
        ),
        "mean_pairwise_alias_response_hamming": float(np.mean(pairwise)),
        "minimum_pairwise_alias_response_hamming": float(np.min(pairwise)),
    }


def json_dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
