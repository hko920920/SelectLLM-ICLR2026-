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
from scipy import sparse
from torch import nn
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parent
DATE = "2026-08-13"
DATASET_ID = "mteb/banking77"
DATASET_REVISION = "18072d2685ea682290f7b8924d94c62acc19c0b2"
UPSTREAM_DATASET_ID = "PolyAI/banking77"
UPSTREAM_DATASET_REVISION = "90d4e2ee5521c04fc1488f065b8b083658768c57"
BASE_MODEL = "distilbert/distilbert-base-uncased"
BASE_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
BASE_MODEL_SHA256 = "5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063"
SPLIT_SALT = "step93-source-faithful-abstention-v1"
ROOT_FRACTIONS = (0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.00)
ELIGIBLE_PARENTS = (4, 5, 6, 7, 8, 9, 10)
ROOT_SEED_BASE = 93300
ABSTENTION_SEEDS = (93400, 93401, 93402, 93403)
N_CLASSES = 77
N_ROOTS = 12
N_ALIASES = 4
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


def array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def stable_u64(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def train_partition(index: int) -> str:
    residue = stable_u64(SPLIT_SALT, "train", int(index)) % 10
    if residue <= 5:
        return "root_train"
    if residue <= 7:
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


class AbstentionAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(HIDDEN_SIZE)
        self.down = nn.Linear(HIDDEN_SIZE, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.out(torch.nn.functional.gelu(self.down(self.norm(hidden))))[:, 0]


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


def load_abstention(path: Path, device: torch.device) -> AbstentionAdapter:
    model = AbstentionAdapter()
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


def _batch_order(size: int, seed: int, epoch: int, namespace: str) -> np.ndarray:
    rng = np.random.default_rng(stable_u64(namespace, seed, epoch))
    return rng.permutation(size)


def train_classifier(
    hidden: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    seed: int,
    epochs: int = 30,
    batch_size: int = 256,
) -> ClassificationAdapter:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClassificationAdapter().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x = torch.from_numpy(hidden[indices]).float()
    y = torch.from_numpy(labels[indices]).long()
    model.train()
    for epoch in range(epochs):
        epoch_order = _batch_order(len(indices), seed, epoch, "step93-root-batch")
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


def train_abstention_adapter(
    hidden: np.ndarray,
    parent_predictions: np.ndarray,
    labels: np.ndarray,
    seed: int,
    epochs: int = 40,
    batch_size: int = 256,
) -> AbstentionAdapter:
    if not (len(hidden) == len(parent_predictions) == len(labels)):
        raise ValueError("abstention-training arrays have inconsistent lengths")
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AbstentionAdapter().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    targets_all = (parent_predictions != labels).astype(np.float32)
    rng = np.random.default_rng(stable_u64("step93-abstention-bootstrap", seed))
    bootstrap = rng.integers(0, len(labels), size=len(labels), dtype=np.int64)
    x = torch.from_numpy(hidden[bootstrap]).float()
    y = torch.from_numpy(targets_all[bootstrap]).float()
    positives = float(torch.sum(y).item())
    negatives = float(len(y) - positives)
    if positives <= 0 or negatives <= 0:
        raise AssertionError({"seed": seed, "positives": positives, "negatives": negatives})
    positive_weight = torch.tensor(negatives / positives, dtype=torch.float32, device=device)
    model.train()
    for epoch in range(epochs):
        epoch_order = _batch_order(len(bootstrap), seed, epoch, "step93-abstention-batch")
        for start in range(0, len(bootstrap), batch_size):
            order = torch.from_numpy(epoch_order[start : start + batch_size]).long()
            xb = x[order].to(device)
            yb = y[order].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, yb, pos_weight=positive_weight
            )
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


def infer_error_score(module: AbstentionAdapter, hidden: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = module.to(device).eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(hidden), batch_size):
            value = torch.from_numpy(hidden[start : start + batch_size]).float().to(device)
            outputs.append(torch.sigmoid(module(value)).float().cpu().numpy())
    module.cpu()
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
    while (
        feasible_k > 0
        and feasible_k < n
        and scores[order[feasible_k - 1]] == scores[order[feasible_k]]
    ):
        feasible_k -= 1
    if feasible_k == 0:
        threshold = float(np.nextafter(np.max(scores), np.inf))
    elif feasible_k == n:
        threshold = float(np.nextafter(np.min(scores), -np.inf))
    else:
        upper = float(scores[order[feasible_k - 1]])
        lower = float(scores[order[feasible_k]])
        if not upper > lower:
            raise AssertionError((upper, lower, feasible_k))
        threshold = float(lower + (upper - lower) / 2.0)
    trigger = scores > threshold
    realized_loss = float(np.mean(trigger & parent_correct))
    realized_trigger = float(np.mean(trigger))
    if realized_loss > float(loss_cap) + 1e-15 or realized_trigger > float(trigger_cap) + 1e-15:
        raise AssertionError(
            {
                "loss_cap": loss_cap,
                "realized_loss": realized_loss,
                "trigger_cap": trigger_cap,
                "realized_trigger": realized_trigger,
            }
        )
    return threshold, trigger, {
        "threshold": threshold,
        "trigger_count": int(np.sum(trigger)),
        "trigger_fraction": realized_trigger,
        "utility_loss_count": int(np.sum(trigger & parent_correct)),
        "utility_loss": realized_loss,
        "mean_score_triggered": float(np.mean(scores[trigger])) if np.any(trigger) else None,
        "mean_score_untriggered": float(np.mean(scores[~trigger])) if np.any(~trigger) else None,
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
    if len(thresholds) != N_ALIASES:
        raise ValueError(len(thresholds))
    triggers = error_scores > np.asarray(thresholds, dtype=np.float64)[None, :]
    codes = np.repeat(parent_predictions[:, None], N_ALIASES, axis=1)
    for alias in range(N_ALIASES):
        codes[triggers[:, alias], alias] = N_CLASSES + alias
    return codes.astype(np.int64), triggers


# This is deliberately the one and only response similarity used by the
# source-faithful Step 93 selector. Both acquisition and posterior feedback call
# this exact function; run_active/run_fixed accept references, never a separate
# feedback or utility matrix.
def exact_match_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.equal(left, right)


def exact_match_feedback(codes: np.ndarray, references: np.ndarray) -> np.ndarray:
    return exact_match_similarity(codes, references[:, None]).astype(np.float64)


def exact_match_agreement(codes: np.ndarray) -> np.ndarray:
    return exact_match_similarity(codes[:, :, None], codes[:, None, :]).astype(np.float64)


@dataclass(frozen=True)
class ExactGroupStructure:
    membership: sparse.csr_matrix
    group_query: np.ndarray
    pool_size: int


def build_source_faithful_groups(codes: np.ndarray) -> ExactGroupStructure:
    codes = np.asarray(codes, dtype=np.int64)
    pool_size, entries = codes.shape
    group_indices = np.empty((pool_size, entries), dtype=np.int64)
    group_query: list[int] = []
    offset = 0
    for query in range(pool_size):
        representatives: list[int] = []
        for entry in range(entries):
            value = int(codes[query, entry])
            if representatives:
                equal = exact_match_similarity(np.asarray(representatives), np.asarray(value))
                matches = np.flatnonzero(equal)
            else:
                matches = np.empty(0, dtype=np.int64)
            if len(matches):
                group = int(matches[0])
            else:
                group = len(representatives)
                representatives.append(value)
                group_query.append(query)
            group_indices[query, entry] = offset + group
        offset += len(representatives)
    rows = group_indices.reshape(-1)
    columns = np.tile(np.arange(entries, dtype=np.int64), pool_size)
    membership = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, columns)),
        shape=(offset, entries),
    )
    return ExactGroupStructure(
        membership=membership,
        group_query=np.asarray(group_query, dtype=np.int64),
        pool_size=pool_size,
    )


def select_source_faithful_acquisition(
    groups: ExactGroupStructure, posterior: np.ndarray
) -> np.ndarray:
    mass = np.asarray(groups.membership @ posterior).reshape(-1)
    return np.bincount(
        groups.group_query,
        weights=mass * mass,
        minlength=groups.pool_size,
    )


def sample_pools(n: int, seeds: tuple[int, ...], pool_size: int = POOL_SIZE) -> np.ndarray:
    return np.asarray(
        [sorted(random.Random(int(seed)).sample(range(n), min(pool_size, n))) for seed in seeds],
        dtype=np.int64,
    )


def choose_query(tied: np.ndarray, pool: np.ndarray, seed: int, step: int) -> int:
    return min(
        (int(position) for position in tied),
        key=lambda position: (
            stable_u64("step93", "query", seed, step, int(pool[position])),
            int(pool[position]),
        ),
    )


def choose_root(tied_entries: np.ndarray, parents: np.ndarray, seed: int, step: int) -> int:
    roots = sorted(set(int(parents[int(entry)]) for entry in tied_entries))
    return min(roots, key=lambda root: (stable_u64("step93", "root", seed, step, root), root))


@dataclass(frozen=True)
class RunSummary:
    terminal: np.ndarray
    cumulative: np.ndarray
    queries: np.ndarray
    roots: np.ndarray


def run_active(
    codes: np.ndarray,
    references: np.ndarray,
    parents: np.ndarray,
    core_codes: np.ndarray,
    pools: np.ndarray,
    seeds: tuple[int, ...],
    budget: int,
    tau: float,
) -> RunSummary:
    codes = np.asarray(codes, dtype=np.int64)
    references = np.asarray(references, dtype=np.int64)
    core_codes = np.asarray(core_codes, dtype=np.int64)
    feedback = exact_match_feedback(codes, references)
    core_feedback = exact_match_feedback(core_codes, references)
    runs = len(seeds)
    terminal = np.zeros(runs, dtype=np.float64)
    cumulative = np.zeros(runs, dtype=np.float64)
    queries = np.full((runs, budget), -1, dtype=np.int64)
    roots = np.full((runs, budget), -1, dtype=np.int64)
    for run_index, seed in enumerate(seeds):
        pool = pools[run_index]
        groups = build_source_faithful_groups(codes[pool])
        active = np.ones(len(pool), dtype=bool)
        scores = np.zeros(codes.shape[1], dtype=np.float64)
        root_quality = np.mean(core_feedback[pool], axis=0)
        best_quality = float(np.max(root_quality))
        for step in range(budget):
            shifted = scores / float(tau)
            shifted -= float(np.max(shifted))
            posterior = np.exp(shifted)
            posterior /= float(np.sum(posterior))
            acquisition = select_source_faithful_acquisition(groups, posterior)
            acquisition[~active] = np.inf
            tied = np.flatnonzero(acquisition == float(np.min(acquisition)))
            position = choose_query(tied, pool, int(seed), step)
            active[position] = False
            query = int(pool[position])
            queries[run_index, step] = query
            scores += feedback[query]
            tied_entries = np.flatnonzero(scores == float(np.max(scores)))
            root = choose_root(tied_entries, parents, int(seed), step)
            roots[run_index, step] = root
            regret = best_quality - float(root_quality[root])
            cumulative[run_index] += regret
            if step == budget - 1:
                terminal[run_index] = regret
    return RunSummary(terminal, cumulative, queries, roots)


def run_fixed(
    codes: np.ndarray,
    references: np.ndarray,
    parents: np.ndarray,
    core_codes: np.ndarray,
    pools: np.ndarray,
    queries: np.ndarray,
    seeds: tuple[int, ...],
) -> RunSummary:
    codes = np.asarray(codes, dtype=np.int64)
    references = np.asarray(references, dtype=np.int64)
    core_codes = np.asarray(core_codes, dtype=np.int64)
    feedback = exact_match_feedback(codes, references)
    core_feedback = exact_match_feedback(core_codes, references)
    runs, budget = queries.shape
    terminal = np.zeros(runs, dtype=np.float64)
    cumulative = np.zeros(runs, dtype=np.float64)
    roots = np.full((runs, budget), -1, dtype=np.int64)
    for run_index, seed in enumerate(seeds):
        pool = pools[run_index]
        root_quality = np.mean(core_feedback[pool], axis=0)
        best_quality = float(np.max(root_quality))
        scores = np.zeros(codes.shape[1], dtype=np.float64)
        for step, query in enumerate(queries[run_index]):
            scores += feedback[int(query)]
            tied_entries = np.flatnonzero(scores == float(np.max(scores)))
            root = choose_root(tied_entries, parents, int(seed), step)
            roots[run_index, step] = root
            regret = best_quality - float(root_quality[root])
            cumulative[run_index] += regret
            if step == budget - 1:
                terminal[run_index] = regret
    return RunSummary(terminal, cumulative, np.asarray(queries), roots)


def bootstrap_interval(values: np.ndarray, seed: int, repetitions: int = 10_000) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    blocks: list[np.ndarray] = []
    remaining = repetitions
    while remaining:
        count = min(1000, remaining)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        blocks.append(np.mean(values[indices], axis=1))
        remaining -= count
    return [float(value) for value in np.quantile(np.concatenate(blocks), [0.025, 0.975])]


def signflip_pvalue(values: np.ndarray, seed: int, repetitions: int = 100_000) -> float:
    values = np.asarray(values, dtype=np.float64)
    observed = float(np.mean(values))
    rng = np.random.default_rng(seed)
    exceed = 0
    remaining = repetitions
    while remaining:
        count = min(1000, remaining)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(count, len(values)))
        exceed += int(np.sum(np.mean(signs * values[None, :], axis=1) >= observed))
        remaining -= count
    return float((exceed + 1) / (repetitions + 1))


def effect_summary(values: np.ndarray, bootstrap_seed: int, signflip_seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "bootstrap_95": bootstrap_interval(values, bootstrap_seed, 10_000),
        "one_sided_signflip_p": signflip_pvalue(values, signflip_seed, 100_000),
        "positive_fraction": float(np.mean(values > 0)),
        "negative_fraction": float(np.mean(values < 0)),
        "zero_fraction": float(np.mean(values == 0)),
    }
