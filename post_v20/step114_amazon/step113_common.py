from __future__ import annotations

import gc
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from scipy.stats import rankdata
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import step93_common as selector


ROOT = Path(__file__).resolve().parent
DATE = "2026-08-14"
PROTOCOL = ROOT / f"STEP113_BOOLQ_POSTREVIEW_PROSPECTIVE_PROTOCOL_{DATE}.md"
METADATA = ROOT / f"STEP113_PUBLIC_METADATA_SNAPSHOT_{DATE}.json"
PREDATA_LOCK = ROOT / f"STEP113_PREDATA_LOCK_{DATE}.json"
STAGE0_MANIFEST = ROOT / f"STEP113_STAGE0_DATA_AND_SEAL_MANIFEST_{DATE}.json"
DEVELOPMENT_LEDGER = ROOT / f"STEP113_DEVELOPMENT_LEDGER_{DATE}.json"
DEVELOPMENT_ARRAYS = ROOT / f"STEP113_DEVELOPMENT_ARRAYS_{DATE}.npz"
DEVELOPMENT_GO_LOCK = ROOT / f"STEP113_DEVELOPMENT_GO_LOCK_{DATE}.json"
PREOUTCOME_LEDGER = ROOT / f"STEP113_PREOUTCOME_PREDICTION_LEDGER_{DATE}.json"
PREOUTCOME_PREDICTIONS = ROOT / f"STEP113_PREOUTCOME_PREDICTIONS_{DATE}.npz"
PREOUTCOME_LOCK = ROOT / f"STEP113_PREOUTCOME_LOCK_{DATE}.json"
PRIMARY_LEDGER = ROOT / f"STEP113_PRIMARY_CONFIRMATORY_LEDGER_{DATE}.json"
PRIMARY_ARRAYS = ROOT / f"STEP113_PRIMARY_CONFIRMATORY_ARRAYS_{DATE}.npz"
INDEPENDENT_VALIDATION = ROOT / f"STEP113_PRIMARY_INDEPENDENT_VALIDATION_{DATE}.json"
MODEL_DIR = ROOT / "step113_models"

DATASET_ID = "google/boolq"
DATASET_REVISION = "35b264d03638db9f4ce671b711558bf7ff0f80d5"
TRAIN_BLOB = "data/train-00000-of-00001.parquet"
VALIDATION_BLOB = "data/validation-00000-of-00001.parquet"
TRAIN_BYTES = 3_685_146
VALIDATION_BYTES = 1_257_630
TRAIN_ROWS = 9_427
OUTCOME_ROWS = 3_270
TRAIN_ROWS_HEAD = 6_599
TRAIN_ROWS_THRESHOLD = 1_414
TRAIN_ROWS_SAFETY = 1_414
TRAIN_SPLIT_SALT = "step113-boolq-train-partition-v1"
HALF_SPLIT_SALT = "step113-boolq-diagnostic-halves-v1"
MAX_LENGTH = 256
N_CLASSES = 2
N_ROOTS = 4
N_ALIASES = 4
HEAD_SEEDS = (113_100, 113_101, 113_102, 113_103)
HEAD_EPOCHS = 30
HEAD_BATCH_SIZE = 256
HEAD_LEARNING_RATE = 1e-3
HEAD_WEIGHT_DECAY = 1e-3
THRESHOLD_QUANTILE = 0.99
SAFETY_TRIGGER_MAX = 0.02
QUALITY_LOSS_MAX = 0.01
POOL_SIZE = 500
BUDGET = 2
TAU = 0.05
TEST_SEEDS = tuple(range(1_213_000, 1_216_000))
BOOTSTRAP_SEEDS = (113_800, 113_801, 113_802)
SIGNFLIP_SEEDS = (113_900, 113_901, 113_902)


@dataclass(frozen=True)
class CandidateSpec:
    key: str
    repo_id: str
    revision: str
    batch_size: int
    safe_weights: bool
    parent: bool = False


CANDIDATES: tuple[CandidateSpec, ...] = (
    CandidateSpec(
        "nfliu_deberta_v3_large",
        "nfliu/deberta-v3-large_boolq",
        "9df67c219a86b2611c177f288b8ccc82d7b97707",
        4,
        True,
        parent=True,
    ),
    CandidateSpec(
        "nfliu_roberta_large",
        "nfliu/roberta-large_boolq",
        "efc939f590968d9b5055127d4aeb8a930ffa0826",
        8,
        False,
    ),
    CandidateSpec(
        "nfliu_minilmv2_l6_h768",
        "nfliu/MiniLMv2-L6-H768-distilled-from-RoBERTa-Large_boolq",
        "f31b1a94395e97a0472054793e5236fd8b7b2416",
        32,
        True,
    ),
    CandidateSpec(
        "andi611_distilbert",
        "andi611/distilbert-base-uncased-qa-boolq",
        "168ef0953f2bf4083670a8519a14db558a6f043c",
        32,
        False,
    ),
)
PARENT = CANDIDATES[0]

sha256_path = selector.sha256_path
array_sha256 = selector.array_sha256
canonical_json_sha256 = selector.canonical_json_sha256
sample_pools = selector.sample_pools
run_active = selector.run_active
run_fixed = selector.run_fixed
effect_summary = selector.effect_summary
exact_match_feedback = selector.exact_match_feedback


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def uid(split: str, index: int, question: str, passage: str) -> str:
    payload = "\x1f".join(
        (DATASET_REVISION, split, str(index), question, passage)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_train(uids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(uids) != TRAIN_ROWS or len(set(uids)) != TRAIN_ROWS:
        raise AssertionError("invalid BoolQ train UID vector")
    order = sorted(
        range(TRAIN_ROWS),
        key=lambda index: (
            hashlib.sha256(
                f"{TRAIN_SPLIT_SALT}\x1f{uids[index]}".encode("utf-8")
            ).hexdigest(),
            index,
        ),
    )
    first = TRAIN_ROWS_HEAD
    second = first + TRAIN_ROWS_THRESHOLD
    if second + TRAIN_ROWS_SAFETY != TRAIN_ROWS:
        raise AssertionError("partition constants do not sum to train rows")
    return (
        np.asarray(sorted(order[:first]), dtype=np.int64),
        np.asarray(sorted(order[first:second]), dtype=np.int64),
        np.asarray(sorted(order[second:]), dtype=np.int64),
    )


def diagnostic_halves(uids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    order = sorted(
        range(len(uids)),
        key=lambda index: (
            hashlib.sha256(
                f"{HALF_SPLIT_SALT}\x1f{uids[index]}".encode("utf-8")
            ).hexdigest(),
            index,
        ),
    )
    middle = len(order) // 2
    return (
        np.asarray(sorted(order[:middle]), dtype=np.int64),
        np.asarray(sorted(order[middle:]), dtype=np.int64),
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _model_label_audit(model: nn.Module, spec: CandidateSpec) -> dict[str, str]:
    labels = {
        str(key): str(value) for key, value in dict(model.config.id2label).items()
    }
    normalized = {key: value.lower() for key, value in labels.items()}
    if normalized != {"0": "false", "1": "true"}:
        raise AssertionError({"candidate": spec.key, "id2label": labels})
    if int(model.config.num_labels) != N_CLASSES:
        raise AssertionError({"candidate": spec.key, "num_labels": model.config.num_labels})
    return labels


def infer_candidate_pairs(
    spec: CandidateSpec,
    questions: list[str],
    passages: list[str],
    *,
    return_features: bool = False,
    local_files_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
    if len(questions) != len(passages):
        raise ValueError("question/passage length mismatch")
    source = spec.repo_id
    revision = spec.revision
    tokenizer = AutoTokenizer.from_pretrained(
        source, revision=revision, local_files_only=local_files_only
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs: dict[str, Any] = {
        "revision": revision,
        "local_files_only": local_files_only,
        "use_safetensors": spec.safe_weights,
    }
    if device.type == "cuda":
        kwargs["torch_dtype"] = torch.float16
    model = AutoModelForSequenceClassification.from_pretrained(source, **kwargs)
    labels = _model_label_audit(model, spec)
    model.to(device).eval()
    prediction_parts: list[np.ndarray] = []
    logits_parts: list[np.ndarray] = []
    feature_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(questions), spec.batch_size):
            stop = start + spec.batch_size
            encoded = tokenizer(
                questions[start:stop],
                passages[start:stop],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded, output_hidden_states=return_features)
            logits = output.logits.float().cpu().numpy().astype(np.float32)
            logits_parts.append(logits)
            prediction_parts.append(np.argmax(logits, axis=1).astype(np.int64))
            if return_features:
                hidden = output.hidden_states[-1][:, 0, :].float().cpu().numpy()
                feature_parts.append(
                    np.concatenate([hidden, logits], axis=1).astype(np.float32)
                )
    predictions = np.concatenate(prediction_parts)
    logits = np.concatenate(logits_parts)
    features = np.concatenate(feature_parts) if return_features else None
    audit = {
        "key": spec.key,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "model_class": model.__class__.__name__,
        "tokenizer_class": tokenizer.__class__.__name__,
        "model_type": str(model.config.model_type),
        "id2label": labels,
        "max_length": MAX_LENGTH,
        "samples": len(predictions),
        "prediction_sha256": array_sha256(predictions),
        "logits_sha256": array_sha256(logits),
        "features_sha256": array_sha256(features) if features is not None else None,
        "feature_dimension": int(features.shape[1]) if features is not None else None,
    }
    model.cpu()
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions, logits, features, audit


class LinearErrorHead(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.linear = nn.Linear(dimension, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear(values).squeeze(-1)


def train_error_head(
    features: np.ndarray, targets: np.ndarray, seed: int
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    values = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int64)
    if len(values) != len(targets):
        raise ValueError("feature/target length mismatch")
    positives = np.flatnonzero(targets == 1)
    negatives = np.flatnonzero(targets == 0)
    if len(positives) < 25 or len(negatives) < 25:
        raise AssertionError({"positive": len(positives), "negative": len(negatives)})
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    normalized = ((values - mean) / scale).astype(np.float32)
    rng = np.random.default_rng(seed)
    count = len(positives)
    bootstrap = np.concatenate(
        [
            rng.choice(positives, size=count, replace=True),
            rng.choice(negatives, size=count, replace=True),
        ]
    ).astype(np.int64)
    rng.shuffle(bootstrap)
    seed_everything(seed)
    # The head is a single linear layer; CPU training avoids small-kernel GPU
    # launch overhead and is bitwise stable across the four bootstraps.
    device = torch.device("cpu")
    model = LinearErrorHead(normalized.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=HEAD_LEARNING_RATE, weight_decay=HEAD_WEIGHT_DECAY
    )
    loss_fn = nn.BCEWithLogitsLoss()
    losses: list[float] = []
    for epoch in range(HEAD_EPOCHS):
        epoch_rng = np.random.default_rng(
            int.from_bytes(
                hashlib.sha256(
                    f"step113-error-head-epoch\x1f{seed}\x1f{epoch}".encode("utf-8")
                ).digest()[:8],
                "big",
            )
        )
        order = bootstrap[epoch_rng.permutation(len(bootstrap))]
        total = 0.0
        seen = 0
        model.train()
        for start in range(0, len(order), HEAD_BATCH_SIZE):
            batch = order[start : start + HEAD_BATCH_SIZE]
            x = torch.from_numpy(normalized[batch]).to(device)
            y = torch.from_numpy(targets[batch].astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(batch)
            seen += len(batch)
        losses.append(total / seen)
    state = {
        "linear.weight": model.linear.weight.detach().cpu().contiguous(),
        "linear.bias": model.linear.bias.detach().cpu().contiguous(),
        "feature_mean": torch.from_numpy(mean).contiguous(),
        "feature_scale": torch.from_numpy(scale).contiguous(),
    }
    audit = {
        "seed": seed,
        "feature_dimension": int(values.shape[1]),
        "train_rows": len(values),
        "parent_error_rows": len(positives),
        "balanced_bootstrap_rows": len(bootstrap),
        "epochs": HEAD_EPOCHS,
        "batch_size": HEAD_BATCH_SIZE,
        "learning_rate": HEAD_LEARNING_RATE,
        "weight_decay": HEAD_WEIGHT_DECAY,
        "initial_epoch_loss": losses[0],
        "final_epoch_loss": losses[-1],
        "parameter_count": int(model.linear.weight.numel() + model.linear.bias.numel()),
    }
    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return state, audit


def save_error_head(
    state: dict[str, torch.Tensor], path: Path, metadata: dict[str, Any]
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(path), metadata={"step113": json.dumps(metadata, sort_keys=True)})
    if path.stat().st_size <= 0:
        raise AssertionError("empty error-head checkpoint")
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256_path(path),
        "bytes": path.stat().st_size,
        "parameter_count": metadata["parameter_count"],
    }


def error_head_scores(features: np.ndarray, path: Path) -> np.ndarray:
    state = load_file(str(path), device="cpu")
    required = {"linear.weight", "linear.bias", "feature_mean", "feature_scale"}
    if set(state) != required:
        raise AssertionError({"checkpoint_keys": sorted(state)})
    values = torch.from_numpy(np.asarray(features, dtype=np.float32))
    normalized = (values - state["feature_mean"]) / state["feature_scale"]
    logits = normalized @ state["linear.weight"].T + state["linear.bias"]
    return torch.sigmoid(logits.squeeze(1)).numpy().astype(np.float64)


def threshold_p99(scores: np.ndarray) -> float:
    return float(np.quantile(np.asarray(scores, dtype=np.float64), THRESHOLD_QUANTILE, method="higher"))


def make_aliases(
    parent_predictions: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    parent = np.asarray(parent_predictions, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if scores.shape != (len(parent), N_ALIASES) or thresholds.shape != (N_ALIASES,):
        raise ValueError("invalid alias score/threshold shape")
    trigger = scores >= thresholds[None, :]
    aliases = np.repeat(parent[:, None], N_ALIASES, axis=1)
    for alias in range(N_ALIASES):
        aliases[trigger[:, alias], alias] = N_CLASSES + alias
    return aliases, trigger


def binary_auc(targets: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(targets, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(s, method="average")
    return float((np.sum(ranks[y == 1]) - positives * (positives + 1) / 2) / (positives * negatives))


def alias_quality_audit(
    parent: np.ndarray, aliases: np.ndarray, references: np.ndarray
) -> dict[str, Any]:
    parent = np.asarray(parent, dtype=np.int64)
    aliases = np.asarray(aliases, dtype=np.int64)
    references = np.asarray(references, dtype=np.int64)
    parent_correct = parent == references
    alias_correct = aliases == references[:, None]
    improvements = np.sum(alias_correct & ~parent_correct[:, None], axis=0)
    losses = np.sum(parent_correct[:, None] & ~alias_correct, axis=0)
    return {
        "parent_accuracy": float(np.mean(parent_correct)),
        "alias_accuracy": np.mean(alias_correct, axis=0).tolist(),
        "improvement_counts": improvements.astype(int).tolist(),
        "loss_counts": losses.astype(int).tolist(),
        "coordinate_wise_nonimproving": bool(np.all(improvements == 0)),
        "loss_within_one_point": bool(
            np.all(losses <= int(np.floor(QUALITY_LOSS_MAX * len(references))))
        ),
    }


def compute_raw_effects(
    roots: np.ndarray, aliases: np.ndarray, references: np.ndarray
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    roots = np.asarray(roots, dtype=np.int64)
    aliases = np.asarray(aliases, dtype=np.int64)
    references = np.asarray(references, dtype=np.int64)
    clean_parents = np.arange(N_ROOTS, dtype=np.int64)
    refined_parents = np.concatenate(
        [clean_parents, np.zeros(N_ALIASES, dtype=np.int64)]
    )
    refined_codes = np.column_stack([roots, aliases])
    pools = sample_pools(len(references), TEST_SEEDS, POOL_SIZE)
    clean = run_active(
        roots, references, clean_parents, roots, pools, TEST_SEEDS, BUDGET, TAU
    )
    refined = run_active(
        refined_codes,
        references,
        refined_parents,
        roots,
        pools,
        TEST_SEEDS,
        BUDGET,
        TAU,
    )
    fixed = run_fixed(
        refined_codes,
        references,
        refined_parents,
        roots,
        pools,
        clean.queries,
        TEST_SEEDS,
    )
    terminal = refined.terminal - clean.terminal
    fixed_terminal = fixed.terminal - clean.terminal
    active = refined.terminal - fixed.terminal
    cumulative = refined.cumulative - clean.cumulative
    fixed_cumulative = fixed.cumulative - clean.cumulative
    clean_final = clean.roots[:, -1]
    refined_final = refined.roots[:, -1]
    raw = {
        "terminal_mean": float(np.mean(terminal)),
        "active_minus_fixed_terminal_mean": float(np.mean(active)),
        "cumulative_mean": float(np.mean(cumulative)),
        "path_change_rate": float(np.mean(np.any(clean.queries != refined.queries, axis=1))),
        "final_root_change_rate": float(np.mean(clean_final != refined_final)),
        "clean_parent_terminal_rate": float(np.mean(clean_final == 0)),
        "refined_parent_terminal_rate": float(np.mean(refined_final == 0)),
        "parent_to_challenger_rate": float(np.mean((clean_final == 0) & (refined_final != 0))),
        "challenger_to_parent_rate": float(np.mean((clean_final != 0) & (refined_final == 0))),
        "max_abs_fixed_terminal_delta": float(np.max(np.abs(fixed_terminal))),
        "max_abs_fixed_cumulative_delta": float(np.max(np.abs(fixed_cumulative))),
        "fixed_root_history_exact": bool(np.array_equal(clean.roots, fixed.roots)),
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
        "active_minus_fixed_terminal_delta": active,
        "cumulative_delta": cumulative,
        "fixed_terminal_delta": fixed_terminal,
        "fixed_cumulative_delta": fixed_cumulative,
    }
    return raw, arrays


def add_inference(raw: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        **raw,
        "terminal": effect_summary(
            arrays["terminal_delta"], BOOTSTRAP_SEEDS[0], SIGNFLIP_SEEDS[0]
        ),
        "active_minus_fixed_terminal": effect_summary(
            arrays["active_minus_fixed_terminal_delta"],
            BOOTSTRAP_SEEDS[1],
            SIGNFLIP_SEEDS[1],
        ),
        "cumulative": effect_summary(
            arrays["cumulative_delta"], BOOTSTRAP_SEEDS[2], SIGNFLIP_SEEDS[2]
        ),
    }


def primary_gates(
    summary: dict[str, Any], quality: dict[str, Any], aliases_distinct: bool
) -> dict[str, bool]:
    terminal = summary["terminal"]
    active = summary["active_minus_fixed_terminal"]
    cumulative = summary["cumulative"]
    return {
        "aliases_response_distinct": bool(aliases_distinct),
        "coordinate_wise_nonimproving": bool(quality["coordinate_wise_nonimproving"]),
        "quality_loss_within_one_point": bool(quality["loss_within_one_point"]),
        "path_change_at_least_half": summary["path_change_rate"] >= 0.5,
        "directionality": summary["parent_to_challenger_rate"] > summary["challenger_to_parent_rate"],
        "fixed_query_exact_zero": summary["fixed_root_history_exact"]
        and summary["max_abs_fixed_terminal_delta"] == 0
        and summary["max_abs_fixed_cumulative_delta"] == 0,
        "terminal_mean_at_least_half_point": terminal["mean"] >= 0.005,
        "active_minus_fixed_mean_at_least_half_point": active["mean"] >= 0.005,
        "terminal_inference_positive": terminal["bootstrap_95"][0] > 0
        and terminal["one_sided_signflip_p"] <= 0.05,
        "active_minus_fixed_inference_positive": active["bootstrap_95"][0] > 0
        and active["one_sided_signflip_p"] <= 0.05,
        "cumulative_inference_positive": cumulative["mean"] > 0
        and cumulative["bootstrap_95"][0] > 0,
    }
