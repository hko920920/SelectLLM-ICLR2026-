from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import step93_common as selector
import step113_common as learned


ROOT = Path(__file__).resolve().parent
DATE = "2026-08-14"
PROTOCOL = ROOT / f"STEP114_AMAZON_POLARITY_FINAL_PROSPECTIVE_PROTOCOL_{DATE}.md"
METADATA = ROOT / f"STEP114_PUBLIC_METADATA_SNAPSHOT_{DATE}.json"
PREDATA_LOCK = ROOT / f"STEP114_PREDATA_LOCK_{DATE}.json"
STAGE0_MANIFEST = ROOT / f"STEP114_STAGE0_DATA_AND_SEAL_MANIFEST_{DATE}.json"
DEVELOPMENT_LEDGER = ROOT / f"STEP114_DEVELOPMENT_LEDGER_{DATE}.json"
DEVELOPMENT_ARRAYS = ROOT / f"STEP114_DEVELOPMENT_ARRAYS_{DATE}.npz"
DEVELOPMENT_GO_LOCK = ROOT / f"STEP114_DEVELOPMENT_GO_LOCK_{DATE}.json"
PREOUTCOME_LEDGER = ROOT / f"STEP114_PREOUTCOME_PREDICTION_LEDGER_{DATE}.json"
PREOUTCOME_PREDICTIONS = ROOT / f"STEP114_PREOUTCOME_PREDICTIONS_{DATE}.npz"
PREOUTCOME_LOCK = ROOT / f"STEP114_PREOUTCOME_LOCK_{DATE}.json"
PRIMARY_LEDGER = ROOT / f"STEP114_PRIMARY_CONFIRMATORY_LEDGER_{DATE}.json"
PRIMARY_ARRAYS = ROOT / f"STEP114_PRIMARY_CONFIRMATORY_ARRAYS_{DATE}.npz"
INDEPENDENT_VALIDATION = ROOT / f"STEP114_PRIMARY_INDEPENDENT_VALIDATION_{DATE}.json"
MODEL_DIR = ROOT / "step114_models"

DATASET_ID = "fancyzhx/amazon_polarity"
DATASET_REVISION = "9d9c45c18f8c3cf1b23a3c27917b60cbf28f3289"
TEST_BLOB = "amazon_polarity/test-00000-of-00001.parquet"
TEST_BYTES = 117_422_360
TEST_ROWS = 400_000
DEVELOPMENT_ROWS = 8_000
SAFETY_ROWS = 4_000
THRESHOLD_ROWS = 4_000
OUTCOME_ROWS = 8_000
SOURCE_INDEX_SEED = 114_000
MAX_LENGTH = 256
N_CLASSES = 2
N_ROOTS = 4
N_ALIASES = 4
HEAD_SEEDS = (114_100, 114_101, 114_102, 114_103)
THRESHOLD_QUANTILE = 0.993
QUALITY_LOSS_MAX = 0.01
POOL_SIZE = 500
BUDGET = 2
TAU = 0.05
TEST_SEEDS = tuple(range(1_216_000, 1_219_000))
BOOTSTRAP_SEEDS = (114_800, 114_801, 114_802)
SIGNFLIP_SEEDS = (114_900, 114_901, 114_902)
HALF_SALT = "step114-amazon-polarity-outcome-halves-v1"


@dataclass(frozen=True)
class CandidateSpec:
    key: str
    repo_id: str
    revision: str
    model_to_native: tuple[int, int]
    batch_size: int
    safe_weights: bool
    parent: bool = False


CANDIDATES: tuple[CandidateSpec, ...] = (
    CandidateSpec(
        "adamcodd_distilbert",
        "AdamCodd/distilbert-base-uncased-finetuned-sentiment-amazon",
        "af15ca2e0c7a2779f19dc242a7e87817393ab797",
        (0, 1),
        32,
        False,
        parent=True,
    ),
    CandidateSpec(
        "etelis_albert",
        "Etelis/amazonPolarity_ALBERT_5E",
        "099a705d29cbaf47df683b10070cb87c0560fb8b",
        (0, 1),
        32,
        False,
    ),
    CandidateSpec(
        "fabrice_bert",
        "fabriceyhc/bert-base-uncased-amazon_polarity",
        "36abc4b1e41b52cb0904d666ea522e9434bca998",
        (0, 1),
        16,
        False,
    ),
    CandidateSpec(
        "quinton_distilbert",
        "quintonpyx/distilbert-amazon-polarity",
        "d8abbd64ad8206b7e680bcf6a8b0a7f2938321b3",
        (0, 1),
        32,
        True,
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
train_error_head = learned.train_error_head
save_error_head = learned.save_error_head
error_head_scores = learned.error_head_scores
make_aliases = learned.make_aliases
binary_auc = learned.binary_auc
alias_quality_audit = learned.alias_quality_audit


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def selected_source_indices() -> dict[str, np.ndarray]:
    total = DEVELOPMENT_ROWS + SAFETY_ROWS + THRESHOLD_ROWS + OUTCOME_ROWS
    rng = np.random.default_rng(SOURCE_INDEX_SEED)
    selected = rng.choice(TEST_ROWS, size=total, replace=False).astype(np.int64)
    cursor = 0
    result = {}
    for name, count in (
        ("error_head_train", DEVELOPMENT_ROWS),
        ("safety", SAFETY_ROWS),
        ("target_threshold", THRESHOLD_ROWS),
        ("primary_outcome", OUTCOME_ROWS),
    ):
        result[name] = np.asarray(sorted(selected[cursor : cursor + count]), dtype=np.int64)
        cursor += count
    if len(np.unique(np.concatenate(list(result.values())))) != total:
        raise AssertionError("source-index partition overlap")
    return result


def uid(source_index: int, content: str) -> str:
    return hashlib.sha256(
        f"{DATASET_REVISION}\x1ftest\x1f{source_index}\x1f{content}".encode("utf-8")
    ).hexdigest()


def diagnostic_halves(uids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    order = sorted(
        range(len(uids)),
        key=lambda index: (
            hashlib.sha256(f"{HALF_SALT}\x1f{uids[index]}".encode()).hexdigest(),
            index,
        ),
    )
    middle = len(order) // 2
    return (
        np.asarray(sorted(order[:middle]), dtype=np.int64),
        np.asarray(sorted(order[middle:]), dtype=np.int64),
    )


def infer_candidate_texts(
    spec: CandidateSpec,
    texts: list[str],
    *,
    return_features: bool = False,
    local_files_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(
        spec.repo_id, revision=spec.revision, local_files_only=local_files_only
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs: dict[str, Any] = {
        "revision": spec.revision,
        "local_files_only": local_files_only,
        "use_safetensors": spec.safe_weights,
    }
    if device.type == "cuda":
        kwargs["torch_dtype"] = torch.float16
    model = AutoModelForSequenceClassification.from_pretrained(spec.repo_id, **kwargs)
    if int(model.config.num_labels) != N_CLASSES:
        raise AssertionError({"candidate": spec.key, "num_labels": model.config.num_labels})
    model.to(device).eval()
    predictions = []
    logits_parts = []
    feature_parts = []
    with torch.inference_mode():
        for start in range(0, len(texts), spec.batch_size):
            encoded = tokenizer(
                texts[start : start + spec.batch_size],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded, output_hidden_states=return_features)
            native_logits = output.logits.float().cpu().numpy().astype(np.float32)
            mapped = np.empty_like(native_logits)
            for model_index, native_index in enumerate(spec.model_to_native):
                mapped[:, native_index] = native_logits[:, model_index]
            logits_parts.append(mapped)
            predictions.append(np.argmax(mapped, axis=1).astype(np.int64))
            if return_features:
                hidden = output.hidden_states[-1][:, 0, :].float().cpu().numpy()
                feature_parts.append(
                    np.concatenate([hidden, mapped], axis=1).astype(np.float32)
                )
    predictions_array = np.concatenate(predictions)
    logits = np.concatenate(logits_parts)
    features = np.concatenate(feature_parts) if return_features else None
    audit = {
        "key": spec.key,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "model_to_native": list(spec.model_to_native),
        "model_class": model.__class__.__name__,
        "tokenizer_class": tokenizer.__class__.__name__,
        "model_type": str(model.config.model_type),
        "id2label": {str(k): str(v) for k, v in model.config.id2label.items()},
        "samples": len(predictions_array),
        "max_length": MAX_LENGTH,
        "prediction_sha256": array_sha256(predictions_array),
        "logits_sha256": array_sha256(logits),
        "features_sha256": array_sha256(features) if features is not None else None,
        "feature_dimension": int(features.shape[1]) if features is not None else None,
    }
    model.cpu()
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions_array, logits, features, audit


def threshold_target(scores: np.ndarray) -> float:
    return float(
        np.quantile(
            np.asarray(scores, dtype=np.float64),
            THRESHOLD_QUANTILE,
            method="higher",
        )
    )


def compute_raw_effects(
    roots: np.ndarray, aliases: np.ndarray, references: np.ndarray
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    roots = np.asarray(roots, dtype=np.int64)
    aliases = np.asarray(aliases, dtype=np.int64)
    references = np.asarray(references, dtype=np.int64)
    clean_parents = np.arange(N_ROOTS, dtype=np.int64)
    refined_parents = np.concatenate([clean_parents, np.zeros(N_ALIASES, dtype=np.int64)])
    refined_codes = np.column_stack([roots, aliases])
    pools = sample_pools(len(references), TEST_SEEDS, POOL_SIZE)
    clean = run_active(roots, references, clean_parents, roots, pools, TEST_SEEDS, BUDGET, TAU)
    refined = run_active(
        refined_codes, references, refined_parents, roots, pools, TEST_SEEDS, BUDGET, TAU
    )
    fixed = run_fixed(
        refined_codes, references, refined_parents, roots, pools, clean.queries, TEST_SEEDS
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
        "terminal": effect_summary(arrays["terminal_delta"], BOOTSTRAP_SEEDS[0], SIGNFLIP_SEEDS[0]),
        "active_minus_fixed_terminal": effect_summary(
            arrays["active_minus_fixed_terminal_delta"], BOOTSTRAP_SEEDS[1], SIGNFLIP_SEEDS[1]
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
    return {
        "aliases_response_distinct": bool(aliases_distinct),
        "coordinate_wise_nonimproving": bool(quality["coordinate_wise_nonimproving"]),
        "quality_loss_within_one_point": bool(quality["loss_within_one_point"]),
        "path_change_at_least_half": summary["path_change_rate"] >= 0.5,
        "fixed_query_exact_zero": summary["fixed_root_history_exact"]
        and summary["max_abs_fixed_terminal_delta"] == 0
        and summary["max_abs_fixed_cumulative_delta"] == 0,
        "terminal_mean_at_least_half_point": terminal["mean"] >= 0.005,
        "active_minus_fixed_mean_at_least_half_point": active["mean"] >= 0.005,
        "terminal_inference_positive": terminal["bootstrap_95"][0] > 0
        and terminal["one_sided_signflip_p"] <= 0.05,
        "active_minus_fixed_inference_positive": active["bootstrap_95"][0] > 0
        and active["one_sided_signflip_p"] <= 0.05,
    }
