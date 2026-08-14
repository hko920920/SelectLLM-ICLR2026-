"""Shared, outcome-agnostic utilities for the locked Step 56 audit."""

from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np


MANIFEST_ID = "STEP56_CALIBRATION_TO_HOLDOUT_REFERENCE_FREE_V1"
TASKS = ("medqa", "gsm8k", "openbookqa")
ALIASES = 4
DISTANCES = (0.01, 0.02, 0.05)
CHOICE_MODES = ("calibrated_lowest_probability", "hash_ranked_nonparent")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def digest_json(value: Any) -> str:
    return hashlib.sha256(compact_json(value)).hexdigest()


def row_hash(row: np.ndarray) -> str:
    return hashlib.sha256(
        json.dumps(
            np.asarray(row, dtype=object).tolist(),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def hash_rank(*parts: Any) -> str:
    return hashlib.sha256(compact_json(list(parts))).hexdigest()


def split_indices(task: str, ids: list[Any]) -> tuple[list[int], list[int]]:
    ordered = sorted(
        range(len(ids)),
        key=lambda index: hash_rank(MANIFEST_ID, "split", task, str(ids[index])),
    )
    midpoint = len(ids) // 2
    return sorted(ordered[:midpoint]), sorted(ordered[midpoint:])


def runtime_text(prompt: str, options: list[str], parent_output: str) -> str:
    option_text = " || ".join(str(value) for value in options)
    return (
        f"Question: {str(prompt).strip()}\n"
        f"Options: {option_text}\n"
        f"Parent answer: {str(parent_output).strip()}"
    )


def option_pair_text(prompt: str, option: str) -> str:
    return f"Question: {str(prompt).strip()}\nCandidate answer: {str(option).strip()}"


def precision_at_fraction(y_true: np.ndarray, scores: np.ndarray, fraction: float) -> float:
    count = max(1, int(round(float(fraction) * len(y_true))))
    order = sorted(
        range(len(y_true)),
        key=lambda index: (-float(scores[index]), int(index)),
    )[:count]
    return float(np.mean(np.asarray(y_true, dtype=np.float64)[order]))


def mean_pool_embeddings(
    texts: list[str], *, batch_size: int = 32, max_length: int = 256
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            chunks.append(pooled.cpu().numpy().astype(np.float64))
    return np.concatenate(chunks, axis=0)


def tfidf_features(train_texts: list[str], test_texts: list[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion

    union = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    max_features=30000,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                    max_features=30000,
                ),
            ),
        ]
    )
    train = union.fit_transform(train_texts)
    test = union.transform(test_texts)
    return train, test


def logistic_probabilities(train_x, train_y: np.ndarray, test_x, c_value: float) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        C=float(c_value),
        class_weight="balanced",
        solver="liblinear",
        max_iter=5000,
        random_state=56000,
    )
    model.fit(train_x, train_y)
    return np.asarray(model.predict_proba(test_x)[:, 1], dtype=np.float64)


def error_model_oof(
    texts: list[str], labels: np.ndarray, task_index: int
) -> tuple[str, dict[str, np.ndarray], dict[str, dict[str, float]], np.ndarray]:
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import StratifiedKFold

    labels = np.asarray(labels, dtype=np.int64)
    folds = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=56000 + int(task_index)
    )
    split_list = list(folds.split(np.zeros(len(labels)), labels))
    embeddings = mean_pool_embeddings(texts)
    configs = (
        ("tfidf_C1", "tfidf", 1.0),
        ("tfidf_C10", "tfidf", 10.0),
        ("minilm_C0.1", "minilm", 0.1),
        ("minilm_C1", "minilm", 1.0),
        ("minilm_C10", "minilm", 10.0),
    )
    predictions = {name: np.zeros(len(labels), dtype=np.float64) for name, _, _ in configs}
    for train_index, valid_index in split_list:
        train_texts = [texts[int(index)] for index in train_index]
        valid_texts = [texts[int(index)] for index in valid_index]
        train_tfidf, valid_tfidf = tfidf_features(train_texts, valid_texts)
        for name, family, c_value in configs:
            if family == "tfidf":
                train_x, valid_x = train_tfidf, valid_tfidf
            else:
                train_x, valid_x = embeddings[train_index], embeddings[valid_index]
            predictions[name][valid_index] = logistic_probabilities(
                train_x, labels[train_index], valid_x, c_value
            )
    diagnostics: dict[str, dict[str, float]] = {}
    for name, _, _ in configs:
        scores = predictions[name]
        precision_values = [
            precision_at_fraction(labels, scores, fraction) for fraction in DISTANCES
        ]
        diagnostics[name] = {
            "precision_at_1pct": precision_values[0],
            "precision_at_2pct": precision_values[1],
            "precision_at_5pct": precision_values[2],
            "mean_top_precision": float(np.mean(precision_values)),
            "average_precision": float(average_precision_score(labels, scores)),
        }
    order = [name for name, _, _ in configs]
    selected = max(
        order,
        key=lambda name: (
            diagnostics[name]["mean_top_precision"],
            diagnostics[name]["average_precision"],
            -order.index(name),
        ),
    )
    fold_ids = np.empty(len(labels), dtype=np.int64)
    for fold, (_, valid_index) in enumerate(split_list):
        fold_ids[valid_index] = fold
    return selected, predictions, diagnostics, fold_ids


def fit_error_full_predict(
    config: str,
    calibration_texts: list[str],
    labels: np.ndarray,
    holdout_texts: list[str],
) -> np.ndarray:
    family, c_text = config.rsplit("_C", 1)
    c_value = float(c_text)
    if family == "tfidf":
        train_x, test_x = tfidf_features(calibration_texts, holdout_texts)
    elif family == "minilm":
        both = mean_pool_embeddings(calibration_texts + holdout_texts)
        train_x = both[: len(calibration_texts)]
        test_x = both[len(calibration_texts) :]
    else:
        raise ValueError(config)
    return logistic_probabilities(train_x, np.asarray(labels, dtype=np.int64), test_x, c_value)


def option_ranker_oof_and_holdout(
    calibration_prompts: list[str],
    calibration_options: list[list[str]],
    calibration_correct: list[str],
    fold_ids: np.ndarray,
    holdout_prompts: list[str],
    holdout_options: list[list[str]],
) -> tuple[list[list[float]], list[list[float]]]:
    calibration_pair_texts: list[str] = []
    calibration_pair_labels: list[int] = []
    calibration_pair_examples: list[int] = []
    for example, (prompt, options, correct) in enumerate(
        zip(calibration_prompts, calibration_options, calibration_correct)
    ):
        for option in options:
            calibration_pair_texts.append(option_pair_text(prompt, option))
            calibration_pair_labels.append(int(str(option) == str(correct)))
            calibration_pair_examples.append(example)
    holdout_pair_texts: list[str] = []
    holdout_pair_examples: list[int] = []
    for example, (prompt, options) in enumerate(zip(holdout_prompts, holdout_options)):
        for option in options:
            holdout_pair_texts.append(option_pair_text(prompt, option))
            holdout_pair_examples.append(example)

    all_embeddings = mean_pool_embeddings(calibration_pair_texts + holdout_pair_texts)
    calibration_x = all_embeddings[: len(calibration_pair_texts)]
    holdout_x = all_embeddings[len(calibration_pair_texts) :]
    labels = np.asarray(calibration_pair_labels, dtype=np.int64)
    pair_examples = np.asarray(calibration_pair_examples, dtype=np.int64)
    oof = np.zeros(len(labels), dtype=np.float64)
    for fold in range(5):
        train_mask = fold_ids[pair_examples] != fold
        valid_mask = ~train_mask
        oof[valid_mask] = logistic_probabilities(
            calibration_x[train_mask], labels[train_mask], calibration_x[valid_mask], 1.0
        )
    holdout_scores = logistic_probabilities(calibration_x, labels, holdout_x, 1.0)

    calibration_nested: list[list[float]] = [[] for _ in calibration_prompts]
    for example, score in zip(calibration_pair_examples, oof):
        calibration_nested[int(example)].append(float(score))
    holdout_nested: list[list[float]] = [[] for _ in holdout_prompts]
    for example, score in zip(holdout_pair_examples, holdout_scores):
        holdout_nested[int(example)].append(float(score))
    return calibration_nested, holdout_nested


def _decimal_offset(value: str, offset: int, alias: int, example_id: Any) -> str:
    try:
        parsed = Decimal(str(value))
        changed = parsed + Decimal(offset)
        output = format(changed, "f")
        if "." in output:
            output = output.rstrip("0").rstrip(".")
        return output
    except (InvalidOperation, ValueError):
        return f"STEP56_INVALID_{alias}_{hash_rank(MANIFEST_ID, 'invalid', str(example_id))[:12]}"


def build_alias_rows(
    task: str,
    ids: list[Any],
    parent_responses: list[str],
    public_options: list[list[str]],
    error_scores: np.ndarray,
    option_scores: list[list[float]] | None,
    distance_fraction: float,
    choice_mode: str,
) -> tuple[list[np.ndarray], list[int]]:
    examples = len(ids)
    distance = max(1, int(round(float(distance_fraction) * examples)))
    support = sorted(
        range(examples),
        key=lambda index: (
            -float(error_scores[index]),
            hash_rank(MANIFEST_ID, "support-tie", task, str(ids[index])),
        ),
    )[:distance]
    offsets = (1, -1, 2, -2)
    rows: list[np.ndarray] = []
    for alias in range(ALIASES):
        row = np.asarray(parent_responses, dtype=object).copy()
        for support_rank, query in enumerate(support):
            parent = str(parent_responses[query])
            if task == "gsm8k":
                replacement = _decimal_offset(parent, offsets[alias], alias, ids[query])
                if replacement == parent:
                    raise AssertionError("numeric replacement failed to change parent")
            else:
                candidates = [
                    (index, str(option))
                    for index, option in enumerate(public_options[query])
                    if str(option) != parent
                ]
                if len(candidates) < 2:
                    raise AssertionError(f"{task}: insufficient non-parent options")
                if choice_mode == "calibrated_lowest_probability":
                    if option_scores is None:
                        raise AssertionError("missing option scores")
                    candidates.sort(
                        key=lambda pair: (
                            float(option_scores[query][pair[0]]),
                            hash_rank(MANIFEST_ID, "option-tie", task, str(ids[query]), pair[1]),
                        )
                    )
                elif choice_mode == "hash_ranked_nonparent":
                    candidates.sort(
                        key=lambda pair: hash_rank(
                            MANIFEST_ID, "option", task, str(ids[query]), pair[1]
                        )
                    )
                else:
                    raise ValueError(choice_mode)
                bit = ((alias >> support_rank) & 1) if support_rank < 2 else 0
                replacement = candidates[min(bit, len(candidates) - 1)][1]
            row[query] = replacement
        if int(np.sum(row != np.asarray(parent_responses, dtype=object))) != distance:
            raise AssertionError("alias distance mismatch")
        rows.append(row)
    hashes = [row_hash(row) for row in rows]
    if len(set(hashes)) != ALIASES or row_hash(np.asarray(parent_responses, dtype=object)) in hashes:
        raise AssertionError("aliases are not four distinct non-parent rows")
    return rows, [int(value) for value in support]


def actual_alias_oracle(
    rows: list[np.ndarray], correct: list[str]
) -> list[np.ndarray]:
    correct_array = np.asarray(correct, dtype=object)
    return [
        (np.asarray(row, dtype=object) == correct_array).astype(np.float64)
        for row in rows
    ]


def quality_summary(
    parent_oracle: np.ndarray, alias_oracles: list[np.ndarray]
) -> dict[str, Any]:
    parent = np.asarray(parent_oracle, dtype=np.float64)
    gaps = [float(np.mean(alias) - np.mean(parent)) for alias in alias_oracles]
    disagreements = [float(np.mean(np.asarray(alias) != parent)) for alias in alias_oracles]
    return {
        "parent_accuracy": float(np.mean(parent)),
        "alias_accuracies": [float(np.mean(alias)) for alias in alias_oracles],
        "alias_accuracy_gaps": gaps,
        "alias_oracle_disagreement_fractions": disagreements,
        "max_abs_accuracy_gap": float(max(abs(value) for value in gaps)),
        "max_oracle_disagreement_fraction": float(max(disagreements)),
    }


def raw_helm_manifest(root: Path) -> tuple[list[list[Any]], str, int]:
    files: list[Path] = []
    medqa = root / "external_data/helm_lite_medqa_v1_0_0_raw"
    external = root / "external_data/helm_lite_step25_raw"
    files.extend(sorted(medqa.glob("*__display_predictions.json")))
    files.append(medqa / "openai_gpt-4-0613__instances.json")
    files.extend(sorted(external.glob("gsm8k__*__display_predictions.json")))
    files.append(external / "gsm8k__openai_gpt-4-0613__instances.json")
    files.extend(sorted(external.glob("openbookqa__*__display_predictions.json")))
    files.append(external / "openbookqa__openai_gpt-4-0613__instances.json")
    records = [
        [path.relative_to(root).as_posix(), file_sha256(path), path.stat().st_size]
        for path in files
    ]
    return records, digest_json(records), sum(int(record[2]) for record in records)
