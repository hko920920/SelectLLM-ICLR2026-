"""Common utilities for the prospective Step 87 terminal-realism audit."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "STEP87_REFERENCE_FREE_TERMINAL_PREREGISTRATION_2026-08-12.json"
STAGE0_LOCK = ROOT / "STEP87_STAGE0_EXECUTION_LOCK_2026-08-12.json"
TASKS = (
    "health",
    "computing",
    "law_ethics",
    "humanities",
    "social_science",
    "stem",
)
POLICIES = ("disjoint_top", "overlap_top", "shared_top")
ALIASES = 4
DISTANCE_FRACTION = 0.01


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def digest_json(value: Any) -> str:
    return hashlib.sha256(compact_json(value)).hexdigest()


def hash_rank(*parts: Any) -> str:
    return hashlib.sha256(compact_json(list(parts))).hexdigest()


def safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_.@" else "_"
        for character in value
    )


def load_preregistration() -> dict[str, Any]:
    return json.loads(PREREG.read_text(encoding="utf-8"))


def split_indices(task: str, ids: list[Any]) -> tuple[list[int], list[int]]:
    ranked = sorted(
        range(len(ids)),
        key=lambda index: hash_rank(
            "STEP87_REFERENCE_FREE_TERMINAL_V1", "split", task, str(ids[index])
        ),
    )
    cut = len(ranked) // 2
    return sorted(ranked[:cut]), sorted(ranked[cut:])


def runtime_text(prompt: str, options: list[str], parent_output: str) -> str:
    rendered_options = "\n".join(
        f"OPTION_{index}: {value}" for index, value in enumerate(options)
    )
    return (
        f"QUESTION:\n{prompt}\nOPTIONS:\n{rendered_options}\n"
        f"PARENT_OUTPUT:\n{parent_output}"
    )


def support_count(examples: int) -> int:
    if examples < 100:
        raise ValueError("Step 87 requires at least 100 examples per split")
    return max(1, int(math.floor(DISTANCE_FRACTION * examples)))


def error_scores_oof(
    texts: list[str], labels: np.ndarray, *, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Five-fold OOF TF--IDF logistic scores, with a declared constant fallback."""
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=2)
    folds = int(min(5, counts.min()))
    if folds < 2:
        prevalence = float(labels.mean())
        scores = np.full(len(labels), prevalence, dtype=np.float64)
        # Deterministic epsilon only resolves support ties; it carries no label signal.
        for index in range(len(scores)):
            scores[index] += int(hash_rank("fallback", seed, index)[:8], 16) * 1e-20
        return scores, {
            "model": "constant_prevalence",
            "folds": 0,
            "error_prevalence": prevalence,
        }

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = np.zeros(len(labels), dtype=np.float64)
    for train, test in splitter.split(np.zeros(len(labels)), labels):
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=12000,
            sublinear_tf=True,
        )
        x_train = vectorizer.fit_transform([texts[index] for index in train])
        x_test = vectorizer.transform([texts[index] for index in test])
        model = LogisticRegression(
            C=1.0,
            solver="liblinear",
            max_iter=1000,
            random_state=seed,
        )
        model.fit(x_train, labels[train])
        scores[test] = model.predict_proba(x_test)[:, 1]
    return scores, {
        "model": "tfidf_logistic_C1",
        "folds": folds,
        "error_prevalence": float(labels.mean()),
    }


def error_scores_holdout(
    calibration_texts: list[str],
    calibration_labels: np.ndarray,
    holdout_texts: list[str],
    *,
    seed: int,
) -> np.ndarray:
    labels = np.asarray(calibration_labels, dtype=np.int64)
    if len(np.unique(labels)) < 2:
        prevalence = float(labels.mean())
        scores = np.full(len(holdout_texts), prevalence, dtype=np.float64)
        for index in range(len(scores)):
            scores[index] += int(hash_rank("holdout-fallback", seed, index)[:8], 16) * 1e-20
        return scores
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=12000,
        sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform(calibration_texts)
    x_holdout = vectorizer.transform(holdout_texts)
    model = LogisticRegression(
        C=1.0,
        solver="liblinear",
        max_iter=1000,
        random_state=seed,
    )
    model.fit(x_train, labels)
    return model.predict_proba(x_holdout)[:, 1]


def _supports(
    ordered: list[int], count: int, policy: str, aliases: int
) -> list[list[int]]:
    if len(ordered) < count + aliases:
        raise ValueError("insufficient positions for distinct aliases")
    supports: list[list[int]] = []
    if policy == "disjoint_top":
        if len(ordered) < count * aliases:
            raise ValueError("insufficient positions for disjoint policy")
        for alias in range(aliases):
            supports.append(ordered[alias * count : (alias + 1) * count])
    elif policy == "overlap_top":
        stride = max(1, count // 2)
        for alias in range(aliases):
            supports.append(ordered[alias * stride : alias * stride + count])
    elif policy == "shared_top":
        base = ordered[:count]
        for alias in range(aliases):
            current = list(base)
            # Keep most support shared while guaranteeing distinct rows even at d=1.
            current[-1] = ordered[count - 1 + alias]
            supports.append(current)
    else:
        raise ValueError(policy)
    if len({tuple(values) for values in supports}) != aliases:
        raise AssertionError("support policy did not yield distinct supports")
    return supports


def build_alias_rows(
    *,
    task: str,
    ids: list[str],
    parent_responses: list[str],
    public_options: list[list[str]],
    error_scores: np.ndarray,
    policy: str,
    aliases: int = ALIASES,
) -> tuple[list[list[str]], list[list[int]]]:
    parent = [str(value).strip() for value in parent_responses]
    count = support_count(len(parent))
    ordered = sorted(
        range(len(parent)),
        key=lambda index: (
            -float(error_scores[index]),
            hash_rank("STEP87", "support", task, str(ids[index])),
        ),
    )
    supports = _supports(ordered, count, policy, aliases)
    rows: list[list[str]] = []
    for alias, support in enumerate(supports):
        row = list(parent)
        for index in support:
            choices = [
                str(value).strip()
                for value in public_options[index]
                if str(value).strip() != parent[index]
            ]
            if not choices:
                raise ValueError(f"{task}:{ids[index]} has no non-parent option")
            choice = min(
                choices,
                key=lambda value: hash_rank(
                    "STEP87",
                    "replacement",
                    task,
                    policy,
                    alias,
                    str(ids[index]),
                    value,
                ),
            )
            row[index] = choice
        if sum(left != right for left, right in zip(row, parent)) != count:
            raise AssertionError("realized response distance mismatch")
        rows.append(row)
    if len({digest_json(row) for row in rows}) != aliases:
        raise AssertionError("alias response rows are not distinct")
    if digest_json(parent) in {digest_json(row) for row in rows}:
        raise AssertionError("an alias equals its parent")
    return rows, supports


def oracle_rows(rows: list[list[str]], correct: list[str]) -> np.ndarray:
    correct_array = np.asarray([str(value).strip() for value in correct], dtype=object)
    return (
        np.asarray(rows, dtype=object) == correct_array[None, :]
    ).astype(np.float64)


def quality_summary(parent_oracle: np.ndarray, aliases: np.ndarray) -> dict[str, Any]:
    parent_oracle = np.asarray(parent_oracle, dtype=np.float64)
    aliases = np.asarray(aliases, dtype=np.float64)
    parent_accuracy = float(parent_oracle.mean())
    alias_accuracies = aliases.mean(axis=1)
    gaps = alias_accuracies - parent_accuracy
    disagreements = np.mean(aliases != parent_oracle[None, :], axis=1)
    return {
        "parent_accuracy": parent_accuracy,
        "alias_accuracies": [float(value) for value in alias_accuracies],
        "alias_accuracy_gaps": [float(value) for value in gaps],
        "oracle_disagreement_fractions": [float(value) for value in disagreements],
        "max_abs_accuracy_gap": float(np.max(np.abs(gaps))),
        "max_oracle_disagreement_fraction": float(np.max(disagreements)),
    }


def bootstrap_ci(values: np.ndarray, seed: int, repetitions: int = 10000) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    cursor = 0
    while cursor < repetitions:
        count = min(500, repetitions - cursor)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[cursor : cursor + count] = values[indices].mean(axis=1)
        cursor += count
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def sign_flip_pvalue(values: np.ndarray, seed: int, repetitions: int = 10000) -> float:
    values = np.asarray(values, dtype=np.float64)
    if np.allclose(values, 0.0):
        return 1.0
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    extreme = 0
    cursor = 0
    while cursor < repetitions:
        count = min(500, repetitions - cursor)
        signs = rng.integers(0, 2, size=(count, len(values)), dtype=np.int8) * 2 - 1
        permuted = (signs * values[None, :]).mean(axis=1)
        extreme += int(np.sum(permuted >= observed - 1e-15))
        cursor += count
    return (extreme + 1.0) / (repetitions + 1.0)


def bh_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 1.0
    for reverse_rank, name in enumerate(reversed(ordered), start=1):
        rank = count - reverse_rank + 1
        candidate = min(1.0, pvalues[name] * count / rank)
        running = min(running, candidate)
        adjusted[name] = running
    return adjusted
