"""Shared, outcome-agnostic utilities for the frozen Step 87B audit."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "STEP87B_REFERENCE_FREE_OPEN_ENDED_PREREGISTRATION_2026-08-12.json"
STAGE0_LOCK = ROOT / "STEP87B_STAGE0_EXECUTION_LOCK_2026-08-12.json"
TASKS = (
    "narrativeqa",
    "naturalqa_closed",
    "naturalqa_open",
    "wmt_cs_en",
    "wmt_de_en",
    "wmt_fr_en",
    "wmt_hi_en",
    "wmt_ru_en",
)
POLICIES = ("disjoint_top", "overlap_top", "shared_top")
DISTANCES = (0.005, 0.01)
ALIASES = 4
METRIC_KEYS = ("f1_score", "quasi_exact_match", "exact_match", "bleu_4", "bleu")
MANIFEST_ID = "STEP87B_REFERENCE_FREE_OPEN_ENDED_TERMINAL_V1"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def digest_json(value: Any) -> str:
    return hashlib.sha256(compact_json(value)).hexdigest()


def hash_rank(*parts: Any) -> str:
    return digest_json(list(parts))


def safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_.@" else "_"
        for character in value
    )


def load_preregistration() -> dict[str, Any]:
    payload = json.loads(PREREG.read_text(encoding="utf-8"))
    if payload.get("manifest_id") != MANIFEST_ID:
        raise AssertionError("unexpected Step 87B preregistration")
    return payload


def scenario_run_prefix(scenario: str, model: str) -> str:
    separator = ":" if "=" not in scenario else ","
    return f"{scenario}{separator}model={model}"


def split_indices(task: str, ids: list[str]) -> tuple[list[int], list[int]]:
    ranked = sorted(
        range(len(ids)),
        key=lambda index: hash_rank(MANIFEST_ID, "split", task, str(ids[index])),
    )
    cut = len(ranked) // 2
    return sorted(ranked[:cut]), sorted(ranked[cut:])


def runtime_text(prompt: str, parent_output: str) -> str:
    return f"QUESTION_OR_SOURCE:\n{prompt}\nPARENT_OUTPUT:\n{parent_output}"


def support_count(examples: int, fraction: float) -> int:
    if fraction not in DISTANCES:
        raise ValueError(f"unfrozen distance {fraction}")
    if examples < 100:
        raise ValueError("Step 87B requires at least 100 examples per split")
    return max(1, int(math.floor(fraction * examples)))


def pools_from_seeds(seeds: list[int], queries: int, pool_size: int) -> np.ndarray:
    if pool_size > queries:
        raise ValueError("pool larger than task split")
    return np.asarray(
        [sorted(random.Random(seed).sample(range(queries), pool_size)) for seed in seeds],
        dtype=np.int64,
    )


def error_scores_oof(
    texts: list[str], utilities: np.ndarray, *, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    utilities = np.asarray(utilities, dtype=np.float64)
    target = 1.0 - utilities
    folds = min(5, len(texts))
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = np.zeros(len(texts), dtype=np.float64)
    for train, test in splitter.split(np.arange(len(texts))):
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=16000,
            sublinear_tf=True,
        )
        x_train = vectorizer.fit_transform([texts[index] for index in train])
        x_test = vectorizer.transform([texts[index] for index in test])
        model = Ridge(alpha=1.0, solver="lsqr")
        model.fit(x_train, target[train])
        scores[test] = model.predict(x_test)
    return scores, {
        "model": "tfidf_bigram_ridge_alpha1_lsqr",
        "folds": folds,
        "mean_parent_error": float(target.mean()),
        "oof_prediction_digest": hashlib.sha256(scores.tobytes()).hexdigest(),
    }


def error_scores_holdout(
    calibration_texts: list[str],
    calibration_utilities: np.ndarray,
    holdout_texts: list[str],
) -> np.ndarray:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=16000,
        sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform(calibration_texts)
    x_holdout = vectorizer.transform(holdout_texts)
    model = Ridge(alpha=1.0, solver="lsqr")
    model.fit(x_train, 1.0 - np.asarray(calibration_utilities, dtype=np.float64))
    return np.asarray(model.predict(x_holdout), dtype=np.float64)


def _supports(
    ordered: list[int], count: int, policy: str, aliases: int = ALIASES
) -> list[list[int]]:
    supports: list[list[int]] = []
    if policy == "disjoint_top":
        if len(ordered) < count * aliases:
            raise ValueError("insufficient positions for disjoint supports")
        for alias in range(aliases):
            supports.append(ordered[alias * count : (alias + 1) * count])
    elif policy == "overlap_top":
        stride = max(1, count // 2)
        if len(ordered) < (aliases - 1) * stride + count:
            raise ValueError("insufficient positions for overlap supports")
        for alias in range(aliases):
            supports.append(ordered[alias * stride : alias * stride + count])
    elif policy == "shared_top":
        if len(ordered) < count + aliases - 1:
            raise ValueError("insufficient positions for shared supports")
        base = ordered[:count]
        for alias in range(aliases):
            current = list(base)
            current[-1] = ordered[count - 1 + alias]
            supports.append(current)
    else:
        raise ValueError(policy)
    if len({tuple(values) for values in supports}) != aliases:
        raise AssertionError("support policy did not yield distinct supports")
    return supports


def opaque_token(task: str, alias: int) -> str:
    task_digest = hashlib.sha256(task.encode("utf-8")).hexdigest()[:16].upper()
    return f"ZZZQXVSTEP87B{task_digest}ALIAS{alias}ABSTAIN"


def alphanumeric_tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value).casefold()))


def validate_tokens_against_references(task: str, references: list[list[str]]) -> None:
    reference_tokens: set[str] = set()
    for rows in references:
        for value in rows:
            reference_tokens.update(alphanumeric_tokens(value))
    for alias in range(ALIASES):
        token = opaque_token(task, alias).casefold()
        if token in reference_tokens:
            raise AssertionError(f"{task}: opaque abstention token collides with a reference")


def build_alias_rows(
    *,
    task: str,
    ids: list[str],
    parent_responses: list[str],
    error_scores: np.ndarray,
    policy: str,
    distance: float,
) -> tuple[list[list[str]], list[list[int]]]:
    parent = [str(value) for value in parent_responses]
    count = support_count(len(parent), distance)
    ordered = sorted(
        range(len(parent)),
        key=lambda index: (
            -float(error_scores[index]),
            hash_rank(MANIFEST_ID, "support", task, str(ids[index])),
        ),
    )
    supports = _supports(ordered, count, policy)
    rows: list[list[str]] = []
    for alias, support in enumerate(supports):
        row = list(parent)
        token = opaque_token(task, alias)
        for index in support:
            if row[index] == token:
                raise AssertionError("opaque token already equals parent response")
            row[index] = token
        if sum(left != right for left, right in zip(row, parent)) != count:
            raise AssertionError("realized response distance mismatch")
        rows.append(row)
    hashes = {digest_json(row) for row in rows}
    if len(hashes) != ALIASES or digest_json(parent) in hashes:
        raise AssertionError("aliases are not four distinct non-parent rows")
    return rows, supports


def alias_utilities(parent: np.ndarray, supports: list[list[int]]) -> np.ndarray:
    parent = np.asarray(parent, dtype=np.float64)
    rows = []
    for support in supports:
        row = parent.copy()
        row[np.asarray(support, dtype=np.int64)] = 0.0
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def quality_summary(parent: np.ndarray, aliases: np.ndarray) -> dict[str, Any]:
    parent = np.asarray(parent, dtype=np.float64)
    aliases = np.asarray(aliases, dtype=np.float64)
    gaps = aliases.mean(axis=1) - parent.mean()
    return {
        "parent_mean_utility": float(parent.mean()),
        "alias_mean_utilities": [float(value) for value in aliases.mean(axis=1)],
        "alias_utility_gaps": [float(value) for value in gaps],
        "minimum_gap": float(gaps.min()),
        "maximum_gap": float(gaps.max()),
        "all_coordinatewise_nonimproving": bool(np.all(aliases <= parent[None, :] + 1e-12)),
    }


def bootstrap_ci(values: np.ndarray, seed: int, repetitions: int = 10000) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if np.allclose(values, values[0]):
        return [float(values[0]), float(values[0])]
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    cursor = 0
    while cursor < repetitions:
        count = min(500, repetitions - cursor)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[cursor : cursor + count] = values[indices].mean(axis=1)
        cursor += count
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def one_sided_sign_flip_pvalue(
    values: np.ndarray, seed: int, repetitions: int = 10000
) -> float:
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
    return float((extreme + 1.0) / (repetitions + 1.0))


def bh_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    count = len(ordered)
    output: dict[str, float] = {}
    running = 1.0
    for reverse_rank, name in enumerate(reversed(ordered), start=1):
        rank = count - reverse_rank + 1
        candidate = min(1.0, float(pvalues[name]) * count / rank)
        running = min(running, candidate)
        output[name] = float(running)
    return output

