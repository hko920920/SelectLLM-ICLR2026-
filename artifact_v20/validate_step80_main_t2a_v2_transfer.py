"""Independent validator for Step 80.

This file intentionally does not import the Step 80 runner. It reconstructs the
matrix-free aliases, V2 canonical acquisition, controls, statistics, and frozen
adjudication from locked upstream inputs and compares them with raw/result files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np

import run_step54_matrix_free_realism_audit as step54


ROOT = Path(__file__).resolve().parent
MANIFEST_ID = "STEP80_MAIN_T2A_V2_TRANSFER_V1"
PREREG = ROOT / "STEP80_MAIN_T2A_V2_TRANSFER_PREREGISTRATION_2026-08-10.json"
PREREG_LOCK = ROOT / "STEP80_MAIN_T2A_V2_TRANSFER_PREREGISTRATION_LOCK_2026-08-10.json"
EXECUTION_LOCK = ROOT / "STEP80_MAIN_T2A_V2_TRANSFER_EXECUTION_LOCK_2026-08-10.json"
RESULT = ROOT / "STEP80_MAIN_T2A_V2_TRANSFER_RESULTS_2026-08-10.json"
RAW = ROOT / "STEP80_MAIN_T2A_V2_TRANSFER_RAW_2026-08-10.npz"
VALIDATION = ROOT / "STEP80_MAIN_T2A_V2_TRANSFER_VALIDATION_2026-08-10.json"
RUNNER = ROOT / "run_step80_main_t2a_v2_transfer.py"
VALIDATOR = ROOT / "validate_step80_main_t2a_v2_transfer.py"
AUDITOR = ROOT / "audit_step80_main_t2a_v2_preoutcome.py"
PREOUTCOME_RESULT = ROOT / "STEP80_MAIN_T2A_V2_PREOUTCOME_AUDIT_RESULTS_2026-08-10.json"

TASKS = ("medqa", "gsm8k", "openbookqa")
QUERY_POLICIES = (
    "minimum_global_query_id_among_exact_minima",
    "maximum_global_query_id_among_exact_minima",
    "sha256_seeded_choice_among_exact_minima",
)
ROOT_POLICIES = (
    "minimum_canonical_root_id",
    "maximum_canonical_root_id",
    "sha256_seeded_root_choice",
)
PRIMARY_QUERY = QUERY_POLICIES[0]
PRIMARY_ROOT = ROOT_POLICIES[0]
RUN_SEEDS = tuple(range(91000, 91500))
PERM_SEEDS = tuple(range(803000, 803016))
EPS = 1e-12


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def keyed_hash(*values: object) -> str:
    return hashlib.sha256("|".join(map(str, values)).encode("utf-8")).hexdigest()


def vector_hash(row: np.ndarray) -> str:
    values = list(np.asarray(row, dtype=object))
    if not all(isinstance(value, str) for value in values):
        raise AssertionError("non-string response")
    return hashlib.sha256(
        json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def text_order(value: object) -> tuple[str, bytes]:
    if not isinstance(value, str):
        raise AssertionError("non-string response")
    payload = value.encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), payload


def normalize_registry(registry: dict[str, Any]) -> dict[str, Any]:
    labels = [str(value) for value in registry["labels"]]
    hashes = [vector_hash(row) for row in np.asarray(registry["responses"], dtype=object)]
    if hashes != [str(value) for value in registry["response_hashes"]]:
        raise AssertionError("registry vector hashes do not verify")
    permutation = sorted(
        range(len(labels)),
        key=lambda i: (hashes[i], int(registry["parents"][i]), keyed_hash(labels[i]), labels[i]),
    )
    response_rows = np.asarray(registry["responses"], dtype=object)[permutation]
    score_rows = np.asarray(registry["oracle"], dtype=np.float64)[permutation]
    parents = np.asarray(registry["parents"], dtype=np.int64)[permutation]
    sorted_labels = [labels[i] for i in permutation]
    sorted_hashes = [hashes[i] for i in permutation]
    q_count = response_rows.shape[1]
    codebook = np.zeros((q_count, len(permutation)), dtype=np.int16)
    for q in range(q_count):
        unique = sorted(set(response_rows[:, q]), key=text_order)
        lookup = {value: code for code, value in enumerate(unique)}
        codebook[q] = [lookup[value] for value in response_rows[:, q]]
    digest_rows = [
        {"response_hash": h, "parent": int(p), "label_hash": keyed_hash(label)}
        for h, p, label in zip(sorted_hashes, parents, sorted_labels)
    ]
    digest = hashlib.sha256(
        json.dumps(digest_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "responses": response_rows,
        "scores": score_rows,
        "parents": parents,
        "codes": codebook,
        "digest": digest,
    }


def query_tie_choice(
    rule: str, positions: np.ndarray, pool: np.ndarray, task: str, seed: int, step: int
) -> int:
    candidates = [int(pool[int(position)]) for position in positions]
    if rule == QUERY_POLICIES[0]:
        query = min(candidates)
    elif rule == QUERY_POLICIES[1]:
        query = max(candidates)
    elif rule == QUERY_POLICIES[2]:
        query = min(
            candidates,
            key=lambda q: (keyed_hash(MANIFEST_ID, task, seed, step, "query", q), q),
        )
    else:
        raise ValueError(rule)
    hits = np.flatnonzero(pool == query)
    if len(hits) != 1:
        raise AssertionError("non-unique pool")
    return int(hits[0])


def root_tie_choice(
    rule: str, entries: np.ndarray, parents: np.ndarray, task: str, seed: int, step: int
) -> int:
    candidates = sorted({int(parents[int(entry)]) for entry in entries})
    if rule == ROOT_POLICIES[0]:
        return min(candidates)
    if rule == ROOT_POLICIES[1]:
        return max(candidates)
    if rule == ROOT_POLICIES[2]:
        return min(
            candidates,
            key=lambda root: (
                keyed_hash(MANIFEST_ID, task, seed, step, "root", root),
                root,
            ),
        )
    raise ValueError(rule)


def replay_active(
    registry: dict[str, Any],
    source_scores: np.ndarray,
    pools: np.ndarray,
    task: str,
    budget: int,
    temperature: float,
    query_rule: str,
) -> dict[str, Any]:
    normalized = normalize_registry(registry)
    codes = normalized["codes"].astype(np.int64)
    scores = normalized["scores"]
    parents = normalized["parents"]
    source_scores = np.asarray(source_scores, dtype=np.float64)
    candidates = len(parents)
    picked_queries = np.full((500, budget), -1, dtype=np.int64)
    picked_values = np.full((500, budget), np.nan, dtype=np.float64)
    roots = {rule: np.full((500, budget), -1, dtype=np.int64) for rule in ROOT_POLICIES}
    regrets = {rule: np.zeros((500, budget), dtype=np.float64) for rule in ROOT_POLICIES}
    for run, seed in enumerate(RUN_SEEDS):
        pool = np.asarray(pools[run], dtype=np.int64)
        source_pool_scores = source_scores[:, pool].mean(axis=1)
        optimum = float(source_pool_scores.max())
        local_codes = codes[pool]
        bins = (
            np.arange(len(pool), dtype=np.int64)[:, None] * candidates + local_codes
        ).reshape(-1)
        remaining = np.ones(len(pool), dtype=bool)
        evidence = np.zeros(candidates, dtype=np.float64)
        for step_index in range(budget):
            logits = evidence / temperature
            logits -= logits.max()
            unnormalized = np.exp(logits)
            probability = unnormalized / math.fsum(float(x) for x in unnormalized)
            grouped = np.bincount(
                bins,
                weights=np.tile(probability, len(pool)),
                minlength=len(pool) * candidates,
            ).reshape(len(pool), candidates)
            grouped.sort(axis=1)
            criterion = np.sum(grouped * grouped, axis=1, dtype=np.float64)
            criterion[~remaining] = np.inf
            best_value = float(criterion.min())
            tied_positions = np.flatnonzero(criterion == best_value)
            position = query_tie_choice(
                query_rule, tied_positions, pool, task, seed, step_index
            )
            remaining[position] = False
            query = int(pool[position])
            picked_queries[run, step_index] = query
            picked_values[run, step_index] = criterion[position]
            evidence += scores[:, query]
            tied_entries = np.flatnonzero(evidence == float(evidence.max()))
            for root_rule in ROOT_POLICIES:
                root = root_tie_choice(
                    root_rule, tied_entries, parents, task, seed, step_index
                )
                roots[root_rule][run, step_index] = root
                regrets[root_rule][run, step_index] = optimum - source_pool_scores[root]
    return {
        "queried": picked_queries,
        "selected_values": picked_values,
        "roots": roots,
        "regrets": regrets,
        "digest": normalized["digest"],
    }


def replay_fixed(
    registry: dict[str, Any],
    source_scores: np.ndarray,
    pools: np.ndarray,
    queries: np.ndarray,
    task: str,
) -> dict[str, Any]:
    normalized = normalize_registry(registry)
    scores = normalized["scores"]
    parents = normalized["parents"]
    source_scores = np.asarray(source_scores, dtype=np.float64)
    budget = queries.shape[1]
    roots = {rule: np.full((500, budget), -1, dtype=np.int64) for rule in ROOT_POLICIES}
    regrets = {rule: np.zeros((500, budget), dtype=np.float64) for rule in ROOT_POLICIES}
    for run, seed in enumerate(RUN_SEEDS):
        source_pool_scores = source_scores[:, pools[run]].mean(axis=1)
        optimum = float(source_pool_scores.max())
        evidence = np.zeros(len(parents), dtype=np.float64)
        for step_index, query_value in enumerate(queries[run]):
            query = int(query_value)
            evidence += scores[:, query]
            tied_entries = np.flatnonzero(evidence == float(evidence.max()))
            for root_rule in ROOT_POLICIES:
                root = root_tie_choice(root_rule, tied_entries, parents, task, seed, step_index)
                roots[root_rule][run, step_index] = root
                regrets[root_rule][run, step_index] = optimum - source_pool_scores[root]
    return {
        "queried": np.asarray(queries, dtype=np.int64),
        "roots": roots,
        "regrets": regrets,
        "digest": normalized["digest"],
    }


def arrays_for(run: dict[str, Any], root: str = PRIMARY_ROOT) -> dict[str, np.ndarray]:
    regret = np.asarray(run["regrets"][root], dtype=np.float64)
    return {
        "queried": np.asarray(run["queried"], dtype=np.int64),
        "selected_root_path": np.asarray(run["roots"][root], dtype=np.int64),
        "root_regret_path": regret,
        "cumulative_root_regret": regret.sum(axis=1),
        "terminal_root_regret": regret[:, -1],
        "exact_auc_per_run": (regret <= EPS).mean(axis=1),
        "near_auc_per_run": (regret <= 0.01 + EPS).mean(axis=1),
    }


def boot(values: np.ndarray, seed: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    estimates = np.empty(10000, dtype=np.float64)
    for start in range(0, 10000, 500):
        count = min(500, 10000 - start)
        draw = generator.integers(0, len(values), size=(count, len(values)))
        estimates[start : start + count] = values[draw].mean(axis=1)
    return [float(x) for x in np.quantile(estimates, (0.025, 0.975))]


def flip(values: np.ndarray, seed: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    observed = abs(float(values.mean()))
    generator = np.random.default_rng(seed)
    extreme = 0
    for start in range(0, 10000, 500):
        count = min(500, 10000 - start)
        signs = 2 * generator.integers(0, 2, size=(count, len(values)), dtype=np.int8) - 1
        extreme += int(np.sum(np.abs((signs * values).mean(axis=1)) >= observed - 1e-15))
    return float((extreme + 1) / 10001)


def stat(values: np.ndarray, bootstrap_seed: int, sign_seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci95": boot(values, bootstrap_seed),
        "signflip_p": flip(values, sign_seed),
        "positive_fraction": float(np.mean(values > EPS)),
        "negative_fraction": float(np.mean(values < -EPS)),
        "zero_fraction": float(np.mean(np.abs(values) <= EPS)),
    }


def independent_public_registry(
    task: str, data: dict[str, Any], clean: dict[str, Any], target: int, distance: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    parent = np.asarray(clean["responses"][target], dtype=object)
    parent_scores = np.asarray(clean["oracle"][target], dtype=np.float64)
    correct = np.asarray(data["correct"], dtype=object)
    ids = list(data["ids"])
    options = data["public_options"]
    eligible = [
        q
        for q in range(len(ids))
        if step54.usable_wrong_coordinate(
            task, q, float(parent_scores[q]), parent, correct, options
        )
    ]
    rows: list[np.ndarray] = []
    supports: list[list[int]] = []
    for alias in range(4):
        ordered = sorted(
            eligible,
            key=lambda q: step54.hash_rank(
                step54.MANIFEST_ID, "public-support", task, alias, ids[q]
            ),
        )
        selected = [int(q) for q in ordered[:distance]]
        row = parent.copy()
        for rank, q in enumerate(selected):
            row[q] = step54.wrong_answer(
                task,
                query=q,
                rank=rank,
                alias=alias,
                example_id=ids[q],
                parent_response=parent[q],
                correct=correct[q],
                public_options=(options[q] if options is not None else None),
            )
        rows.append(row)
        supports.append(selected)
    registry = step54.step46.phase_a.make_scenario(
        np.concatenate([np.asarray(clean["responses"], dtype=object), np.asarray(rows, dtype=object)]),
        np.concatenate(
            [np.asarray(clean["oracle"], dtype=np.float64), np.asarray([parent_scores.copy() for _ in range(4)])]
        ),
        list(np.asarray(clean["parents"], dtype=np.int64)) + [target] * 4,
        list(clean["labels"])
        + [f"STEP54:matrix_free_public_reference_d{distance}:{step54.step46.phase_a.MODELS[target]}:{alias}" for alias in range(4)],
        f"independent_step80_{task}",
    )
    metadata = {
        "alias_hashes": [vector_hash(row) for row in rows],
        "support_digests": [step54.digest_jsonable(values) for values in supports],
        "distances": [int(np.sum(row != parent)) for row in rows],
        "oracle_mismatches": [0, 0, 0, 0],
    }
    return registry, metadata


def permuted(registry: dict[str, Any], seed: int) -> dict[str, Any]:
    order = np.random.default_rng(seed).permutation(len(registry["labels"]))
    return step54.step46.phase_a.make_scenario(
        np.asarray(registry["responses"], dtype=object)[order],
        np.asarray(registry["oracle"], dtype=np.float64)[order],
        [int(registry["parents"][i]) for i in order],
        [str(registry["labels"][i]) for i in order],
        "independent_permutation",
    )


def assert_array(raw: Any, key: str, expected: np.ndarray, checks: list[str]) -> None:
    observed = np.asarray(raw[key])
    expected = np.asarray(expected)
    if observed.dtype.kind == "f" or expected.dtype.kind == "f":
        if observed.shape != expected.shape or not np.array_equal(
            observed.view(np.uint64), expected.astype(np.float64, copy=False).view(np.uint64)
        ):
            raise AssertionError(f"raw float array mismatch: {key}")
    elif not np.array_equal(observed, expected):
        raise AssertionError(f"raw array mismatch: {key}")
    checks.append(key)


def assert_close(actual: Any, expected: Any, label: str, checks: list[str]) -> None:
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError(f"key mismatch: {label}")
        for key in expected:
            assert_close(actual[key], expected[key], f"{label}.{key}", checks)
        return
    if isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(f"list length mismatch: {label}")
        for index, value in enumerate(expected):
            assert_close(actual[index], value, f"{label}[{index}]", checks)
        return
    if isinstance(expected, float):
        if not np.isclose(float(actual), expected, rtol=0.0, atol=1e-12, equal_nan=True):
            raise AssertionError(f"float mismatch {label}: {actual} != {expected}")
    elif actual != expected:
        raise AssertionError(f"value mismatch {label}: {actual} != {expected}")
    checks.append(label)


def expected_method_summary(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    regret = arrays["root_regret_path"]
    exact = (regret <= EPS).mean(axis=0)
    near = (regret <= 0.01 + EPS).mean(axis=0)
    def labels(curve: np.ndarray, threshold: float) -> int | None:
        hits = np.flatnonzero(curve >= threshold)
        return None if len(hits) == 0 else int(hits[0] + 1)
    return {
        "mean_cumulative_root_regret": float(arrays["cumulative_root_regret"].mean()),
        "mean_normalized_cumulative_root_regret": float(arrays["cumulative_root_regret"].mean() / regret.shape[1]),
        "mean_terminal_root_regret": float(arrays["terminal_root_regret"].mean()),
        "exact_identification_curve": [float(x) for x in exact],
        "near_identification_curve": [float(x) for x in near],
        "exact_identification_auc": float(exact.mean()),
        "near_identification_auc": float(near.mean()),
        "labels_to_exact_identification": {str(t): labels(exact, t) for t in (0.70, 0.80, 0.90)},
        "near_best_definition": "root regret <= 0.01 + 1e-12; inherited descriptive V2 metric",
    }


def adjust(pvalues: dict[str, float]) -> dict[str, float]:
    names = sorted(pvalues, key=pvalues.get)
    n = len(names)
    answer: dict[str, float] = {}
    ceiling = 1.0
    for reverse_index, name in enumerate(reversed(names), 1):
        rank = n - reverse_index + 1
        ceiling = min(ceiling, min(1.0, pvalues[name] * n / rank))
        answer[name] = float(ceiling)
    return answer


def validate() -> None:
    if VALIDATION.exists():
        raise FileExistsError("Step80 validation output already exists")
    prereg = read_json(PREREG)
    execution_lock = read_json(EXECUTION_LOCK)
    result = read_json(RESULT)
    if result.get("manifest_id") != "STEP80_MAIN_T2A_V2_TRANSFER_RESULTS_V1":
        raise AssertionError("unexpected result manifest")
    code_hashes = {
        "runner": hash_file(RUNNER),
        "validator": hash_file(VALIDATOR),
        "preoutcome_auditor": hash_file(AUDITOR),
    }
    if execution_lock.get("code_sha256") != code_hashes:
        raise AssertionError("execution lock code mismatch")
    if result.get("runner_sha256") != code_hashes["runner"] or result.get("validator_sha256") != code_hashes["validator"]:
        raise AssertionError("result code binding mismatch")
    if result.get("raw_sha256") != hash_file(RAW):
        raise AssertionError("raw hash mismatch")
    if result.get("execution_lock_sha256") != hash_file(EXECUTION_LOCK):
        raise AssertionError("result execution-lock mismatch")
    if result.get("preoutcome_result_sha256") != hash_file(PREOUTCOME_RESULT):
        raise AssertionError("result pre-outcome binding mismatch")
    step54.step46.configure_legacy_paths()
    data_all = step54.step46.phase_a.load_all_data()
    historical = read_json(ROOT / "STEP54_MATRIX_FREE_REALISM_RESULTS_2026-08-09.json")
    checks: list[str] = []
    reconstructed: dict[str, Any] = {}
    with np.load(RAW, allow_pickle=False) as raw:
        for task_index, task in enumerate(TASKS):
            spec = prereg["frozen_tasks"][task]
            data = data_all[task]
            clean = step54.step46.phase_a.base_scenario(data)
            target = int(spec["target_index"])
            refined, alias_meta = independent_public_registry(
                task, data, clean, target, int(spec["distance_queries"])
            )
            reported_meta = result["tasks"][task]["constructor_metadata"]
            if alias_meta["alias_hashes"] != reported_meta["alias_response_hashes"]:
                raise AssertionError(f"{task}: independent alias hashes mismatch")
            if alias_meta["support_digests"] != reported_meta["changed_set_digests"]:
                raise AssertionError(f"{task}: independent supports mismatch")
            if alias_meta["distances"] != [int(spec["distance_queries"])] * 4:
                raise AssertionError(f"{task}: independent distance mismatch")
            pools = step54.step46.phase_a.pools_from_seeds(
                list(RUN_SEEDS), int(spec["examples"]), int(spec["pool_size"])
            )
            if step54.digest_array(pools) != historical["tasks"][task]["pool_matrix_digest"]:
                raise AssertionError(f"{task}: historical pool mismatch")
            fixed_queries = step54.fixed_random_queries(pools, int(spec["budget"]), task_index)
            random_queries = np.asarray(
                [
                    random.Random(seed + 99173).sample([int(x) for x in pool], int(spec["budget"]))
                    for seed, pool in zip(RUN_SEEDS, pools)
                ],
                dtype=np.int64,
            )
            clean_active = replay_active(
                clean, data["oracle"], pools, task, int(spec["budget"]), float(spec["temperature"]), PRIMARY_QUERY
            )
            refined_active = replay_active(
                refined, data["oracle"], pools, task, int(spec["budget"]), float(spec["temperature"]), PRIMARY_QUERY
            )
            random_run = replay_fixed(clean, data["oracle"], pools, random_queries, task)
            fixed_clean = replay_fixed(clean, data["oracle"], pools, fixed_queries, task)
            fixed_refined = replay_fixed(refined, data["oracle"], pools, fixed_queries, task)
            methods = {
                "clean_active_v2": arrays_for(clean_active),
                "matrix_free_refined_active_v2": arrays_for(refined_active),
                "clean_random_v2_root": arrays_for(random_run),
                "clean_fixed_query_v2_root": arrays_for(fixed_clean),
                "matrix_free_refined_fixed_query_v2_root": arrays_for(fixed_refined),
            }
            for method_name, arrays in methods.items():
                keys = result["tasks"][task]["array_keys"]["methods"][method_name]
                for array_name, expected in arrays.items():
                    assert_array(raw, keys[array_name], expected, checks)
                assert_close(
                    result["tasks"][task]["methods"][method_name],
                    expected_method_summary(arrays),
                    f"{task}.method.{method_name}",
                    checks,
                )
            active_penalty = methods["matrix_free_refined_active_v2"]["cumulative_root_regret"] - methods["clean_active_v2"]["cumulative_root_regret"]
            fixed_penalty = methods["matrix_free_refined_fixed_query_v2_root"]["cumulative_root_regret"] - methods["clean_fixed_query_v2_root"]["cumulative_root_regret"]
            values = {
                "clean_random_minus_clean_active_cumulative": methods["clean_random_v2_root"]["cumulative_root_regret"] - methods["clean_active_v2"]["cumulative_root_regret"],
                "refined_active_minus_clean_active_cumulative": active_penalty,
                "refined_active_minus_clean_random_cumulative": methods["matrix_free_refined_active_v2"]["cumulative_root_regret"] - methods["clean_random_v2_root"]["cumulative_root_regret"],
                "clean_active_minus_clean_random_exact_auc": methods["clean_active_v2"]["exact_auc_per_run"] - methods["clean_random_v2_root"]["exact_auc_per_run"],
                "refined_fixed_minus_clean_fixed_cumulative": fixed_penalty,
                "sequential_excess_cumulative": active_penalty - fixed_penalty,
            }
            for contrast_index, (name, vector) in enumerate(values.items()):
                key = result["tasks"][task]["array_keys"]["contrasts"][name]
                assert_array(raw, key, vector, checks)
                expected = stat(
                    vector,
                    804000 + task_index * 1000 + contrast_index * 10,
                    805000 + task_index * 1000 + contrast_index * 10,
                )
                expected["bh_q"] = result["tasks"][task]["contrasts"][name]["bh_q"]
                assert_close(result["tasks"][task]["contrasts"][name], expected, f"{task}.contrast.{name}", checks)

            permutation_pass = True
            for scenario_name, scenario, reference in (
                ("clean", clean, clean_active),
                ("matrix_free_refined", refined, refined_active),
            ):
                expected_mismatch = {
                    "query_mismatch_counts": np.zeros((16, 500), dtype=np.int64),
                    "root_mismatch_counts": np.zeros((16, 500), dtype=np.int64),
                    "regret_mismatch_counts": np.zeros((16, 500), dtype=np.int64),
                    "selected_value_mismatch_counts": np.zeros((16, 500), dtype=np.int64),
                    "canonical_digest_match": np.zeros(16, dtype=np.int8),
                }
                for pi, seed in enumerate(PERM_SEEDS):
                    rerun = replay_active(
                        permuted(scenario, seed), data["oracle"], pools, task,
                        int(spec["budget"]), float(spec["temperature"]), PRIMARY_QUERY
                    )
                    expected_mismatch["query_mismatch_counts"][pi] = np.sum(rerun["queried"] != reference["queried"], axis=1)
                    expected_mismatch["root_mismatch_counts"][pi] = np.sum(rerun["roots"][PRIMARY_ROOT] != reference["roots"][PRIMARY_ROOT], axis=1)
                    expected_mismatch["regret_mismatch_counts"][pi] = np.sum(rerun["regrets"][PRIMARY_ROOT].view(np.uint64) != reference["regrets"][PRIMARY_ROOT].view(np.uint64), axis=1)
                    expected_mismatch["selected_value_mismatch_counts"][pi] = np.sum(rerun["selected_values"].view(np.uint64) != reference["selected_values"].view(np.uint64), axis=1)
                    expected_mismatch["canonical_digest_match"][pi] = int(rerun["digest"] == reference["digest"])
                keys = result["tasks"][task]["permutation"]["scenarios"][scenario_name]["array_keys"]
                for name, expected in expected_mismatch.items():
                    assert_array(raw, keys[name], expected, checks)
                local_pass = bool(
                    all(int(x.sum()) == 0 for name, x in expected_mismatch.items() if name != "canonical_digest_match")
                    and np.all(expected_mismatch["canonical_digest_match"] == 1)
                )
                if result["tasks"][task]["permutation"]["scenarios"][scenario_name]["all_exact_checks_pass"] != local_pass:
                    raise AssertionError(f"{task}: permutation decision mismatch")
                permutation_pass = permutation_pass and local_pass
            if result["tasks"][task]["permutation"]["all_exact_checks_pass"] != permutation_pass:
                raise AssertionError(f"{task}: permutation aggregate mismatch")

            query_runs: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {PRIMARY_QUERY: (clean_active, refined_active)}
            for query_rule in QUERY_POLICIES[1:]:
                query_runs[query_rule] = (
                    replay_active(clean, data["oracle"], pools, task, int(spec["budget"]), float(spec["temperature"]), query_rule),
                    replay_active(refined, data["oracle"], pools, task, int(spec["budget"]), float(spec["temperature"]), query_rule),
                )
            for qi, query_rule in enumerate(QUERY_POLICIES):
                left, right = query_runs[query_rule]
                changed = np.sum(left["queried"] != right["queried"], axis=1)
                jaccard = np.asarray([
                    len(set(map(int, a)) & set(map(int, b))) / len(set(map(int, a)) | set(map(int, b)))
                    for a, b in zip(left["queried"], right["queried"])
                ])
                for ri, root_rule in enumerate(ROOT_POLICIES):
                    cumulative = right["regrets"][root_rule].sum(axis=1) - left["regrets"][root_rule].sum(axis=1)
                    terminal = right["regrets"][root_rule][:, -1] - left["regrets"][root_rule][:, -1]
                    cell_name = f"{query_rule}|{root_rule}"
                    cell = result["tasks"][task]["tie_grid"]["cells"][cell_name]
                    expected_arrays = {
                        "clean_queries": left["queried"],
                        "refined_queries": right["queried"],
                        "changed_positions": changed,
                        "query_jaccard": jaccard,
                        "clean_selected_roots": left["roots"][root_rule],
                        "refined_selected_roots": right["roots"][root_rule],
                        "clean_root_regret_path": left["regrets"][root_rule],
                        "refined_root_regret_path": right["regrets"][root_rule],
                        "delta_cumulative_root_regret": cumulative,
                        "delta_terminal_root_regret": terminal,
                    }
                    for name, expected in expected_arrays.items():
                        assert_array(raw, cell["array_keys"][name], expected, checks)
                    cumulative_stat = stat(cumulative, 904000 + task_index * 10000 + qi * 1000 + ri * 100, 905000 + task_index * 10000 + qi * 1000 + ri * 100)
                    terminal_stat = stat(terminal, 904050 + task_index * 10000 + qi * 1000 + ri * 100, 905050 + task_index * 10000 + qi * 1000 + ri * 100)
                    cumulative_stat["bh_q"] = cell["cumulative_root_regret_delta"]["bh_q"]
                    terminal_stat["bh_q"] = cell["terminal_root_regret_delta"]["bh_q"]
                    expected_cell = {
                        "query_policy": query_rule,
                        "root_policy": root_rule,
                        "path_change_fraction": float(np.mean(changed > 0)),
                        "mean_changed_query_positions": float(changed.mean()),
                        "mean_query_jaccard": float(jaccard.mean()),
                        "cumulative_root_regret_delta": cumulative_stat,
                        "terminal_root_regret_delta": terminal_stat,
                        "final_root_change_fraction": float(np.mean(left["roots"][root_rule][:, -1] != right["roots"][root_rule][:, -1])),
                        "array_keys": cell["array_keys"],
                    }
                    assert_close(cell, expected_cell, f"{task}.tie.{cell_name}", checks)
            reconstructed[task] = {"methods": methods, "contrasts": values}

    # Independently recompute BH values and the frozen classification.
    for contrast in result["tasks"][TASKS[0]]["contrasts"]:
        qvalues = adjust({task: result["tasks"][task]["contrasts"][contrast]["signflip_p"] for task in TASKS})
        for task in TASKS:
            assert_close(result["tasks"][task]["contrasts"][contrast]["bh_q"], qvalues[task], f"bh.{contrast}.{task}", checks)
    for query in QUERY_POLICIES:
        for root in ROOT_POLICIES:
            cell = f"{query}|{root}"
            for statistic in ("cumulative_root_regret_delta", "terminal_root_regret_delta"):
                qvalues = adjust({task: result["tasks"][task]["tie_grid"]["cells"][cell][statistic]["signflip_p"] for task in TASKS})
                for task in TASKS:
                    assert_close(result["tasks"][task]["tie_grid"]["cells"][cell][statistic]["bh_q"], qvalues[task], f"bh.{cell}.{statistic}.{task}", checks)

    def positive(task: str, contrast: str) -> bool:
        row = result["tasks"][task]["contrasts"][contrast]
        return row["mean"] > 0 and row["ci95"][0] > 0 and row["bh_q"] < 0.05
    integrity = all(all(result["tasks"][task]["binding_checks"].values()) for task in TASKS)
    permutation = all(result["tasks"][task]["permutation"]["all_exact_checks_pass"] for task in TASKS)
    active_count = sum(positive(task, "clean_random_minus_clean_active_cumulative") for task in TASKS)
    refinement_all = all(positive(task, "refined_active_minus_clean_active_cumulative") for task in TASKS)
    worse_count = sum(positive(task, "refined_active_minus_clean_random_cumulative") for task in TASKS)
    sequential_all = all(result["tasks"][task]["contrasts"]["sequential_excess_cumulative"]["ci95"][0] > 0 for task in TASKS)
    main_direction = all(result["tasks"][task]["contrasts"]["refined_active_minus_clean_active_cumulative"]["mean"] > 0 for task in TASKS)
    path_all = all(result["tasks"][task]["tie_grid"]["cells"][f"{q}|{r}"]["path_change_fraction"] >= 0.90 for task in TASKS for q in QUERY_POLICIES for r in ROOT_POLICIES)
    mean_all = all(result["tasks"][task]["tie_grid"]["cells"][f"{q}|{r}"]["cumulative_root_regret_delta"]["mean"] > 0 for task in TASKS for q in QUERY_POLICIES for r in ROOT_POLICIES)
    ci_all = all(result["tasks"][task]["tie_grid"]["cells"][f"{q}|{r}"]["cumulative_root_regret_delta"]["ci95"][0] > 0 for task in TASKS for q in QUERY_POLICIES for r in ROOT_POLICIES)
    terminal_counts: dict[str, dict[str, int]] = {}
    terminal_pass = True
    for task in ("medqa", "gsm8k"):
        terminal_counts[task] = {}
        for q in QUERY_POLICIES:
            count = sum(result["tasks"][task]["tie_grid"]["cells"][f"{q}|{r}"]["terminal_root_regret_delta"]["mean"] > 0 for r in ROOT_POLICIES)
            terminal_counts[task][q] = int(count)
            terminal_pass = terminal_pass and count >= 2
    first_six = integrity and active_count >= 2 and refinement_all and worse_count >= 2 and sequential_all and permutation and path_all and mean_all and ci_all
    if first_six and terminal_pass:
        classification = "STRONG_TRANSFER"
    elif first_six:
        classification = "CUMULATIVE_TRANSFER"
    elif integrity and permutation and main_direction:
        classification = "AUDIT_ONLY_TRANSFER"
    else:
        classification = "BLOCKED"
    if result["adjudication"]["classification"] != classification:
        raise AssertionError("classification mismatch")
    if result["adjudication"]["terminal_positive_root_policy_counts"] != terminal_counts:
        raise AssertionError("terminal-count mismatch")
    validation = {
        "manifest_id": "STEP80_MAIN_T2A_V2_TRANSFER_VALIDATION_V1",
        "status": "PASS_STEP80_INDEPENDENT_VALIDATION",
        "result_sha256": hash_file(RESULT),
        "raw_sha256": hash_file(RAW),
        "execution_lock_sha256": hash_file(EXECUTION_LOCK),
        "runner_sha256": hash_file(RUNNER),
        "validator_sha256": hash_file(VALIDATOR),
        "runner_imported": False,
        "independent_alias_reconstruction": True,
        "independent_acquisition_recomputation": True,
        "independent_statistics_recomputation": True,
        "independent_adjudication": True,
        "checks_completed": len(checks),
        "validated_classification": classification,
        "final_status": "INDEPENDENTLY_VALIDATED",
    }
    VALIDATION.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2))


def preflight() -> None:
    source = VALIDATOR.read_text(encoding="utf-8")
    forbidden = "import " + "run_step80_main_t2a_v2_transfer"
    if forbidden in source:
        raise AssertionError("validator imports runner")
    prereg = read_json(PREREG)
    lock = read_json(PREREG_LOCK)
    if prereg.get("manifest_id") != MANIFEST_ID:
        raise AssertionError("preregistration mismatch")
    if lock.get("manifest_id") != "STEP80_MAIN_T2A_V2_TRANSFER_PREREGISTRATION_LOCK_V1":
        raise AssertionError("preregistration lock mismatch")
    if RESULT.exists() or RAW.exists() or VALIDATION.exists():
        raise FileExistsError("reserved output exists during validator preflight")
    print(json.dumps({"status": "PASS_STEP80_VALIDATOR_PREFLIGHT", "outcomes_run": False, "runner_imported": False}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight()
    else:
        validate()


if __name__ == "__main__":
    main()
