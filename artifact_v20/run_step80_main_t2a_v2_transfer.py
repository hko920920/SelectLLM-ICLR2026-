"""Step 80: V2 transfer on the paper's exact Step 54 matrix-free condition.

Scientific outputs are refused until a separate execution lock binds this runner,
the independently implemented validator, and the pre-outcome audit evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

import run_step54_matrix_free_realism_audit as step54


ROOT = Path(__file__).resolve().parent
MANIFEST_ID = "STEP80_MAIN_T2A_V2_TRANSFER_V1"
PREREG = ROOT / "STEP80_MAIN_T2A_V2_TRANSFER_PREREGISTRATION_2026-08-10.json"
PROTOCOL = ROOT / "STEP80_MAIN_T2A_V2_TRANSFER_PROTOCOL_2026-08-10.md"
SCOPE_AUDIT = ROOT / "STEP80_MAIN_T2A_V2_SCOPE_GAP_AUDIT_2026-08-10.md"
PREREG_LOCK = ROOT / "STEP80_MAIN_T2A_V2_TRANSFER_PREREGISTRATION_LOCK_2026-08-10.json"
EXECUTION_LOCK = ROOT / "STEP80_MAIN_T2A_V2_TRANSFER_EXECUTION_LOCK_2026-08-10.json"
PREOUTCOME_RESULT = ROOT / "STEP80_MAIN_T2A_V2_PREOUTCOME_AUDIT_RESULTS_2026-08-10.json"
RUNNER = ROOT / "run_step80_main_t2a_v2_transfer.py"
VALIDATOR = ROOT / "validate_step80_main_t2a_v2_transfer.py"
AUDITOR = ROOT / "audit_step80_main_t2a_v2_preoutcome.py"
OUT = ROOT / "STEP80_MAIN_T2A_V2_TRANSFER_RESULTS_2026-08-10.json"
RAW = ROOT / "STEP80_MAIN_T2A_V2_TRANSFER_RAW_2026-08-10.npz"

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
PRIMARY_QUERY_POLICY = QUERY_POLICIES[0]
PRIMARY_ROOT_POLICY = ROOT_POLICIES[0]
PAIRED_SEEDS = tuple(range(91000, 91500))
PERMUTATION_SEEDS = tuple(range(803000, 803016))
TOL = 1e-12

EXPECTED_FROZEN_SHA256 = {
    "preregistration": "a4644ece8c9e36c66469b95301978963dc0bf4c36142f3416fcf2a1e75e6e607",
    "protocol": "e0f45b8d3a8bdff4544a850889d03b49fd63a57aac51a80e18ffdf1f401309e3",
    "scope_audit": "0ded1d19e7733d1c49b36fc46fd087c332257c9e12161d76bf805c9371bd3ac9",
    "preregistration_lock": "6645607e15f871ffc06885ba6d87963ddeb736bb800c3b375c1dbf96f094c56c",
    "step54_preregistration": "cf45cdd3e6b1425db2f2c17d4e4283bde95d58356a3fd1ade09c2ef6813e8ed9",
    "step54_execution_lock": "2d213d8090b35a4cd14edc6292ddc3d73e934e288c14a24743daabf5b32d5bb3",
    "step54_results": "4ab5428c368991f8c9696edda63660ac76b1d516bb7c78d52eeb553ad56a41dd",
    "step54_runner": "f6fedf37dcbbef82b68353526b7ecd5b0f2156fb85918574fda139db23df4f7d",
    "v2_runner": "4f8059efc834c9983b5856217f5319672c0e6cd9c28e6ccd9adad8fadd3c7577",
}

FROZEN_FILES = {
    "preregistration": PREREG,
    "protocol": PROTOCOL,
    "scope_audit": SCOPE_AUDIT,
    "preregistration_lock": PREREG_LOCK,
    "step54_preregistration": ROOT / "STEP54_MATRIX_FREE_REALISM_PREREGISTRATION_2026-08-09.json",
    "step54_execution_lock": ROOT / "STEP54_MATRIX_FREE_REALISM_EXECUTION_LOCK_2026-08-09.json",
    "step54_results": ROOT / "STEP54_MATRIX_FREE_REALISM_RESULTS_2026-08-09.json",
    "step54_runner": ROOT / "run_step54_matrix_free_realism_audit.py",
    "v2_runner": ROOT / "run_step77_v2_stable_acquisition.py",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_digest(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\x00" + array.tobytes(order="C")).hexdigest()


def verify_frozen_inputs() -> tuple[dict[str, Any], dict[str, str]]:
    observed = {name: sha256_file(path) for name, path in FROZEN_FILES.items()}
    if observed != EXPECTED_FROZEN_SHA256:
        raise AssertionError({"expected": EXPECTED_FROZEN_SHA256, "observed": observed})
    prereg = load_json(PREREG)
    if prereg.get("manifest_id") != MANIFEST_ID:
        raise AssertionError("unexpected Step80 preregistration")
    if prereg.get("status") != "PREREGISTERED_BEFORE_IMPLEMENTATION_OR_TRANSFER_OUTCOMES":
        raise AssertionError("Step80 preregistration status drift")
    lock = load_json(PREREG_LOCK)
    if lock.get("manifest_id") != "STEP80_MAIN_T2A_V2_TRANSFER_PREREGISTRATION_LOCK_V1":
        raise AssertionError("unexpected Step80 preregistration lock")
    for name, record in lock["locked_documents"].items():
        mapped = {"scope_gap_audit": "scope_audit"}.get(name, name)
        if observed[mapped] != record["sha256"]:
            raise AssertionError(f"preregistration lock mismatch: {name}")
    return prereg, observed


def verify_execution_lock() -> dict[str, Any]:
    if not EXECUTION_LOCK.exists():
        raise FileNotFoundError("Step80 execution lock is absent")
    lock = load_json(EXECUTION_LOCK)
    if lock.get("manifest_id") != "STEP80_MAIN_T2A_V2_TRANSFER_EXECUTION_LOCK_V1":
        raise AssertionError("unexpected Step80 execution lock")
    expected_code = {
        "runner": sha256_file(RUNNER),
        "validator": sha256_file(VALIDATOR),
        "preoutcome_auditor": sha256_file(AUDITOR),
    }
    if lock.get("code_sha256") != expected_code:
        raise AssertionError({"locked_code": lock.get("code_sha256"), "observed": expected_code})
    if lock.get("frozen_input_sha256") != EXPECTED_FROZEN_SHA256:
        raise AssertionError("execution lock frozen-input mismatch")
    audit = load_json(PREOUTCOME_RESULT)
    if audit.get("status") != "PASS_STEP80_PREOUTCOME_AUDIT":
        raise AssertionError("pre-outcome audit did not pass")
    if audit.get("outcomes_run") is not False:
        raise AssertionError("pre-outcome audit claims outcomes were run")
    if audit.get("code_sha256") != expected_code:
        raise AssertionError("pre-outcome audit is stale")
    if lock.get("preoutcome_result_sha256") != sha256_file(PREOUTCOME_RESULT):
        raise AssertionError("execution lock does not bind pre-outcome evidence")
    return lock


def response_key(value: object) -> tuple[str, bytes]:
    if not isinstance(value, str):
        raise AssertionError("responses must be exact strings")
    encoded = value.encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), encoded


def response_vector_hash(row: np.ndarray) -> str:
    values = list(np.asarray(row, dtype=object))
    if not all(isinstance(value, str) for value in values):
        raise AssertionError("response vectors must contain exact strings")
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_entry_order(scenario: dict[str, Any]) -> list[int]:
    labels = [str(value) for value in scenario["labels"]]
    if len(labels) != len(set(labels)):
        raise AssertionError("candidate labels must be unique")
    responses = np.asarray(scenario["responses"], dtype=object)
    expected = [response_vector_hash(row) for row in responses]
    if expected != [str(value) for value in scenario["response_hashes"]]:
        raise AssertionError("response-vector hash mismatch")
    return sorted(
        range(len(labels)),
        key=lambda index: (
            expected[index],
            int(scenario["parents"][index]),
            stable_digest(labels[index]),
            labels[index],
        ),
    )


def canonical_encode_responses(responses: np.ndarray) -> np.ndarray:
    responses = np.asarray(responses, dtype=object)
    entries, queries = responses.shape
    codes = np.zeros((queries, entries), dtype=np.int16)
    for query in range(queries):
        column = list(responses[:, query])
        if not all(isinstance(value, str) for value in column):
            raise AssertionError("responses must be exact strings")
        values = sorted(set(column), key=response_key)
        mapping = {value: index for index, value in enumerate(values)}
        for entry in range(entries):
            codes[query, entry] = mapping[responses[entry, query]]
    return codes


def canonical_scenario(scenario: dict[str, Any], kind: str) -> dict[str, Any]:
    order = canonical_entry_order(scenario)
    responses = np.asarray(scenario["responses"], dtype=object)[order]
    oracle = np.asarray(scenario["oracle"], dtype=np.float64)[order]
    parents = np.asarray(scenario["parents"], dtype=np.int64)[order]
    labels = [str(scenario["labels"][index]) for index in order]
    hashes = [str(scenario["response_hashes"][index]) for index in order]
    digest_payload = [
        {"response_hash": h, "parent": int(parent), "label_hash": stable_digest(label)}
        for h, parent, label in zip(hashes, parents, labels)
    ]
    return {
        "kind": kind,
        "responses": responses,
        "oracle": oracle,
        "parents": parents,
        "labels": labels,
        "response_hashes": hashes,
        "codes": canonical_encode_responses(responses),
        "canonical_digest": hashlib.sha256(
            json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def choose_query_position(
    policy: str,
    tied_positions: np.ndarray,
    pool: np.ndarray,
    task: str,
    paired_seed: int,
    step: int,
) -> int:
    queries = [int(pool[int(position)]) for position in tied_positions]
    if policy == QUERY_POLICIES[0]:
        selected = min(queries)
    elif policy == QUERY_POLICIES[1]:
        selected = max(queries)
    elif policy == QUERY_POLICIES[2]:
        selected = min(
            queries,
            key=lambda query: (
                stable_digest(MANIFEST_ID, task, paired_seed, step, "query", query),
                query,
            ),
        )
    else:
        raise ValueError(policy)
    matches = np.flatnonzero(pool == selected)
    if len(matches) != 1:
        raise AssertionError("pool queries must be unique")
    return int(matches[0])


def choose_root(
    policy: str,
    tied_entries: np.ndarray,
    parents: np.ndarray,
    task: str,
    paired_seed: int,
    step: int,
) -> int:
    roots = sorted(set(int(parents[int(entry)]) for entry in tied_entries))
    if policy == ROOT_POLICIES[0]:
        return min(roots)
    if policy == ROOT_POLICIES[1]:
        return max(roots)
    if policy == ROOT_POLICIES[2]:
        return min(
            roots,
            key=lambda root: (
                stable_digest(MANIFEST_ID, task, paired_seed, step, "root", root),
                root,
            ),
        )
    raise ValueError(policy)


def active_runs(
    scenario: dict[str, Any],
    original_oracle: np.ndarray,
    pools: np.ndarray,
    paired_seeds: tuple[int, ...],
    budget: int,
    tau: float,
    task: str,
    query_policy: str,
) -> dict[str, Any]:
    canonical = canonical_scenario(scenario, f"step80_{task}_{scenario['kind']}")
    codes = np.asarray(canonical["codes"], dtype=np.int64)
    oracle = np.asarray(canonical["oracle"], dtype=np.float64)
    parents = np.asarray(canonical["parents"], dtype=np.int64)
    original_oracle = np.asarray(original_oracle, dtype=np.float64)
    entries = len(parents)
    queried = np.full((len(pools), budget), -1, dtype=np.int64)
    selected_values = np.full((len(pools), budget), np.nan, dtype=np.float64)
    selected_roots = {
        policy: np.full((len(pools), budget), -1, dtype=np.int64)
        for policy in ROOT_POLICIES
    }
    root_regrets = {
        policy: np.zeros((len(pools), budget), dtype=np.float64)
        for policy in ROOT_POLICIES
    }
    for run_index, paired_seed in enumerate(paired_seeds):
        pool = np.asarray(pools[run_index], dtype=np.int64)
        original_scores = original_oracle[:, pool].mean(axis=1)
        best_original = float(original_scores.max())
        pool_codes = codes[pool]
        flat_bins = (
            np.arange(len(pool), dtype=np.int64)[:, None] * entries + pool_codes
        ).ravel()
        active = np.ones(len(pool), dtype=bool)
        cumulative = np.zeros(entries, dtype=np.float64)
        for step_index in range(budget):
            shifted = cumulative / tau
            shifted -= shifted.max()
            exponentials = np.exp(shifted)
            denominator = math.fsum(float(value) for value in exponentials)
            posterior = exponentials / denominator
            masses = np.bincount(
                flat_bins,
                weights=np.tile(posterior, len(pool)),
                minlength=len(pool) * entries,
            ).reshape(len(pool), entries)
            ordered = np.sort(masses, axis=1)
            acquisition = np.sum(ordered * ordered, axis=1, dtype=np.float64)
            acquisition[~active] = np.inf
            minimum = float(acquisition.min())
            tied = np.flatnonzero(acquisition == minimum)
            position = choose_query_position(
                query_policy, tied, pool, task, int(paired_seed), step_index
            )
            active[position] = False
            query = int(pool[position])
            queried[run_index, step_index] = query
            selected_values[run_index, step_index] = acquisition[position]
            cumulative += oracle[:, query]
            maximum = float(cumulative.max())
            tied_entries = np.flatnonzero(cumulative == maximum)
            for root_policy in ROOT_POLICIES:
                root = choose_root(
                    root_policy,
                    tied_entries,
                    parents,
                    task,
                    int(paired_seed),
                    step_index,
                )
                selected_roots[root_policy][run_index, step_index] = root
                root_regrets[root_policy][run_index, step_index] = best_original - original_scores[root]
    return {
        "queried": queried,
        "selected_acquisition_values": selected_values,
        "selected_root_paths": selected_roots,
        "root_regret_paths": root_regrets,
        "canonical_digest": canonical["canonical_digest"],
    }


def fixed_runs(
    scenario: dict[str, Any],
    original_oracle: np.ndarray,
    pools: np.ndarray,
    paired_seeds: tuple[int, ...],
    queries: np.ndarray,
    task: str,
) -> dict[str, Any]:
    canonical = canonical_scenario(scenario, f"step80_{task}_fixed")
    oracle = np.asarray(canonical["oracle"], dtype=np.float64)
    parents = np.asarray(canonical["parents"], dtype=np.int64)
    original_oracle = np.asarray(original_oracle, dtype=np.float64)
    runs, budget = queries.shape
    roots = {
        policy: np.full((runs, budget), -1, dtype=np.int64) for policy in ROOT_POLICIES
    }
    regrets = {
        policy: np.zeros((runs, budget), dtype=np.float64) for policy in ROOT_POLICIES
    }
    for run_index, paired_seed in enumerate(paired_seeds):
        pool = np.asarray(pools[run_index], dtype=np.int64)
        scores = original_oracle[:, pool].mean(axis=1)
        best = float(scores.max())
        cumulative = np.zeros(len(parents), dtype=np.float64)
        for step_index in range(budget):
            query = int(queries[run_index, step_index])
            cumulative += oracle[:, query]
            tied = np.flatnonzero(cumulative == float(cumulative.max()))
            for root_policy in ROOT_POLICIES:
                root = choose_root(
                    root_policy, tied, parents, task, int(paired_seed), step_index
                )
                roots[root_policy][run_index, step_index] = root
                regrets[root_policy][run_index, step_index] = best - scores[root]
    return {
        "queried": np.asarray(queries, dtype=np.int64),
        "selected_root_paths": roots,
        "root_regret_paths": regrets,
        "canonical_digest": canonical["canonical_digest"],
    }


def primary_view(run: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "queried": np.asarray(run["queried"], dtype=np.int64),
        "selected_root_path": np.asarray(run["selected_root_paths"][PRIMARY_ROOT_POLICY]),
        "root_regret_path": np.asarray(run["root_regret_paths"][PRIMARY_ROOT_POLICY]),
    }


def method_arrays(run: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    regret = np.asarray(run["root_regret_path"], dtype=np.float64)
    exact = regret <= TOL
    near = regret <= 0.01 + TOL
    return {
        "queried": np.asarray(run["queried"], dtype=np.int64),
        "selected_root_path": np.asarray(run["selected_root_path"], dtype=np.int64),
        "root_regret_path": regret,
        "cumulative_root_regret": regret.sum(axis=1),
        "terminal_root_regret": regret[:, -1],
        "exact_auc_per_run": exact.mean(axis=1),
        "near_auc_per_run": near.mean(axis=1),
    }


def labels_to_threshold(curve: np.ndarray, threshold: float) -> int | None:
    positions = np.flatnonzero(np.asarray(curve) >= threshold)
    return None if len(positions) == 0 else int(positions[0] + 1)


def summarize_method(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    regret = arrays["root_regret_path"]
    exact_curve = (regret <= TOL).mean(axis=0)
    near_curve = (regret <= 0.01 + TOL).mean(axis=0)
    return {
        "mean_cumulative_root_regret": float(arrays["cumulative_root_regret"].mean()),
        "mean_normalized_cumulative_root_regret": float(
            arrays["cumulative_root_regret"].mean() / regret.shape[1]
        ),
        "mean_terminal_root_regret": float(arrays["terminal_root_regret"].mean()),
        "exact_identification_curve": [float(value) for value in exact_curve],
        "near_identification_curve": [float(value) for value in near_curve],
        "exact_identification_auc": float(exact_curve.mean()),
        "near_identification_auc": float(near_curve.mean()),
        "labels_to_exact_identification": {
            str(threshold): labels_to_threshold(exact_curve, threshold)
            for threshold in (0.70, 0.80, 0.90)
        },
        "near_best_definition": "root regret <= 0.01 + 1e-12; inherited descriptive V2 metric",
    }


def bootstrap_ci(values: np.ndarray, seed: int, repetitions: int = 10_000) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    cursor = 0
    while cursor < repetitions:
        count = min(500, repetitions - cursor)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        estimates[cursor : cursor + count] = values[indices].mean(axis=1)
        cursor += count
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def signflip_pvalue(values: np.ndarray, seed: int, repetitions: int = 10_000) -> float:
    values = np.asarray(values, dtype=np.float64)
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    extreme = 0
    cursor = 0
    while cursor < repetitions:
        count = min(500, repetitions - cursor)
        signs = rng.integers(0, 2, size=(count, len(values)), dtype=np.int8) * 2 - 1
        estimates = np.abs((signs * values[None, :]).mean(axis=1))
        extreme += int(np.sum(estimates >= observed - 1e-15))
        cursor += count
    return float((extreme + 1.0) / (repetitions + 1.0))


def vector_summary(values: np.ndarray, bootstrap_seed: int, signflip_seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci95": bootstrap_ci(values, bootstrap_seed),
        "signflip_p": signflip_pvalue(values, signflip_seed),
        "positive_fraction": float(np.mean(values > TOL)),
        "negative_fraction": float(np.mean(values < -TOL)),
        "zero_fraction": float(np.mean(np.abs(values) <= TOL)),
    }


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


def store_arrays(raw: dict[str, np.ndarray], prefix: str, arrays: dict[str, np.ndarray]) -> dict[str, str]:
    keys: dict[str, str] = {}
    for name, array in arrays.items():
        key = f"{prefix}__{name}"
        raw[key] = np.asarray(array)
        keys[name] = key
    return keys


def permute_scenario(scenario: dict[str, Any], seed: int, kind: str) -> dict[str, Any]:
    order = np.random.default_rng(seed).permutation(len(scenario["labels"]))
    return step54.step46.phase_a.make_scenario(
        np.asarray(scenario["responses"], dtype=object)[order],
        np.asarray(scenario["oracle"], dtype=np.float64)[order],
        [int(scenario["parents"][index]) for index in order],
        [str(scenario["labels"][index]) for index in order],
        kind,
    )


def load_task_context(
    task: str,
    task_index: int,
    spec: dict[str, Any],
    historical: dict[str, Any],
) -> dict[str, Any]:
    phase_a = step54.step46.phase_a
    data = historical["all_data"][task]
    clean = phase_a.base_scenario(data)
    target = int(spec["target_index"])
    if phase_a.MODELS[target] != spec["target"]:
        raise AssertionError(f"{task}: target drift")
    if clean["responses"].shape != (31, int(spec["examples"])):
        raise AssertionError(f"{task}: clean shape drift")
    refined, metadata = step54.build_public_reference_variant(
        task,
        data,
        clean,
        target=target,
        distance_queries=int(spec["distance_queries"]),
    )
    if refined["responses"].shape != (35, int(spec["examples"])):
        raise AssertionError(f"{task}: refined shape drift")
    old_task = historical["step54_results"]["tasks"][task]
    old_meta = old_task["construction"]["matrix_free_public_reference_5pct"]
    if metadata != old_meta:
        raise AssertionError(f"{task}: Step54 constructor metadata drift")
    data_digests = {
        "ids": step54.digest_jsonable(list(data["ids"])),
        "responses": step54.digest_jsonable(np.asarray(data["responses"], dtype=object).tolist()),
        "oracle": step54.digest_array(np.asarray(data["oracle"], dtype=np.float64)),
        "correct": step54.digest_jsonable(np.asarray(data["correct"], dtype=object).tolist()),
    }
    if data_digests != old_task["data_digests"]:
        raise AssertionError(f"{task}: Step54 data digest drift")
    pools = phase_a.pools_from_seeds(
        list(PAIRED_SEEDS), int(spec["examples"]), int(spec["pool_size"])
    )
    if step54.digest_array(pools) != old_task["pool_matrix_digest"]:
        raise AssertionError(f"{task}: Step54 pool digest drift")
    fixed_queries = step54.fixed_random_queries(pools, int(spec["budget"]), task_index)
    if step54.digest_array(fixed_queries) != old_task["fixed_query_control"]["query_matrix_digest_clean"]:
        raise AssertionError(f"{task}: Step54 fixed-query digest drift")
    checks = {
        "constructor_integrity_pass": metadata["constructor_integrity_pass"] is True,
        "zero_peer_response_rows": int(metadata["competitor_response_rows_read_by_constructor"]) == 0,
        "per_example_quality_equal": metadata["per_example_quality_equal"] is True,
        "construction_before_pool_generation": old_task["construction_completed_before_pool_generation"] is True,
        "alias_count_exact": int(metadata["aliases"]) == 4,
        "distance_exact": int(metadata["distance_queries"]) == int(spec["distance_queries"]),
        "clean_entries_exact": len(clean["labels"]) == 31,
        "refined_entries_exact": len(refined["labels"]) == 35,
        "pool_count_exact": len(pools) == 500,
    }
    if not all(checks.values()):
        raise AssertionError({task: checks})
    return {
        "task": task,
        "data": data,
        "clean": clean,
        "refined": refined,
        "metadata": metadata,
        "pools": pools,
        "fixed_queries": fixed_queries,
        "checks": checks,
        "data_digests": data_digests,
        "pool_digest": step54.digest_array(pools),
        "fixed_query_digest": step54.digest_array(fixed_queries),
    }


def evaluate_task(
    task: str,
    task_index: int,
    spec: dict[str, Any],
    context: dict[str, Any],
    raw: dict[str, np.ndarray],
) -> dict[str, Any]:
    budget = int(spec["budget"])
    tau = float(spec["temperature"])
    data = context["data"]
    pools = context["pools"]
    clean = context["clean"]
    refined = context["refined"]
    clean_active = active_runs(
        clean, data["oracle"], pools, PAIRED_SEEDS, budget, tau, task, PRIMARY_QUERY_POLICY
    )
    refined_active = active_runs(
        refined, data["oracle"], pools, PAIRED_SEEDS, budget, tau, task, PRIMARY_QUERY_POLICY
    )
    random_queries = np.asarray(
        [
            random.Random(seed + 99173).sample([int(value) for value in pool], budget)
            for seed, pool in zip(PAIRED_SEEDS, pools)
        ],
        dtype=np.int64,
    )
    clean_random = fixed_runs(clean, data["oracle"], pools, PAIRED_SEEDS, random_queries, task)
    fixed_clean = fixed_runs(
        clean, data["oracle"], pools, PAIRED_SEEDS, context["fixed_queries"], task
    )
    fixed_refined = fixed_runs(
        refined, data["oracle"], pools, PAIRED_SEEDS, context["fixed_queries"], task
    )
    method_runs = {
        "clean_active_v2": primary_view(clean_active),
        "matrix_free_refined_active_v2": primary_view(refined_active),
        "clean_random_v2_root": primary_view(clean_random),
        "clean_fixed_query_v2_root": primary_view(fixed_clean),
        "matrix_free_refined_fixed_query_v2_root": primary_view(fixed_refined),
    }
    methods = {name: method_arrays(run) for name, run in method_runs.items()}
    method_keys = {
        name: store_arrays(raw, f"step80__{task}__method__{name}", arrays)
        for name, arrays in methods.items()
    }
    active_penalty = (
        methods["matrix_free_refined_active_v2"]["cumulative_root_regret"]
        - methods["clean_active_v2"]["cumulative_root_regret"]
    )
    fixed_penalty = (
        methods["matrix_free_refined_fixed_query_v2_root"]["cumulative_root_regret"]
        - methods["clean_fixed_query_v2_root"]["cumulative_root_regret"]
    )
    contrast_values = {
        "clean_random_minus_clean_active_cumulative": (
            methods["clean_random_v2_root"]["cumulative_root_regret"]
            - methods["clean_active_v2"]["cumulative_root_regret"]
        ),
        "refined_active_minus_clean_active_cumulative": active_penalty,
        "refined_active_minus_clean_random_cumulative": (
            methods["matrix_free_refined_active_v2"]["cumulative_root_regret"]
            - methods["clean_random_v2_root"]["cumulative_root_regret"]
        ),
        "clean_active_minus_clean_random_exact_auc": (
            methods["clean_active_v2"]["exact_auc_per_run"]
            - methods["clean_random_v2_root"]["exact_auc_per_run"]
        ),
        "refined_fixed_minus_clean_fixed_cumulative": fixed_penalty,
        "sequential_excess_cumulative": active_penalty - fixed_penalty,
    }
    contrasts: dict[str, Any] = {}
    contrast_keys: dict[str, str] = {}
    for contrast_index, (name, values) in enumerate(contrast_values.items()):
        key = f"step80__{task}__contrast__{name}"
        raw[key] = np.asarray(values, dtype=np.float64)
        contrast_keys[name] = key
        contrasts[name] = vector_summary(
            values,
            804000 + task_index * 1000 + contrast_index * 10,
            805000 + task_index * 1000 + contrast_index * 10,
        )

    permutation: dict[str, Any] = {"seeds": list(PERMUTATION_SEEDS), "scenarios": {}}
    for scenario_name, scenario, reference in (
        ("clean", clean, clean_active),
        ("matrix_free_refined", refined, refined_active),
    ):
        mismatch_arrays = {
            "query_mismatch_counts": np.zeros((16, 500), dtype=np.int64),
            "root_mismatch_counts": np.zeros((16, 500), dtype=np.int64),
            "regret_mismatch_counts": np.zeros((16, 500), dtype=np.int64),
            "selected_value_mismatch_counts": np.zeros((16, 500), dtype=np.int64),
            "canonical_digest_match": np.zeros(16, dtype=np.int8),
        }
        reference_roots = reference["selected_root_paths"][PRIMARY_ROOT_POLICY]
        reference_regret = reference["root_regret_paths"][PRIMARY_ROOT_POLICY]
        reference_values = reference["selected_acquisition_values"]
        for permutation_index, seed in enumerate(PERMUTATION_SEEDS):
            permuted = permute_scenario(scenario, seed, f"step80_{task}_{scenario_name}_{seed}")
            rerun = active_runs(
                permuted,
                data["oracle"],
                pools,
                PAIRED_SEEDS,
                budget,
                tau,
                task,
                PRIMARY_QUERY_POLICY,
            )
            mismatch_arrays["query_mismatch_counts"][permutation_index] = np.sum(
                rerun["queried"] != reference["queried"], axis=1
            )
            mismatch_arrays["root_mismatch_counts"][permutation_index] = np.sum(
                rerun["selected_root_paths"][PRIMARY_ROOT_POLICY] != reference_roots, axis=1
            )
            mismatch_arrays["regret_mismatch_counts"][permutation_index] = np.sum(
                rerun["root_regret_paths"][PRIMARY_ROOT_POLICY].view(np.uint64)
                != reference_regret.view(np.uint64),
                axis=1,
            )
            mismatch_arrays["selected_value_mismatch_counts"][permutation_index] = np.sum(
                rerun["selected_acquisition_values"].view(np.uint64)
                != reference_values.view(np.uint64),
                axis=1,
            )
            mismatch_arrays["canonical_digest_match"][permutation_index] = int(
                rerun["canonical_digest"] == reference["canonical_digest"]
            )
        keys = store_arrays(raw, f"step80__{task}__permutation__{scenario_name}", mismatch_arrays)
        totals = {
            name: int(values.sum())
            for name, values in mismatch_arrays.items()
            if name != "canonical_digest_match"
        }
        permutation["scenarios"][scenario_name] = {
            **totals,
            "canonical_digest_mismatch_count": int(
                np.sum(mismatch_arrays["canonical_digest_match"] == 0)
            ),
            "all_exact_checks_pass": bool(
                all(value == 0 for value in totals.values())
                and np.all(mismatch_arrays["canonical_digest_match"] == 1)
            ),
            "array_keys": keys,
        }
    permutation["all_exact_checks_pass"] = bool(
        all(row["all_exact_checks_pass"] for row in permutation["scenarios"].values())
    )

    tie_cells: dict[str, Any] = {}
    query_runs: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
        PRIMARY_QUERY_POLICY: (clean_active, refined_active)
    }
    for query_policy in QUERY_POLICIES[1:]:
        query_runs[query_policy] = (
            active_runs(clean, data["oracle"], pools, PAIRED_SEEDS, budget, tau, task, query_policy),
            active_runs(refined, data["oracle"], pools, PAIRED_SEEDS, budget, tau, task, query_policy),
        )
    for query_index, query_policy in enumerate(QUERY_POLICIES):
        clean_run, refined_run = query_runs[query_policy]
        changed_positions = np.sum(clean_run["queried"] != refined_run["queried"], axis=1)
        jaccard = np.asarray(
            [
                len(set(map(int, left)) & set(map(int, right)))
                / len(set(map(int, left)) | set(map(int, right)))
                for left, right in zip(clean_run["queried"], refined_run["queried"])
            ],
            dtype=np.float64,
        )
        for root_index, root_policy in enumerate(ROOT_POLICIES):
            clean_regret = clean_run["root_regret_paths"][root_policy]
            refined_regret = refined_run["root_regret_paths"][root_policy]
            clean_roots = clean_run["selected_root_paths"][root_policy]
            refined_roots = refined_run["selected_root_paths"][root_policy]
            cumulative_delta = refined_regret.sum(axis=1) - clean_regret.sum(axis=1)
            terminal_delta = refined_regret[:, -1] - clean_regret[:, -1]
            cell_key = f"{query_policy}|{root_policy}"
            arrays = {
                "clean_queries": clean_run["queried"],
                "refined_queries": refined_run["queried"],
                "changed_positions": changed_positions,
                "query_jaccard": jaccard,
                "clean_selected_roots": clean_roots,
                "refined_selected_roots": refined_roots,
                "clean_root_regret_path": clean_regret,
                "refined_root_regret_path": refined_regret,
                "delta_cumulative_root_regret": cumulative_delta,
                "delta_terminal_root_regret": terminal_delta,
            }
            keys = store_arrays(
                raw, f"step80__{task}__tie__q{query_index}__r{root_index}", arrays
            )
            tie_cells[cell_key] = {
                "query_policy": query_policy,
                "root_policy": root_policy,
                "path_change_fraction": float(np.mean(changed_positions > 0)),
                "mean_changed_query_positions": float(changed_positions.mean()),
                "mean_query_jaccard": float(jaccard.mean()),
                "cumulative_root_regret_delta": vector_summary(
                    cumulative_delta,
                    904000 + task_index * 10000 + query_index * 1000 + root_index * 100,
                    905000 + task_index * 10000 + query_index * 1000 + root_index * 100,
                ),
                "terminal_root_regret_delta": vector_summary(
                    terminal_delta,
                    904050 + task_index * 10000 + query_index * 1000 + root_index * 100,
                    905050 + task_index * 10000 + query_index * 1000 + root_index * 100,
                ),
                "final_root_change_fraction": float(np.mean(clean_roots[:, -1] != refined_roots[:, -1])),
                "array_keys": keys,
            }
    return {
        "spec": spec,
        "binding_checks": context["checks"],
        "constructor_metadata": context["metadata"],
        "data_digests": context["data_digests"],
        "pool_matrix_digest": context["pool_digest"],
        "fixed_query_matrix_digest": context["fixed_query_digest"],
        "canonical_digests": {
            "clean": clean_active["canonical_digest"],
            "matrix_free_refined": refined_active["canonical_digest"],
        },
        "methods": {name: summarize_method(arrays) for name, arrays in methods.items()},
        "contrasts": contrasts,
        "permutation": permutation,
        "tie_grid": {"cells": tie_cells},
        "array_keys": {"methods": method_keys, "contrasts": contrast_keys},
    }


def add_multiplicity_and_adjudicate(result: dict[str, Any]) -> None:
    contrast_names = list(result["tasks"][TASKS[0]]["contrasts"])
    for contrast in contrast_names:
        adjusted = bh_adjust(
            {
                task: float(result["tasks"][task]["contrasts"][contrast]["signflip_p"])
                for task in TASKS
            }
        )
        for task in TASKS:
            result["tasks"][task]["contrasts"][contrast]["bh_q"] = adjusted[task]
    for query in QUERY_POLICIES:
        for root in ROOT_POLICIES:
            cell = f"{query}|{root}"
            for statistic in ("cumulative_root_regret_delta", "terminal_root_regret_delta"):
                adjusted = bh_adjust(
                    {
                        task: float(
                            result["tasks"][task]["tie_grid"]["cells"][cell][statistic]["signflip_p"]
                        )
                        for task in TASKS
                    }
                )
                for task in TASKS:
                    result["tasks"][task]["tie_grid"]["cells"][cell][statistic]["bh_q"] = adjusted[task]

    def significant_positive(task: str, name: str) -> bool:
        row = result["tasks"][task]["contrasts"][name]
        return bool(row["mean"] > 0 and row["ci95"][0] > 0 and row["bh_q"] < 0.05)

    integrity = all(all(result["tasks"][task]["binding_checks"].values()) for task in TASKS)
    permutation = all(result["tasks"][task]["permutation"]["all_exact_checks_pass"] for task in TASKS)
    active_value_count = sum(
        significant_positive(task, "clean_random_minus_clean_active_cumulative") for task in TASKS
    )
    refinement_all = all(
        significant_positive(task, "refined_active_minus_clean_active_cumulative") for task in TASKS
    )
    worse_random_count = sum(
        significant_positive(task, "refined_active_minus_clean_random_cumulative") for task in TASKS
    )
    sequential_excess_all = all(
        result["tasks"][task]["contrasts"]["sequential_excess_cumulative"]["ci95"][0] > 0
        for task in TASKS
    )
    main_positive_all = all(
        result["tasks"][task]["contrasts"]["refined_active_minus_clean_active_cumulative"]["mean"] > 0
        for task in TASKS
    )
    all_cells_path = all(
        result["tasks"][task]["tie_grid"]["cells"][f"{query}|{root}"]["path_change_fraction"] >= 0.90
        for task in TASKS for query in QUERY_POLICIES for root in ROOT_POLICIES
    )
    all_cells_mean = all(
        result["tasks"][task]["tie_grid"]["cells"][f"{query}|{root}"]["cumulative_root_regret_delta"]["mean"] > 0
        for task in TASKS for query in QUERY_POLICIES for root in ROOT_POLICIES
    )
    all_cells_ci = all(
        result["tasks"][task]["tie_grid"]["cells"][f"{query}|{root}"]["cumulative_root_regret_delta"]["ci95"][0] > 0
        for task in TASKS for query in QUERY_POLICIES for root in ROOT_POLICIES
    )
    terminal_support: dict[str, dict[str, int]] = {}
    terminal_pass = True
    for task in ("medqa", "gsm8k"):
        terminal_support[task] = {}
        for query in QUERY_POLICIES:
            count = sum(
                result["tasks"][task]["tie_grid"]["cells"][f"{query}|{root}"]["terminal_root_regret_delta"]["mean"] > 0
                for root in ROOT_POLICIES
            )
            terminal_support[task][query] = int(count)
            terminal_pass = terminal_pass and count >= 2
    items_1_to_6 = bool(
        integrity
        and active_value_count >= 2
        and refinement_all
        and worse_random_count >= 2
        and sequential_excess_all
        and permutation
        and all_cells_path
        and all_cells_mean
        and all_cells_ci
    )
    if items_1_to_6 and terminal_pass:
        classification = "STRONG_TRANSFER"
    elif items_1_to_6:
        classification = "CUMULATIVE_TRANSFER"
    elif integrity and permutation and main_positive_all:
        classification = "AUDIT_ONLY_TRANSFER"
    else:
        classification = "BLOCKED"
    result["adjudication"] = {
        "classification": classification,
        "construction_integrity": integrity,
        "candidate_permutation_invariance": permutation,
        "significant_clean_active_advantage_task_count": int(active_value_count),
        "significant_refinement_penalty_all_tasks": refinement_all,
        "significantly_worse_than_random_task_count": int(worse_random_count),
        "sequential_excess_positive_lower_ci_all_tasks": sequential_excess_all,
        "main_cumulative_direction_positive_all_tasks": main_positive_all,
        "all_tie_cells_path_change_at_least_90pct": all_cells_path,
        "all_tie_cells_positive_mean_cumulative": all_cells_mean,
        "all_tie_cells_positive_lower_ci_cumulative": all_cells_ci,
        "terminal_positive_root_policy_counts": terminal_support,
        "terminal_support_pass": bool(terminal_pass),
        "items_1_to_6_pass": items_1_to_6,
        "manuscript_permission": {
            "STRONG_TRANSFER": "full cumulative and preregistered MedQA/GSM8K terminal language",
            "CUMULATIVE_TRANSFER": "cumulative/trajectory language; narrow terminal language",
            "AUDIT_ONLY_TRANSFER": "audit framing only; report failed counterfactual",
            "BLOCKED": "no manuscript promotion",
        }[classification],
        "final_status": "PENDING_INDEPENDENT_VALIDATION",
    }


def preflight() -> None:
    prereg, observed = verify_frozen_inputs()
    step54.step46.configure_legacy_paths()
    all_data = step54.step46.phase_a.load_all_data()
    historical = load_json(FROZEN_FILES["step54_results"])
    contexts: dict[str, Any] = {}
    for task_index, task in enumerate(TASKS):
        context = load_task_context(
            task,
            task_index,
            prereg["frozen_tasks"][task],
            {"all_data": all_data, "step54_results": historical},
        )
        contexts[task] = {
            "clean_shape": list(context["clean"]["responses"].shape),
            "refined_shape": list(context["refined"]["responses"].shape),
            "pool_shape": list(context["pools"].shape),
            "checks": context["checks"],
            "clean_canonical_digest": canonical_scenario(context["clean"], f"preflight_{task}_clean")["canonical_digest"],
            "refined_canonical_digest": canonical_scenario(context["refined"], f"preflight_{task}_refined")["canonical_digest"],
        }
    print(json.dumps({"status": "PASS_STEP80_PREFLIGHT", "outcomes_run": False, "inputs": observed, "tasks": contexts}, indent=2))


def execute() -> None:
    started = time.time()
    if OUT.exists() or RAW.exists():
        raise FileExistsError("Step80 reserved scientific output already exists")
    prereg, observed = verify_frozen_inputs()
    execution_lock = verify_execution_lock()
    step54.step46.configure_legacy_paths()
    all_data = step54.step46.phase_a.load_all_data()
    historical = load_json(FROZEN_FILES["step54_results"])
    raw: dict[str, np.ndarray] = {}
    tasks: dict[str, Any] = {}
    for task_index, task in enumerate(TASKS):
        context = load_task_context(
            task,
            task_index,
            prereg["frozen_tasks"][task],
            {"all_data": all_data, "step54_results": historical},
        )
        tasks[task] = evaluate_task(
            task, task_index, prereg["frozen_tasks"][task], context, raw
        )
        print(f"completed {task}", flush=True)
    np.savez_compressed(RAW, **raw)
    result: dict[str, Any] = {
        "manifest_id": "STEP80_MAIN_T2A_V2_TRANSFER_RESULTS_V1",
        "complete": True,
        "frozen_input_sha256": observed,
        "preregistration_sha256": sha256_file(PREREG),
        "preregistration_lock_sha256": sha256_file(PREREG_LOCK),
        "execution_lock_sha256": sha256_file(EXECUTION_LOCK),
        "runner_sha256": sha256_file(RUNNER),
        "validator_sha256": sha256_file(VALIDATOR),
        "preoutcome_result_sha256": sha256_file(PREOUTCOME_RESULT),
        "raw_sha256": sha256_file(RAW),
        "frozen_seed_block": [91000, 91499],
        "permutation_seeds": [803000, 803015],
        "bootstrap_repetitions": 10000,
        "signflip_repetitions": 10000,
        "scientific_condition": "Step54 matrix_free_public_reference_5pct under V2 canonical acquisition",
        "historical_step54_results_preserved": True,
        "tasks": tasks,
        "elapsed_seconds": time.time() - started,
    }
    add_multiplicity_and_adjudicate(result)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["adjudication"], indent=2), flush=True)
    print(f"WROTE {RAW}", flush=True)
    print(f"WROTE {OUT}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight()
    else:
        execute()


if __name__ == "__main__":
    main()
