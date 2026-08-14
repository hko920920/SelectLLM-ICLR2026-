"""Step 77 V2 stable-acquisition runner.

The runner may reconstruct inputs in ``--preflight`` mode. Outcome execution is
refused unless a later execution lock binds this file, the independent validator,
the Step 76 preregistration, and every frozen input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import psutil

import run_step73_active_value_order_tie_audit as step73


ROOT = Path(__file__).resolve().parent
MANIFEST_ID = "STEP76_V2_STABLE_ACQUISITION_PREREGISTRATION_V1"
PREREG = ROOT / "STEP76_V2_STABLE_ACQUISITION_PREREGISTRATION_2026-08-10.json"
PROTOCOL = ROOT / "STEP76_V2_STABLE_ACQUISITION_PROTOCOL_2026-08-10.md"
PREREG_LOCK = ROOT / "STEP76_V2_PREREGISTRATION_LOCK_2026-08-10.json"
EXECUTION_LOCK = ROOT / "STEP78_V2_EXECUTION_LOCK_2026-08-10.json"
OUT = ROOT / "STEP79_V2_STABLE_ACQUISITION_RESULTS_2026-08-10.json"
RAW = ROOT / "STEP79_V2_STABLE_ACQUISITION_RAW_2026-08-10.npz"
RUNNER = ROOT / "run_step77_v2_stable_acquisition.py"
VALIDATOR = ROOT / "validate_step77_v2_stable_acquisition.py"
UNIT_AUDIT = ROOT / "audit_step78_v2_unit_closure.py"
UNIT_RESULT = ROOT / "STEP78_V2_UNIT_CLOSURE_RESULTS_2026-08-10.json"
UNIT_REPORT = ROOT / "STEP78_V2_UNIT_CLOSURE_REPORT_2026-08-10.md"
PREOUTCOME_AUDIT = ROOT / "audit_step78b_v2_preoutcome_contract.py"
PREOUTCOME_RESULT = ROOT / "STEP78B_V2_PREOUTCOME_CONTRACT_AUDIT_2026-08-10.json"
PREOUTCOME_REPORT = ROOT / "STEP78B_V2_PREOUTCOME_CONTRACT_AUDIT_REPORT_2026-08-10.md"

INPUTS = {
    "preregistration": PREREG,
    "protocol": PROTOCOL,
    "preregistration_lock": PREREG_LOCK,
    "step73_preregistration": ROOT / "STEP73_ACTIVE_VALUE_ORDER_TIE_PREREGISTRATION_2026-08-10.json",
    "step73_amendment": ROOT / "STEP73A_PREREGISTRATION_AMENDMENT_2026-08-10.json",
    "step73_clarification": ROOT / "STEP73B_CLASSIFICATION_CLARIFICATION_2026-08-10.json",
    "step73_execution_lock": ROOT / "STEP73_EXECUTION_LOCK_2026-08-10.json",
    "step73_results": ROOT / "STEP73_ACTIVE_VALUE_ORDER_TIE_RESULTS_2026-08-10.json",
    "step73_raw": ROOT / "STEP73_ACTIVE_VALUE_ORDER_TIE_RAW_2026-08-10.npz",
    "step73_validation": ROOT / "STEP73_ACTIVE_VALUE_ORDER_TIE_VALIDATION_2026-08-10.json",
    "step75_diagnosis": ROOT / "STEP75_CANDIDATE_PERMUTATION_NUMERIC_DIAGNOSIS_2026-08-10.json",
    "step75_report": ROOT / "STEP75_CANDIDATE_PERMUTATION_DIAGNOSIS_REPORT_2026-08-10.md",
    "step75_runner": ROOT / "run_step75_candidate_permutation_numeric_diagnosis.py",
    "phase_a_runner": ROOT / "run_step29_locked_phase_a.py",
    "phase_b2_runner": ROOT / "run_step31_phase_b2_grid.py",
}
EXPECTED_INPUT_SHA256 = {
    "preregistration": "f1e88fc7e73be146fd08f9835de62c95832fd2773dc3af6c9afb93ff4b6db4c9",
    "protocol": "1782bab58ff31011a3f9f29f728658f0360ed9d6f59cc1c1cd2367ac14f2ba3a",
    "preregistration_lock": "e0a2e2e65a6f3fa4c5ddbf7df51d2bce42eeb5fe916ad7da4546f52c365c88e1",
    "step73_preregistration": "eb40ae184f372cf4902ea2b696d53e8347a20d88205a3e7183c417f4cd09d9c9",
    "step73_amendment": "494ec02c3dd596bf395fc33c2c7d8547d6c9101faca365e2681f47dc34dc3cac",
    "step73_clarification": "94a43dbc30fac208cd04a3a0713f8c4cd915fdda97eac3c189577a256cdb86d9",
    "step73_execution_lock": "2ddd88fc93c2a44e633cc344c36e7a3e7d9a45f43aa1c57ad69d3c2f8b356cf3",
    "step73_results": "ff8fcb02a625e0fd08eefec3dd23183548650f513a8d62848f52ad2d40be9685",
    "step73_raw": "94e7efa12a534fec349b37ffd52d32425da2396fd1aa6379da94c96c448f8447",
    "step73_validation": "aab596e22901bcf47758cac622d5f48c7dfa24ae1238de842e0e113cc8e16e8e",
    "step75_diagnosis": "f895d09f55fd3fdf2f6ceb11d1a20ee4860a7799e0d1b2061ef738fc127d602b",
    "step75_report": "fcba1f82f37c026f924e8ecc470ceb667eaddef2476416f2e0e9d5f1fcf3aa4b",
    "step75_runner": "75806e7f1b075b50deaa9eba4dac1d802496bfc7fc38463a1d356854e04e759c",
    "phase_a_runner": "8019d116e17bf669f876f5f20ed4ca90c2f0b276868e049005a8cdcc4a1f0839",
    "phase_b2_runner": "8c28ec9a2fb4e91cbc5ea2b485abf270cda0a582c9e8809dcc6d3ff0cbf56587",
}

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
PERMUTATION_SEEDS = tuple(range(733000, 733016))
PAIRED_SEEDS = tuple(range(50000, 50500))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_digest(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_inputs() -> tuple[dict[str, str], dict[str, Any]]:
    observed = {name: sha256_file(path) for name, path in INPUTS.items()}
    if observed != EXPECTED_INPUT_SHA256:
        raise AssertionError({"expected": EXPECTED_INPUT_SHA256, "observed": observed})
    prereg = load_json(PREREG)
    if prereg.get("manifest_id") != MANIFEST_ID:
        raise AssertionError("unexpected V2 preregistration")
    lock = load_json(PREREG_LOCK)
    if lock.get("manifest_id") != "STEP76_V2_PREREGISTRATION_LOCK_V1":
        raise AssertionError("unexpected V2 preregistration lock")
    if lock["preregistration"]["sha256"] != observed["preregistration"]:
        raise AssertionError("V2 preregistration lock mismatch")
    if lock["protocol"]["sha256"] != observed["protocol"]:
        raise AssertionError("V2 protocol lock mismatch")
    return observed, prereg


def preoutcome_evidence_sha256() -> dict[str, str]:
    paths = {
        "unit_audit": UNIT_AUDIT,
        "unit_result": UNIT_RESULT,
        "unit_report": UNIT_REPORT,
        "preoutcome_audit": PREOUTCOME_AUDIT,
        "preoutcome_result": PREOUTCOME_RESULT,
        "preoutcome_report": PREOUTCOME_REPORT,
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError({"missing_preoutcome_evidence": missing})
    unit = load_json(UNIT_RESULT)
    if (
        unit.get("status") != "PASS_STEP78_V2_UNIT_CLOSURE"
        or unit.get("outcomes_run") is not False
        or unit.get("code_sha256", {}).get("runner") != sha256_file(RUNNER)
        or unit.get("code_sha256", {}).get("validator") != sha256_file(VALIDATOR)
        or unit.get("code_sha256", {}).get("unit_audit") != sha256_file(UNIT_AUDIT)
    ):
        raise AssertionError("unit-closure evidence is stale or failed")
    audit = load_json(PREOUTCOME_RESULT)
    if (
        audit.get("status") != "PASS_STEP78B_V2_PREOUTCOME_CONTRACT_AUDIT"
        or audit.get("outcomes_run") is not False
        or audit.get("code_sha256", {}).get("runner") != sha256_file(RUNNER)
        or audit.get("code_sha256", {}).get("validator") != sha256_file(VALIDATOR)
        or audit.get("code_sha256", {}).get("unit_audit") != sha256_file(UNIT_AUDIT)
        or audit.get("unit_closure_result_sha256") != sha256_file(UNIT_RESULT)
    ):
        raise AssertionError("pre-outcome contract audit is stale or failed")
    return {name: sha256_file(path) for name, path in paths.items()}


def verify_execution_lock() -> dict[str, Any]:
    if not EXECUTION_LOCK.exists():
        raise FileNotFoundError("V2 execution lock does not exist")
    lock = load_json(EXECUTION_LOCK)
    if lock.get("manifest_id") != "STEP78_V2_EXECUTION_LOCK_V1":
        raise AssertionError("unexpected V2 execution lock")
    if lock.get("runner_sha256") != sha256_file(RUNNER):
        raise AssertionError("V2 runner hash mismatch")
    if lock.get("validator_sha256") != sha256_file(VALIDATOR):
        raise AssertionError("V2 validator hash mismatch")
    if lock.get("input_sha256") != EXPECTED_INPUT_SHA256:
        raise AssertionError("V2 execution-lock input mismatch")
    if lock.get("preoutcome_evidence_sha256") != preoutcome_evidence_sha256():
        raise AssertionError("V2 execution-lock pre-outcome evidence mismatch")
    return lock


def configure_paths() -> None:
    step73.configure_paths()


def response_key(value: object) -> tuple[str, bytes]:
    if not isinstance(value, str):
        raise AssertionError("responses must be exact strings")
    encoded = value.encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), encoded


def response_vector_hash(row: np.ndarray) -> str:
    values = list(np.asarray(row, dtype=object))
    if not all(isinstance(value, str) for value in values):
        raise AssertionError("response vectors must contain exact strings")
    return hashlib.sha256(
        json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def canonical_entry_order(scenario: dict[str, Any]) -> list[int]:
    labels = [str(value) for value in scenario["labels"]]
    if len(labels) != len(set(labels)):
        raise AssertionError("candidate labels must be unique")
    responses = np.asarray(scenario["responses"], dtype=object)
    expected_hashes = [response_vector_hash(row) for row in responses]
    observed_hashes = [str(value) for value in scenario["response_hashes"]]
    if observed_hashes != expected_hashes:
        raise AssertionError("response-vector hash mismatch")
    return sorted(
        range(len(labels)),
        key=lambda index: (
            str(scenario["response_hashes"][index]),
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
    response_hashes = [str(scenario["response_hashes"][index]) for index in order]
    payload = [
        {
            "response_hash": response_hash,
            "parent": int(parent),
            "label_hash": stable_digest(label),
        }
        for response_hash, parent, label in zip(response_hashes, parents, labels)
    ]
    return {
        "kind": kind,
        "responses": responses,
        "oracle": oracle,
        "parents": parents,
        "labels": labels,
        "response_hashes": response_hashes,
        "codes": canonical_encode_responses(responses),
        "canonical_digest": hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def build_primary_scenarios(
    task: str, locks: dict[str, dict[str, Any]], prereg: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    step31 = step73.step46.step31
    phase_a = step73.step46.phase_a
    context = step31.task_context(task, locks)
    spec = prereg["frozen_scientific_conditions"]["tasks"][task]
    if context["target"] != spec["target"]:
        raise AssertionError(f"{task}: target drift")
    if int(context["examples"]) != int(spec["examples"]):
        raise AssertionError(f"{task}: example drift")
    if context["pools"].shape != (500, int(spec["pool_size"])):
        raise AssertionError(f"{task}: pool drift")
    if float(context["selected_tau"]) != float(spec["temperature"]):
        raise AssertionError(f"{task}: temperature drift")
    if int(context["max_budget"]) != int(spec["budget"]):
        raise AssertionError(f"{task}: budget drift")
    refinement = prereg["frozen_scientific_conditions"]["refinement"]
    if float(refinement["nominal_distance"]) != 0.05:
        raise AssertionError("unexpected refinement distance")
    if int(context["examples"]) % 20 != 0:
        raise AssertionError("5-percent distance must be integral")
    distance_queries = int(context["examples"]) // 20
    aliases = int(refinement["aliases"])
    refined, metadata = phase_a.build_wrapper(
        context["data"],
        context["clean"],
        target=context["target_index"],
        distance_queries=distance_queries,
        aliases=aliases,
        reference_aware=True,
    )
    expected_metadata = {
        "target": context["target"],
        "distance_queries": distance_queries,
        "distance_fraction": 0.05,
        "aliases": aliases,
        "reference_aware": True,
        "online_adaptation": False,
        "per_example_quality_equal": True,
    }
    observed_metadata = {key: metadata.get(key) for key in expected_metadata}
    if observed_metadata != expected_metadata:
        raise AssertionError({f"{task}: refinement metadata drift": observed_metadata})
    if any(float(value) != 0.0 for value in metadata["alias_full_score_gaps"]):
        raise AssertionError(f"{task}: nonzero alias quality gap")
    return context, context["clean"], refined, metadata


def choose_query_position(
    policy: str,
    positions: np.ndarray,
    pool: np.ndarray,
    task: str,
    paired_seed: int,
    step: int,
) -> int:
    global_queries = [int(pool[int(position)]) for position in positions]
    if policy == QUERY_POLICIES[0]:
        chosen = min(global_queries)
    elif policy == QUERY_POLICIES[1]:
        chosen = max(global_queries)
    elif policy == QUERY_POLICIES[2]:
        chosen = min(
            global_queries,
            key=lambda query: (
                stable_digest(MANIFEST_ID, task, paired_seed, step, "query", query),
                query,
            ),
        )
    else:
        raise ValueError(policy)
    matches = np.flatnonzero(pool == chosen)
    if len(matches) != 1:
        raise AssertionError("query is not unique in the no-replacement pool")
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


def stable_policy_runs(
    scenario: dict[str, Any],
    original_oracle: np.ndarray,
    pools: np.ndarray,
    paired_seeds: tuple[int, ...] | list[int],
    budget: int,
    tau: float,
    task: str,
    query_policy: str,
) -> dict[str, Any]:
    canonical = canonical_scenario(scenario, f"v2_{task}_{scenario['kind']}")
    codes = np.asarray(canonical["codes"], dtype=np.int64)
    scenario_oracle = np.asarray(canonical["oracle"], dtype=np.float64)
    parents = np.asarray(canonical["parents"], dtype=np.int64)
    original_oracle = np.asarray(original_oracle, dtype=np.float64)
    entries = len(parents)
    runs = len(pools)
    queried = np.full((runs, budget), -1, dtype=np.int64)
    selected_values = np.full((runs, budget), np.nan, dtype=np.float64)
    selected_roots = {
        policy: np.full((runs, budget), -1, dtype=np.int64)
        for policy in ROOT_POLICIES
    }
    root_regrets = {
        policy: np.zeros((runs, budget), dtype=np.float64)
        for policy in ROOT_POLICIES
    }

    for run_index, paired_seed in enumerate(paired_seeds):
        pool = np.asarray(pools[run_index], dtype=np.int64)
        pool_size = len(pool)
        original_scores = original_oracle[:, pool].mean(axis=1)
        best_original = float(original_scores.max())
        pool_codes = codes[pool]
        flat_bins = (
            np.arange(pool_size, dtype=np.int64)[:, None] * entries + pool_codes
        ).ravel()
        active = np.ones(pool_size, dtype=bool)
        cumulative = np.zeros(entries, dtype=np.float64)
        for step in range(budget):
            shifted = cumulative / tau
            shifted -= shifted.max()
            exponentials = np.exp(shifted)
            denominator = math.fsum(float(value) for value in exponentials)
            posterior = exponentials / denominator
            masses = np.bincount(
                flat_bins,
                weights=np.tile(posterior, pool_size),
                minlength=pool_size * entries,
            ).reshape(pool_size, entries)
            ordered_masses = np.sort(masses, axis=1)
            acquisition = np.sum(
                ordered_masses * ordered_masses, axis=1, dtype=np.float64
            )
            acquisition[~active] = np.inf
            minimum = float(acquisition.min())
            tied_positions = np.flatnonzero(acquisition == minimum)
            position = choose_query_position(
                query_policy,
                tied_positions,
                pool,
                task,
                int(paired_seed),
                step,
            )
            active[position] = False
            query = int(pool[position])
            queried[run_index, step] = query
            selected_values[run_index, step] = acquisition[position]
            cumulative += scenario_oracle[:, query]
            maximum = float(cumulative.max())
            tied_entries = np.flatnonzero(cumulative == maximum)
            for root_policy in ROOT_POLICIES:
                root = choose_root(
                    root_policy,
                    tied_entries,
                    parents,
                    task,
                    int(paired_seed),
                    step,
                )
                selected_roots[root_policy][run_index, step] = root
                root_regrets[root_policy][run_index, step] = (
                    best_original - original_scores[root]
                )
    return {
        "queried": queried,
        "selected_acquisition_values": selected_values,
        "selected_root_paths": selected_roots,
        "root_regret_paths": root_regrets,
        "canonical_digest": canonical["canonical_digest"],
    }


def stable_fixed_query_runs(
    scenario: dict[str, Any],
    original_oracle: np.ndarray,
    pools: np.ndarray,
    paired_seeds: tuple[int, ...] | list[int],
    queries: np.ndarray,
    task: str,
) -> dict[str, Any]:
    canonical = canonical_scenario(scenario, f"v2_{task}_fixed")
    oracle = np.asarray(canonical["oracle"], dtype=np.float64)
    parents = np.asarray(canonical["parents"], dtype=np.int64)
    original_oracle = np.asarray(original_oracle, dtype=np.float64)
    runs, budget = queries.shape
    selected_roots = np.full((runs, budget), -1, dtype=np.int64)
    root_regret = np.zeros((runs, budget), dtype=np.float64)
    for run_index, paired_seed in enumerate(paired_seeds):
        pool = np.asarray(pools[run_index], dtype=np.int64)
        original_scores = original_oracle[:, pool].mean(axis=1)
        best_original = float(original_scores.max())
        cumulative = np.zeros(len(parents), dtype=np.float64)
        for step in range(budget):
            query = int(queries[run_index, step])
            cumulative += oracle[:, query]
            maximum = float(cumulative.max())
            tied_entries = np.flatnonzero(cumulative == maximum)
            root = choose_root(
                PRIMARY_ROOT_POLICY,
                tied_entries,
                parents,
                task,
                int(paired_seed),
                step,
            )
            selected_roots[run_index, step] = root
            root_regret[run_index, step] = best_original - original_scores[root]
    return {
        "queried": np.asarray(queries, dtype=np.int64),
        "selected_root_path": selected_roots,
        "root_regret_path": root_regret,
        "canonical_digest": canonical["canonical_digest"],
    }


def primary_view(run: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "queried": np.asarray(run["queried"], dtype=np.int64),
        "selected_root_path": np.asarray(
            run["selected_root_paths"][PRIMARY_ROOT_POLICY], dtype=np.int64
        ),
        "root_regret_path": np.asarray(
            run["root_regret_paths"][PRIMARY_ROOT_POLICY], dtype=np.float64
        ),
    }


def array_digest(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    header = json.dumps(
        {"dtype": value.dtype.str, "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\x00" + value.tobytes(order="C")).hexdigest()


def store_arrays(
    raw: dict[str, np.ndarray], prefix: str, values: dict[str, np.ndarray]
) -> dict[str, str]:
    keys: dict[str, str] = {}
    for name, array in values.items():
        key = f"{prefix}__{name}"
        raw[key] = np.asarray(array)
        keys[name] = key
    return keys


def evaluate_gate_a(
    task: str,
    task_index: int,
    context: dict[str, Any],
    clean: dict[str, Any],
    refined: dict[str, Any],
    metadata: dict[str, Any],
    prereg: dict[str, Any],
    raw: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = prereg["frozen_scientific_conditions"]["tasks"][task]
    budget = int(spec["budget"])
    tau = float(spec["temperature"])
    clean_run = stable_policy_runs(
        clean, context["data"]["oracle"], context["pools"], PAIRED_SEEDS,
        budget, tau, task, PRIMARY_QUERY_POLICY
    )
    refined_run = stable_policy_runs(
        refined, context["data"]["oracle"], context["pools"], PAIRED_SEEDS,
        budget, tau, task, PRIMARY_QUERY_POLICY
    )
    random_queries = np.asarray(
        [
            random.Random(seed + 99173).sample(pool.tolist(), budget)
            for seed, pool in zip(PAIRED_SEEDS, context["pools"])
        ],
        dtype=np.int64,
    )
    random_run = stable_fixed_query_runs(
        clean,
        context["data"]["oracle"],
        context["pools"],
        PAIRED_SEEDS,
        random_queries,
        task,
    )
    methods = {
        "clean_active_v2": step73.method_arrays(primary_view(clean_run)),
        "refined_active_v2": step73.method_arrays(primary_view(refined_run)),
        "clean_random_with_v2_root_policy": step73.method_arrays(random_run),
    }
    method_keys = {
        name: store_arrays(raw, f"v2_gate_a__{task}__{name}", arrays)
        for name, arrays in methods.items()
    }
    selected_keys = {
        "clean_active_v2": store_arrays(
            raw,
            f"v2_gate_a__{task}__clean_active_v2_aux",
            {"selected_acquisition_values": clean_run["selected_acquisition_values"]},
        ),
        "refined_active_v2": store_arrays(
            raw,
            f"v2_gate_a__{task}__refined_active_v2_aux",
            {"selected_acquisition_values": refined_run["selected_acquisition_values"]},
        ),
    }
    contrast_values = {
        "clean_active_advantage": methods["clean_random_with_v2_root_policy"]["cumulative_root_regret"]
        - methods["clean_active_v2"]["cumulative_root_regret"],
        "refinement_penalty": methods["refined_active_v2"]["cumulative_root_regret"]
        - methods["clean_active_v2"]["cumulative_root_regret"],
        "refined_relative_to_random": methods["refined_active_v2"]["cumulative_root_regret"]
        - methods["clean_random_with_v2_root_policy"]["cumulative_root_regret"],
        "clean_active_identification_advantage": methods["clean_active_v2"]["exact_auc_per_run"]
        - methods["clean_random_with_v2_root_policy"]["exact_auc_per_run"],
    }
    contrasts: dict[str, Any] = {}
    contrast_keys: dict[str, str] = {}
    for contrast_index, (name, values) in enumerate(contrast_values.items()):
        key = f"v2_gate_a__{task}__contrast__{name}"
        raw[key] = np.asarray(values, dtype=np.float64)
        contrast_keys[name] = key
        contrasts[name] = step73.vector_summary(
            values,
            731000 + task_index * 1000 + contrast_index * 10,
            732000 + task_index * 1000 + contrast_index * 10,
        )
    binding_checks = {
        "clean_canonical_digest_present": bool(clean_run["canonical_digest"]),
        "refined_canonical_digest_present": bool(refined_run["canonical_digest"]),
        "per_example_quality_equal": bool(metadata["per_example_quality_equal"]),
        "nominal_distance_exact": float(metadata["distance_fraction"]) == 0.05,
        "alias_count_exact": int(metadata["aliases"]) == 4,
        "reference_aware": metadata["reference_aware"] is True,
        "online_adaptation_disabled": metadata["online_adaptation"] is False,
        "alias_full_score_gaps_exact_zero": all(
            float(value) == 0.0 for value in metadata["alias_full_score_gaps"]
        ),
        "paired_pool_count": len(context["pools"]) == 500,
        "paired_seed_count": len(PAIRED_SEEDS) == 500,
    }
    if not all(binding_checks.values()):
        raise AssertionError({task: binding_checks})
    return (
        {
            "methods": {
                name: step73.summarize_method(arrays) for name, arrays in methods.items()
            },
            "contrasts": contrasts,
            "array_keys": {
                "methods": method_keys,
                "auxiliary": selected_keys,
                "contrasts": contrast_keys,
            },
            "binding_checks": binding_checks,
            "canonical_digests": {
                "clean": clean_run["canonical_digest"],
                "refined": refined_run["canonical_digest"],
            },
        },
        {"clean": clean_run, "refined": refined_run, "methods": methods},
    )


def evaluate_v2_vs_v1_descriptive(
    task: str,
    v2_methods: dict[str, dict[str, np.ndarray]],
    raw: dict[str, np.ndarray],
) -> dict[str, Any]:
    v1_result = load_json(INPUTS["step73_results"])
    v1_gate_a = v1_result["tasks"][task]["gate_a"]
    method_map = {
        "clean_active_v2": "clean_active",
        "refined_active_v2": "refined_active",
        "clean_random_with_v2_root_policy": "clean_random",
    }
    output: dict[str, Any] = {
        "non_gate_descriptive_only": True,
        "v1_result_sha256": EXPECTED_INPUT_SHA256["step73_results"],
        "v1_raw_sha256": EXPECTED_INPUT_SHA256["step73_raw"],
        "methods": {},
    }
    with np.load(INPUTS["step73_raw"], allow_pickle=False) as v1_raw:
        for v2_name, v1_name in method_map.items():
            v2 = v2_methods[v2_name]
            v1_keys = v1_gate_a["array_keys"]["methods"][v1_name]
            v1 = {name: np.asarray(v1_raw[key]) for name, key in v1_keys.items()}
            for name in v2:
                if np.asarray(v2[name]).shape != np.asarray(v1[name]).shape:
                    raise AssertionError(f"{task}/{v2_name}/{name}: V1/V2 shape drift")
            query_mismatch = np.sum(v2["queried"] != v1["queried"], axis=1).astype(np.int64)
            root_mismatch = np.sum(
                v2["selected_root_path"] != v1["selected_root_path"], axis=1
            ).astype(np.int64)
            regret_delta = np.asarray(
                v2["root_regret_path"] - v1["root_regret_path"], dtype=np.float64
            )
            regret_bitwise_changed = (
                np.asarray(v2["root_regret_path"], dtype=np.float64).view(np.uint64)
                != np.asarray(v1["root_regret_path"], dtype=np.float64).view(np.uint64)
            ).astype(np.int8)
            deltas = {
                "query_mismatch_counts": query_mismatch,
                "selected_root_mismatch_counts": root_mismatch,
                "root_regret_path_delta": regret_delta,
                "root_regret_path_bitwise_changed": regret_bitwise_changed,
                "cumulative_root_regret_delta": np.asarray(
                    v2["cumulative_root_regret"] - v1["cumulative_root_regret"],
                    dtype=np.float64,
                ),
                "terminal_root_regret_delta": np.asarray(
                    v2["terminal_root_regret"] - v1["terminal_root_regret"],
                    dtype=np.float64,
                ),
                "exact_auc_delta": np.asarray(
                    v2["exact_auc_per_run"] - v1["exact_auc_per_run"],
                    dtype=np.float64,
                ),
                "near_auc_delta": np.asarray(
                    v2["near_auc_per_run"] - v1["near_auc_per_run"],
                    dtype=np.float64,
                ),
            }
            keys = store_arrays(raw, f"v2_vs_v1__{task}__{v2_name}", deltas)
            total_steps = int(np.prod(v2["queried"].shape))
            output["methods"][v2_name] = {
                "v1_method": v1_name,
                "runs_with_query_change_fraction": float(np.mean(query_mismatch > 0)),
                "query_step_change_fraction": float(query_mismatch.sum() / total_steps),
                "runs_with_selected_root_change_fraction": float(np.mean(root_mismatch > 0)),
                "selected_root_step_change_fraction": float(root_mismatch.sum() / total_steps),
                "root_regret_bitwise_change_fraction": float(regret_bitwise_changed.mean()),
                "mean_absolute_root_regret_difference": float(np.abs(regret_delta).mean()),
                "maximum_absolute_root_regret_difference": float(np.abs(regret_delta).max()),
                "mean_v2_minus_v1_cumulative_root_regret": float(
                    deltas["cumulative_root_regret_delta"].mean()
                ),
                "mean_v2_minus_v1_terminal_root_regret": float(
                    deltas["terminal_root_regret_delta"].mean()
                ),
                "mean_v2_minus_v1_exact_auc": float(deltas["exact_auc_delta"].mean()),
                "mean_v2_minus_v1_near_auc": float(deltas["near_auc_delta"].mean()),
                "array_keys": keys,
            }
    return output


def evaluate_permutations(
    task: str,
    context: dict[str, Any],
    clean: dict[str, Any],
    refined: dict[str, Any],
    primary: dict[str, Any],
    prereg: dict[str, Any],
    raw: dict[str, np.ndarray],
) -> dict[str, Any]:
    spec = prereg["frozen_scientific_conditions"]["tasks"][task]
    budget = int(spec["budget"])
    tau = float(spec["temperature"])
    output: dict[str, Any] = {"seeds": list(PERMUTATION_SEEDS), "scenarios": {}}
    for scenario_name, scenario in (("clean", clean), ("refined", refined)):
        reference = primary[scenario_name]
        reference_roots = reference["selected_root_paths"][PRIMARY_ROOT_POLICY]
        reference_regret = reference["root_regret_paths"][PRIMARY_ROOT_POLICY]
        reference_values = reference["selected_acquisition_values"]
        shape = (len(PERMUTATION_SEEDS), len(context["pools"]), budget)
        perm_queries = np.full(shape, -1, dtype=np.int64)
        perm_roots = np.full(shape, -1, dtype=np.int64)
        perm_regret = np.zeros(shape, dtype=np.float64)
        perm_values = np.full(shape, np.nan, dtype=np.float64)
        digest_match = np.zeros(len(PERMUTATION_SEEDS), dtype=np.int8)
        selected_value_digest_match = np.zeros(len(PERMUTATION_SEEDS), dtype=np.int8)
        reference_value_digest = array_digest(reference_values)
        for permutation_index, seed in enumerate(PERMUTATION_SEEDS):
            permuted = step73.permute_scenario(
                scenario, seed, f"v2_{task}_{scenario_name}_perm_{seed}"
            )
            run = stable_policy_runs(
                permuted,
                context["data"]["oracle"],
                context["pools"],
                PAIRED_SEEDS,
                budget,
                tau,
                task,
                PRIMARY_QUERY_POLICY,
            )
            perm_queries[permutation_index] = run["queried"]
            perm_roots[permutation_index] = run["selected_root_paths"][PRIMARY_ROOT_POLICY]
            perm_regret[permutation_index] = run["root_regret_paths"][PRIMARY_ROOT_POLICY]
            perm_values[permutation_index] = run["selected_acquisition_values"]
            digest_match[permutation_index] = int(
                run["canonical_digest"] == reference["canonical_digest"]
            )
            selected_value_digest_match[permutation_index] = int(
                array_digest(run["selected_acquisition_values"]) == reference_value_digest
            )
        query_mismatch = np.sum(
            perm_queries != reference["queried"][None, :, :], axis=2
        ).astype(np.int64)
        root_mismatch = np.sum(
            perm_roots != reference_roots[None, :, :], axis=2
        ).astype(np.int64)
        regret_mismatch = np.sum(
            perm_regret.view(np.uint64) != reference_regret[None, :, :].view(np.uint64),
            axis=2,
        ).astype(np.int64)
        value_mismatch = np.sum(
            perm_values.view(np.uint64) != reference_values[None, :, :].view(np.uint64),
            axis=2,
        ).astype(np.int64)
        keys = store_arrays(
            raw,
            f"v2_gate_b1__{task}__{scenario_name}",
            {
                "permuted_queries": perm_queries,
                "permuted_selected_roots": perm_roots,
                "permuted_root_regret": perm_regret,
                "permuted_selected_acquisition_values": perm_values,
                "query_mismatch_counts": query_mismatch,
                "root_mismatch_counts": root_mismatch,
                "regret_mismatch_counts": regret_mismatch,
                "selected_value_mismatch_counts": value_mismatch,
                "canonical_digest_match": digest_match,
                "selected_acquisition_value_digest_match": selected_value_digest_match,
            },
        )
        per_seed = []
        for permutation_index, seed in enumerate(PERMUTATION_SEEDS):
            row = {
                "seed": int(seed),
                "query_mismatch_total": int(query_mismatch[permutation_index].sum()),
                "root_mismatch_total": int(root_mismatch[permutation_index].sum()),
                "regret_mismatch_total": int(regret_mismatch[permutation_index].sum()),
                "selected_value_mismatch_total": int(value_mismatch[permutation_index].sum()),
                "canonical_digest_match": bool(digest_match[permutation_index]),
                "selected_acquisition_value_digest_match": bool(
                    selected_value_digest_match[permutation_index]
                ),
            }
            row["all_exact_checks_pass"] = bool(
                row["query_mismatch_total"] == 0
                and row["root_mismatch_total"] == 0
                and row["regret_mismatch_total"] == 0
                and row["selected_value_mismatch_total"] == 0
                and row["canonical_digest_match"]
                and row["selected_acquisition_value_digest_match"]
            )
            per_seed.append(row)
        output["scenarios"][scenario_name] = {
            "query_mismatch_total": int(query_mismatch.sum()),
            "root_mismatch_total": int(root_mismatch.sum()),
            "regret_mismatch_total": int(regret_mismatch.sum()),
            "selected_value_mismatch_total": int(value_mismatch.sum()),
            "canonical_digest_mismatch_count": int(np.sum(digest_match == 0)),
            "selected_acquisition_value_digest_mismatch_count": int(
                np.sum(selected_value_digest_match == 0)
            ),
            "per_seed": per_seed,
            "all_exact_checks_pass": bool(
                query_mismatch.sum() == 0
                and root_mismatch.sum() == 0
                and regret_mismatch.sum() == 0
                and value_mismatch.sum() == 0
                and np.all(digest_match == 1)
                and np.all(selected_value_digest_match == 1)
            ),
            "array_keys": keys,
        }
    output["all_exact_checks_pass"] = bool(
        all(row["all_exact_checks_pass"] for row in output["scenarios"].values())
    )
    return output


def evaluate_tie_policies(
    task: str,
    task_index: int,
    context: dict[str, Any],
    clean: dict[str, Any],
    refined: dict[str, Any],
    prereg: dict[str, Any],
    raw: dict[str, np.ndarray],
) -> dict[str, Any]:
    spec = prereg["frozen_scientific_conditions"]["tasks"][task]
    budget = int(spec["budget"])
    tau = float(spec["temperature"])
    cells: dict[str, Any] = {}
    for query_index, query_policy in enumerate(QUERY_POLICIES):
        clean_run = stable_policy_runs(
            clean, context["data"]["oracle"], context["pools"], PAIRED_SEEDS,
            budget, tau, task, query_policy
        )
        refined_run = stable_policy_runs(
            refined, context["data"]["oracle"], context["pools"], PAIRED_SEEDS,
            budget, tau, task, query_policy
        )
        query_arrays = step73.query_pair_arrays(clean_run["queried"], refined_run["queried"])
        query_keys = store_arrays(
            raw,
            f"v2_gate_b2__{task}__q{query_index}",
            {
                "clean_queries": clean_run["queried"],
                "refined_queries": refined_run["queried"],
                "clean_selected_acquisition_values": clean_run["selected_acquisition_values"],
                "refined_selected_acquisition_values": refined_run["selected_acquisition_values"],
                **query_arrays,
            },
        )
        for root_index, root_policy in enumerate(ROOT_POLICIES):
            clean_regret = clean_run["root_regret_paths"][root_policy]
            refined_regret = refined_run["root_regret_paths"][root_policy]
            clean_roots = clean_run["selected_root_paths"][root_policy]
            refined_roots = refined_run["selected_root_paths"][root_policy]
            cumulative_delta = refined_regret.sum(axis=1) - clean_regret.sum(axis=1)
            terminal_delta = refined_regret[:, -1] - clean_regret[:, -1]
            final_root_changed = (refined_roots[:, -1] != clean_roots[:, -1]).astype(np.int8)
            cell_key = f"{query_policy}|{root_policy}"
            cell_keys = dict(query_keys)
            cell_keys.update(
                store_arrays(
                    raw,
                    f"v2_gate_b2__{task}__q{query_index}__r{root_index}",
                    {
                        "clean_root_regret_path": clean_regret,
                        "refined_root_regret_path": refined_regret,
                        "clean_selected_root_path": clean_roots,
                        "refined_selected_root_path": refined_roots,
                        "delta_cumulative_root_regret": cumulative_delta,
                        "delta_terminal_root_regret": terminal_delta,
                        "final_root_changed": final_root_changed,
                    },
                )
            )
            cells[cell_key] = {
                "query_policy": query_policy,
                "root_policy": root_policy,
                "path_change_fraction": float(np.mean(query_arrays["changed_positions"] > 0)),
                "mean_changed_query_positions": float(query_arrays["changed_positions"].mean()),
                "mean_query_jaccard": float(query_arrays["query_jaccard"].mean()),
                "cumulative_root_regret_delta": step73.vector_summary(
                    cumulative_delta,
                    735000 + task_index * 10000 + query_index * 1000 + root_index * 100,
                    736000 + task_index * 10000 + query_index * 1000 + root_index * 100,
                ),
                "terminal_root_regret_delta": step73.vector_summary(
                    terminal_delta,
                    737000 + task_index * 10000 + query_index * 1000 + root_index * 100,
                    738000 + task_index * 10000 + query_index * 1000 + root_index * 100,
                ),
                "final_root_change_fraction": float(final_root_changed.mean()),
                "array_keys": cell_keys,
            }
    return {"cells": cells}


def apply_gate_a_classification(result: dict[str, Any]) -> None:
    contrasts = tuple(result["tasks"][TASKS[0]]["gate_a"]["contrasts"].keys())
    for contrast in contrasts:
        pvalues = {
            task: float(result["tasks"][task]["gate_a"]["contrasts"][contrast]["signflip_p"])
            for task in TASKS
        }
        adjusted = step73.bh_adjust(pvalues)
        for task in TASKS:
            result["tasks"][task]["gate_a"]["contrasts"][contrast]["bh_q"] = float(adjusted[task])

    def positive(task: str, contrast: str) -> bool:
        row = result["tasks"][task]["gate_a"]["contrasts"][contrast]
        return bool(row["mean"] > 0 and row["ci95"][0] > 0 and row["bh_q"] < 0.05)

    active = sum(positive(task, "clean_active_advantage") for task in TASKS)
    refinement = sum(positive(task, "refinement_penalty") for task in TASKS)
    worse_random = sum(positive(task, "refined_relative_to_random") for task in TASKS)
    bindings = all(
        all(result["tasks"][task]["gate_a"]["binding_checks"].values()) for task in TASKS
    )
    if not bindings:
        classification = "INVALID"
    elif active >= 2 and worse_random >= 2:
        classification = "STRONG_GREEN"
    elif active >= 2 and refinement == 3:
        classification = "GREEN"
    elif active == 0 and worse_random == 0:
        classification = "RED"
    else:
        classification = "YELLOW"
    result["gate_a_decision"] = {
        "classification": classification,
        "significant_clean_active_advantage_tasks": active,
        "significant_refinement_penalty_tasks": refinement,
        "significantly_worse_than_random_tasks": worse_random,
        "binding_checks_pass": bindings,
    }


def apply_gate_b_classification(result: dict[str, Any]) -> None:
    permutation_pass = all(
        result["tasks"][task]["gate_b"]["permutation"]["all_exact_checks_pass"]
        for task in TASKS
    )
    path_green = all(
        result["tasks"][task]["gate_b"]["ties"]["cells"][f"{query}|{ROOT_POLICIES[0]}"]["path_change_fraction"] >= 0.90
        for task in TASKS for query in QUERY_POLICIES
    )
    mean_cumulative_positive = all(
        result["tasks"][task]["gate_b"]["ties"]["cells"][f"{query}|{root}"]["cumulative_root_regret_delta"]["mean"] > 0
        for task in TASKS for query in QUERY_POLICIES for root in ROOT_POLICIES
    )
    cumulative_ci_green = all(
        result["tasks"][task]["gate_b"]["ties"]["cells"][f"{query}|{root}"]["cumulative_root_regret_delta"]["ci95"][0] > 0
        for task in ("medqa", "gsm8k") for query in QUERY_POLICIES for root in ROOT_POLICIES
    )
    terminal_support: dict[str, int] = {}
    for task in ("medqa", "gsm8k"):
        terminal_support[task] = sum(
            sum(
                result["tasks"][task]["gate_b"]["ties"]["cells"][f"{query}|{root}"]["terminal_root_regret_delta"]["mean"] > 0
                for root in ROOT_POLICIES
            ) >= 2
            for query in QUERY_POLICIES
        )
    terminal_green = all(value == 3 for value in terminal_support.values())
    scientific_green = bool(
        path_green and mean_cumulative_positive and cumulative_ci_green and terminal_green
    )
    positive_tasks = sum(
        all(
            result["tasks"][task]["gate_b"]["ties"]["cells"][f"{query}|{root}"]["cumulative_root_regret_delta"]["mean"] > 0
            for query in QUERY_POLICIES for root in ROOT_POLICIES
        )
        for task in TASKS
    )
    if not permutation_pass:
        classification = "RED_TECHNICAL"
    elif not path_green or positive_tasks < 2:
        classification = "RED_SCIENTIFIC"
    elif scientific_green:
        classification = "GREEN"
    else:
        classification = "YELLOW"
    result["gate_b_decision"] = {
        "classification": classification,
        "candidate_permutation_pass": permutation_pass,
        "path_green": path_green,
        "mean_cumulative_positive_all_cells": mean_cumulative_positive,
        "cumulative_ci_green": cumulative_ci_green,
        "terminal_positive_query_policy_counts": terminal_support,
        "terminal_green": terminal_green,
        "scientific_green": scientific_green,
    }


def apply_overall_decision(result: dict[str, Any]) -> None:
    gate_a = result["gate_a_decision"]["classification"]
    gate_b = result["gate_b_decision"]["classification"]
    if gate_a == "STRONG_GREEN" and gate_b == "GREEN":
        decision = "EMPIRICAL_GATES_CLOSED"
    elif gate_b == "GREEN" and gate_a in {"GREEN", "YELLOW"}:
        decision = "TECHNICALLY_REPAIRED_BUT_BORDERLINE"
    else:
        decision = "BLOCKED"
    result["conditional_overall_decision_after_independent_validation"] = decision
    result["overall_decision"] = "PENDING_INDEPENDENT_VALIDATION"


def preflight() -> None:
    observed, prereg = verify_inputs()
    configure_paths()
    locks = step73.step46.step31.verify_locks()
    tasks: dict[str, Any] = {}
    for task in TASKS:
        context, clean, refined, metadata = build_primary_scenarios(task, locks, prereg)
        clean_canonical = canonical_scenario(clean, f"preflight_{task}_clean")
        refined_canonical = canonical_scenario(refined, f"preflight_{task}_refined")
        tasks[task] = {
            "pool_shape": list(context["pools"].shape),
            "clean_entries": len(clean_canonical["labels"]),
            "refined_entries": len(refined_canonical["labels"]),
            "clean_canonical_digest": clean_canonical["canonical_digest"],
            "refined_canonical_digest": refined_canonical["canonical_digest"],
            "per_example_quality_equal": bool(metadata["per_example_quality_equal"]),
        }
    print(
        json.dumps(
            {
                "status": "PASS_STEP77_PREFLIGHT_NO_OUTCOMES_RUN",
                "input_sha256": observed,
                "tasks": tasks,
                "execution_lock_exists": EXECUTION_LOCK.exists(),
                "outputs_exist": {"result": OUT.exists(), "raw": RAW.exists()},
            },
            indent=2,
            sort_keys=True,
        )
    )


def run_outcomes() -> None:
    if OUT.exists() or RAW.exists():
        raise FileExistsError("V2 outputs already exist; refusing overwrite")
    observed, prereg = verify_inputs()
    execution_lock = verify_execution_lock()
    v1_result = load_json(INPUTS["step73_results"])
    configure_paths()
    locks = step73.step46.step31.verify_locks()
    started = time.time()
    tracemalloc.start()
    raw: dict[str, np.ndarray] = {}
    result: dict[str, Any] = {
        "manifest_id": "STEP79_V2_STABLE_ACQUISITION_RESULTS_V1",
        "complete": False,
        "preregistration_sha256": EXPECTED_INPUT_SHA256["preregistration"],
        "execution_lock_sha256": sha256_file(EXECUTION_LOCK),
        "input_sha256": observed,
        "code_sha256": {
            "runner": sha256_file(RUNNER),
            "validator": sha256_file(VALIDATOR),
            "unit_audit": sha256_file(UNIT_AUDIT),
            "preoutcome_audit": sha256_file(PREOUTCOME_AUDIT),
        },
        "preoutcome_evidence_sha256": execution_lock["preoutcome_evidence_sha256"],
        "historical_v1": {
            "preserved": True,
            "result_sha256": EXPECTED_INPUT_SHA256["step73_results"],
            "raw_sha256": EXPECTED_INPUT_SHA256["step73_raw"],
            "gate_a_classification": v1_result["gate_a_decision"]["classification"],
            "gate_b_classification": v1_result["gate_b_decision"]["classification"],
            "step75_diagnosis_sha256": EXPECTED_INPUT_SHA256["step75_diagnosis"],
        },
        "tasks": {},
    }
    for task_index, task in enumerate(TASKS):
        task_started = time.time()
        context, clean, refined, metadata = build_primary_scenarios(task, locks, prereg)
        gate_a, primary = evaluate_gate_a(
            task, task_index, context, clean, refined, metadata, prereg, raw
        )
        v2_vs_v1 = evaluate_v2_vs_v1_descriptive(task, primary["methods"], raw)
        permutation = evaluate_permutations(
            task, context, clean, refined, primary, prereg, raw
        )
        ties = evaluate_tie_policies(
            task, task_index, context, clean, refined, prereg, raw
        )
        result["tasks"][task] = {
            "settings": prereg["frozen_scientific_conditions"]["tasks"][task],
            "scenario": {
                "clean_entries": len(clean["labels"]),
                "refined_entries": len(refined["labels"]),
                "per_example_quality_equal": bool(metadata["per_example_quality_equal"]),
                "realized_distance": float(metadata["distance_fraction"]),
                "aliases": int(metadata["aliases"]),
                "reference_aware": bool(metadata["reference_aware"]),
                "online_adaptation": bool(metadata["online_adaptation"]),
                "alias_full_score_gaps": [
                    float(value) for value in metadata["alias_full_score_gaps"]
                ],
            },
            "gate_a": gate_a,
            "gate_b": {"permutation": permutation, "ties": ties},
            "v2_vs_v1_descriptive": v2_vs_v1,
            "runtime_seconds": float(time.time() - task_started),
        }
        print(f"[V2] completed {task} in {time.time()-task_started:.1f}s", flush=True)
    apply_gate_a_classification(result)
    apply_gate_b_classification(result)
    apply_overall_decision(result)
    raw_tmp = RAW.with_suffix(".tmp.npz")
    with raw_tmp.open("wb") as handle:
        np.savez_compressed(handle, **raw)
    raw_tmp.replace(RAW)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory = psutil.Process().memory_info()
    result["raw_arrays"] = {
        "file": RAW.name,
        "sha256": sha256_file(RAW),
        "array_count": len(raw),
    }
    result["runtime"] = {
        "wall_seconds": float(time.time() - started),
        "python_peak_tracemalloc_bytes": int(peak),
        "process_peak_working_set_bytes": int(
            getattr(memory, "peak_wset", memory.rss)
        ),
        "new_model_or_api_calls": 0,
    }
    result["complete"] = True
    temporary = OUT.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(OUT)
    print(
        json.dumps(
            {
                "gate_a": result["gate_a_decision"],
                "gate_b": result["gate_b_decision"],
                "conditional_overall_after_validation": result[
                    "conditional_overall_decision_after_independent_validation"
                ],
                "overall": result["overall_decision"],
                "result_sha256": sha256_file(OUT),
                "raw_sha256": sha256_file(RAW),
                "execution_lock": execution_lock["manifest_id"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight()
    else:
        run_outcomes()


if __name__ == "__main__":
    main()
