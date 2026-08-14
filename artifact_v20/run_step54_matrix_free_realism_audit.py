"""Step 54: preregistered competitor-matrix-free realism kill test."""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable

import numpy as np

import run_step46_clone_robust_soft_weighting as step46
import run_step50_natural_alias_audit as step50


ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "STEP54_MATRIX_FREE_REALISM_PREREGISTRATION_2026-08-09.json"
EXECUTION_LOCK = ROOT / "STEP54_MATRIX_FREE_REALISM_EXECUTION_LOCK_2026-08-09.json"
OUT = ROOT / "STEP54_MATRIX_FREE_REALISM_RESULTS_2026-08-09.json"
RUNNER = ROOT / "run_step54_matrix_free_realism_audit.py"
MANIFEST_ID = "STEP54_MATRIX_FREE_REALISM_V1"
EXPECTED_PREREG_SHA256 = "cf45cdd3e6b1425db2f2c17d4e4283bde95d58356a3fd1ade09c2ef6813e8ed9"

INPUTS = {
    "step28_manifest": ROOT / "STEP28_FINAL_MATRIX_MANIFEST_2026-08-06.json",
    "step29_results": ROOT / "STEP29_LOCKED_PHASE_A_RESULTS_2026-08-06.json",
    "step30_target_lock": ROOT
    / "STEP30_PHASE_B1_TARGET_SELECTION_LOCK_2026-08-06.json",
    "step31_preregistration": ROOT
    / "STEP31_PHASE_B2_PREREGISTRATION_2026-08-06.json",
}
CODE_DEPENDENCIES = {
    "run_step24_real_response_current_method_gate.py": ROOT
    / "run_step24_real_response_current_method_gate.py",
    "run_step25_current_method_external_generality_gate.py": ROOT
    / "run_step25_current_method_external_generality_gate.py",
    "run_step29_locked_phase_a.py": ROOT / "run_step29_locked_phase_a.py",
    "run_step46_clone_robust_soft_weighting.py": ROOT
    / "run_step46_clone_robust_soft_weighting.py",
    "run_step50_natural_alias_audit.py": ROOT
    / "run_step50_natural_alias_audit.py",
}
EXPECTED_INPUT_SHA256 = {
    "step28_manifest": "a16f9a262f8d100899a443e96e3ae141ca266f7b5121cdf5e628fc1a67c201ca",
    "step29_results": "9a075b143db6a8f11a42dbbbc313a9262a4547249fe81d39caa3f1c7a1540aaa",
    "step30_target_lock": "98af2c94f8cd3f831ff83fd78fbf6c6755f0fda49460d4c16778b57be1974c7d",
    "step31_preregistration": "a757631ef6f43ed90fd86c40f042a8ab9259e6008704fe460e04b8bc612b2af0",
}
TASKS = ("medqa", "gsm8k", "openbookqa")
ALIASES = 4


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_jsonable(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def digest_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def hash_rank(*parts: Any) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_locks() -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    if file_sha256(PREREG) != EXPECTED_PREREG_SHA256:
        raise AssertionError("Step 54 preregistration hash mismatch")
    prereg = load_json(PREREG)
    if prereg["manifest_id"] != MANIFEST_ID:
        raise AssertionError("unexpected Step 54 preregistration")
    observed_inputs = {name: file_sha256(path) for name, path in INPUTS.items()}
    if observed_inputs != EXPECTED_INPUT_SHA256:
        raise AssertionError(
            {"expected_inputs": EXPECTED_INPUT_SHA256, "observed": observed_inputs}
        )
    for name, record in prereg["inputs"].items():
        if record["sha256"] != observed_inputs[name]:
            raise AssertionError(f"preregistration input mismatch: {name}")
    lock = load_json(EXECUTION_LOCK)
    if lock["manifest_id"] != "STEP54_MATRIX_FREE_REALISM_EXECUTION_LOCK_V1":
        raise AssertionError("unexpected Step 54 execution lock")
    if lock["preregistration_sha256"] != EXPECTED_PREREG_SHA256:
        raise AssertionError("execution lock does not bind preregistration")
    if lock["runner_sha256"] != file_sha256(RUNNER):
        raise AssertionError("execution lock does not bind runner")
    if lock["input_sha256"] != observed_inputs:
        raise AssertionError("execution lock input hash mismatch")
    observed_code = {
        name: file_sha256(path) for name, path in CODE_DEPENDENCIES.items()
    }
    if lock["code_dependency_sha256"] != observed_code:
        raise AssertionError(
            {"locked_code": lock["code_dependency_sha256"], "observed": observed_code}
        )
    return prereg, lock, observed_inputs


def normalize_mc(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def multiple_choice_alternatives(
    options: list[Any], correct: Any, parent_response: Any
) -> list[str]:
    correct_norm = normalize_mc(correct)
    parent_norm = normalize_mc(parent_response)
    choices: list[str] = []
    for value in options:
        candidate = str(value)
        normalized = normalize_mc(candidate)
        if normalized not in {correct_norm, parent_norm} and candidate not in choices:
            choices.append(candidate)
    return choices


def parse_decimal(value: Any) -> Decimal | None:
    cleaned = str(value).strip().replace(",", "")
    try:
        parsed = Decimal(cleaned)
    except InvalidOperation:
        return None
    if not parsed.is_finite():
        return None
    return parsed


def format_decimal(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def wrong_answer(
    task: str,
    *,
    query: int,
    rank: int,
    alias: int,
    example_id: Any,
    parent_response: Any,
    correct: Any,
    public_options: list[Any] | None,
) -> str:
    if task in {"medqa", "openbookqa"}:
        if public_options is None:
            raise AssertionError(f"{task}: missing public options")
        alternatives = multiple_choice_alternatives(
            public_options, correct, parent_response
        )
        if len(alternatives) < 2:
            raise AssertionError(f"{task}:{query}: fewer than two safe wrong options")
        if rank < 2:
            choice_index = (alias >> rank) & 1
        else:
            choice_index = int(
                hash_rank(MANIFEST_ID, "answer", task, alias, example_id), 16
            ) % len(alternatives)
        answer = alternatives[choice_index]
        if normalize_mc(answer) in {
            normalize_mc(correct),
            normalize_mc(parent_response),
        }:
            raise AssertionError("multiple-choice replacement is not safely distinct")
        return answer

    if task == "gsm8k":
        correct_number = parse_decimal(correct)
        if correct_number is None:
            raise AssertionError(f"gsm8k:{query}: unparseable correct answer {correct!r}")
        parent_number = parse_decimal(parent_response)
        offset = Decimal(alias + 1)
        candidate = correct_number + offset
        while candidate == parent_number:
            offset += Decimal(ALIASES)
            candidate = correct_number + offset
        if candidate == correct_number or candidate == parent_number:
            raise AssertionError("GSM8K replacement is not safely wrong/distinct")
        return format_decimal(candidate)

    raise AssertionError(f"unsupported task: {task}")


def usable_wrong_coordinate(
    task: str,
    query: int,
    target_score: float,
    target_response: np.ndarray,
    correct: np.ndarray,
    public_options: list[list[Any]] | None,
) -> bool:
    if target_score != 0.0:
        return False
    if task in {"medqa", "openbookqa"}:
        assert public_options is not None
        return len(
            multiple_choice_alternatives(
                public_options[query], correct[query], target_response[query]
            )
        ) >= 2
    return parse_decimal(correct[query]) is not None


def assemble_refined_scenario(
    clean: dict[str, Any],
    target: int,
    alias_rows: list[np.ndarray],
    alias_scores: list[np.ndarray],
    kind: str,
) -> dict[str, Any]:
    phase_a = step46.phase_a
    responses = np.concatenate(
        [np.asarray(clean["responses"], dtype=object), np.asarray(alias_rows, dtype=object)]
    )
    oracle = np.concatenate(
        [np.asarray(clean["oracle"], dtype=np.float64), np.asarray(alias_scores)]
    )
    return phase_a.make_scenario(
        responses,
        oracle,
        list(np.asarray(clean["parents"], dtype=np.int64)) + [target] * ALIASES,
        list(clean["labels"])
        + [f"STEP54:{kind}:{phase_a.MODELS[target]}:{alias}" for alias in range(ALIASES)],
        kind,
    )


def validate_and_describe(
    *,
    scenario: dict[str, Any],
    clean: dict[str, Any],
    target: int,
    target_oracle: np.ndarray,
    changed_sets: list[list[int]],
    distance_queries: int,
    variant: str,
    labels_inspected: int,
    inspected_indices: list[int],
) -> dict[str, Any]:
    alias_rows = np.asarray(scenario["responses"][-ALIASES:], dtype=object)
    alias_oracle = np.asarray(scenario["oracle"][-ALIASES:], dtype=np.float64)
    parent = np.asarray(clean["responses"][target], dtype=object)
    distances = [int(np.sum(row != parent)) for row in alias_rows]
    oracle_mismatches = [int(np.sum(row != target_oracle)) for row in alias_oracle]
    hashes = list(scenario["response_hashes"][-ALIASES:])
    parent_hash = clean["response_hashes"][target]
    distinct = len(set(hashes)) == ALIASES and parent_hash not in hashes
    support_valid = all(len(values) == distance_queries for values in changed_sets)
    integrity = bool(
        distances == [distance_queries] * ALIASES
        and oracle_mismatches == [0] * ALIASES
        and distinct
        and support_valid
    )
    if not integrity:
        raise AssertionError(
            {
                "variant": variant,
                "distances": distances,
                "oracle_mismatches": oracle_mismatches,
                "hashes_distinct": distinct,
                "support_valid": support_valid,
            }
        )
    return {
        "variant": variant,
        "aliases": ALIASES,
        "distance_queries": distance_queries,
        "distance_fraction": distance_queries / len(target_oracle),
        "response_distance_queries_by_alias": distances,
        "per_example_oracle_mismatches_by_alias": oracle_mismatches,
        "per_example_quality_equal": True,
        "four_distinct_nonparent_hashes": distinct,
        "alias_response_hashes": hashes,
        "parent_response_hash": parent_hash,
        "changed_set_digests": [digest_jsonable(values) for values in changed_sets],
        "changed_set_overlap": [
            [len(set(left) & set(right)) for right in changed_sets]
            for left in changed_sets
        ],
        "reference_labels_inspected": labels_inspected,
        "reference_fraction_inspected": labels_inspected / len(target_oracle),
        "inspected_index_digest": digest_jsonable(inspected_indices),
        "competitor_response_rows_read_by_constructor": 0,
        "pool_seed_access_during_construction": False,
        "online_adaptation": False,
        "constructor_integrity_pass": integrity,
    }


def build_public_reference_variant(
    task: str,
    data: dict[str, Any],
    clean: dict[str, Any],
    *,
    target: int,
    distance_queries: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target_response = np.asarray(clean["responses"][target], dtype=object)
    target_oracle = np.asarray(clean["oracle"][target], dtype=np.float64)
    correct = np.asarray(data["correct"], dtype=object)
    ids = list(data["ids"])
    public_options = data["public_options"]
    eligible = [
        query
        for query in range(len(ids))
        if usable_wrong_coordinate(
            task,
            query,
            float(target_oracle[query]),
            target_response,
            correct,
            public_options,
        )
    ]
    if len(eligible) < distance_queries:
        raise AssertionError(f"{task}: insufficient public-reference coordinates")
    alias_rows: list[np.ndarray] = []
    alias_scores: list[np.ndarray] = []
    changed_sets: list[list[int]] = []
    for alias in range(ALIASES):
        ordered = sorted(
            eligible,
            key=lambda query: hash_rank(
                MANIFEST_ID, "public-support", task, alias, ids[query]
            ),
        )
        changed = ordered[:distance_queries]
        row = target_response.copy()
        for rank, query in enumerate(changed):
            row[query] = wrong_answer(
                task,
                query=query,
                rank=rank,
                alias=alias,
                example_id=ids[query],
                parent_response=target_response[query],
                correct=correct[query],
                public_options=(public_options[query] if public_options is not None else None),
            )
        alias_rows.append(row)
        alias_scores.append(target_oracle.copy())
        changed_sets.append([int(value) for value in changed])
    scenario = assemble_refined_scenario(
        clean,
        target,
        alias_rows,
        alias_scores,
        f"matrix_free_public_reference_d{distance_queries}",
    )
    metadata = validate_and_describe(
        scenario=scenario,
        clean=clean,
        target=target,
        target_oracle=target_oracle,
        changed_sets=changed_sets,
        distance_queries=distance_queries,
        variant="matrix_free_public_reference",
        labels_inspected=len(ids),
        inspected_indices=list(range(len(ids))),
    )
    metadata["eligible_target_wrong_coordinates"] = len(eligible)
    metadata["constructor_inputs"] = [
        "target_response_vector",
        "public_example_ids",
        "public_options_or_numeric_answer_syntax",
        "public_reference_vector",
    ]
    return scenario, metadata


def build_sequential_reference_variant(
    task: str,
    data: dict[str, Any],
    clean: dict[str, Any],
    *,
    target: int,
    distance_queries: int,
    label_provider: Callable[[int], tuple[float, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    target_response = np.asarray(clean["responses"][target], dtype=object)
    target_oracle = np.asarray(clean["oracle"][target], dtype=np.float64)
    ids = list(data["ids"])
    public_options = data["public_options"]
    order = sorted(
        range(len(ids)),
        key=lambda query: hash_rank(
            MANIFEST_ID, "sequential-support", task, ids[query]
        ),
    )
    inspected: list[int] = []
    revealed_correct: dict[int, Any] = {}
    changed: list[int] = []
    for query in order:
        target_score, correct_value = label_provider(query)
        inspected.append(int(query))
        revealed_correct[query] = correct_value
        correct_proxy = np.asarray([revealed_correct.get(i, "") for i in range(len(ids))], dtype=object)
        if usable_wrong_coordinate(
            task,
            query,
            float(target_score),
            target_response,
            correct_proxy,
            public_options,
        ):
            changed.append(int(query))
            if len(changed) == distance_queries:
                break
    if len(changed) != distance_queries:
        raise AssertionError(f"{task}: sequential labels exhausted before support filled")
    alias_rows: list[np.ndarray] = []
    alias_scores: list[np.ndarray] = []
    changed_sets: list[list[int]] = []
    for alias in range(ALIASES):
        row = target_response.copy()
        for rank, query in enumerate(changed):
            row[query] = wrong_answer(
                task,
                query=query,
                rank=rank,
                alias=alias,
                example_id=ids[query],
                parent_response=target_response[query],
                correct=revealed_correct[query],
                public_options=(public_options[query] if public_options is not None else None),
            )
        alias_rows.append(row)
        alias_scores.append(target_oracle.copy())
        changed_sets.append(list(changed))
    scenario = assemble_refined_scenario(
        clean,
        target,
        alias_rows,
        alias_scores,
        f"matrix_free_sequential_reference_d{distance_queries}",
    )
    metadata = validate_and_describe(
        scenario=scenario,
        clean=clean,
        target=target,
        target_oracle=target_oracle,
        changed_sets=changed_sets,
        distance_queries=distance_queries,
        variant="matrix_free_sequential_reference",
        labels_inspected=len(inspected),
        inspected_indices=inspected,
    )
    metadata["constructor_inputs"] = [
        "target_response_vector",
        "public_example_ids",
        "public_options_or_numeric_answer_syntax",
        "sequential_label_provider",
    ]
    metadata["stopped_immediately_after_support_filled"] = True
    return scenario, metadata


def fixed_random_queries(
    pools: np.ndarray, budget: int, task_index: int
) -> np.ndarray:
    rows = []
    for run_index, pool in enumerate(pools):
        rng = random.Random(191000 + task_index * 1000 + run_index)
        rows.append(rng.sample([int(value) for value in pool], budget))
    return np.asarray(rows, dtype=np.int64)


def effect_pass(effect: dict[str, Any]) -> bool:
    return float(effect["delta_cumulative_deployed_regret_95ci"][0]) > 0.0


def evaluate_task(
    task: str,
    task_index: int,
    data: dict[str, Any],
    spec: dict[str, Any],
) -> dict[str, Any]:
    phase_a = step46.phase_a
    clean = phase_a.base_scenario(data)
    target = int(spec["target_index"])
    if phase_a.MODELS[target] != spec["target"]:
        raise AssertionError(f"{task}: target identity drift")
    if clean["responses"].shape[1] != int(spec["examples"]):
        raise AssertionError(f"{task}: example-count drift")

    primary_d = int(spec["primary_distance_queries"])
    secondary_d = int(spec["secondary_distance_queries"])
    public5, public5_meta = build_public_reference_variant(
        task, data, clean, target=target, distance_queries=primary_d
    )
    sequential5, sequential5_meta = build_sequential_reference_variant(
        task,
        data,
        clean,
        target=target,
        distance_queries=primary_d,
        label_provider=lambda query: (
            float(clean["oracle"][target, query]),
            data["correct"][query],
        ),
    )
    public1, public1_meta = build_public_reference_variant(
        task, data, clean, target=target, distance_queries=secondary_d
    )
    legacy5, legacy5_meta = phase_a.build_wrapper(
        data,
        clean,
        target=target,
        distance_queries=primary_d,
        aliases=ALIASES,
        reference_aware=True,
    )

    # Fresh pools are deliberately generated only after every wrapper is frozen.
    seeds = list(range(91000, 91500))
    pools = phase_a.pools_from_seeds(
        seeds, int(spec["examples"]), int(spec["pool_size"])
    )
    budget = int(spec["budget"])
    temperature = float(spec["temperature"])
    clean_runs = phase_a.run_scenario(
        clean, data["oracle"], pools, budget, temperature
    )
    conditions = {
        "legacy_matrix_t2_5pct": legacy5,
        "matrix_free_public_reference_5pct": public5,
        "matrix_free_sequential_reference_5pct": sequential5,
        "matrix_free_public_reference_1pct": public1,
    }
    effects: dict[str, Any] = {}
    for condition_index, (name, scenario) in enumerate(conditions.items()):
        changed_runs = phase_a.run_scenario(
            scenario, data["oracle"], pools, budget, temperature
        )
        effects[name] = phase_a.compare_runs(
            clean_runs,
            changed_runs,
            954000 + task_index * 1000 + condition_index * 10,
            include_arrays=True,
        )

    fixed_queries = fixed_random_queries(pools, budget, task_index)
    fixed_clean = step50.run_fixed_queries(clean, data["oracle"], pools, fixed_queries)
    fixed_public = step50.run_fixed_queries(
        public5, data["oracle"], pools, fixed_queries
    )
    fixed_effect = phase_a.compare_runs(
        fixed_clean,
        fixed_public,
        958000 + task_index * 1000,
        include_arrays=False,
    )
    fixed_digest_clean = digest_array(fixed_clean["queried"])
    fixed_digest_public = digest_array(fixed_public["queried"])

    return {
        "task": task,
        "target_index": target,
        "target": spec["target"],
        "target_full_score": float(np.mean(clean["oracle"][target])),
        "temperature": temperature,
        "budget": budget,
        "examples": int(spec["examples"]),
        "pool_size": int(spec["pool_size"]),
        "fresh_seed_start": seeds[0],
        "fresh_seed_stop": seeds[-1],
        "pool_matrix_digest": digest_array(pools),
        "data_digests": {
            "ids": digest_jsonable(list(data["ids"])),
            "responses": digest_jsonable(
                np.asarray(data["responses"], dtype=object).tolist()
            ),
            "oracle": digest_array(np.asarray(data["oracle"], dtype=np.float64)),
            "correct": digest_jsonable(np.asarray(data["correct"], dtype=object).tolist()),
        },
        "construction_completed_before_pool_generation": True,
        "construction": {
            "legacy_matrix_t2_5pct": legacy5_meta,
            "matrix_free_public_reference_5pct": public5_meta,
            "matrix_free_sequential_reference_5pct": sequential5_meta,
            "matrix_free_public_reference_1pct": public1_meta,
        },
        "effects": effects,
        "fixed_query_control": {
            "query_matrix_digest_clean": fixed_digest_clean,
            "query_matrix_digest_refined": fixed_digest_public,
            "query_matrices_byte_identical": bool(
                np.array_equal(fixed_clean["queried"], fixed_public["queried"])
            ),
            "descriptive_selection_effect_under_fixed_queries": fixed_effect,
        },
    }


def adjudicate(tasks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    matrix_free_variants = (
        "matrix_free_public_reference_5pct",
        "matrix_free_sequential_reference_5pct",
        "matrix_free_public_reference_1pct",
    )
    g0_by_task: dict[str, bool] = {}
    g1_by_task: dict[str, bool] = {}
    g2_by_task: dict[str, bool] = {}
    g3_cost_by_task: dict[str, dict[str, Any]] = {}
    g4_by_task: dict[str, bool] = {}
    for task, result in tasks.items():
        g0_by_task[task] = all(
            result["construction"][variant]["constructor_integrity_pass"]
            and result["construction"][variant][
                "competitor_response_rows_read_by_constructor"
            ]
            == 0
            for variant in matrix_free_variants
        )
        g1_by_task[task] = effect_pass(result["effects"]["legacy_matrix_t2_5pct"])
        g2_by_task[task] = effect_pass(
            result["effects"]["matrix_free_public_reference_5pct"]
        )
        seq_meta = result["construction"][
            "matrix_free_sequential_reference_5pct"
        ]
        fraction = float(seq_meta["reference_fraction_inspected"])
        g3_cost_by_task[task] = {
            "labels_inspected": int(seq_meta["reference_labels_inspected"]),
            "examples": int(result["examples"]),
            "fraction": fraction,
            "at_most_80_percent": fraction <= 0.8 + 1e-12,
            "at_most_35_percent": fraction <= 0.35 + 1e-12,
        }
        g4_by_task[task] = bool(
            result["fixed_query_control"]["query_matrices_byte_identical"]
        )

    g0 = all(g0_by_task.values())
    g1 = all(g1_by_task.values())
    positive_tasks = [task for task, passed in g2_by_task.items() if passed]
    g2_status = (
        "STRONG_PASS"
        if len(positive_tasks) == 3
        else "NARROW_PASS"
        if len(positive_tasks) == 2
        else "FAIL"
    )
    g3 = all(
        value["at_most_80_percent"] for value in g3_cost_by_task.values()
    ) and sum(
        int(value["at_most_35_percent"])
        for value in g3_cost_by_task.values()
    ) >= 2
    g4 = all(g4_by_task.values())
    if g0 and g1 and g3 and g4 and g2_status == "STRONG_PASS":
        decision = "STRONG_GO"
    elif g0 and g1 and g3 and g4 and g2_status == "NARROW_PASS":
        decision = "NARROWED_GO"
    else:
        decision = "NO_GO"
    return {
        "G0_constructor_integrity": {"pass": g0, "by_task": g0_by_task},
        "G1_fresh_legacy_reproduction": {"pass": g1, "by_task": g1_by_task},
        "G2_primary_matrix_free_effect": {
            "status": g2_status,
            "positive_lower_ci_tasks": positive_tasks,
            "by_task": g2_by_task,
        },
        "G3_sequential_reference_cost": {
            "pass": g3,
            "by_task": g3_cost_by_task,
        },
        "G4_fixed_query_control": {"pass": g4, "by_task": g4_by_task},
        "decision": decision,
    }


def main() -> None:
    started = time.time()
    prereg, lock, observed_inputs = verify_locks()
    step46.configure_legacy_paths()
    data = step46.phase_a.load_all_data()
    results: dict[str, dict[str, Any]] = {}
    for task_index, task in enumerate(TASKS):
        results[task] = evaluate_task(
            task,
            task_index,
            data[task],
            prereg["frozen_tasks"][task],
        )
        effect = results[task]["effects"][
            "matrix_free_public_reference_5pct"
        ]
        print(
            task,
            effect["mean_delta_cumulative_deployed_regret"],
            effect["delta_cumulative_deployed_regret_95ci"],
            flush=True,
        )
    gates = adjudicate(results)
    payload = {
        "manifest_id": "STEP54_MATRIX_FREE_REALISM_RESULTS_V1",
        "complete": True,
        "preregistration_sha256": EXPECTED_PREREG_SHA256,
        "execution_lock_sha256": file_sha256(EXECUTION_LOCK),
        "runner_sha256": file_sha256(RUNNER),
        "input_sha256": observed_inputs,
        "code_dependency_sha256": lock["code_dependency_sha256"],
        "fresh_seed_block": [91000, 91499],
        "tasks": results,
        "gates": gates,
        "elapsed_seconds": time.time() - started,
    }
    OUT.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(gates, indent=2), flush=True)
    print(f"WROTE {OUT}", flush=True)


if __name__ == "__main__":
    main()
