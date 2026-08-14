"""Step 50: locked natural exact-alias registry-refinement audit.

The base registry removes the later of the unique cross-task exact-response
pair.  The refined registry is the original 31-entry roster, with the later
alias mapped to the retained model's deployment root.  Selection outcomes are
computed only after the preregistration and this runner are hash locked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

import run_step46_clone_robust_soft_weighting as step46


ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "STEP50_NATURAL_ALIAS_PREREGISTRATION_2026-08-09.json"
EXECUTION_LOCK = ROOT / "STEP50_NATURAL_ALIAS_EXECUTION_LOCK_2026-08-09.json"
OUT = ROOT / "STEP50_NATURAL_ALIAS_RESULTS_2026-08-09.json"
RUNNER = ROOT / "run_step50_natural_alias_audit.py"

INPUTS = {
    "step28_manifest": ROOT / "STEP28_FINAL_MATRIX_MANIFEST_2026-08-06.json",
    "step29_results": ROOT / "STEP29_LOCKED_PHASE_A_RESULTS_2026-08-06.json",
    "step31_preregistration": ROOT / "STEP31_PHASE_B2_PREREGISTRATION_2026-08-06.json",
    "step48_audit": ROOT / "STEP48_ADVERSARIAL_REVIEW_AUDIT_2026-08-09.json",
}
CODE_DEPENDENCIES = {
    "run_step29_locked_phase_a.py": ROOT / "run_step29_locked_phase_a.py",
    "run_step31_phase_b2_grid.py": ROOT / "run_step31_phase_b2_grid.py",
    "run_step46_clone_robust_soft_weighting.py": ROOT
    / "run_step46_clone_robust_soft_weighting.py",
}
EXPECTED_INPUT_SHA256 = {
    "step28_manifest": "a16f9a262f8d100899a443e96e3ae141ca266f7b5121cdf5e628fc1a67c201ca",
    "step29_results": "9a075b143db6a8f11a42dbbbc313a9262a4547249fe81d39caa3f1c7a1540aaa",
    "step31_preregistration": "a757631ef6f43ed90fd86c40f042a8ab9259e6008704fe460e04b8bc612b2af0",
    "step48_audit": "c102f5671897efccec3c3baa3e8bbe9f9860966b027d16590d1ba07b3d4bf3ba",
}
EXPECTED_PREREG_SHA256 = "4e5201cb67e8603378477510366b94847797905cf79d40a63fc43842fa2bedda"
TASKS = ("medqa", "gsm8k", "openbookqa")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def configure_paths() -> None:
    step46.configure_legacy_paths()


def verify_locks() -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    observed = {name: file_sha256(path) for name, path in INPUTS.items()}
    if observed != EXPECTED_INPUT_SHA256:
        raise AssertionError({"expected_inputs": EXPECTED_INPUT_SHA256, "observed": observed})
    if file_sha256(PREREG) != EXPECTED_PREREG_SHA256:
        raise AssertionError("Step 50 preregistration hash mismatch")
    prereg = load_json(PREREG)
    if prereg["manifest_id"] != "STEP50_NATURAL_EXACT_ALIAS_V1":
        raise AssertionError("unexpected Step 50 preregistration")
    lock = load_json(EXECUTION_LOCK)
    if lock["manifest_id"] != "STEP50_NATURAL_EXACT_ALIAS_EXECUTION_LOCK_V1":
        raise AssertionError("unexpected Step 50 execution lock")
    if lock["preregistration_sha256"] != EXPECTED_PREREG_SHA256:
        raise AssertionError("execution lock does not bind the preregistration")
    if lock["runner_sha256"] != file_sha256(RUNNER):
        raise AssertionError("execution lock does not bind this runner")
    if lock["input_sha256"] != EXPECTED_INPUT_SHA256:
        raise AssertionError("execution lock input hashes differ")
    dependency_hashes = {
        name: file_sha256(path) for name, path in CODE_DEPENDENCIES.items()
    }
    if lock["code_dependency_sha256"] != dependency_hashes:
        raise AssertionError(
            {
                "locked_code_dependencies": lock["code_dependency_sha256"],
                "observed": dependency_hashes,
            }
        )
    return prereg, lock, observed


def exact_pairs(contexts: dict[str, dict[str, Any]]) -> dict[str, list[list[int]]]:
    result: dict[str, list[list[int]]] = {}
    for task, context in contexts.items():
        responses = np.asarray(context["clean"]["responses"], dtype=object)
        oracle = np.asarray(context["clean"]["oracle"], dtype=np.float64)
        pairs: list[list[int]] = []
        for left in range(responses.shape[0]):
            for right in range(left + 1, responses.shape[0]):
                if np.array_equal(responses[left], responses[right]) and np.array_equal(
                    oracle[left], oracle[right]
                ):
                    pairs.append([left, right])
        result[task] = pairs
    return result


def build_registries(
    clean: dict[str, Any], retained: int, added: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    phase_a = step46.phase_a
    indices = [index for index in range(len(phase_a.MODELS)) if index != added]
    base = phase_a.make_scenario(
        clean["responses"][indices],
        clean["oracle"][indices],
        indices,
        [phase_a.MODELS[index] for index in indices],
        "natural_alias_collapsed_base",
    )
    refined_parents = list(range(len(phase_a.MODELS)))
    refined_parents[added] = retained
    refined = phase_a.make_scenario(
        clean["responses"],
        clean["oracle"],
        refined_parents,
        list(phase_a.MODELS),
        "natural_alias_refined_registry",
    )
    return base, refined


def truncate_runs(runs: dict[str, np.ndarray], scenario: dict[str, Any], budget: int) -> dict:
    queried = runs["queried"][:, :budget]
    selected = runs["selected_local_path"][:, :budget]
    deploy = runs["deploy_regret_path"][:, :budget]
    expanded = runs["expanded_regret_path"][:, :budget]
    root = runs["root_regret_path"][:, :budget]
    final_local = selected[:, -1]
    final_parent = np.asarray(scenario["parents"], dtype=np.int64)[final_local]
    return {
        "queried": queried,
        "selected_local_path": selected,
        "final_local": final_local,
        "final_parent": final_parent,
        "deploy_regret_path": deploy,
        "expanded_regret_path": expanded,
        "root_regret_path": root,
        "cumulative_deploy_regret": deploy.sum(axis=1),
        "final_deploy_regret": deploy[:, -1],
        "cumulative_expanded_regret": expanded.sum(axis=1),
        "final_expanded_regret": expanded[:, -1],
        "cumulative_root_regret": root.sum(axis=1),
        "final_root_regret": root[:, -1],
    }


def run_fixed_queries(
    scenario: dict[str, Any],
    original_oracle: np.ndarray,
    pools: np.ndarray,
    queries: np.ndarray,
) -> dict[str, np.ndarray]:
    scenario_oracle = np.asarray(scenario["oracle"], dtype=np.float64)
    parents = np.asarray(scenario["parents"], dtype=np.int64)
    runs, budget = queries.shape
    selected = np.full((runs, budget), -1, dtype=np.int64)
    deploy = np.zeros((runs, budget), dtype=np.float64)
    expanded = np.zeros((runs, budget), dtype=np.float64)
    root = np.zeros((runs, budget), dtype=np.float64)
    for run_index in range(runs):
        pool = pools[run_index]
        entry_scores = scenario_oracle[:, pool].mean(axis=1)
        original_scores = original_oracle[:, pool].mean(axis=1)
        best_original = float(original_scores.max())
        best_expanded = float(entry_scores.max())
        cumulative = np.zeros(scenario_oracle.shape[0], dtype=np.float64)
        for step, query_value in enumerate(queries[run_index]):
            query = int(query_value)
            cumulative += scenario_oracle[:, query]
            local = int(np.argmax(cumulative))
            selected[run_index, step] = local
            deploy[run_index, step] = best_original - entry_scores[local]
            expanded[run_index, step] = best_expanded - entry_scores[local]
            root[run_index, step] = best_original - original_scores[parents[local]]
    final_local = selected[:, -1]
    final_parent = parents[final_local]
    return {
        "queried": queries.copy(),
        "selected_local_path": selected,
        "final_local": final_local,
        "final_parent": final_parent,
        "deploy_regret_path": deploy,
        "expanded_regret_path": expanded,
        "root_regret_path": root,
        "cumulative_deploy_regret": deploy.sum(axis=1),
        "final_deploy_regret": deploy[:, -1],
        "cumulative_expanded_regret": expanded.sum(axis=1),
        "final_expanded_regret": expanded[:, -1],
        "cumulative_root_regret": root.sum(axis=1),
        "final_root_regret": root[:, -1],
    }


def compare(
    base: dict[str, np.ndarray],
    refined: dict[str, np.ndarray],
    base_scenario: dict[str, Any],
    refined_scenario: dict[str, Any],
    seed: int,
    *,
    include_arrays: bool,
) -> dict[str, Any]:
    phase_a = step46.phase_a
    delta_cumulative_root = (
        refined["cumulative_root_regret"] - base["cumulative_root_regret"]
    )
    delta_final_root = refined["final_root_regret"] - base["final_root_regret"]
    delta_cumulative_deploy = (
        refined["cumulative_deploy_regret"] - base["cumulative_deploy_regret"]
    )
    delta_final_deploy = refined["final_deploy_regret"] - base["final_deploy_regret"]
    changed_queries = np.sum(base["queried"] != refined["queried"], axis=1)
    base_root_path = np.asarray(base_scenario["parents"])[base["selected_local_path"]]
    refined_root_path = np.asarray(refined_scenario["parents"])[
        refined["selected_local_path"]
    ]
    changed_root_positions = np.sum(base_root_path != refined_root_path, axis=1)
    jaccards = []
    for left, right in zip(base["queried"], refined["queried"]):
        left_set = set(int(value) for value in left)
        right_set = set(int(value) for value in right)
        jaccards.append(len(left_set & right_set) / len(left_set | right_set))
    result: dict[str, Any] = {
        "runs": int(len(delta_cumulative_root)),
        "query_matrix_digest_base": digest_array(base["queried"]),
        "query_matrix_digest_refined": digest_array(refined["queried"]),
        "root_selected_path_digest_base": digest_array(base_root_path),
        "root_selected_path_digest_refined": digest_array(refined_root_path),
        "path_change_fraction": float(np.mean(changed_queries > 0)),
        "mean_changed_query_positions": float(changed_queries.mean()),
        "mean_query_set_jaccard": float(np.mean(jaccards)),
        "selected_root_path_change_fraction": float(np.mean(changed_root_positions > 0)),
        "mean_changed_selected_root_positions": float(changed_root_positions.mean()),
        "final_selected_root_change_fraction": float(
            np.mean(base["final_parent"] != refined["final_parent"])
        ),
        "mean_delta_cumulative_root_regret": float(delta_cumulative_root.mean()),
        "median_delta_cumulative_root_regret": float(np.median(delta_cumulative_root)),
        "delta_cumulative_root_regret_95ci": phase_a.bootstrap_ci(
            delta_cumulative_root, seed
        ),
        "delta_cumulative_root_regret_signflip_p": phase_a.sign_flip_pvalue(
            delta_cumulative_root, seed + 1
        ),
        "mean_delta_final_root_regret": float(delta_final_root.mean()),
        "delta_final_root_regret_95ci": phase_a.bootstrap_ci(delta_final_root, seed + 2),
        "delta_final_root_regret_signflip_p": phase_a.sign_flip_pvalue(
            delta_final_root, seed + 3
        ),
        "mean_delta_cumulative_deployed_regret": float(
            delta_cumulative_deploy.mean()
        ),
        "delta_cumulative_deployed_regret_95ci": phase_a.bootstrap_ci(
            delta_cumulative_deploy, seed + 4
        ),
        "mean_delta_final_deployed_regret": float(delta_final_deploy.mean()),
        "root_and_entry_deltas_exactly_equal": bool(
            np.array_equal(delta_cumulative_root, delta_cumulative_deploy)
            and np.array_equal(delta_final_root, delta_final_deploy)
        ),
        "harmful_fraction": float(np.mean(delta_cumulative_root > 1e-12)),
        "improved_fraction": float(np.mean(delta_cumulative_root < -1e-12)),
        "tied_fraction": float(np.mean(np.abs(delta_cumulative_root) <= 1e-12)),
    }
    if include_arrays:
        result["paired_delta_cumulative_root_regret"] = [
            float(value) for value in delta_cumulative_root
        ]
        result["paired_delta_final_root_regret"] = [
            float(value) for value in delta_final_root
        ]
    return result


def cell_id(tau: float, budget: int) -> str:
    return f"tau={tau:g}|B={budget}"


def cell_seed(task_index: int, tau_index: int, budget_index: int) -> int:
    return 500000 + task_index * 10000 + tau_index * 100 + budget_index * 10


def evaluate_task(
    task: str,
    task_index: int,
    context: dict[str, Any],
    prereg: dict[str, Any],
) -> dict[str, Any]:
    started = time.time()
    phase_a = step46.phase_a
    spec = prereg["evaluation"]["tasks"][task]
    retained = int(prereg["natural_pair"]["retained_index"])
    added = int(prereg["natural_pair"]["added_index"])
    clean = context["clean"]
    data = context["data"]
    pools = context["pools"]
    base, refined = build_registries(clean, retained, added)

    response_mismatches = int(
        np.sum(clean["responses"][retained] != clean["responses"][added])
    )
    oracle_mismatches = int(
        np.sum(clean["oracle"][retained] != clean["oracle"][added])
    )
    if response_mismatches or oracle_mismatches:
        raise AssertionError(f"{task}: frozen natural pair is not exact")
    if int(context["examples"]) != int(spec["examples"]):
        raise AssertionError(f"{task}: example count drift")
    if pools.shape != (500, int(spec["pool_size"])):
        raise AssertionError(f"{task}: paired pool shape drift")
    if float(context["selected_tau"]) != float(spec["selected_temperature"]):
        raise AssertionError(f"{task}: frozen temperature drift")
    if [int(v) for v in context["budgets"]] != [
        int(v) for v in spec["robustness_budgets"]
    ]:
        raise AssertionError(f"{task}: budget grid drift")

    temperatures = [float(value) for value in prereg["evaluation"]["temperature_grid"]]
    primary_budget = int(spec["primary_budget"])
    selected_tau = float(spec["selected_temperature"])
    max_budget = max(int(value) for value in spec["robustness_budgets"])
    run_cache: dict[float, tuple[dict, dict]] = {}
    for tau in temperatures:
        tau_started = time.time()
        base_run = phase_a.run_scenario(base, data["oracle"], pools, max_budget, tau)
        refined_run = phase_a.run_scenario(
            refined, data["oracle"], pools, max_budget, tau
        )
        run_cache[tau] = (base_run, refined_run)
        print(f"[{task}] tau={tau:g} base/refined {time.time()-tau_started:.1f}s", flush=True)

    configurations = {
        (tau, primary_budget) for tau in temperatures
    } | {
        (selected_tau, int(budget)) for budget in spec["robustness_budgets"]
    }
    cells: dict[str, Any] = {}
    for tau, budget in sorted(configurations):
        base_full, refined_full = run_cache[tau]
        base_run = truncate_runs(base_full, base, budget)
        refined_run = truncate_runs(refined_full, refined, budget)
        tau_index = temperatures.index(tau)
        budget_index = [int(v) for v in spec["robustness_budgets"]].index(budget)
        identifier = cell_id(tau, budget)
        cells[identifier] = {
            "temperature": tau,
            "budget": budget,
            "is_primary": tau == selected_tau and budget == primary_budget,
            "comparison": compare(
                base_run,
                refined_run,
                base,
                refined,
                cell_seed(task_index, tau_index, budget_index),
                include_arrays=True,
            ),
        }

    final_seeds = list(range(50000, 50500))
    random_queries = np.asarray(
        [
            random.Random(seed + 99173).sample(pool.tolist(), primary_budget)
            for seed, pool in zip(final_seeds, pools)
        ],
        dtype=np.int64,
    )
    random_base = run_fixed_queries(base, data["oracle"], pools, random_queries)
    random_refined = run_fixed_queries(refined, data["oracle"], pools, random_queries)
    random_comparison = compare(
        random_base,
        random_refined,
        base,
        refined,
        590000 + task_index * 100,
        include_arrays=False,
    )
    random_pass = bool(
        random_comparison["query_matrix_digest_base"]
        == random_comparison["query_matrix_digest_refined"]
        and random_comparison["root_selected_path_digest_base"]
        == random_comparison["root_selected_path_digest_refined"]
        and random_comparison["mean_delta_cumulative_root_regret"] == 0.0
        and random_comparison["mean_delta_final_root_regret"] == 0.0
        and random_comparison["path_change_fraction"] == 0.0
        and random_comparison["selected_root_path_change_fraction"] == 0.0
    )
    return {
        "task": task,
        "pair_checks": {
            "examples": int(context["examples"]),
            "response_mismatches": response_mismatches,
            "oracle_vector_mismatches": oracle_mismatches,
            "retained_quality": float(clean["oracle"][retained].mean()),
            "added_quality": float(clean["oracle"][added].mean()),
            "quality_equal": bool(
                float(clean["oracle"][retained].mean())
                == float(clean["oracle"][added].mean())
            ),
        },
        "registry_checks": {
            "base_entries": len(base["labels"]),
            "refined_entries": len(refined["labels"]),
            "base_summary": phase_a.scenario_summary(base),
            "refined_summary": phase_a.scenario_summary(refined),
            "added_parent_root": int(refined["parents"][added]),
            "stable_base_global_indices": [
                int(value) for value in np.asarray(base["parents"], dtype=np.int64)
            ],
        },
        "primary_cell_id": cell_id(selected_tau, primary_budget),
        "cells": cells,
        "random_query_control": {
            "pass": random_pass,
            "query_seed_offset": 99173,
            "comparison": random_comparison,
        },
        "runtime_seconds": float(time.time() - started),
    }


def apply_multiplicity_and_classify(
    tasks: dict[str, dict[str, Any]], pair_valid: bool
) -> dict[str, Any]:
    phase_a = step46.phase_a
    primary_p = {
        task: float(
            tasks[task]["cells"][tasks[task]["primary_cell_id"]]["comparison"][
                "delta_cumulative_root_regret_signflip_p"
            ]
        )
        for task in TASKS
    }
    primary_q = phase_a.bh_adjust(primary_p)
    all_p: dict[str, float] = {}
    for task in TASKS:
        for identifier, cell in tasks[task]["cells"].items():
            all_p[f"{task}|{identifier}"] = float(
                cell["comparison"]["delta_cumulative_root_regret_signflip_p"]
            )
    all_q = phase_a.bh_adjust(all_p)

    path_material_tasks: list[str] = []
    outcome_material_tasks: list[str] = []
    harm_tasks: list[str] = []
    benefit_tasks: list[str] = []
    primary_rows = []
    for task in TASKS:
        identifier = tasks[task]["primary_cell_id"]
        comparison = tasks[task]["cells"][identifier]["comparison"]
        comparison["primary_family_bh_q"] = float(primary_q[task])
        comparison["all_21_cell_family_bh_q"] = float(all_q[f"{task}|{identifier}"])
        path_material = bool(
            comparison["path_change_fraction"] >= 0.10
            and comparison["mean_changed_query_positions"] >= 1.0
        )
        interval = comparison["delta_cumulative_root_regret_95ci"]
        excludes_zero = bool(interval[0] > 0.0 or interval[1] < 0.0)
        outcome_material = bool(
            abs(comparison["mean_delta_cumulative_root_regret"]) >= 0.10
            and excludes_zero
            and primary_q[task] < 0.05
        )
        if path_material:
            path_material_tasks.append(task)
        if outcome_material:
            outcome_material_tasks.append(task)
            if comparison["mean_delta_cumulative_root_regret"] > 0:
                harm_tasks.append(task)
            else:
                benefit_tasks.append(task)
        primary_rows.append(
            {
                "task": task,
                "cell_id": identifier,
                "path_material": path_material,
                "outcome_material": outcome_material,
                "direction": (
                    "harm"
                    if outcome_material
                    and comparison["mean_delta_cumulative_root_regret"] > 0
                    else "benefit"
                    if outcome_material
                    else "not_outcome_material"
                ),
                "path_change_fraction": comparison["path_change_fraction"],
                "mean_changed_query_positions": comparison[
                    "mean_changed_query_positions"
                ],
                "mean_delta_cumulative_root_regret": comparison[
                    "mean_delta_cumulative_root_regret"
                ],
                "delta_cumulative_root_regret_95ci": interval,
                "primary_family_bh_q": float(primary_q[task]),
            }
        )

    for task in TASKS:
        for identifier, cell in tasks[task]["cells"].items():
            cell["comparison"]["all_21_cell_family_bh_q"] = float(
                all_q[f"{task}|{identifier}"]
            )

    control_pass = all(tasks[task]["random_query_control"]["pass"] for task in TASKS)
    if not pair_valid or not control_pass:
        classification = "INVALID_CONTROL_OR_PAIR"
    elif len(path_material_tasks) >= 2 and len(outcome_material_tasks) >= 2:
        classification = "STRONG_NATURAL_OUTCOME_ANCHOR"
    elif len(path_material_tasks) >= 2 and len(outcome_material_tasks) == 1:
        classification = "LIMITED_NATURAL_OUTCOME_ANCHOR"
    elif len(path_material_tasks) >= 2:
        classification = "NATURAL_TRANSCRIPT_ANCHOR_ONLY"
    else:
        classification = "TASK_LOCAL_OR_NULL_NATURAL_EFFECT"
    return {
        "classification": classification,
        "pair_valid": pair_valid,
        "random_query_controls_pass": control_pass,
        "path_material_tasks": path_material_tasks,
        "outcome_material_tasks": outcome_material_tasks,
        "outcome_harm_tasks": harm_tasks,
        "outcome_benefit_tasks": benefit_tasks,
        "primary_rows": primary_rows,
        "primary_family_bh_q": {key: float(value) for key, value in primary_q.items()},
        "all_21_cell_family_size": len(all_q),
    }


def run() -> None:
    started = time.time()
    configure_paths()
    prereg, execution_lock, input_hashes = verify_locks()
    legacy_locks = step46.step31.verify_locks()
    contexts = {
        task: step46.step31.task_context(task, legacy_locks) for task in TASKS
    }
    pair_lists = exact_pairs(contexts)
    intersections = set(tuple(pair) for pair in pair_lists[TASKS[0]])
    for task in TASKS[1:]:
        intersections &= set(tuple(pair) for pair in pair_lists[task])
    expected_pair = (
        int(prereg["natural_pair"]["retained_index"]),
        int(prereg["natural_pair"]["added_index"]),
    )
    pair_valid = bool(intersections == {expected_pair})
    if not pair_valid:
        raise AssertionError(
            {"expected_cross_task_pair": expected_pair, "observed": sorted(intersections)}
        )

    tasks: dict[str, dict[str, Any]] = {}
    for task_index, task in enumerate(TASKS):
        print(f"[{task}] starting locked natural-alias audit", flush=True)
        tasks[task] = evaluate_task(task, task_index, contexts[task], prereg)
    decision = apply_multiplicity_and_classify(tasks, pair_valid)
    result = {
        "manifest_id": "STEP50_NATURAL_EXACT_ALIAS_RESULTS_V1",
        "created_local_date": "2026-08-09",
        "preregistration_sha256": file_sha256(PREREG),
        "execution_lock_sha256": file_sha256(EXECUTION_LOCK),
        "runner_sha256": file_sha256(RUNNER),
        "input_sha256": input_hashes,
        "execution_lock": execution_lock,
        "natural_pair": {
            "retained_index": expected_pair[0],
            "retained_model": step46.phase_a.MODELS[expected_pair[0]],
            "added_index": expected_pair[1],
            "added_model": step46.phase_a.MODELS[expected_pair[1]],
            "exact_pairs_by_task": pair_lists,
            "cross_task_exact_pair_intersection": [list(pair) for pair in sorted(intersections)],
            "unique_cross_task_pair_pass": pair_valid,
        },
        "tasks": tasks,
        "decision": decision,
        "runtime_seconds": float(time.time() - started),
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(decision, indent=2), flush=True)
    print(f"wrote {OUT}", flush=True)


def validate_only() -> None:
    configure_paths()
    prereg, lock, hashes = verify_locks()
    print(
        json.dumps(
            {
                "status": "LOCKS_VALID",
                "manifest_id": prereg["manifest_id"],
                "runner_sha256": lock["runner_sha256"],
                "input_sha256": hashes,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate_only()
    else:
        run()


if __name__ == "__main__":
    main()
