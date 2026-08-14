"""Run the locked Step 31 Phase B2 robustness and matched-defense grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

import run_step29_locked_phase_a as phase_a


ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "STEP31_PHASE_B2_PREREGISTRATION_2026-08-06.json"
STEP28 = ROOT / "STEP28_FINAL_MATRIX_MANIFEST_2026-08-06.json"
PHASE_A = ROOT / "STEP29_LOCKED_PHASE_A_RESULTS_2026-08-06.json"
SELECTION = ROOT / "STEP30_PHASE_B1_TARGET_SELECTION_LOCK_2026-08-06.json"
PHASE_B1 = ROOT / "STEP30_PHASE_B1_ALL_TARGET_RESULTS_2026-08-06.json"
OUT = ROOT / "STEP31_PHASE_B2_RESULTS_2026-08-06.json"

EXPECTED = {
    "prereg": "a757631ef6f43ed90fd86c40f042a8ab9259e6008704fe460e04b8bc612b2af0",
    "step28": "a16f9a262f8d100899a443e96e3ae141ca266f7b5121cdf5e628fc1a67c201ca",
    "phase_a": "9a075b143db6a8f11a42dbbbc313a9262a4547249fe81d39caa3f1c7a1540aaa",
    "selection": "98af2c94f8cd3f831ff83fd78fbf6c6755f0fda49460d4c16778b57be1974c7d",
    "phase_b1": "53f45f99edf2116a2e2f871fd7faac425e8ec0d38aa2944fb6e867c3420760bd",
}
TASK_ORDER = ("medqa", "gsm8k", "openbookqa")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_locks() -> dict[str, dict]:
    paths = {
        "prereg": PREREG,
        "step28": STEP28,
        "phase_a": PHASE_A,
        "selection": SELECTION,
        "phase_b1": PHASE_B1,
    }
    observed = {name: file_sha256(path) for name, path in paths.items()}
    if observed != EXPECTED:
        raise AssertionError({"expected": EXPECTED, "observed": observed})
    values = {name: load_json(path) for name, path in paths.items()}
    if values["prereg"]["manifest_id"] != "STEP31_PHASE_B2_V1":
        raise AssertionError("unexpected Step 31 manifest")
    if values["phase_b1"]["binding_gate"]["decision"] != "PASS_CONTINUE_TO_PHASE_B2":
        raise AssertionError("Phase B1 did not authorize Phase B2")
    return values


def load_one_task(name: str) -> dict:
    if name == "medqa":
        ids, responses, oracle, options, correct, provenance = phase_a.med_source.load_helm_medqa()
        data = {
            "ids": ids,
            "responses": responses,
            "oracle": oracle,
            "public_options": options,
            "correct": correct,
            "provenance": provenance,
            "public_option_mode": True,
        }
    else:
        loaded = phase_a.external_source.load_task(
            name, phase_a.external_source.TASKS[name]
        )
        data = {
            "ids": loaded["ids"],
            "responses": loaded["responses"],
            "oracle": loaded["oracle"],
            "public_options": loaded["alternatives"] if name == "openbookqa" else None,
            "correct": loaded["correct"],
            "provenance": loaded["provenance"],
            "public_option_mode": name == "openbookqa",
        }
    responses = np.asarray(data["responses"], dtype=object)
    oracle = np.asarray(data["oracle"], dtype=np.float64)
    correct = np.asarray(data["correct"], dtype=object)
    if responses.shape != oracle.shape or responses.shape[0] != len(phase_a.MODELS):
        raise AssertionError(f"{name}: malformed response/oracle matrix")
    reconstructed = (responses == correct[None, :]).astype(np.float64)
    data["raw_string_vs_recorded_oracle_mismatches"] = int(
        np.sum(reconstructed != oracle)
    )
    data["responses"] = responses
    data["oracle"] = oracle
    data["correct"] = correct
    return data


def truncate_runs(runs: dict, scenario: dict, budget: int) -> dict:
    queried = runs["queried"][:, :budget]
    selected = runs["selected_local_path"][:, :budget]
    deploy = runs["deploy_regret_path"][:, :budget]
    expanded = runs["expanded_regret_path"][:, :budget]
    root = runs["root_regret_path"][:, :budget]
    final_local = selected[:, -1]
    final_parent = scenario["parents"][final_local]
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


def comparison_seed(
    task_index: int,
    distance_index: int,
    alias_index: int,
    temperature_index: int,
    budget_index: int,
    *,
    defense_offset: int = 0,
) -> int:
    return (
        340000
        + task_index * 100000
        + distance_index * 10000
        + alias_index * 1000
        + temperature_index * 100
        + budget_index * 10
        + defense_offset
    )


def cell_id(nominal: float, aliases: int, tau: float, budget: int) -> str:
    return f"d={nominal:g}|a={aliases}|tau={tau:g}|B={budget}"


def defense_cell_id(nominal: float, tau: float, budget: int) -> str:
    return f"d={nominal:g}|a=4|tau={tau:g}|B={budget}"


def extended_cover_cost(clean: dict, components: list[list[int]]) -> dict:
    base = phase_a.cover_cost(clean, components)
    exact_components = phase_a.components_at_radius(clean["responses"], 0.0)
    exact_removed = len(phase_a.MODELS) - len(exact_components)
    scores = clean["oracle"].mean(axis=1)
    details = []
    all_pairs = 0
    for component in components:
        if len(component) <= 1:
            continue
        all_pairs += len(component) * (len(component) - 1) // 2
        details.append(
            {
                "members": [
                    {
                        "index": int(index),
                        "model": phase_a.MODELS[index],
                        "full_score": float(scores[index]),
                    }
                    for index in component
                ],
                "quality_span": float(
                    max(scores[index] for index in component)
                    - min(scores[index] for index in component)
                ),
            }
        )
    base.update(
        {
            "exact_response_models_removed": exact_removed,
            "additional_models_removed_beyond_exact": base["models_removed"]
            - exact_removed,
            "all_natural_pairs_in_nontrivial_components": all_pairs,
            "components_with_scores": details,
        }
    )
    return base


def task_context(name: str, locks: dict[str, dict]) -> dict:
    prereg = locks["prereg"]
    step28 = locks["step28"]
    phase_result = locks["phase_a"]
    selection = locks["selection"]
    data = load_one_task(name)
    clean = phase_a.base_scenario(data)
    config = step28["data"]["tasks"][name]
    target_index = int(selection["tasks"][name]["selected_target_index"])
    target = phase_a.MODELS[target_index]
    if target != prereg["frozen_targets"][name]:
        raise AssertionError(f"{name}: frozen target mismatch")
    budgets = [int(value) for value in prereg["attack"]["budgets"][name]]
    if budgets != [int(value) for value in config["robustness_budgets"]]:
        raise AssertionError(f"{name}: budget grid drift")
    tau_selected = float(
        phase_result["tasks"][name]["temperature_tuning"]["selected_tau"]
    )
    examples = int(clean["responses"].shape[1])
    seeds = list(
        range(
            int(prereg["evaluation"]["seeds_start"]),
            int(prereg["evaluation"]["seeds_stop_inclusive"]) + 1,
        )
    )
    pools = phase_a.pools_from_seeds(
        seeds, examples, int(prereg["evaluation"]["pool_sizes"][name])
    )
    return {
        "data": data,
        "clean": clean,
        "target_index": target_index,
        "target": target,
        "budgets": budgets,
        "max_budget": max(budgets),
        "selected_tau": tau_selected,
        "examples": examples,
        "pools": pools,
    }


def evaluate_task(name: str) -> dict:
    started = time.time()
    locks = verify_locks()
    prereg = locks["prereg"]
    context = task_context(name, locks)
    data = context["data"]
    clean = context["clean"]
    target_index = context["target_index"]
    budgets = context["budgets"]
    max_budget = context["max_budget"]
    pools = context["pools"]
    examples = context["examples"]
    task_index = TASK_ORDER.index(name)
    temperatures = [float(value) for value in prereg["attack"]["temperature_grid"]]
    alias_counts = [int(value) for value in prereg["attack"]["alias_counts"]]
    distances = [float(value) for value in prereg["attack"]["nominal_distance_fractions"]]

    print(f"[{name}] clean runs", flush=True)
    clean_runs = {
        tau: phase_a.run_scenario(
            clean, data["oracle"], pools, max_budget, tau
        )
        for tau in temperatures
    }

    result: dict[str, Any] = {
        "task": name,
        "target_index": target_index,
        "target": phase_a.MODELS[target_index],
        "target_full_score": float(data["oracle"][target_index].mean()),
        "examples": examples,
        "budgets": budgets,
        "selected_temperature": context["selected_tau"],
        "temperatures": temperatures,
        "alias_counts": alias_counts,
        "nominal_distances": distances,
        "scenarios": {},
        "ineligible_scenarios": [],
        "grid_cells": [],
        "matched_cover": [],
    }

    scenario_cache: dict[tuple[float, int], tuple[dict, dict] | None] = {}
    for distance_index, nominal in enumerate(distances):
        distance_queries = max(1, int(round(nominal * examples)))
        actual_distance = distance_queries / examples
        for alias_index, aliases in enumerate(alias_counts):
            key = (nominal, aliases)
            built = phase_a.build_wrapper
            try:
                scenario, metadata = built(
                    data,
                    clean,
                    target=target_index,
                    distance_queries=distance_queries,
                    aliases=aliases,
                    reference_aware=True,
                )
                scenario_cache[key] = (scenario, metadata)
            except ValueError:
                scenario_cache[key] = None
                result["ineligible_scenarios"].append(
                    {
                        "nominal_distance": nominal,
                        "realized_distance": actual_distance,
                        "distance_queries": distance_queries,
                        "aliases": aliases,
                        "status": "INELIGIBLE_TARGET_AT_DISTANCE",
                    }
                )
                print(
                    f"[{name}] ineligible d={nominal:g} a={aliases}", flush=True
                )
                continue
            if metadata["per_example_quality_equal"] is not True:
                raise AssertionError(f"{name}: per-example quality equality failed")
            if any(abs(float(value)) > 1e-12 for value in metadata["alias_full_score_gaps"]):
                raise AssertionError(f"{name}: nonzero alias quality gap")
            scenario_key = f"d={nominal:g}|a={aliases}"
            result["scenarios"][scenario_key] = {
                "nominal_distance": nominal,
                "realized_distance": actual_distance,
                "distance_queries": distance_queries,
                "aliases": aliases,
                "metadata": metadata,
                "scenario_summary": phase_a.scenario_summary(scenario),
            }
            for temperature_index, tau in enumerate(temperatures):
                run_started = time.time()
                changed_runs = phase_a.run_scenario(
                    scenario, data["oracle"], pools, max_budget, tau
                )
                for budget_index, budget in enumerate(budgets):
                    comparison = phase_a.compare_runs(
                        truncate_runs(clean_runs[tau], clean, budget),
                        truncate_runs(changed_runs, scenario, budget),
                        comparison_seed(
                            task_index,
                            distance_index,
                            alias_index,
                            temperature_index,
                            budget_index,
                        ),
                        include_arrays=True,
                    )
                    comparison.update(
                        {
                            "cell_id": cell_id(nominal, aliases, tau, budget),
                            "nominal_distance": nominal,
                            "realized_distance": actual_distance,
                            "distance_queries": distance_queries,
                            "aliases": aliases,
                            "temperature": tau,
                            "budget": budget,
                            "scenario_key": scenario_key,
                        }
                    )
                    result["grid_cells"].append(comparison)
                print(
                    f"[{name}] attack d={nominal:g} a={aliases} tau={tau:g} "
                    f"({time.time()-run_started:.1f}s)",
                    flush=True,
                )

    # A conservative all-grid BH adjustment is descriptive. Binding families
    # receive their own adjustments in the parent process.
    grid_pvalues = {
        row["cell_id"]: float(row["delta_cumulative_deployed_regret_signflip_p"])
        for row in result["grid_cells"]
    }
    grid_adjusted = phase_a.bh_adjust(grid_pvalues)
    for row in result["grid_cells"]:
        row["within_task_full_grid_BH_adjusted_p"] = float(
            grid_adjusted[row["cell_id"]]
        )

    print(f"[{name}] matched behavioral covers", flush=True)
    for distance_index, nominal in enumerate(distances):
        distance_queries = max(1, int(round(nominal * examples)))
        actual_distance = distance_queries / examples
        cached = scenario_cache[(nominal, 4)]
        if cached is None:
            result["matched_cover"].append(
                {
                    "nominal_distance": nominal,
                    "realized_radius": actual_distance,
                    "status": "INELIGIBLE_TARGET_AT_DISTANCE",
                }
            )
            continue
        attack_scenario, _ = cached
        defended_clean, clean_components = phase_a.quotient_scenario(
            clean, actual_distance
        )
        defended_attack, attack_components = phase_a.quotient_scenario(
            attack_scenario, actual_distance
        )
        aliases_kept = sum(
            label.startswith("T2_REFERENCE") for label in defended_attack["labels"]
        )
        distance_result: dict[str, Any] = {
            "nominal_distance": nominal,
            "realized_radius": actual_distance,
            "distance_queries": distance_queries,
            "status": "EVALUATED",
            "clean_cover_cost_structure": extended_cover_cost(clean, clean_components),
            "defended_clean_entries": len(defended_clean["labels"]),
            "defended_attack_entries": len(defended_attack["labels"]),
            "T2_alias_representatives_kept": aliases_kept,
            "attack_component_count": len(attack_components),
            "cells": [],
        }
        for temperature_index, tau in enumerate(temperatures):
            defended_clean_runs = phase_a.run_scenario(
                defended_clean, data["oracle"], pools, max_budget, tau
            )
            defended_attack_runs = phase_a.run_scenario(
                defended_attack, data["oracle"], pools, max_budget, tau
            )
            for budget_index, budget in enumerate(budgets):
                clean_cost = phase_a.compare_runs(
                    truncate_runs(clean_runs[tau], clean, budget),
                    truncate_runs(defended_clean_runs, defended_clean, budget),
                    comparison_seed(
                        task_index,
                        distance_index,
                        0,
                        temperature_index,
                        budget_index,
                        defense_offset=1,
                    ),
                    include_arrays=True,
                )
                residual = phase_a.compare_runs(
                    truncate_runs(defended_clean_runs, defended_clean, budget),
                    truncate_runs(defended_attack_runs, defended_attack, budget),
                    comparison_seed(
                        task_index,
                        distance_index,
                        0,
                        temperature_index,
                        budget_index,
                        defense_offset=5,
                    ),
                    include_arrays=True,
                )
                distance_result["cells"].append(
                    {
                        "cell_id": defense_cell_id(nominal, tau, budget),
                        "temperature": tau,
                        "budget": budget,
                        "clean_cover_cost": clean_cost,
                        "residual_attack": residual,
                    }
                )
        result["matched_cover"].append(distance_result)
        print(
            f"[{name}] cover rho={actual_distance:g} aliases-kept={aliases_kept} "
            f"removed={distance_result['clean_cover_cost_structure']['models_removed']} "
            f"span={distance_result['clean_cover_cost_structure']['maximum_component_quality_span']:.6f}",
            flush=True,
        )

    # The B1 cell must reproduce bit-for-bit despite new bootstrap seeds.
    b1_selected = locks["phase_b1"]["tasks"][name]["selected_final_confirmation"]
    overlap_id = cell_id(0.05, 4, context["selected_tau"], max_budget)
    overlap = next(row for row in result["grid_cells"] if row["cell_id"] == overlap_id)
    exact_overlap = bool(
        overlap["paired_delta_cumulative_deployed_regret"]
        == b1_selected["paired_delta_cumulative_deployed_regret"]
        and overlap["paired_delta_final_deployed_regret"]
        == b1_selected["paired_delta_final_deployed_regret"]
    )
    if not exact_overlap:
        raise AssertionError(f"{name}: B1 overlap failed")
    result["phase_b1_overlap"] = {
        "cell_id": overlap_id,
        "exact_paired_arrays": exact_overlap,
        "b1_mean": b1_selected["mean_delta_cumulative_deployed_regret"],
        "b2_mean": overlap["mean_delta_cumulative_deployed_regret"],
    }
    result["authenticated_lineage_oracle"] = {
        "description": "Retain the frozen parent and reject the four authenticated aliases using true lineage.",
        "attack_residual": 0.0,
        "clean_registry_change": 0.0,
        "is_response_only": False,
        "is_new_method_claim": False,
    }
    result["elapsed_seconds"] = time.time() - started
    print(f"[{name}] complete in {result['elapsed_seconds']:.1f}s", flush=True)
    return result


def find_grid_cell(task: dict, nominal: float, aliases: int, tau: float, budget: int) -> dict | None:
    matches = [
        row
        for row in task["grid_cells"]
        if row["nominal_distance"] == nominal
        and row["aliases"] == aliases
        and row["temperature"] == tau
        and row["budget"] == budget
    ]
    if len(matches) > 1:
        raise AssertionError("duplicate grid cell")
    return matches[0] if matches else None


def find_defense_cell(task: dict, nominal: float, tau: float, budget: int) -> tuple[dict, dict] | None:
    distances = [
        row for row in task["matched_cover"] if row["nominal_distance"] == nominal
    ]
    if len(distances) != 1 or distances[0]["status"] != "EVALUATED":
        return None
    matches = [
        row
        for row in distances[0]["cells"]
        if row["temperature"] == tau and row["budget"] == budget
    ]
    if len(matches) != 1:
        raise AssertionError("missing or duplicate defense cell")
    return distances[0], matches[0]


def apply_binding_gates(result: dict) -> None:
    robustness = {}
    for name in TASK_ORDER:
        task = result["tasks"][name]
        primary_budget = max(task["budgets"])
        selected_tau = float(task["selected_temperature"])
        temperature_family = [
            find_grid_cell(task, 0.05, 4, tau, primary_budget)
            for tau in task["temperatures"]
        ]
        if any(row is None for row in temperature_family):
            raise AssertionError(f"{name}: missing temperature-family cell")
        temp_p = {
            str(row["temperature"]): row["delta_cumulative_deployed_regret_signflip_p"]
            for row in temperature_family
        }
        temp_adjusted = phase_a.bh_adjust(temp_p)
        for row in temperature_family:
            row["binding_temperature_family_BH_adjusted_p"] = float(
                temp_adjusted[str(row["temperature"])]
            )
        temp_positive = sum(
            row["mean_delta_cumulative_deployed_regret"] > 0
            for row in temperature_family
        )
        temp_positive_significant = sum(
            row["delta_cumulative_deployed_regret_95ci"][0] > 0
            and row["binding_temperature_family_BH_adjusted_p"] < 0.05
            for row in temperature_family
        )
        temp_pass = temp_positive >= 4 and temp_positive_significant >= 3

        budget_family = [
            find_grid_cell(task, 0.05, 4, selected_tau, budget)
            for budget in task["budgets"]
        ]
        if any(row is None for row in budget_family):
            raise AssertionError(f"{name}: missing budget-family cell")
        budget_p = {
            str(row["budget"]): row["delta_cumulative_deployed_regret_signflip_p"]
            for row in budget_family
        }
        budget_adjusted = phase_a.bh_adjust(budget_p)
        for row in budget_family:
            row["binding_budget_family_BH_adjusted_p"] = float(
                budget_adjusted[str(row["budget"])]
            )
        budget_positive = sum(
            row["mean_delta_cumulative_deployed_regret"] > 0 for row in budget_family
        )
        budget_positive_significant = sum(
            row["delta_cumulative_deployed_regret_95ci"][0] > 0
            and row["binding_budget_family_BH_adjusted_p"] < 0.05
            for row in budget_family
        )
        budget_pass = budget_positive == 3 and budget_positive_significant >= 2
        robustness[name] = {
            "temperature_family": {
                "positive_points": temp_positive,
                "positive_significant": temp_positive_significant,
                "pass": temp_pass,
                "cells": [row["cell_id"] for row in temperature_family],
            },
            "budget_family": {
                "positive_points": budget_positive,
                "positive_significant": budget_positive_significant,
                "pass": budget_pass,
                "cells": [row["cell_id"] for row in budget_family],
            },
            "pass": bool(temp_pass and budget_pass),
        }
    robustness_pass = all(row["pass"] for row in robustness.values())

    defense = {}
    for name in TASK_ORDER:
        task = result["tasks"][name]
        selected_tau = float(task["selected_temperature"])
        primary_budget = max(task["budgets"])
        attack = find_grid_cell(task, 0.1, 4, selected_tau, primary_budget)
        defended = find_defense_cell(task, 0.1, selected_tau, primary_budget)
        if attack is None or defended is None:
            defense[name] = {
                "eligible": False,
                "qualifies": False,
                "reason": "INELIGIBLE_TARGET_AT_10_PERCENT",
            }
            continue
        distance_row, defense_cell = defended
        residual = defense_cell["residual_attack"]
        clean_cost = defense_cell["clean_cover_cost"]
        structure = distance_row["clean_cover_cost_structure"]
        attack_harm = attack["delta_cumulative_deployed_regret_95ci"][0] > 0
        exact_suppression = bool(
            residual["path_change_fraction"] == 0.0
            and abs(residual["mean_delta_cumulative_deployed_regret"]) <= 1e-12
            and abs(residual["mean_delta_final_deployed_regret"]) <= 1e-12
        )
        positive_clean_cost = clean_cost["delta_cumulative_deployed_regret_95ci"][0] > 0
        material_span = structure["maximum_component_quality_span"] >= 0.05
        costful = bool(positive_clean_cost or material_span)
        qualifies = bool(attack_harm and exact_suppression and costful)
        defense[name] = {
            "eligible": True,
            "attack_mean": attack["mean_delta_cumulative_deployed_regret"],
            "attack_ci": attack["delta_cumulative_deployed_regret_95ci"],
            "attack_harm": attack_harm,
            "exact_suppression": exact_suppression,
            "clean_cost_mean": clean_cost["mean_delta_cumulative_deployed_regret"],
            "clean_cost_ci": clean_cost["delta_cumulative_deployed_regret_95ci"],
            "positive_clean_cost": positive_clean_cost,
            "models_removed": structure["models_removed"],
            "additional_models_removed_beyond_exact": structure[
                "additional_models_removed_beyond_exact"
            ],
            "maximum_component_quality_span": structure[
                "maximum_component_quality_span"
            ],
            "material_span_at_least_5pp": material_span,
            "qualifies": qualifies,
        }
    defense_qualifying = sum(row["qualifies"] for row in defense.values())
    defense_pass = defense_qualifying >= 2
    final_pass = robustness_pass and defense_pass
    result["binding_gates"] = {
        "R_robust_harm_surface": {
            "tasks": robustness,
            "pass": robustness_pass,
        },
        "D_matched_defense_cost": {
            "tasks": defense,
            "qualifying_tasks": defense_qualifying,
            "pass": defense_pass,
        },
        "final": {
            "pass": final_pass,
            "decision": "FINAL_PAPER_GO"
            if final_pass
            else "FINAL_TOPIC_NO_GO_CURRENT_FORM",
        },
    }


def smoke_test() -> None:
    locks = verify_locks()
    prereg = locks["prereg"]
    for name in TASK_ORDER:
        data = load_one_task(name)
        clean = phase_a.base_scenario(data)
        target_index = int(locks["selection"]["tasks"][name]["selected_target_index"])
        examples = clean["responses"].shape[1]
        distance_queries = max(1, int(round(0.05 * examples)))
        scenario, metadata = phase_a.build_wrapper(
            data,
            clean,
            target=target_index,
            distance_queries=distance_queries,
            aliases=4,
            reference_aware=True,
        )
        pools = phase_a.pools_from_seeds(
            list(range(2000, 2005)), examples, min(100, examples)
        )
        clean_runs = phase_a.run_scenario(clean, data["oracle"], pools, 10, 1.0)
        attack_runs = phase_a.run_scenario(scenario, data["oracle"], pools, 10, 1.0)
        short = truncate_runs(attack_runs, scenario, 5)
        if short["queried"].shape != (5, 5):
            raise AssertionError("prefix truncation failed")
        defended_clean, _ = phase_a.quotient_scenario(clean, metadata["distance_fraction"])
        defended_attack, _ = phase_a.quotient_scenario(
            scenario, metadata["distance_fraction"]
        )
        dc = phase_a.run_scenario(defended_clean, data["oracle"], pools, 10, 1.0)
        da = phase_a.run_scenario(defended_attack, data["oracle"], pools, 10, 1.0)
        residual = phase_a.compare_runs(dc, da, 990000, include_arrays=False)
        print(
            json.dumps(
                {
                    "task": name,
                    "per_example_quality_equal": metadata["per_example_quality_equal"],
                    "attack_path_change": phase_a.compare_runs(
                        clean_runs, attack_runs, 990100, include_arrays=False
                    )["path_change_fraction"],
                    "matched_cover_residual": residual[
                        "mean_delta_cumulative_deployed_regret"
                    ],
                    "matched_cover_path_change": residual["path_change_fraction"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    print("PASS_DEVELOPMENT_SMOKE", flush=True)


def run_full() -> None:
    started = time.time()
    locks = verify_locks()
    if OUT.exists():
        existing = load_json(OUT)
        if existing.get("complete") is True:
            raise FileExistsError(f"completed result already exists: {OUT}")
    result: dict[str, Any] = {
        "gate": "STEP31_PHASE_B2_FINAL_GRID",
        "complete": False,
        "preregistration_sha256": EXPECTED["prereg"],
        "phase_b1_selection_lock_sha256": EXPECTED["selection"],
        "phase_b1_result_sha256": EXPECTED["phase_b1"],
        "method_source": "https://arxiv.org/html/2605.24981v1",
        "tasks": {},
    }
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(evaluate_task, name): name for name in TASK_ORDER}
        for future in as_completed(futures):
            name = futures[future]
            task_result = future.result()
            result["tasks"][name] = task_result
            result["elapsed_seconds_so_far"] = time.time() - started
            OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
            print(f"[main] stored {name}", flush=True)
    result["tasks"] = {name: result["tasks"][name] for name in TASK_ORDER}
    apply_binding_gates(result)
    result["complete"] = True
    result.pop("elapsed_seconds_so_far", None)
    result["elapsed_seconds"] = time.time() - started
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["binding_gates"], indent=2, sort_keys=True), flush=True)
    print(f"result_sha256={file_sha256(OUT)}", flush=True)
    print(f"completed in {result['elapsed_seconds']:.1f}s -> {OUT}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke_test()
    else:
        run_full()


if __name__ == "__main__":
    main()
