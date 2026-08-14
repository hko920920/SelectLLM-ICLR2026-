"""Independent structural/statistical validator for Step 46 results."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "STEP46_CLONE_ROBUST_SOFT_WEIGHTING_RESULTS_2026-08-08.json"
PREREG = ROOT / "STEP46_CLONE_ROBUST_SOFT_WEIGHTING_PREREGISTRATION_2026-08-08.json"
RUNNER = ROOT / "run_step46_clone_robust_soft_weighting.py"
LOCK = ROOT / "STEP46_CLONE_ROBUST_SOFT_WEIGHTING_EXECUTION_LOCK_2026-08-08.json"
STEP31 = ROOT / "STEP31_PHASE_B2_RESULTS_2026-08-06.json"

EXPECTED_RESULT_HASH = "15b693f06f708e4583aac52ab21dd85cdc8326c87683ff88938c128ab7b7b136"
EXPECTED_PREREG_HASH = "9cdfaeeab69bb4e516e99f452fd932bd34323d3ed3a225c5e2833e8a1544b8e2"
EXPECTED_RUNNER_HASH = "fffc9bf093978fd0b4340ef06ae1da8adf8bba6c1483aadf55f966ec28ff55cf"
TASKS = ("medqa", "gsm8k", "openbookqa")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def validate_comparison(comparison: dict) -> None:
    cumulative = np.asarray(
        comparison["paired_delta_cumulative_deployed_regret"], dtype=np.float64
    )
    final = np.asarray(comparison["paired_delta_final_deployed_regret"], dtype=np.float64)
    if len(cumulative) != 500 or len(final) != 500 or comparison["runs"] != 500:
        raise AssertionError("paired comparison length drift")
    if not close(cumulative.mean(), comparison["mean_delta_cumulative_deployed_regret"]):
        raise AssertionError("cumulative mean mismatch")
    if not close(np.median(cumulative), comparison["median_delta_cumulative_deployed_regret"]):
        raise AssertionError("cumulative median mismatch")
    if not close(final.mean(), comparison["mean_delta_final_deployed_regret"]):
        raise AssertionError("final mean mismatch")
    tolerance = 1e-12
    harmful = float(np.mean(cumulative > tolerance))
    improved = float(np.mean(cumulative < -tolerance))
    tied = float(np.mean(np.abs(cumulative) <= tolerance))
    if not close(harmful, comparison["harmful_fraction"]):
        raise AssertionError("harmful fraction mismatch")
    if not close(improved, comparison["improved_fraction"]):
        raise AssertionError("improved fraction mismatch")
    if not close(tied, comparison["tied_fraction"]):
        raise AssertionError("tied fraction mismatch")
    for key in (
        "delta_cumulative_deployed_regret_95ci",
        "delta_final_deployed_regret_95ci",
    ):
        interval = comparison[key]
        if len(interval) != 2 or float(interval[0]) > float(interval[1]):
            raise AssertionError(f"malformed interval: {key}")
    if not 0.0 <= float(comparison["path_change_fraction"]) <= 1.0:
        raise AssertionError("invalid path fraction")
    if not 0.0 <= float(comparison["mean_query_set_jaccard"]) <= 1.0:
        raise AssertionError("invalid Jaccard")


def locked_step31_cell(step31: dict, task: str, cell_id: str) -> dict:
    return next(
        row for row in step31["tasks"][task]["grid_cells"] if row["cell_id"] == cell_id
    )


def main() -> None:
    if sha256(RESULT) != EXPECTED_RESULT_HASH:
        raise AssertionError("Step 46 result hash drift")
    if sha256(PREREG) != EXPECTED_PREREG_HASH:
        raise AssertionError("Step 46 preregistration hash drift")
    if sha256(RUNNER) != EXPECTED_RUNNER_HASH:
        raise AssertionError("Step 46 runner hash drift")

    result = json.loads(RESULT.read_text(encoding="utf-8"))
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    step31 = json.loads(STEP31.read_text(encoding="utf-8"))
    if lock["completed_full_task_outcomes_before_lock"] != 0:
        raise AssertionError("execution was not outcome-blind")
    if lock["preregistration_sha256"].lower() != EXPECTED_PREREG_HASH:
        raise AssertionError("execution lock prereg hash mismatch")
    if lock["runner_sha256"].lower() != EXPECTED_RUNNER_HASH:
        raise AssertionError("execution lock runner hash mismatch")
    if result["runner_sha256"] != EXPECTED_RUNNER_HASH:
        raise AssertionError("result runner hash mismatch")
    if result["manifest_sha256"] != EXPECTED_PREREG_HASH:
        raise AssertionError("result manifest hash mismatch")
    if set(result["tasks"]) != set(TASKS):
        raise AssertionError("task set mismatch")
    if not all(test["passed"] for test in result["synthetic_tests"].values()):
        raise AssertionError("synthetic test failure")

    rules = prereg["grid"]["graph_rules"]
    alphas = [float(value) for value in prereg["grid"]["alpha_values"]]
    expected_config_ids = {f"{rule}|alpha={alpha:g}" for rule in rules for alpha in alphas}
    total_clean_cells = 0
    total_attack_cells = 0
    cheap_count = 0
    suppressed_count = 0
    observed_summary: dict[str, dict[str, float | int | bool]] = {}

    for task in TASKS:
        task_result = result["tasks"][task]
        if not all(task_result["uniform_runner_parity"].values()):
            raise AssertionError(f"{task}: uniform runner parity failed")
        configs = task_result["configs"]
        if {cell["config_id"] for cell in configs} != expected_config_ids:
            raise AssertionError(f"{task}: config grid mismatch")
        if len(configs) != 21:
            raise AssertionError(f"{task}: expected 21 clean cells")
        total_clean_cells += len(configs)

        for distance_text, undefended in task_result["undefended"].items():
            validate_comparison(undefended["comparison"])
            nominal = float(distance_text)
            locked_id = (
                f"d={nominal:g}|a=4|tau={float(task_result['temperature']):g}"
                f"|B={int(task_result['budget'])}"
            )
            locked = locked_step31_cell(step31, task, locked_id)
            current = undefended["comparison"]
            if current["paired_delta_cumulative_deployed_regret"] != locked[
                "paired_delta_cumulative_deployed_regret"
            ]:
                raise AssertionError(f"{task} {distance_text}: locked cumulative array mismatch")
            if current["paired_delta_final_deployed_regret"] != locked[
                "paired_delta_final_deployed_regret"
            ]:
                raise AssertionError(f"{task} {distance_text}: locked final array mismatch")

        for cell in configs:
            prior = np.asarray(cell["clean_prior"], dtype=np.float64)
            if len(prior) != 31 or np.any(prior <= 0) or not close(prior.sum(), 1.0):
                raise AssertionError(f"{task} {cell['config_id']}: invalid clean prior")
            validate_comparison(cell["clean_cost"])
            clean_cost = cell["clean_cost"]
            cheap = bool(
                abs(float(clean_cost["mean_delta_cumulative_deployed_regret"])) <= 0.10
                and float(clean_cost["delta_cumulative_deployed_regret_95ci"][0]) <= 0.0
                <= float(clean_cost["delta_cumulative_deployed_regret_95ci"][1])
                and float(clean_cost["delta_final_deployed_regret_95ci"][0]) <= 0.0
                <= float(clean_cost["delta_final_deployed_regret_95ci"][1])
            )
            if cheap != bool(cell["cheap_clean"]):
                raise AssertionError("cheap-clean classification mismatch")
            cheap_count += int(cheap)

            for distance_text, attack in cell["attacks"].items():
                total_attack_cells += 1
                attack_prior = np.asarray(attack["attack_prior"], dtype=np.float64)
                if (
                    len(attack_prior) != 35
                    or np.any(attack_prior <= 0)
                    or not close(attack_prior.sum(), 1.0)
                ):
                    raise AssertionError("invalid attacked prior")
                target = int(task_result["target_index"])
                family_mass = float(attack_prior[target] + attack_prior[-4:].sum())
                if not close(family_mass, attack["attacked_parent_plus_alias_mass"]):
                    raise AssertionError("family mass mismatch")
                if not close(attack_prior[-4:].sum(), attack["attacked_alias_only_mass"]):
                    raise AssertionError("alias mass mismatch")
                collapsed = attack_prior[:31].copy()
                collapsed[target] += attack_prior[-4:].sum()
                tv = float(0.5 * np.abs(collapsed - prior).sum())
                if not close(tv, attack["root_collapsed_prior_total_variation"]):
                    raise AssertionError("root prior TV mismatch")
                validate_comparison(attack["residual"])
                undefended_mean = float(
                    task_result["undefended"][distance_text]["comparison"]
                    ["mean_delta_cumulative_deployed_regret"]
                )
                residual_mean = float(
                    attack["residual"]["mean_delta_cumulative_deployed_regret"]
                )
                ratio = abs(residual_mean) / max(abs(undefended_mean), 1e-15)
                if not close(ratio, attack["absolute_residual_to_undefended_ratio"]):
                    raise AssertionError("residual ratio mismatch")
                suppressed = bool(
                    ratio <= 0.10 + 1e-12
                    and float(attack["residual"]["path_change_fraction"]) <= 0.10 + 1e-12
                )
                if suppressed != bool(attack["attack_suppressed"]):
                    raise AssertionError("suppression classification mismatch")
                if attack["quality_invariance"] is not True:
                    raise AssertionError("quality-invariance flag failed")
                suppressed_count += int(suppressed)

            observed_summary[f"{task}|{cell['config_id']}"] = {
                "cheap": cheap,
                "attack_count": len(cell["attacks"]),
            }

    if total_clean_cells != 63 or total_attack_cells != 105:
        raise AssertionError(
            {"clean_cells": total_clean_cells, "attack_cells": total_attack_cells}
        )

    recomputed_rows = []
    for rule in rules:
        for alpha in alphas:
            config_id = f"{rule}|alpha={alpha:g}"
            cheap_tasks = 0
            suppressed_cells = 0
            eligible_cells = 0
            for task in TASKS:
                cell = next(
                    row for row in result["tasks"][task]["configs"] if row["config_id"] == config_id
                )
                cheap_tasks += int(cell["cheap_clean"])
                for attack in cell["attacks"].values():
                    eligible_cells += 1
                    suppressed_cells += int(attack["attack_suppressed"])
            recomputed_rows.append(
                {
                    "config_id": config_id,
                    "suppressed_attack_cells": suppressed_cells,
                    "eligible_attack_cells": eligible_cells,
                    "cheap_clean_tasks": cheap_tasks,
                    "dominant": suppressed_cells == 5 and cheap_tasks == 3,
                    "partial": suppressed_cells >= 3 and cheap_tasks >= 2,
                }
            )

    dominant = [row for row in recomputed_rows if row["dominant"]]
    partial = [row for row in recomputed_rows if row["partial"]]
    classification = (
        "DOMINANT_SOFT_FIX"
        if dominant
        else "PARTIAL_SOFT_FIX"
        if partial
        else "TRADEOFF_OR_FAILURE"
    )
    if classification != result["decision"]["classification"]:
        raise AssertionError("aggregate decision mismatch")
    if classification != "TRADEOFF_OR_FAILURE":
        raise AssertionError("unexpected locked outcome")

    all_paths_changed = all(
        close(attack["residual"]["path_change_fraction"], 1.0)
        for task in result["tasks"].values()
        for cell in task["configs"]
        for attack in cell["attacks"].values()
    )
    if not all_paths_changed:
        raise AssertionError("expected full residual path sensitivity in every cell")

    print("PASS_STEP46_CLONE_ROBUST_SOFT_WEIGHTING")
    print(f"classification = {classification}")
    print(f"clean_cells = {total_clean_cells}")
    print(f"attack_cells = {total_attack_cells}")
    print(f"cheap_clean_cells = {cheap_count}")
    print(f"suppressed_attack_cells = {suppressed_count}")
    print(f"all_105_residual_path_change_fractions_equal_one = {all_paths_changed}")


if __name__ == "__main__":
    main()
