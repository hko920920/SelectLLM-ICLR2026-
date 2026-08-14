"""Independent consistency validator for Step 31 Phase B2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import run_step29_locked_phase_a as phase_a


ROOT = Path(__file__).resolve().parent
PATHS = {
    "prereg": ROOT / "STEP31_PHASE_B2_PREREGISTRATION_2026-08-06.json",
    "phase_b1": ROOT / "STEP30_PHASE_B1_ALL_TARGET_RESULTS_2026-08-06.json",
    "result": ROOT / "STEP31_PHASE_B2_RESULTS_2026-08-06.json",
}
EXPECTED = {
    "prereg": "a757631ef6f43ed90fd86c40f042a8ab9259e6008704fe460e04b8bc612b2af0",
    "phase_b1": "53f45f99edf2116a2e2f871fd7faac425e8ec0d38aa2944fb6e867c3420760bd",
    "result": "67249d225606a2bfc9ef5280e8430a37252c4f5dd7d047be9a90ccfed3eee15b",
}
TASK_ORDER = ("medqa", "gsm8k", "openbookqa")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(left: float, right: float, atol: float = 1e-12) -> bool:
    return bool(abs(float(left) - float(right)) <= atol)


def seed_for(
    task_index: int,
    distance_index: int,
    alias_index: int,
    temperature_index: int,
    budget_index: int,
    offset: int = 0,
) -> int:
    return (
        340000
        + task_index * 100000
        + distance_index * 10000
        + alias_index * 1000
        + temperature_index * 100
        + budget_index * 10
        + offset
    )


def verify_comparison(row: dict, seed: int, label: str) -> None:
    cumulative = np.asarray(
        row["paired_delta_cumulative_deployed_regret"], dtype=np.float64
    )
    final = np.asarray(row["paired_delta_final_deployed_regret"], dtype=np.float64)
    if cumulative.shape != (500,) or final.shape != (500,):
        raise AssertionError(f"{label}: paired array shape")
    if not close(cumulative.mean(), row["mean_delta_cumulative_deployed_regret"]):
        raise AssertionError(f"{label}: cumulative mean")
    if not close(np.median(cumulative), row["median_delta_cumulative_deployed_regret"]):
        raise AssertionError(f"{label}: cumulative median")
    if not close(final.mean(), row["mean_delta_final_deployed_regret"]):
        raise AssertionError(f"{label}: final mean")
    harmful = float(np.mean(cumulative > 1e-12))
    improved = float(np.mean(cumulative < -1e-12))
    tied = float(np.mean(np.abs(cumulative) <= 1e-12))
    if not close(harmful, row["harmful_fraction"]):
        raise AssertionError(f"{label}: harmful fraction")
    if not close(improved, row["improved_fraction"]):
        raise AssertionError(f"{label}: improved fraction")
    if not close(tied, row["tied_fraction"]):
        raise AssertionError(f"{label}: tied fraction")
    cumulative_ci = phase_a.bootstrap_ci(cumulative, seed)
    pvalue = phase_a.sign_flip_pvalue(cumulative, seed + 1)
    final_ci = phase_a.bootstrap_ci(final, seed + 2)
    if not np.array_equal(
        np.asarray(cumulative_ci),
        np.asarray(row["delta_cumulative_deployed_regret_95ci"]),
    ):
        raise AssertionError(f"{label}: cumulative CI")
    if not close(pvalue, row["delta_cumulative_deployed_regret_signflip_p"]):
        raise AssertionError(f"{label}: sign-flip p")
    if not np.array_equal(
        np.asarray(final_ci), np.asarray(row["delta_final_deployed_regret_95ci"])
    ):
        raise AssertionError(f"{label}: final CI")


def find_grid(task: dict, d: float, a: int, tau: float, budget: int) -> dict | None:
    rows = [
        row
        for row in task["grid_cells"]
        if row["nominal_distance"] == d
        and row["aliases"] == a
        and row["temperature"] == tau
        and row["budget"] == budget
    ]
    if len(rows) > 1:
        raise AssertionError("duplicate grid cell")
    return rows[0] if rows else None


def find_defense(task: dict, d: float, tau: float, budget: int) -> tuple[dict, dict] | None:
    distance_rows = [row for row in task["matched_cover"] if row["nominal_distance"] == d]
    if len(distance_rows) != 1 or distance_rows[0]["status"] != "EVALUATED":
        return None
    cells = [
        row
        for row in distance_rows[0]["cells"]
        if row["temperature"] == tau and row["budget"] == budget
    ]
    if len(cells) != 1:
        raise AssertionError("defense cell missing or duplicated")
    return distance_rows[0], cells[0]


def main() -> None:
    hashes = {name: sha256(path) for name, path in PATHS.items()}
    if hashes != EXPECTED:
        raise AssertionError({"expected": EXPECTED, "observed": hashes})
    prereg = json.loads(PATHS["prereg"].read_text(encoding="utf-8"))
    b1 = json.loads(PATHS["phase_b1"].read_text(encoding="utf-8"))
    result = json.loads(PATHS["result"].read_text(encoding="utf-8"))
    if result["complete"] is not True:
        raise AssertionError("incomplete result")
    if result["preregistration_sha256"] != EXPECTED["prereg"]:
        raise AssertionError("result/prereg binding mismatch")
    if result["phase_b1_result_sha256"] != EXPECTED["phase_b1"]:
        raise AssertionError("result/B1 binding mismatch")

    expected_grid_counts = {"medqa": 240, "gsm8k": 180, "openbookqa": 240}
    expected_defense_counts = {"medqa": 60, "gsm8k": 45, "openbookqa": 60}
    robustness = {}
    defense = {}
    audit_summary = {}

    for task_index, name in enumerate(TASK_ORDER):
        task = result["tasks"][name]
        temperatures = [float(v) for v in prereg["attack"]["temperature_grid"]]
        aliases = [int(v) for v in prereg["attack"]["alias_counts"]]
        distances = [float(v) for v in prereg["attack"]["nominal_distance_fractions"]]
        budgets = [int(v) for v in prereg["attack"]["budgets"][name]]
        if len(task["grid_cells"]) != expected_grid_counts[name]:
            raise AssertionError(f"{name}: grid count")
        observed_defense_count = sum(
            len(row.get("cells", [])) for row in task["matched_cover"]
        )
        if observed_defense_count != expected_defense_counts[name]:
            raise AssertionError(f"{name}: defense count")
        if task["target"] != prereg["frozen_targets"][name]:
            raise AssertionError(f"{name}: target substitution")

        scenario_metadata = task["scenarios"]
        for scenario_key, scenario in scenario_metadata.items():
            metadata = scenario["metadata"]
            if metadata["per_example_quality_equal"] is not True:
                raise AssertionError(f"{name}/{scenario_key}: quality equality")
            if any(abs(float(v)) > 1e-12 for v in metadata["alias_full_score_gaps"]):
                raise AssertionError(f"{name}/{scenario_key}: quality gap")
            if len(set(metadata["alias_hashes"])) != metadata["aliases"]:
                raise AssertionError(f"{name}/{scenario_key}: duplicate alias hashes")
            if not close(metadata["distance_fraction"], scenario["realized_distance"]):
                raise AssertionError(f"{name}/{scenario_key}: distance mismatch")

        full_pvalues = {}
        for row in task["grid_cells"]:
            d_index = distances.index(float(row["nominal_distance"]))
            a_index = aliases.index(int(row["aliases"]))
            t_index = temperatures.index(float(row["temperature"]))
            b_index = budgets.index(int(row["budget"]))
            seed = seed_for(task_index, d_index, a_index, t_index, b_index)
            verify_comparison(row, seed, f"{name}/attack/{row['cell_id']}")
            full_pvalues[row["cell_id"]] = row[
                "delta_cumulative_deployed_regret_signflip_p"
            ]
        adjusted_full = phase_a.bh_adjust(full_pvalues)
        for row in task["grid_cells"]:
            if not close(
                adjusted_full[row["cell_id"]],
                row["within_task_full_grid_BH_adjusted_p"],
            ):
                raise AssertionError(f"{name}/{row['cell_id']}: full-grid BH")

        for distance_row in task["matched_cover"]:
            if distance_row["status"] != "EVALUATED":
                continue
            d_index = distances.index(float(distance_row["nominal_distance"]))
            if distance_row["T2_alias_representatives_kept"] != 0:
                raise AssertionError(f"{name}: matched cover retained alias")
            for cell in distance_row["cells"]:
                t_index = temperatures.index(float(cell["temperature"]))
                b_index = budgets.index(int(cell["budget"]))
                base_seed = seed_for(task_index, d_index, 0, t_index, b_index)
                verify_comparison(
                    cell["clean_cover_cost"],
                    base_seed + 1,
                    f"{name}/cover-cost/{cell['cell_id']}",
                )
                verify_comparison(
                    cell["residual_attack"],
                    base_seed + 5,
                    f"{name}/residual/{cell['cell_id']}",
                )

        b1_row = b1["tasks"][name]["selected_final_confirmation"]
        overlap = find_grid(
            task, 0.05, 4, float(task["selected_temperature"]), max(budgets)
        )
        if overlap is None:
            raise AssertionError(f"{name}: missing B1 overlap")
        if (
            overlap["paired_delta_cumulative_deployed_regret"]
            != b1_row["paired_delta_cumulative_deployed_regret"]
            or overlap["paired_delta_final_deployed_regret"]
            != b1_row["paired_delta_final_deployed_regret"]
        ):
            raise AssertionError(f"{name}: B1 arrays changed")

        primary_budget = max(budgets)
        selected_tau = float(task["selected_temperature"])
        temp_family = [find_grid(task, 0.05, 4, tau, primary_budget) for tau in temperatures]
        if any(row is None for row in temp_family):
            raise AssertionError(f"{name}: incomplete temperature family")
        temp_p = {
            str(row["temperature"]): row["delta_cumulative_deployed_regret_signflip_p"]
            for row in temp_family
        }
        temp_bh = phase_a.bh_adjust(temp_p)
        for row in temp_family:
            if not close(
                temp_bh[str(row["temperature"])],
                row["binding_temperature_family_BH_adjusted_p"],
            ):
                raise AssertionError(f"{name}: temperature BH")
        temp_points = sum(row["mean_delta_cumulative_deployed_regret"] > 0 for row in temp_family)
        temp_sig = sum(
            row["delta_cumulative_deployed_regret_95ci"][0] > 0
            and temp_bh[str(row["temperature"])] < 0.05
            for row in temp_family
        )
        temp_pass = temp_points >= 4 and temp_sig >= 3

        budget_family = [find_grid(task, 0.05, 4, selected_tau, b) for b in budgets]
        budget_p = {
            str(row["budget"]): row["delta_cumulative_deployed_regret_signflip_p"]
            for row in budget_family
        }
        budget_bh = phase_a.bh_adjust(budget_p)
        for row in budget_family:
            if not close(
                budget_bh[str(row["budget"])],
                row["binding_budget_family_BH_adjusted_p"],
            ):
                raise AssertionError(f"{name}: budget BH")
        budget_points = sum(
            row["mean_delta_cumulative_deployed_regret"] > 0 for row in budget_family
        )
        budget_sig = sum(
            row["delta_cumulative_deployed_regret_95ci"][0] > 0
            and budget_bh[str(row["budget"])] < 0.05
            for row in budget_family
        )
        budget_pass = budget_points == 3 and budget_sig >= 2
        robustness[name] = {
            "temperature_positive": temp_points,
            "temperature_significant": temp_sig,
            "budget_positive": budget_points,
            "budget_significant": budget_sig,
            "pass": bool(temp_pass and budget_pass),
        }

        anchor_attack = find_grid(task, 0.1, 4, selected_tau, primary_budget)
        anchor_defense = find_defense(task, 0.1, selected_tau, primary_budget)
        if anchor_attack is None or anchor_defense is None:
            defense[name] = {"eligible": False, "qualifies": False}
        else:
            distance_row, cell = anchor_defense
            residual = cell["residual_attack"]
            clean_cost = cell["clean_cover_cost"]
            structure = distance_row["clean_cover_cost_structure"]
            attack_harm = anchor_attack["delta_cumulative_deployed_regret_95ci"][0] > 0
            exact_suppression = bool(
                residual["path_change_fraction"] == 0
                and abs(residual["mean_delta_cumulative_deployed_regret"]) <= 1e-12
                and abs(residual["mean_delta_final_deployed_regret"]) <= 1e-12
            )
            positive_cost = clean_cost["delta_cumulative_deployed_regret_95ci"][0] > 0
            material_span = structure["maximum_component_quality_span"] >= 0.05
            qualifies = bool(
                attack_harm and exact_suppression and (positive_cost or material_span)
            )
            defense[name] = {
                "eligible": True,
                "attack_harm": attack_harm,
                "exact_suppression": exact_suppression,
                "positive_clean_cost": positive_cost,
                "material_span": material_span,
                "qualifies": qualifies,
            }

        audit_summary[name] = {
            "grid_cells": len(task["grid_cells"]),
            "defense_cells": observed_defense_count,
            "B1_overlap": True,
        }

    r_pass = all(row["pass"] for row in robustness.values())
    d_count = sum(row["qualifies"] for row in defense.values())
    d_pass = d_count >= 2
    final_pass = r_pass and d_pass
    expected_decision = "FINAL_PAPER_GO" if final_pass else "FINAL_TOPIC_NO_GO_CURRENT_FORM"
    stored = result["binding_gates"]
    if stored["R_robust_harm_surface"]["pass"] != r_pass:
        raise AssertionError("stored R gate mismatch")
    if stored["D_matched_defense_cost"]["qualifying_tasks"] != d_count:
        raise AssertionError("stored D count mismatch")
    if stored["D_matched_defense_cost"]["pass"] != d_pass:
        raise AssertionError("stored D gate mismatch")
    if stored["final"]["decision"] != expected_decision or stored["final"]["pass"] != final_pass:
        raise AssertionError("stored final decision mismatch")

    print(
        json.dumps(
            {
                "status": "PASS_RESULT_CONSISTENCY",
                "hashes": hashes,
                "robustness": robustness,
                "defense": defense,
                "audit": audit_summary,
                "decision": expected_decision,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
