from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
from typing import Any

import numpy as np

import step102_common as common
from step100_stagea_develop import run_condition


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / f"STEP103_IMDB_ROBUST_DUAL_DEVELOPMENT_SELECTION_PROTOCOL_{common.DATE}.md"
STEP102_LEDGER = ROOT / f"STEP102_STAGEA_COMPLETE_LEDGER_{common.DATE}.json"
STEP102_GRID = ROOT / f"STEP102_STAGEA_COMPLETE_GRID_{common.DATE}.json"
STEP102_ARRAYS = ROOT / f"STEP102_STAGEA_DEVELOPMENT_ARRAYS_{common.DATE}.npz"
OUTPUT_GRID = ROOT / f"STEP103_COMPLETE_DUAL_DEVELOPMENT_GRID_{common.DATE}.json"
OUTPUT_ARRAYS = ROOT / f"STEP103_SELECTED_DEVELOPMENT_ARRAYS_{common.DATE}.npz"
OUTPUT_LEDGER = ROOT / f"STEP103_ROBUST_DUAL_DEVELOPMENT_LEDGER_{common.DATE}.json"


def worker(loss_cap: float) -> dict[str, Any]:
    ledger = json.loads(STEP102_LEDGER.read_text(encoding="utf-8"))
    arrays = np.load(STEP102_ARRAYS, allow_pickle=False)
    predictions = arrays["root_predictions"].astype(np.int64)
    labels = arrays["labels"].astype(np.int64)
    scores = arrays["error_scores"].astype(np.float64)
    indices = arrays["selector_verify_indices"].astype(np.int64)
    pair_id = f"l{loss_cap:.4f}_t0.025"
    thresholds = ledger["threshold_pairs"][pair_id]["thresholds"]
    aliases, _ = common.make_alias_codes(predictions[:, 0], scores, thresholds)
    rows: list[dict[str, Any]] = []
    vectors: dict[str, dict[str, np.ndarray]] = {}
    for budget in common.BUDGETS:
        for tau in common.TAUS:
            key = f"l{loss_cap:.4f}_b{budget}_tau{tau:.3f}"
            effect, raw = run_condition(
                predictions, labels, indices, aliases, common.VERIFY_SEEDS,
                budget, tau,
            )
            rows.append({
                "unique_condition_id": key,
                "loss_cap": loss_cap,
                "budget": budget,
                "tau": tau,
                "effect": effect,
            })
            vectors[key] = {
                "terminal": raw["terminal"],
                "active": raw["active"],
                "cumulative": raw["cumulative"],
                "fixed_terminal": raw["fixed_terminal"],
                "fixed_cumulative": raw["fixed_cumulative"],
            }
    return {"loss_cap": loss_cap, "rows": rows, "vectors": vectors}


def main() -> None:
    for path in (OUTPUT_GRID, OUTPUT_ARRAYS, OUTPUT_LEDGER):
        if path.exists():
            raise FileExistsError(path)
    step102 = json.loads(STEP102_LEDGER.read_text(encoding="utf-8"))
    search_grid = json.loads(STEP102_GRID.read_text(encoding="utf-8"))
    if step102["decision"] != "NO_GO_STEP102_LEARNED_DEVELOPMENT_STOP":
        raise AssertionError("Step 102 non-relabeling drift")
    if step102["selected"]["cell_id"] != "l0.0025_t0.025_b5_tau0.050":
        raise AssertionError("Step 102 selected-cell provenance drift")
    equivalence_classes: dict[str, list[str]] = {}
    for loss_cap in common.LOSS_CAPS:
        ids = [f"l{loss_cap:.4f}_t{trigger:.3f}" for trigger in common.TRIGGER_CAPS]
        threshold_values = [tuple(step102["threshold_pairs"][key]["thresholds"]) for key in ids]
        if len(set(threshold_values)) != 1:
            raise AssertionError({"nonidentical_trigger_cap_class": ids})
        equivalence_classes[f"loss_{loss_cap:.4f}"] = ids

    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(worker, common.LOSS_CAPS))
    unique_rows = {
        row["unique_condition_id"]: row
        for result in results for row in result["rows"]
    }
    unique_vectors = {
        key: value
        for result in results for key, value in result["vectors"].items()
    }
    complete: list[dict[str, Any]] = []
    for search in search_grid["rows"]:
        unique_id = f"l{search['loss_cap']:.4f}_b{search['budget']}_tau{search['tau']:.3f}"
        verify = unique_rows[unique_id]["effect"]
        verify_gates = {
            "quality": bool(search["gates"]["quality"]),
            "path_change_at_least_half": verify["path_change_rate"] >= 0.50,
            "directionality": verify["parent_to_challenger_rate"] > verify["challenger_to_parent_rate"],
            "terminal_at_least_half_point": verify["mean_terminal_delta"] >= 0.005,
            "active_at_least_half_point": verify["mean_active_minus_fixed_terminal_delta"] >= 0.005,
            "cumulative_positive": verify["mean_cumulative_delta"] > 0,
            "fixed_exact_zero": verify["max_abs_fixed_terminal_delta"] == 0
            and verify["max_abs_fixed_cumulative_delta"] == 0
            and verify["fixed_root_history_exact"],
        }
        complete.append({
            "cell_id": search["cell_id"],
            "unique_condition_id": unique_id,
            "loss_cap": search["loss_cap"],
            "trigger_cap": search["trigger_cap"],
            "budget": search["budget"],
            "tau": search["tau"],
            "search": search["effect"],
            "search_gates": search["gates"],
            "verify": verify,
            "verify_gates": verify_gates,
            "dual_eligible": bool(search["eligible"] and all(verify_gates.values())),
            "robust_terminal": min(
                search["effect"]["mean_terminal_delta"], verify["mean_terminal_delta"]
            ),
            "robust_cumulative": min(
                search["effect"]["mean_cumulative_delta"], verify["mean_cumulative_delta"]
            ),
            "robust_path": min(
                search["effect"]["path_change_rate"], verify["path_change_rate"]
            ),
        })
    eligible = [row for row in complete if row["dual_eligible"]]
    selected = sorted(
        eligible,
        key=lambda row: (
            -row["robust_terminal"], -row["robust_cumulative"], -row["robust_path"],
            row["loss_cap"], row["trigger_cap"], row["budget"], row["tau"],
            row["cell_id"],
        ),
    )[0] if eligible else None
    inference = None
    inference_gates: dict[str, bool] = {}
    selected_vectors = None
    if selected:
        selected_vectors = unique_vectors[selected["unique_condition_id"]]
        inference = {
            "terminal": common.effect_summary(
                selected_vectors["terminal"], 103700, 103701
            ),
            "active_minus_fixed_terminal": common.effect_summary(
                selected_vectors["active"], 103701, 103702
            ),
            "cumulative": common.effect_summary(
                selected_vectors["cumulative"], 103702, 103703
            ),
        }
        inference_gates = {
            "terminal_positive": inference["terminal"]["bootstrap_95"][0] > 0
            and inference["terminal"]["one_sided_signflip_p"] <= 0.05,
            "active_positive": inference["active_minus_fixed_terminal"]["bootstrap_95"][0] > 0
            and inference["active_minus_fixed_terminal"]["one_sided_signflip_p"] <= 0.05,
            "cumulative_positive": inference["cumulative"]["mean"] > 0
            and inference["cumulative"]["bootstrap_95"][0] > 0,
        }
    decision = (
        "GO_STEP103_TO_HELDOUT_LOCK"
        if selected and all(inference_gates.values())
        else "NO_GO_STEP103_ROBUST_DEVELOPMENT_STOP"
    )
    common.json_dump(OUTPUT_GRID, {
        "grid_id": "STEP103_COMPLETE_81_CELL_DUAL_DEVELOPMENT_GRID_V1",
        "equivalence_classes": equivalence_classes,
        "unique_condition_count": len(unique_rows),
        "expanded_cell_count": len(complete),
        "cells": complete,
        "selected_cell_id": selected["cell_id"] if selected else None,
    })
    if selected_vectors is not None:
        np.savez_compressed(
            OUTPUT_ARRAYS,
            terminal=selected_vectors["terminal"],
            active=selected_vectors["active"],
            cumulative=selected_vectors["cumulative"],
            fixed_terminal=selected_vectors["fixed_terminal"],
            fixed_cumulative=selected_vectors["fixed_cumulative"],
        )
    else:
        np.savez_compressed(OUTPUT_ARRAYS, empty=np.asarray([], dtype=np.float64))
    ledger: dict[str, Any] = {
        "ledger_id": "STEP103_ROBUST_DUAL_DEVELOPMENT_V1",
        "date": common.DATE,
        "decision": decision,
        "protocol_sha256": common.sha256_path(PROTOCOL),
        "step102_decision_retained": step102["decision"],
        "step102_ledger_sha256": common.sha256_path(STEP102_LEDGER),
        "step102_grid_sha256": common.sha256_path(STEP102_GRID),
        "step102_arrays_sha256": common.sha256_path(STEP102_ARRAYS),
        "equivalence_classes": equivalence_classes,
        "unique_condition_count": len(unique_rows),
        "expanded_cell_count": len(complete),
        "dual_eligible_cell_count": len(eligible),
        "selected": selected,
        "selected_verification_inference": inference,
        "inference_gates": inference_gates,
        "failed_inference_gates": [name for name, passed in inference_gates.items() if not passed],
        "complete_grid_file": OUTPUT_GRID.name,
        "complete_grid_sha256": common.sha256_path(OUTPUT_GRID),
        "selected_arrays_file": OUTPUT_ARRAYS.name,
        "selected_arrays_sha256": common.sha256_path(OUTPUT_ARRAYS),
        "sealed_outcome_opened": False,
        "code_sha256": common.sha256_path(Path(__file__)),
    }
    common.json_dump(OUTPUT_LEDGER, ledger)
    print(json.dumps({
        "decision": decision,
        "unique_condition_count": len(unique_rows),
        "expanded_cell_count": len(complete),
        "dual_eligible_cell_count": len(eligible),
        "selected_cell": selected["cell_id"] if selected else None,
        "search_terminal_pp": 100 * selected["search"]["mean_terminal_delta"] if selected else None,
        "verify_terminal_pp": 100 * selected["verify"]["mean_terminal_delta"] if selected else None,
        "robust_terminal_pp": 100 * selected["robust_terminal"] if selected else None,
        "verify_terminal_ci_pp": [100 * value for value in inference["terminal"]["bootstrap_95"]] if inference else None,
        "failed_inference_gates": ledger["failed_inference_gates"],
        "sealed_outcome_opened": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
