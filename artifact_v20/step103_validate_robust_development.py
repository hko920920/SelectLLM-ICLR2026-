from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import step102_common as common
from step100_stagea_develop import run_condition


ROOT = Path(__file__).resolve().parent
STEP102_LEDGER = ROOT / f"STEP102_STAGEA_COMPLETE_LEDGER_{common.DATE}.json"
STEP102_ARRAYS = ROOT / f"STEP102_STAGEA_DEVELOPMENT_ARRAYS_{common.DATE}.npz"
STEP103_GRID = ROOT / f"STEP103_COMPLETE_DUAL_DEVELOPMENT_GRID_{common.DATE}.json"
STEP103_ARRAYS = ROOT / f"STEP103_SELECTED_DEVELOPMENT_ARRAYS_{common.DATE}.npz"
STEP103_LEDGER = ROOT / f"STEP103_ROBUST_DUAL_DEVELOPMENT_LEDGER_{common.DATE}.json"
OUTPUT = ROOT / f"STEP103_INDEPENDENT_DEVELOPMENT_VALIDATION_{common.DATE}.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    step102 = json.loads(STEP102_LEDGER.read_text(encoding="utf-8"))
    grid = json.loads(STEP103_GRID.read_text(encoding="utf-8"))
    ledger = json.loads(STEP103_LEDGER.read_text(encoding="utf-8"))
    arrays = np.load(STEP102_ARRAYS, allow_pickle=False)
    selected_saved = np.load(STEP103_ARRAYS, allow_pickle=False)
    checks: dict[str, bool] = {}
    checks["step102_no_go_retained"] = (
        step102["decision"] == "NO_GO_STEP102_LEARNED_DEVELOPMENT_STOP"
        and ledger["step102_decision_retained"] == step102["decision"]
    )
    checks["authority_hashes_match"] = (
        ledger["step102_ledger_sha256"] == common.sha256_path(STEP102_LEDGER)
        and ledger["step102_arrays_sha256"] == common.sha256_path(STEP102_ARRAYS)
        and ledger["complete_grid_sha256"] == common.sha256_path(STEP103_GRID)
        and ledger["selected_arrays_sha256"] == common.sha256_path(STEP103_ARRAYS)
    )
    checks["complete_grid_cardinality"] = (
        grid["unique_condition_count"] == 27
        and grid["expanded_cell_count"] == 81
        and len(grid["cells"]) == 81
    )
    eligible = [row for row in grid["cells"] if row["dual_eligible"]]
    selected = sorted(
        eligible,
        key=lambda row: (
            -row["robust_terminal"], -row["robust_cumulative"], -row["robust_path"],
            row["loss_cap"], row["trigger_cap"], row["budget"], row["tau"],
            row["cell_id"],
        ),
    )[0]
    checks["selection_rule_reconstructed"] = (
        len(eligible) == ledger["dual_eligible_cell_count"] == 6
        and selected == ledger["selected"]
        and selected["cell_id"] == "l0.0075_t0.025_b5_tau0.050"
    )
    prediction = arrays["root_predictions"].astype(np.int64)
    labels = arrays["labels"].astype(np.int64)
    scores = arrays["error_scores"].astype(np.float64)
    indices = arrays["selector_verify_indices"].astype(np.int64)
    thresholds = step102["threshold_pairs"]["l0.0075_t0.025"]["thresholds"]
    aliases, _ = common.make_alias_codes(prediction[:, 0], scores, thresholds)
    effect, vectors = run_condition(
        prediction, labels, indices, aliases, common.VERIFY_SEEDS,
        budget=5, tau=0.05,
    )
    checks["selected_effect_reconstructed"] = effect == selected["verify"]
    vector_names = ("terminal", "active", "cumulative", "fixed_terminal", "fixed_cumulative")
    checks["selected_raw_vectors_bitwise"] = all(
        np.array_equal(vectors[name], selected_saved[name]) for name in vector_names
    )
    inference = {
        "terminal": common.effect_summary(vectors["terminal"], 103700, 103701),
        "active_minus_fixed_terminal": common.effect_summary(vectors["active"], 103701, 103702),
        "cumulative": common.effect_summary(vectors["cumulative"], 103702, 103703),
    }
    checks["selected_inference_reconstructed"] = inference == ledger["selected_verification_inference"]
    checks["all_development_go_gates"] = (
        ledger["decision"] == "GO_STEP103_TO_HELDOUT_LOCK"
        and all(ledger["inference_gates"].values())
        and effect["mean_terminal_delta"] >= 0.005
        and effect["mean_active_minus_fixed_terminal_delta"] >= 0.005
        and effect["max_abs_fixed_terminal_delta"] == 0
        and effect["max_abs_fixed_cumulative_delta"] == 0
        and effect["fixed_root_history_exact"]
    )
    checks = {name: bool(value) for name, value in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]
    output: dict[str, Any] = {
        "validation_id": "STEP103_INDEPENDENT_DEVELOPMENT_VALIDATION_V1",
        "date": common.DATE,
        "decision": "PASS_STEP103_INDEPENDENT_DEVELOPMENT_VALIDATION" if not failed else "FAIL_STEP103_INDEPENDENT_DEVELOPMENT_VALIDATION",
        "checks": checks,
        "failed_checks": failed,
        "selected_cell": selected["cell_id"],
        "recomputed": {
            "dual_eligible_cell_count": len(eligible),
            "search_terminal_pp": 100 * selected["search"]["mean_terminal_delta"],
            "verify_terminal_pp": 100 * effect["mean_terminal_delta"],
            "verify_terminal_ci_pp": [100 * value for value in inference["terminal"]["bootstrap_95"]],
        },
        "authority_sha256": {
            STEP102_LEDGER.name: common.sha256_path(STEP102_LEDGER),
            STEP102_ARRAYS.name: common.sha256_path(STEP102_ARRAYS),
            STEP103_GRID.name: common.sha256_path(STEP103_GRID),
            STEP103_ARRAYS.name: common.sha256_path(STEP103_ARRAYS),
            STEP103_LEDGER.name: common.sha256_path(STEP103_LEDGER),
            Path(__file__).name: common.sha256_path(Path(__file__)),
        },
        "sealed_outcome_opened": False,
    }
    common.json_dump(OUTPUT, output)
    print(json.dumps(output, indent=2), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
