"""Independent consistency checks for the locked Step 30 Phase B1 audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import run_step29_locked_phase_a as phase_a


ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "STEP30_PHASE_B1_PREREGISTRATION_2026-08-06.json"
SELECTION = ROOT / "STEP30_PHASE_B1_TARGET_SELECTION_LOCK_2026-08-06.json"
RESULT = ROOT / "STEP30_PHASE_B1_ALL_TARGET_RESULTS_2026-08-06.json"

EXPECTED = {
    "preregistration": "75426cf20eda0e2536b85f1d0274652eef94dd8dbea66faa5e483291a167156a",
    "selection": "98af2c94f8cd3f831ff83fd78fbf6c6755f0fda49460d4c16778b57be1974c7d",
    "result": "53f45f99edf2116a2e2f871fd7faac425e8ec0d38aa2944fb6e867c3420760bd",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return bool(abs(float(left) - float(right)) <= tolerance)


def main() -> None:
    observed_hashes = {
        "preregistration": sha256(PREREG),
        "selection": sha256(SELECTION),
        "result": sha256(RESULT),
    }
    if observed_hashes != EXPECTED:
        raise AssertionError({"expected": EXPECTED, "observed": observed_hashes})

    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    if selection["complete"] is not True or result["complete"] is not True:
        raise AssertionError("incomplete artifact")
    if result["selection_lock_sha256"] != EXPECTED["selection"]:
        raise AssertionError("result is not bound to the frozen selection")

    task_order = ("medqa", "gsm8k", "openbookqa")
    expected_counts = {"medqa": 31, "gsm8k": 31, "openbookqa": 30}
    selected_pvalues = {}
    summary = {}

    for task_offset, name in enumerate(task_order):
        selection_task = selection["tasks"][name]
        result_task = result["tasks"][name]
        development_rows = selection_task["all_development_targets"]
        selected_from_rule = max(
            development_rows,
            key=lambda row: (
                row["development"]["mean_delta_cumulative_deployed_regret"],
                -row["target_index"],
            ),
        )
        selected_index = int(selection_task["selected_target_index"])
        if selected_from_rule["target_index"] != selected_index:
            raise AssertionError(f"{name}: development selection rule mismatch")
        if result_task["development_selected_target_index"] != selected_index:
            raise AssertionError(f"{name}: confirmation target substitution")

        rows = result_task["all_final_targets"]
        if len(rows) != expected_counts[name]:
            raise AssertionError(f"{name}: unexpected eligible-parent count")
        if sum(bool(row["selected_on_development"]) for row in rows) != 1:
            raise AssertionError(f"{name}: selected marker count mismatch")

        raw_pvalues = {}
        for row in rows:
            target = int(row["target_index"])
            delta = np.asarray(
                row["paired_delta_cumulative_deployed_regret"], dtype=np.float64
            )
            final_delta = np.asarray(row["paired_delta_final_deployed_regret"], dtype=np.float64)
            if len(delta) != 500 or len(final_delta) != 500:
                raise AssertionError(f"{name}/{target}: paired-array length mismatch")
            if not close(delta.mean(), row["mean_delta_cumulative_deployed_regret"]):
                raise AssertionError(f"{name}/{target}: mean mismatch")
            if not close(np.median(delta), row["median_delta_cumulative_deployed_regret"]):
                raise AssertionError(f"{name}/{target}: median mismatch")
            if not close(final_delta.mean(), row["mean_delta_final_deployed_regret"]):
                raise AssertionError(f"{name}/{target}: final mean mismatch")

            seed = 310000 + task_offset * 1000 + target
            ci = phase_a.bootstrap_ci(delta, seed)
            pvalue = phase_a.sign_flip_pvalue(delta, seed + 1)
            if not np.allclose(
                ci, row["delta_cumulative_deployed_regret_95ci"], atol=0.0, rtol=0.0
            ):
                raise AssertionError(f"{name}/{target}: bootstrap interval mismatch")
            if not close(pvalue, row["delta_cumulative_deployed_regret_signflip_p"]):
                raise AssertionError(f"{name}/{target}: sign-flip p mismatch")
            metadata = row["wrapper_metadata"]
            if metadata["per_example_quality_equal"] is not True:
                raise AssertionError(f"{name}/{target}: quality equality missing")
            if metadata["aliases"] != 4 or not close(metadata["distance_fraction"], 0.05):
                raise AssertionError(f"{name}/{target}: attack configuration drift")
            if len(set(metadata["alias_hashes"])) != 4:
                raise AssertionError(f"{name}/{target}: aliases not distinct")
            if any(abs(float(value)) > 1e-12 for value in metadata["alias_full_score_gaps"]):
                raise AssertionError(f"{name}/{target}: quality gap")
            raw_pvalues[str(target)] = pvalue

        within_adjusted = phase_a.bh_adjust(raw_pvalues)
        for row in rows:
            target = str(row["target_index"])
            if not close(within_adjusted[target], row["within_task_BH_adjusted_p"]):
                raise AssertionError(f"{name}/{target}: within-task BH mismatch")

        selected = result_task["selected_final_confirmation"]
        matching = [row for row in rows if row["target_index"] == selected_index]
        if len(matching) != 1 or matching[0] != selected:
            # The cross-task adjusted p is appended later only to the copied
            # selected object, so compare after removing that one field.
            selected_copy = dict(selected)
            selected_copy.pop("BH_adjusted_p_across_three_selected_targets", None)
            if len(matching) != 1 or matching[0] != selected_copy:
                raise AssertionError(f"{name}: selected result mismatch")
        selected_pvalues[name] = float(
            selected["delta_cumulative_deployed_regret_signflip_p"]
        )
        if result_task["phase_a_overlap_reproduction"]["exact_paired_array_match"] is not True:
            raise AssertionError(f"{name}: Phase A overlap mismatch")

        means = np.asarray(
            [row["mean_delta_cumulative_deployed_regret"] for row in rows]
        )
        selected_rank = 1 + int(
            np.sum(means > selected["mean_delta_cumulative_deployed_regret"])
        )
        summary[name] = {
            "eligible_targets": len(rows),
            "development_selected_target": selected["target"],
            "selected_final_mean": selected["mean_delta_cumulative_deployed_regret"],
            "selected_final_ci": selected["delta_cumulative_deployed_regret_95ci"],
            "selected_final_rank_by_harm": selected_rank,
            "positive_point_targets": int(np.sum(means > 0)),
            "negative_point_targets": int(np.sum(means < 0)),
        }

    adjusted_selected = phase_a.bh_adjust(selected_pvalues)
    expected_task_gate = {}
    for name in task_order:
        selected = result["tasks"][name]["selected_final_confirmation"]
        observed_adjusted = selected["BH_adjusted_p_across_three_selected_targets"]
        if not close(observed_adjusted, adjusted_selected[name]):
            raise AssertionError(f"{name}: selected-target BH mismatch")
        passed = bool(
            selected["mean_delta_cumulative_deployed_regret"] > 0
            and selected["delta_cumulative_deployed_regret_95ci"][0] > 0
            and adjusted_selected[name] < 0.05
        )
        expected_task_gate[name] = passed
        if result["binding_gate"]["task_results"][name]["pass"] != passed:
            raise AssertionError(f"{name}: gate mismatch")

    overall = all(expected_task_gate.values())
    if result["binding_gate"]["pass"] != overall:
        raise AssertionError("overall gate mismatch")
    expected_decision = "PASS_CONTINUE_TO_PHASE_B2" if overall else "FINAL_TOPIC_NO_GO"
    if result["binding_gate"]["decision"] != expected_decision:
        raise AssertionError("decision mismatch")

    print(
        json.dumps(
            {
                "status": "PASS_RESULT_CONSISTENCY",
                "hashes": observed_hashes,
                "binding_decision": expected_decision,
                "tasks": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
