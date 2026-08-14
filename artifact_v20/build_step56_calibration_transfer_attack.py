"""Stage A: fit on calibration and freeze blind holdout aliases for Step 56."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import run_step54_matrix_free_realism_audit as step54
from step56_reference_free_common import (
    CHOICE_MODES,
    DISTANCES,
    TASKS,
    actual_alias_oracle,
    build_alias_rows,
    digest_json,
    error_model_oof,
    file_sha256,
    fit_error_full_predict,
    option_ranker_oof_and_holdout,
    quality_summary,
    row_hash,
    runtime_text,
)


ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "STEP56_CALIBRATION_TO_HOLDOUT_PREREGISTRATION_2026-08-09.json"
STAGEA_LOCK = ROOT / "STEP56_CALIBRATION_STAGEA_EXECUTION_LOCK_2026-08-09.json"
CALIBRATION = ROOT / "STEP56_LABELED_CALIBRATION_PACKAGE_2026-08-09.json"
BLIND = ROOT / "STEP56_BLIND_HOLDOUT_ATTACK_INPUT_2026-08-09.json"
SPLIT_AUDIT = ROOT / "STEP56_SPLIT_AUDIT_2026-08-09.json"
OUT = ROOT / "STEP56_FROZEN_BLIND_HOLDOUT_ATTACK_2026-08-09.json"
EXPECTED_PREREG_SHA256 = "909626b48342171b82f91d6f61a51b2a1f2d3db0df1a2a75e4611d12e3842896"


def verify_stage_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    assert file_sha256(PREREG) == EXPECTED_PREREG_SHA256
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    lock = json.loads(STAGEA_LOCK.read_text(encoding="utf-8"))
    assert lock["manifest_id"] == "STEP56_CALIBRATION_STAGEA_EXECUTION_LOCK_V1"
    assert lock["outcomes_seen_before_lock"] is False
    assert lock["preregistration_sha256"] == EXPECTED_PREREG_SHA256
    paths = {
        "common_module": ROOT / "step56_reference_free_common.py",
        "preparation_runner": ROOT / "prepare_step56_blind_split.py",
        "attack_builder": Path(__file__).resolve(),
    }
    for name, path in paths.items():
        assert lock[f"{name}_sha256"] == file_sha256(path)
    return prereg, lock


def clean_scenario(calibration_task: dict[str, Any]) -> dict[str, Any]:
    phase_a = step54.step46.phase_a
    responses = np.asarray(calibration_task["registry_responses"], dtype=object)
    oracle = np.asarray(calibration_task["registry_oracle"], dtype=np.float64)
    return phase_a.make_scenario(
        responses,
        oracle,
        list(range(len(phase_a.MODELS))),
        list(phase_a.MODELS),
        "step56_calibration_clean",
    )


def evaluate_calibration_policy(
    *,
    task: str,
    task_index: int,
    calibration_task: dict[str, Any],
    clean: dict[str, Any],
    clean_runs: dict[str, np.ndarray],
    pools: np.ndarray,
    spec: dict[str, Any],
    error_scores: np.ndarray,
    option_scores: list[list[float]] | None,
    distance: float,
    choice_mode: str,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    target = int(spec["target_index"])
    rows, support = build_alias_rows(
        task,
        calibration_task["ids"],
        calibration_task["parent_responses"],
        calibration_task["public_options"],
        error_scores,
        option_scores,
        distance,
        choice_mode,
    )
    alias_oracles = actual_alias_oracle(rows, calibration_task["correct"])
    quality = quality_summary(
        np.asarray(calibration_task["parent_oracle"], dtype=np.float64), alias_oracles
    )
    refined = step54.assemble_refined_scenario(
        clean, target, rows, alias_oracles, f"step56_calibration_{distance}_{choice_mode}"
    )
    phase_a = step54.step46.phase_a
    refined_runs = phase_a.run_scenario(
        refined,
        np.asarray(calibration_task["registry_oracle"], dtype=np.float64),
        pools,
        int(spec["budget"]),
        float(spec["temperature"]),
    )
    comparison = phase_a.compare_runs(
        clean_runs,
        refined_runs,
        955000 + task_index * 1000 + int(round(distance * 1000)) * 10
        + (0 if choice_mode == CHOICE_MODES[0] else 1),
        include_arrays=False,
    )
    feasible = bool(
        quality["max_abs_accuracy_gap"] <= 0.01 + 1e-12
        and quality["max_oracle_disagreement_fraction"] <= 0.02 + 1e-12
    )
    record = {
        "distance_fraction": float(distance),
        "distance_queries": len(support),
        "choice_mode": choice_mode,
        "support_digest": digest_json(support),
        "support_error_precision": float(
            np.mean(
                1.0
                - np.asarray(calibration_task["parent_oracle"], dtype=np.float64)[support]
            )
        ),
        "alias_response_hashes": [row_hash(row) for row in rows],
        "quality": quality,
        "feasible": feasible,
        "comparison": comparison,
    }
    return record, rows


def select_policy(records: list[dict[str, Any]]) -> tuple[dict[str, Any], bool]:
    feasible = [record for record in records if record["feasible"]]
    if feasible:
        selected = max(
            feasible,
            key=lambda record: (
                record["comparison"]["mean_delta_cumulative_deployed_regret"],
                -record["quality"]["max_oracle_disagreement_fraction"],
                -record["distance_fraction"],
                int(record["choice_mode"] == CHOICE_MODES[0]),
                record["choice_mode"],
            ),
        )
        return selected, True
    selected = min(
        records,
        key=lambda record: (
            record["quality"]["max_oracle_disagreement_fraction"],
            -record["comparison"]["mean_delta_cumulative_deployed_regret"],
            record["distance_fraction"],
            record["choice_mode"],
        ),
    )
    return selected, False


def main() -> None:
    started = time.time()
    prereg, lock = verify_stage_lock()
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    blind = json.loads(BLIND.read_text(encoding="utf-8"))
    split = json.loads(SPLIT_AUDIT.read_text(encoding="utf-8"))
    assert calibration["manifest_id"] == "STEP56_LABELED_CALIBRATION_PACKAGE_V1"
    assert blind["manifest_id"] == "STEP56_BLIND_HOLDOUT_ATTACK_INPUT_V1"
    assert split["manifest_id"] == "STEP56_SPLIT_AUDIT_V1"
    serialized_blind = BLIND.read_text(encoding="utf-8").casefold()
    for forbidden_key in (
        '"correct":',
        '"oracle":',
        '"peer_responses":',
        '"registry_responses":',
        '"pool_seed":',
        '"selector_state":',
    ):
        assert forbidden_key not in serialized_blind

    step54.step46.configure_legacy_paths()
    phase_a = step54.step46.phase_a
    tasks_out: dict[str, Any] = {}
    for task_index, task in enumerate(TASKS):
        print(f"[{task}] cross-fitting error learner", flush=True)
        cal = calibration["tasks"][task]
        held = blind["tasks"][task]
        spec = prereg["frozen_tasks"][task]
        calibration_texts = [
            runtime_text(prompt, options, output)
            for prompt, options, output in zip(
                cal["prompts"], cal["public_options"], cal["parent_responses"]
            )
        ]
        holdout_texts = [
            runtime_text(prompt, options, output)
            for prompt, options, output in zip(
                held["prompts"], held["public_options"], held["parent_responses"]
            )
        ]
        error_labels = 1 - np.asarray(cal["parent_oracle"], dtype=np.int64)
        selected_error_model, oof_by_model, error_diagnostics, fold_ids = error_model_oof(
            calibration_texts, error_labels, task_index
        )
        print(f"[{task}] selected error model {selected_error_model}", flush=True)
        if task in {"medqa", "openbookqa"}:
            cal_option_scores, holdout_option_scores = option_ranker_oof_and_holdout(
                cal["prompts"],
                cal["public_options"],
                cal["correct"],
                fold_ids,
                held["prompts"],
                held["public_options"],
            )
            modes = CHOICE_MODES
        else:
            cal_option_scores = None
            holdout_option_scores = None
            modes = ("calibrated_lowest_probability",)

        clean = clean_scenario(cal)
        pools = phase_a.pools_from_seeds(
            list(range(91500, 91700)),
            len(cal["ids"]),
            int(spec["calibration_pool_size"]),
        )
        clean_runs = phase_a.run_scenario(
            clean,
            np.asarray(cal["registry_oracle"], dtype=np.float64),
            pools,
            int(spec["budget"]),
            float(spec["temperature"]),
        )
        calibration_records: list[dict[str, Any]] = []
        for distance in DISTANCES:
            for mode in modes:
                record, _ = evaluate_calibration_policy(
                    task=task,
                    task_index=task_index,
                    calibration_task=cal,
                    clean=clean,
                    clean_runs=clean_runs,
                    pools=pools,
                    spec=spec,
                    error_scores=oof_by_model[selected_error_model],
                    option_scores=cal_option_scores,
                    distance=distance,
                    choice_mode=mode,
                )
                calibration_records.append(record)
                print(
                    f"[{task}] d={distance:.2f} {mode} "
                    f"feasible={record['feasible']} "
                    f"dCReg={record['comparison']['mean_delta_cumulative_deployed_regret']:.6f}",
                    flush=True,
                )
        selected_policy, calibration_feasible = select_policy(calibration_records)
        holdout_error_scores = fit_error_full_predict(
            selected_error_model, calibration_texts, error_labels, holdout_texts
        )
        holdout_rows, holdout_support = build_alias_rows(
            task,
            held["ids"],
            held["parent_responses"],
            held["public_options"],
            holdout_error_scores,
            holdout_option_scores,
            float(selected_policy["distance_fraction"]),
            str(selected_policy["choice_mode"]),
        )
        tasks_out[task] = {
            "target_index": int(spec["target_index"]),
            "target": spec["target"],
            "calibration_count": len(cal["ids"]),
            "holdout_count": len(held["ids"]),
            "calibration_id_digest": digest_json([str(value) for value in cal["ids"]]),
            "holdout_id_digest": digest_json([str(value) for value in held["ids"]]),
            "selected_error_model": selected_error_model,
            "error_model_oof_diagnostics": error_diagnostics,
            "calibration_parent_error_prevalence": float(np.mean(error_labels)),
            "calibration_policy_records": calibration_records,
            "selected_policy": selected_policy,
            "calibration_feasible": calibration_feasible,
            "holdout_runtime_fields": ["prompt", "public_options", "parent_output"],
            "holdout_peer_response_access": False,
            "holdout_reference_or_oracle_access": False,
            "holdout_pool_or_selector_access": False,
            "holdout_error_scores": [float(value) for value in holdout_error_scores],
            "holdout_error_score_digest": digest_json(
                [float(value) for value in holdout_error_scores]
            ),
            "holdout_support_indices": holdout_support,
            "holdout_support_id_digest": digest_json(
                [str(held["ids"][index]) for index in holdout_support]
            ),
            "holdout_parent_response_hash": row_hash(
                np.asarray(held["parent_responses"], dtype=object)
            ),
            "holdout_alias_responses": [row.tolist() for row in holdout_rows],
            "holdout_alias_response_hashes": [row_hash(row) for row in holdout_rows],
        }
        print(
            f"[{task}] FROZEN policy={selected_policy['distance_fraction']:.2f}/"
            f"{selected_policy['choice_mode']} feasible={calibration_feasible}",
            flush=True,
        )

    payload = {
        "manifest_id": "STEP56_FROZEN_BLIND_HOLDOUT_ATTACK_V1",
        "status": "ATTACK_FROZEN_BEFORE_HOLDOUT_OUTCOME",
        "holdout_outcomes_seen": False,
        "preregistration_sha256": EXPECTED_PREREG_SHA256,
        "stagea_execution_lock_sha256": file_sha256(STAGEA_LOCK),
        "calibration_package_sha256": file_sha256(CALIBRATION),
        "blind_holdout_input_sha256": file_sha256(BLIND),
        "split_audit_sha256": file_sha256(SPLIT_AUDIT),
        "attack_builder_sha256": file_sha256(Path(__file__).resolve()),
        "stagea_code_sha256": {
            "common_module": lock["common_module_sha256"],
            "preparation_runner": lock["preparation_runner_sha256"],
            "attack_builder": lock["attack_builder_sha256"],
        },
        "forbidden_holdout_inputs_absent": True,
        "tasks": tasks_out,
        "elapsed_seconds": time.time() - started,
    }
    OUT.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"WROTE {OUT}", flush=True)
    print(f"SHA256 {file_sha256(OUT)}", flush=True)


if __name__ == "__main__":
    main()
