"""Independent validation of the locked Step 56 calibration-to-holdout audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

import run_step54_matrix_free_realism_audit as step54
from step56_reference_free_common import (
    TASKS,
    actual_alias_oracle,
    build_alias_rows,
    digest_json,
    file_sha256,
    fit_error_full_predict,
    option_ranker_oof_and_holdout,
    quality_summary,
    raw_helm_manifest,
    row_hash,
    runtime_text,
    split_indices,
)


ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "STEP56_CALIBRATION_TO_HOLDOUT_PREREGISTRATION_2026-08-09.json"
STAGEA = ROOT / "STEP56_CALIBRATION_STAGEA_EXECUTION_LOCK_2026-08-09.json"
STAGEB = ROOT / "STEP56_HOLDOUT_OUTCOME_EXECUTION_LOCK_2026-08-09.json"
CALIBRATION = ROOT / "STEP56_LABELED_CALIBRATION_PACKAGE_2026-08-09.json"
BLIND = ROOT / "STEP56_BLIND_HOLDOUT_ATTACK_INPUT_2026-08-09.json"
SPLIT = ROOT / "STEP56_SPLIT_AUDIT_2026-08-09.json"
ATTACK = ROOT / "STEP56_FROZEN_BLIND_HOLDOUT_ATTACK_2026-08-09.json"
RESULT = ROOT / "STEP56_CALIBRATION_TO_HOLDOUT_RESULTS_2026-08-09.json"

EXPECTED = {
    "prereg": "909626b48342171b82f91d6f61a51b2a1f2d3db0df1a2a75e4611d12e3842896",
    "stagea": "36703a7bb673c07c5157ff91aba21490b0cb7af75f7c7999dd7d268f58cf2941",
    "stageb": "2ada3f4b4acbfb97f1950db9aedee8475af9985b3d7238c575d99a7e81cf2cfb",
    "calibration": "b6c9b7a764fe73a98def1c6e347f4658d7e4ce9bbb6904de1c40d359a717f3b3",
    "blind": "4aaff2434d0722ec848228d274d4bfe24e7a9360c3803b546529a158538bc9b7",
    "split": "d23de6ac3d8fc7e2696f21cf1d5c3b46add5d4b758858d8bcadc7493e79d37da",
    "attack": "3cf06b698465637cd79b12b329d1d091f2b8e85a6170ac0c8de4039f3dbe9755",
    "result": "318515b15de5841f2da4b340dc5d8823e9191a298733923faa2802ea96ea6001",
}


def assert_close_json(left: Any, right: Any) -> None:
    if isinstance(left, dict):
        assert isinstance(right, dict) and set(left) == set(right)
        for key in left:
            assert_close_json(left[key], right[key])
    elif isinstance(left, list):
        assert isinstance(right, list) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_close_json(a, b)
    elif isinstance(left, float):
        assert np.isclose(left, float(right), rtol=0.0, atol=1e-12), (left, right)
    else:
        assert left == right, (left, right)


def clean_scenario(data: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    phase_a = step54.step46.phase_a
    return phase_a.make_scenario(
        np.asarray(data["responses"], dtype=object)[:, indices],
        np.asarray(data["oracle"], dtype=np.float64)[:, indices],
        list(range(len(phase_a.MODELS))),
        list(phase_a.MODELS),
        "step56_independent_clean",
    )


def validate_file_chain(result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    paths = {
        "prereg": PREREG,
        "stagea": STAGEA,
        "stageb": STAGEB,
        "calibration": CALIBRATION,
        "blind": BLIND,
        "split": SPLIT,
        "attack": ATTACK,
        "result": RESULT,
    }
    for name, path in paths.items():
        assert file_sha256(path) == EXPECTED[name], name
    stagea = json.loads(STAGEA.read_text(encoding="utf-8"))
    stageb = json.loads(STAGEB.read_text(encoding="utf-8"))
    attack = json.loads(ATTACK.read_text(encoding="utf-8"))
    assert stagea["outcomes_seen_before_lock"] is False
    assert stageb["holdout_outcomes_seen_before_lock"] is False
    assert attack["holdout_outcomes_seen"] is False
    assert result["stageb_execution_lock_sha256"] == EXPECTED["stageb"]
    assert result["frozen_attack_sha256"] == EXPECTED["attack"]
    for name, digest in stageb["bound_sha256"].items():
        mapping = {
            "stagea_execution_lock": STAGEA,
            "calibration_package": CALIBRATION,
            "blind_holdout_input": BLIND,
            "split_audit": SPLIT,
            "frozen_attack": ATTACK,
            "common_module": ROOT / "step56_reference_free_common.py",
            "preparation_runner": ROOT / "prepare_step56_blind_split.py",
            "attack_builder": ROOT / "build_step56_calibration_transfer_attack.py",
        }
        assert file_sha256(mapping[name]) == digest
    records, digest, total_bytes = raw_helm_manifest(ROOT)
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    assert len(records) == 96
    assert digest == prereg["locked_sources"]["raw_helm_file_manifest_sha256"]
    assert total_bytes == prereg["locked_sources"]["raw_helm_total_bytes"]
    return prereg, stageb, attack


def reconstruct_holdout_aliases(
    task: str,
    task_index: int,
    cal: dict[str, Any],
    blind: dict[str, Any],
    attack_task: dict[str, Any],
) -> list[np.ndarray]:
    calibration_texts = [
        runtime_text(prompt, options, output)
        for prompt, options, output in zip(
            cal["prompts"], cal["public_options"], cal["parent_responses"]
        )
    ]
    holdout_texts = [
        runtime_text(prompt, options, output)
        for prompt, options, output in zip(
            blind["prompts"], blind["public_options"], blind["parent_responses"]
        )
    ]
    error_labels = 1 - np.asarray(cal["parent_oracle"], dtype=np.int64)
    scores = fit_error_full_predict(
        attack_task["selected_error_model"],
        calibration_texts,
        error_labels,
        holdout_texts,
    )
    assert np.allclose(
        scores,
        np.asarray(attack_task["holdout_error_scores"], dtype=np.float64),
        rtol=0.0,
        atol=1e-10,
    )
    if task in {"medqa", "openbookqa"}:
        folds = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=56000 + task_index
        )
        fold_ids = np.empty(len(error_labels), dtype=np.int64)
        for fold, (_, valid) in enumerate(folds.split(np.zeros(len(error_labels)), error_labels)):
            fold_ids[valid] = fold
        _, holdout_option_scores = option_ranker_oof_and_holdout(
            cal["prompts"],
            cal["public_options"],
            cal["correct"],
            fold_ids,
            blind["prompts"],
            blind["public_options"],
        )
    else:
        holdout_option_scores = None
    rows, support = build_alias_rows(
        task,
        blind["ids"],
        blind["parent_responses"],
        blind["public_options"],
        scores,
        holdout_option_scores,
        float(attack_task["selected_policy"]["distance_fraction"]),
        str(attack_task["selected_policy"]["choice_mode"]),
    )
    assert support == attack_task["holdout_support_indices"]
    assert [row.tolist() for row in rows] == attack_task["holdout_alias_responses"]
    assert [row_hash(row) for row in rows] == attack_task[
        "holdout_alias_response_hashes"
    ]
    return rows


def recompute_task(
    task: str,
    task_index: int,
    data: dict[str, Any],
    cal: dict[str, Any],
    blind: dict[str, Any],
    attack_task: dict[str, Any],
    spec: dict[str, Any],
) -> dict[str, Any]:
    calibration_index, holdout_index = split_indices(task, list(data["ids"]))
    assert [str(data["ids"][index]) for index in calibration_index] == [
        str(value) for value in cal["ids"]
    ]
    assert [str(data["ids"][index]) for index in holdout_index] == [
        str(value) for value in blind["ids"]
    ]
    assert not set(map(str, cal["ids"])) & set(map(str, blind["ids"]))
    assert set(blind) == {
        "target_index",
        "target",
        "ids",
        "prompts",
        "public_options",
        "parent_responses",
        "source_indices_digest",
    }
    rows = reconstruct_holdout_aliases(task, task_index, cal, blind, attack_task)
    target = int(spec["target_index"])
    clean = clean_scenario(data, holdout_index)
    parent = np.asarray(clean["responses"][target], dtype=object)
    parent_oracle = np.asarray(clean["oracle"][target], dtype=np.float64)
    correct = [str(data["correct"][index]) for index in holdout_index]
    alias_oracles = actual_alias_oracle(rows, correct)
    quality = quality_summary(parent_oracle, alias_oracles)
    support = [int(value) for value in attack_task["holdout_support_indices"]]
    support_precision = float(np.mean(1.0 - parent_oracle[support]))
    distance = max(
        1,
        int(
            round(
                float(attack_task["selected_policy"]["distance_fraction"])
                * len(holdout_index)
            )
        ),
    )
    alias_distances = [int(np.sum(row != parent)) for row in rows]
    hashes_distinct = len(set(row_hash(row) for row in rows)) == 4 and row_hash(parent) not in {
        row_hash(row) for row in rows
    }
    refined = step54.assemble_refined_scenario(
        clean, target, rows, alias_oracles, "step56_independent_refined"
    )
    phase_a = step54.step46.phase_a
    original_oracle = np.asarray(clean["oracle"], dtype=np.float64)
    pools = phase_a.pools_from_seeds(
        list(range(92000, 92500)), len(holdout_index), int(spec["holdout_pool_size"])
    )
    clean_runs = phase_a.run_scenario(
        clean, original_oracle, pools, int(spec["budget"]), float(spec["temperature"])
    )
    refined_runs = phase_a.run_scenario(
        refined, original_oracle, pools, int(spec["budget"]), float(spec["temperature"])
    )
    comparison = phase_a.compare_runs(
        clean_runs, refined_runs, 956000 + task_index * 1000, include_arrays=True
    )
    fixed_queries = step54.fixed_random_queries(pools, int(spec["budget"]), 56 + task_index)
    fixed_clean = step54.step50.run_fixed_queries(clean, original_oracle, pools, fixed_queries)
    fixed_refined = step54.step50.run_fixed_queries(
        refined, original_oracle, pools, fixed_queries
    )
    fixed_exact = bool(np.array_equal(fixed_clean["queried"], fixed_refined["queried"]))
    fixed_comparison = phase_a.compare_runs(
        fixed_clean,
        fixed_refined,
        956000 + task_index * 1000 + 100,
        include_arrays=False,
    )
    quality_pass = quality["max_abs_accuracy_gap"] <= 0.01 + 1e-12 and quality[
        "max_oracle_disagreement_fraction"
    ] <= 0.02 + 1e-12
    effect_pass = comparison["delta_cumulative_deployed_regret_95ci"][0] > 0.0
    return {
        "quality": quality,
        "support_precision": support_precision,
        "alias_distances": alias_distances,
        "hashes_distinct": hashes_distinct,
        "comparison": comparison,
        "fixed_exact": fixed_exact,
        "fixed_comparison": fixed_comparison,
        "quality_pass": bool(quality_pass),
        "effect_pass": bool(effect_pass),
    }


def main() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    prereg, _, attack = validate_file_chain(result)
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    blind_payload = json.loads(BLIND.read_text(encoding="utf-8"))
    assert blind_payload["explicitly_absent"] == [
        "correct",
        "oracle",
        "peer_responses",
        "registry_responses",
        "pool_seed",
        "selector_state",
    ]
    step54.step46.configure_legacy_paths()
    all_data = step54.step46.phase_a.load_all_data()
    joint_count = 0
    for task_index, task in enumerate(TASKS):
        recomputed = recompute_task(
            task,
            task_index,
            all_data[task],
            calibration["tasks"][task],
            blind_payload["tasks"][task],
            attack["tasks"][task],
            prereg["frozen_tasks"][task],
        )
        recorded = result["tasks"][task]
        assert_close_json(recomputed["quality"], recorded["quality"])
        assert np.isclose(
            recomputed["support_precision"],
            recorded["support_parent_error_precision"],
        )
        assert recomputed["alias_distances"] == recorded["alias_distances"]
        assert recomputed["hashes_distinct"] == recorded[
            "alias_hashes_distinct_nonparent"
        ]
        assert_close_json(recomputed["comparison"], recorded["comparison"])
        assert recomputed["fixed_exact"] == recorded["fixed_query_control"][
            "query_matrices_byte_identical"
        ]
        assert_close_json(
            recomputed["fixed_comparison"],
            recorded["fixed_query_control"]["comparison"],
        )
        assert recomputed["quality_pass"] == recorded["quality_gate_pass"]
        assert recomputed["effect_pass"] == recorded["effect_gate_pass"]
        joint_count += int(recomputed["quality_pass"] and recomputed["effect_pass"])
    assert joint_count == 2
    assert result["gates"]["G0_blind_construction_integrity"]["pass"] is True
    assert result["gates"]["G1_calibration_feasibility"]["pass"] is True
    assert result["gates"]["G2_holdout_quality"]["pass_count"] == 3
    assert result["gates"]["G3_holdout_effect"]["joint_quality_effect_count"] == 2
    assert result["gates"]["G4_controls"]["pass"] is True
    assert result["gates"]["decision"] == "PROMOTE_LABEL_FREE_T1"
    print("PASS_STEP56_CALIBRATION_TO_HOLDOUT_AUDIT")


if __name__ == "__main__":
    main()
