"""Stage A: calibration-select and freeze Step 87B attacks without holdout outcomes."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

import run_step80_main_t2a_v2_transfer as v2
from step87b_openended_common import (
    ALIASES,
    DISTANCES,
    POLICIES,
    PREREG,
    ROOT,
    TASKS,
    alias_utilities,
    build_alias_rows,
    digest_json,
    error_scores_holdout,
    error_scores_oof,
    file_sha256,
    load_preregistration,
    opaque_token,
    pools_from_seeds,
    quality_summary,
    runtime_text,
)


RUNNER = Path(__file__).resolve()
LOCK = ROOT / "STEP87B_STAGEA_EXECUTION_LOCK_2026-08-12.json"
CALIBRATION = ROOT / "STEP87B_LABELED_CALIBRATION_PACKAGE_2026-08-12.json"
BLIND_MANIFEST = ROOT / "STEP87B_BLIND_PARENT_INPUT_MANIFEST_2026-08-12.json"
SPLIT_AUDIT = ROOT / "STEP87B_SPLIT_AND_SCHEMA_AUDIT_2026-08-12.json"
RAW_MANIFEST = ROOT / "STEP87B_RAW_SOURCE_MANIFEST_2026-08-12.json"
OUT = ROOT / "STEP87B_FROZEN_REFERENCE_FREE_ATTACK_2026-08-12.json"
DEPENDENCIES = {
    "step87b_openended_common.py": ROOT / "step87b_openended_common.py",
    "run_step80_main_t2a_v2_transfer.py": ROOT / "run_step80_main_t2a_v2_transfer.py",
    "run_step54_matrix_free_realism_audit.py": ROOT / "run_step54_matrix_free_realism_audit.py",
    "run_step46_clone_robust_soft_weighting.py": ROOT / "run_step46_clone_robust_soft_weighting.py",
    "run_step50_natural_alias_audit.py": ROOT / "run_step50_natural_alias_audit.py",
    "run_step29_locked_phase_a.py": ROOT / "run_step29_locked_phase_a.py",
}
BUDGETS = (10, 20, 40)
TAUS = (1.0, 3.0, 5.0)
CALIBRATION_SEEDS = tuple(range(88200, 88400))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    prereg = load_preregistration()
    lock = load_json(LOCK)
    if lock.get("manifest_id") != "STEP87B_STAGEA_EXECUTION_LOCK_V1":
        raise AssertionError("unexpected Step 87B Stage-A lock")
    observed_inputs = {
        "preregistration": file_sha256(PREREG),
        "calibration": file_sha256(CALIBRATION),
        "blind_manifest": file_sha256(BLIND_MANIFEST),
        "split_audit": file_sha256(SPLIT_AUDIT),
        "raw_manifest": file_sha256(RAW_MANIFEST),
    }
    observed_code = {
        "stage_a_runner": file_sha256(RUNNER),
        **{name: file_sha256(path) for name, path in DEPENDENCIES.items()},
    }
    if lock.get("input_sha256") != observed_inputs:
        raise AssertionError({"locked_inputs": lock.get("input_sha256"), "observed": observed_inputs})
    if lock.get("code_sha256") != observed_code:
        raise AssertionError({"locked_code": lock.get("code_sha256"), "observed": observed_code})
    if lock.get("sealed_holdout_semantically_opened_before_stage_a") is not False:
        raise AssertionError("Stage-A separation certificate failed")
    return prereg, lock


def make_clean(data: dict[str, Any], models: list[str], task: str) -> dict[str, Any]:
    phase_a = v2.step54.step46.phase_a
    return phase_a.make_scenario(
        np.asarray(data["registry_responses"], dtype=object),
        np.asarray(data["registry_utilities"], dtype=np.float64),
        list(range(len(models))),
        models,
        f"step87b_clean_{task}",
    )


def make_refined(
    clean: dict[str, Any],
    target: int,
    rows: list[list[str]],
    utilities: np.ndarray,
    models: list[str],
    task: str,
    policy: str,
    distance: float,
) -> dict[str, Any]:
    phase_a = v2.step54.step46.phase_a
    return phase_a.make_scenario(
        np.concatenate([np.asarray(clean["responses"], dtype=object), np.asarray(rows, dtype=object)]),
        np.concatenate([np.asarray(clean["oracle"], dtype=np.float64), utilities]),
        list(np.asarray(clean["parents"], dtype=np.int64)) + [target] * ALIASES,
        list(clean["labels"])
        + [f"STEP87B:{task}:{models[target]}:{policy}:{distance}:{alias}" for alias in range(ALIASES)],
        f"step87b_refined_{task}_{target}_{policy}_{distance}",
    )


def fixed_queries(pools: np.ndarray, budget: int, task_index: int) -> np.ndarray:
    rows = []
    for run_index, pool in enumerate(pools):
        rng = random.Random(487000 + task_index * 10000 + run_index)
        rows.append(rng.sample([int(value) for value in pool], budget))
    return np.asarray(rows, dtype=np.int64)


def arrays(run: dict[str, Any]) -> dict[str, np.ndarray]:
    return v2.method_arrays(v2.primary_view(run))


def run_active(
    scenario: dict[str, Any],
    original: np.ndarray,
    pools: np.ndarray,
    budget: int,
    tau: float,
    task: str,
) -> dict[str, np.ndarray]:
    run = v2.active_runs(
        scenario,
        original,
        pools,
        CALIBRATION_SEEDS,
        budget,
        tau,
        task,
        v2.PRIMARY_QUERY_POLICY,
    )
    return arrays(run)


def run_fixed(
    scenario: dict[str, Any],
    original: np.ndarray,
    pools: np.ndarray,
    queries: np.ndarray,
    task: str,
) -> dict[str, np.ndarray]:
    return arrays(
        v2.fixed_runs(
            scenario,
            original,
            pools,
            CALIBRATION_SEEDS,
            queries,
            task,
        )
    )


def screen_targets(
    clean: dict[str, Any], pools: np.ndarray, models: list[str]
) -> dict[str, Any]:
    phase_a = v2.step54.step46.phase_a
    clean_score = phase_a.initial_acquisition(clean["codes"])
    clean_first = np.asarray(
        [int(pool[int(np.argmin(clean_score[pool]))]) for pool in pools], dtype=np.int64
    )
    rows = []
    for target, model in enumerate(models):
        cloned = phase_a.clone_scenario(clean, target, 1)
        score = phase_a.initial_acquisition(cloned["codes"])
        first = np.asarray(
            [int(pool[int(np.argmin(score[pool]))]) for pool in pools], dtype=np.int64
        )
        rows.append(
            {
                "target_index": target,
                "target": model,
                "first_query_change_fraction": float(np.mean(first != clean_first)),
            }
        )
    retained = sorted(
        rows, key=lambda row: (-row["first_query_change_fraction"], row["target"])
    )[:5]
    return {"retained": retained, "all_targets": rows}


def calibration_record(
    *,
    task: str,
    target: int,
    policy: str,
    distance: float,
    budget: int,
    tau: float,
    clean_active: dict[str, np.ndarray],
    refined_active: dict[str, np.ndarray],
    clean_fixed: dict[str, np.ndarray],
    refined_fixed: dict[str, np.ndarray],
    quality: dict[str, Any],
) -> dict[str, Any]:
    active_terminal = refined_active["terminal_root_regret"] - clean_active["terminal_root_regret"]
    fixed_terminal = refined_fixed["terminal_root_regret"] - clean_fixed["terminal_root_regret"]
    active_cumulative = refined_active["cumulative_root_regret"] - clean_active["cumulative_root_regret"]
    fixed_cumulative = refined_fixed["cumulative_root_regret"] - clean_fixed["cumulative_root_regret"]
    return {
        "task": task,
        "target_index": target,
        "policy": policy,
        "distance": distance,
        "budget": budget,
        "tau": tau,
        "quality": quality,
        "mean_active_terminal_root_delta": float(active_terminal.mean()),
        "mean_fixed_terminal_root_delta": float(fixed_terminal.mean()),
        "mean_terminal_root_did": float((active_terminal - fixed_terminal).mean()),
        "mean_active_cumulative_root_delta": float(active_cumulative.mean()),
        "mean_fixed_cumulative_root_delta": float(fixed_cumulative.mean()),
        "mean_cumulative_root_did": float((active_cumulative - fixed_cumulative).mean()),
        "active_path_change_fraction": float(
            np.mean(np.any(clean_active["queried"] != refined_active["queried"], axis=1))
        ),
        "fixed_query_byte_equal": bool(np.array_equal(clean_fixed["queried"], refined_fixed["queried"])),
    }


def main() -> None:
    prereg, lock = verify_lock()
    calibration = load_json(CALIBRATION)
    blind_manifest = load_json(BLIND_MANIFEST)
    models = list(calibration["models"])
    phase_a_models = list(v2.step54.step46.phase_a.MODELS)
    if models != phase_a_models:
        raise AssertionError("frozen model roster does not match the manuscript backend")

    result: dict[str, Any] = {
        "manifest_id": "STEP87B_FROZEN_REFERENCE_FREE_ATTACK_V1",
        "status": "ATTACK_FROZEN_BEFORE_SEALED_HOLDOUT_OUTCOMES",
        "preregistration_sha256": file_sha256(PREREG),
        "stage_a_lock_sha256": file_sha256(LOCK),
        "stage_a_runner_sha256": file_sha256(RUNNER),
        "sealed_holdout_outcomes_read": False,
        "holdout_peer_responses_read": False,
        "holdout_references_or_utilities_read": False,
        "models": models,
        "tasks": {},
    }

    for task_index, task in enumerate(TASKS):
        print(f"[{task}] calibration selection", flush=True)
        data = calibration["tasks"][task]
        responses = np.asarray(data["registry_responses"], dtype=object)
        utilities = np.asarray(data["registry_utilities"], dtype=np.float64)
        clean = make_clean(data, models, task)
        examples = responses.shape[1]
        pool_size = min(250, max(max(BUDGETS) + 10, int(np.floor(0.8 * examples))))
        pools = pools_from_seeds(list(CALIBRATION_SEEDS), examples, pool_size)
        screen = screen_targets(clean, pools, models)

        temperature_rows: dict[str, Any] = {}
        clean_active_by_budget: dict[int, dict[str, np.ndarray]] = {}
        clean_fixed_by_budget: dict[int, dict[str, np.ndarray]] = {}
        fixed_by_budget: dict[int, np.ndarray] = {}
        selected_tau: dict[int, float] = {}
        for budget in BUDGETS:
            candidates = []
            for tau in TAUS:
                current = run_active(clean, utilities, pools, budget, tau, task)
                candidates.append(
                    {
                        "tau": tau,
                        "mean_cumulative_root_regret": float(current["cumulative_root_regret"].mean()),
                        "arrays": current,
                    }
                )
            chosen = min(candidates, key=lambda row: (row["mean_cumulative_root_regret"], row["tau"]))
            selected_tau[budget] = float(chosen["tau"])
            clean_active_by_budget[budget] = chosen.pop("arrays")
            temperature_rows[str(budget)] = {
                "selected_tau": selected_tau[budget],
                "grid": [
                    {key: value for key, value in row.items() if key != "arrays"}
                    for row in candidates
                ],
            }
            queries = fixed_queries(pools, budget, task_index)
            fixed_by_budget[budget] = queries
            clean_fixed_by_budget[budget] = run_fixed(clean, utilities, pools, queries, task)

        candidates: list[dict[str, Any]] = []
        cached: dict[tuple[int, str, float], tuple[list[list[str]], list[list[int]], np.ndarray, dict[str, Any]]] = {}
        for screen_row in screen["retained"]:
            target = int(screen_row["target_index"])
            texts = [
                runtime_text(prompt, output)
                for prompt, output in zip(data["prompts"], responses[target])
            ]
            scores, learner = error_scores_oof(texts, utilities[target], seed=18700 + task_index * 100 + target)
            for policy in POLICIES:
                for distance in DISTANCES:
                    rows, supports = build_alias_rows(
                        task=task,
                        ids=data["ids"],
                        parent_responses=responses[target].tolist(),
                        error_scores=scores,
                        policy=policy,
                        distance=distance,
                    )
                    alias_scores = alias_utilities(utilities[target], supports)
                    quality = quality_summary(utilities[target], alias_scores)
                    refined = make_refined(clean, target, rows, alias_scores, models, task, policy, distance)
                    cached[(target, policy, distance)] = (rows, supports, alias_scores, learner)
                    for budget in BUDGETS:
                        tau = selected_tau[budget]
                        refined_active = run_active(refined, utilities, pools, budget, tau, task)
                        refined_fixed = run_fixed(
                            refined,
                            utilities,
                            pools,
                            fixed_by_budget[budget],
                            task,
                        )
                        candidates.append(
                            calibration_record(
                                task=task,
                                target=target,
                                policy=policy,
                                distance=distance,
                                budget=budget,
                                tau=tau,
                                clean_active=clean_active_by_budget[budget],
                                refined_active=refined_active,
                                clean_fixed=clean_fixed_by_budget[budget],
                                refined_fixed=refined_fixed,
                                quality=quality,
                            )
                        )
        selected = min(
            candidates,
            key=lambda row: (
                -row["mean_terminal_root_did"],
                -row["mean_active_terminal_root_delta"],
                -row["mean_cumulative_root_did"],
                row["distance"],
                row["budget"],
                row["target_index"],
                row["policy"],
            ),
        )
        target = int(selected["target_index"])
        model = models[target]
        policy = str(selected["policy"])
        distance = float(selected["distance"])
        blind_record = blind_manifest["tasks"][task]["models"][model]
        blind_path = ROOT / blind_record["path"]
        if file_sha256(blind_path) != blind_record["sha256"]:
            raise AssertionError(f"{task}: selected blind parent file hash mismatch")
        blind = load_json(blind_path)
        if blind["parent_index"] != target or blind["parent"] != model:
            raise AssertionError(f"{task}: blind parent identity mismatch")
        calibration_texts = [
            runtime_text(prompt, output)
            for prompt, output in zip(data["prompts"], responses[target])
        ]
        holdout_texts = [
            runtime_text(prompt, output)
            for prompt, output in zip(blind["prompts"], blind["parent_outputs"])
        ]
        holdout_scores = error_scores_holdout(calibration_texts, utilities[target], holdout_texts)
        holdout_rows, holdout_supports = build_alias_rows(
            task=task,
            ids=blind["ids"],
            parent_responses=blind["parent_outputs"],
            error_scores=holdout_scores,
            policy=policy,
            distance=distance,
        )
        result["tasks"][task] = {
            "scenario": data["scenario"],
            "metric": data["metric"],
            "calibration_examples": examples,
            "holdout_examples": len(blind["ids"]),
            "pool_size": pool_size,
            "prediction_only_screen": screen,
            "temperature_selection": temperature_rows,
            "calibration_candidates": candidates,
            "selected": {
                **selected,
                "target": model,
                "selection_rule": "frozen lexicographic calibration rule",
            },
            "opened_blind_parent_file": {
                "path": blind_record["path"],
                "sha256": blind_record["sha256"],
            },
            "forbidden_holdout_files_opened": [],
            "holdout_attack": {
                "ids_digest": digest_json(blind["ids"]),
                "parent_response_hash": digest_json(blind["parent_outputs"]),
                "error_score_sha256": hashlib.sha256(holdout_scores.tobytes()).hexdigest(),
                "supports": [[int(value) for value in support] for support in holdout_supports],
                "support_id_digests": [
                    digest_json([blind["ids"][index] for index in support])
                    for support in holdout_supports
                ],
                "opaque_tokens": [opaque_token(task, alias) for alias in range(ALIASES)],
                "alias_response_hashes": [digest_json(row) for row in holdout_rows],
                "distance_queries": len(holdout_supports[0]),
                "distance_fraction_realized": len(holdout_supports[0]) / len(blind["ids"]),
                "constructor_inputs": ["prompt", "parent_output"],
                "references_read": 0,
                "peer_response_rows_read": 0,
                "pool_or_selector_state_read": False,
            },
        }
        print(
            f"[{task}] target={model} policy={policy} d={distance:g} "
            f"B={selected['budget']} cal-DID={selected['mean_terminal_root_did']:.6f}",
            flush=True,
        )

    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"STEP87B_STAGEA_COMPLETE {OUT.name} sha256={file_sha256(OUT)}", flush=True)


if __name__ == "__main__":
    main()

