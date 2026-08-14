"""Execute and seal all response-only Step 61 ERFA scores and baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

import audit_step61_freshness_feasibility as stage0
import prepare_step61a_blind_scenarios as scenario_prep
import step61a_blind_common as common


ROOT = stage0.ROOT
PROTOCOL = ROOT / "STEP61_ERFA_FRESH_CROSS_SELECTOR_PREREGISTRATION_2026-08-10.json"
BLIND_MANIFEST = ROOT / "STEP61A_BLIND_INPUT_MANIFEST_2026-08-10.json"
SCENARIO_MANIFEST = ROOT / "STEP61A_BLIND_SCENARIO_MANIFEST_2026-08-10.json"
EXECUTION_LOCK = ROOT / "STEP61A_BLIND_SCORE_EXECUTION_LOCK_2026-08-10.json"
OUT = ROOT / "STEP61A_BLIND_SCORE_RESULTS_2026-08-10.json"
RISK_OUT = ROOT / "STEP61A_SEALED_HIGH_RISK_IDS_2026-08-10.json"
RUNNER = ROOT / "run_step61a_blind_scores.py"

EPSILON = {
    "domain_drift": 0.50,
    "emotion_detection": 0.47,
    "imagenet": 0.45,
    "imagenet_pytorch_models": 0.46,
    "imagenet_v2_matched-frequency": 0.46,
    "imagenet_v2_threshold-0.7": 0.46,
    "imagenet_v2_top-images": 0.46,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_lock() -> dict[str, Any]:
    lock = read_json(EXECUTION_LOCK)
    if lock["manifest_id"] != "STEP61A_BLIND_SCORE_EXECUTION_LOCK_V1":
        raise AssertionError("unexpected Stage A execution lock")
    paths = {
        "protocol": PROTOCOL,
        "blind_input_manifest": BLIND_MANIFEST,
        "scenario_manifest": SCENARIO_MANIFEST,
        "common_code": ROOT / "step61a_blind_common.py",
        "blind_preparer": ROOT / "prepare_step61a_blind_inputs.py",
        "scenario_preparer": ROOT / "prepare_step61a_blind_scenarios.py",
        "runner": RUNNER,
        "candidate_id_clarification": ROOT / "STEP61A_CANDIDATE_ID_IMPLEMENTATION_CLARIFICATION_2026-08-10.json",
        "tie_stream_clarification": ROOT / "STEP61A_TIE_STREAM_IMPLEMENTATION_CLARIFICATION_2026-08-10.json",
        "integrity_deviation": ROOT / "STEP61A_PRE_SCORE_INTEGRITY_DEVIATION_2026-08-10.md",
    }
    observed = {name: sha256_file(path) for name, path in paths.items()}
    if observed != lock["bound_sha256"]:
        raise AssertionError({"locked": lock["bound_sha256"], "observed": observed})
    if lock["outcomes_permitted"] is not False:
        raise AssertionError("Stage A lock must prohibit outcomes")
    return lock


def load_context(task_id: str, task_manifest: dict[str, Any]) -> dict[str, Any]:
    package = ROOT / task_manifest["blind_package"]["path"]
    with np.load(package, allow_pickle=False) as data:
        responses = data["responses"]
        candidate_ids = [str(value) for value in data["candidate_ids"]]
        observed_task = str(data["task_id"].item())
        family = str(data["selector_family"].item())
    if observed_task != task_id or family != task_manifest["selector_family"]:
        raise AssertionError(f"{task_id}: blind package identity mismatch")
    partition = task_manifest["candidate_partition"]
    core_indices = [int(value) for value in partition["core_original_indices"]]
    reserve_indices = [int(value) for value in partition["reserve_original_indices"]]
    core = responses[:, core_indices]
    reserve = responses[:, reserve_indices]
    core_ids = [candidate_ids[index] for index in core_indices]
    reserve_ids = [candidate_ids[index] for index in reserve_indices]
    roots = [int(row["core_local_index"]) for row in task_manifest["stress_roots"]]
    clean_rewards = common.core_support_rewards(core)
    if common.digest_array(clean_rewards) != task_manifest["response_only"]["clean_proxy_reward_digest"]:
        raise AssertionError(f"{task_id}: clean proxy reward digest mismatch")
    scenarios = list(
        common.generate_registry_scenarios(
            task_id,
            core,
            core_ids,
            reserve,
            reserve_ids,
            roots,
            clean_rewards,
        )
    )
    recorded = task_manifest["scenarios"]
    if [row["scenario_id"] for row in scenarios] != [row["scenario_id"] for row in recorded]:
        raise AssertionError(f"{task_id}: scenario order mismatch")
    for current, expected in zip(scenarios, recorded):
        if common.digest_array(current["responses"]) != expected["response_digest"]:
            raise AssertionError(f"{task_id}: response digest mismatch")
        if common.digest_array(current["rewards"]) != expected["reward_digest"]:
            raise AssertionError(f"{task_id}: reward digest mismatch")
    seeds = [int(value) for value in task_manifest["paired_design"]["seeds"]]
    requested_pool = (
        scenario_prep.MODEL_POOL_SIZE
        if family == "model_selector"
        else scenario_prep.LLM_POOL_SIZE
    )
    pools = np.asarray(
        [common.sample_pool(seed, len(responses), requested_pool) for seed in seeds]
    )
    if common.digest_array(pools) != task_manifest["paired_design"]["pool_matrix_digest"]:
        raise AssertionError(f"{task_id}: pool digest mismatch")
    if family == "model_selector":
        clean_codes = np.asarray(core, dtype=np.int32)
    else:
        clean_codes, _ = common.encode_responses(core)
    return {
        "family": family,
        "full_responses": responses,
        "core": core,
        "clean_rewards": clean_rewards,
        "clean_codes": clean_codes,
        "core_count": len(core_indices),
        "scenarios": scenarios,
        "scenario_records": recorded,
        "seeds": seeds,
        "pools": pools,
        "budget": int(task_manifest["paired_design"]["budget"]),
        "class_count": int(task_manifest["response_only"]["global_response_classes"]),
    }


def run_condition(
    task_id: str,
    family: str,
    codes: np.ndarray,
    rewards: np.ndarray,
    core_count: int,
    pools: np.ndarray,
    seeds: list[int],
    budget: int,
    class_count: int,
) -> list[dict[str, Any]]:
    epsilon = EPSILON.get(task_id)
    return [
        common.run_active_selector(
            codes,
            rewards,
            core_count,
            pool,
            budget,
            seed,
            family,
            epsilon=epsilon,
            tau=1.0,
            class_count=class_count,
        )
        for seed, pool in zip(seeds, pools)
    ]


def score_task(task_id: str) -> dict[str, Any]:
    started = time.perf_counter()
    manifest = read_json(SCENARIO_MANIFEST)
    task_manifest = manifest["tasks"][task_id]
    context = load_context(task_id, task_manifest)
    family = context["family"]
    clean_runs = run_condition(
        task_id,
        family,
        context["clean_codes"],
        context["clean_rewards"],
        context["core_count"],
        context["pools"],
        context["seeds"],
        context["budget"],
        context["class_count"],
    )
    scenario_results: list[dict[str, Any]] = []
    for scenario, recorded in zip(context["scenarios"], context["scenario_records"]):
        if family == "model_selector":
            codes = np.asarray(scenario["responses"], dtype=np.int32)
        else:
            codes, _ = common.encode_responses(scenario["responses"])
        proposed_runs = run_condition(
            task_id,
            family,
            codes,
            scenario["rewards"],
            context["core_count"],
            context["pools"],
            context["seeds"],
            context["budget"],
            context["class_count"],
        )
        delta = np.asarray(
            [
                proposed["cumulative_regret"] - clean["cumulative_regret"]
                for clean, proposed in zip(clean_runs, proposed_runs)
            ],
            dtype=np.float64,
        )
        first = np.asarray(
            [clean["query_path"][0] != proposed["query_path"][0] for clean, proposed in zip(clean_runs, proposed_runs)],
            dtype=np.float64,
        )
        hammings: list[float] = []
        jaccards: list[float] = []
        static: list[float] = []
        for clean, proposed in zip(clean_runs, proposed_runs):
            hamming, jaccard = common.path_metrics(
                clean["query_path"], proposed["query_path"]
            )
            hammings.append(hamming)
            jaccards.append(jaccard)
            static.append(
                common.static_rank_displacement(
                    clean["initial_scores"], proposed["initial_scores"]
                )
            )
        witness_index = int(np.argmax(np.abs(delta)))
        clean_witness = clean_runs[witness_index]
        proposed_witness = proposed_runs[witness_index]
        budget = context["budget"]
        signed_mean = float(np.mean(delta) / budget)
        result = {
            "task_id": task_id,
            "selector_family": family,
            "scenario_id": scenario["scenario_id"],
            "scenario_sha256": scenario["scenario_sha256"],
            "kind": scenario["kind"],
            "aliases": int(scenario["aliases"]),
            "nominal_rho": scenario.get("nominal_rho"),
            "realized_rho": scenario.get("realized_rho"),
            "nearest_core_distances": [float(value) for value in scenario["nearest_core_distances"]],
            "paired_runs": len(delta),
            "budget": budget,
            "scores": {
                "erfa": abs(signed_mean),
                "proxy_signed_mean_delta_normalized": signed_mean,
                "count_distance": float(scenario["count_distance"]),
                "alias_count": int(scenario["aliases"]),
                "mean_nearest_core_distance": float(np.mean(scenario["nearest_core_distances"])),
                "first_query_change_fraction": float(np.mean(first)),
                "path_hamming": float(np.mean(hammings)),
                "path_jaccard": float(np.mean(jaccards)),
                "static_mass_rank_displacement": float(np.mean(static)),
            },
            "paired_proxy_delta_cumulative_regret": [float(value) for value in delta],
            "witness": {
                "paired_index": witness_index,
                "seed": context["seeds"][witness_index],
                "proxy_delta_cumulative_regret": float(delta[witness_index]),
                "clean_query_path_sha256": common.digest_array(clean_witness["query_path"]),
                "proposed_query_path_sha256": common.digest_array(proposed_witness["query_path"]),
                "clean_selected_path_sha256": common.digest_array(clean_witness["selected_path"]),
                "proposed_selected_path_sha256": common.digest_array(proposed_witness["selected_path"]),
            },
            "blind_inputs": {
                "response_digest": recorded["response_digest"],
                "reward_digest": recorded["reward_digest"],
            },
            "high_risk": None,
            "family_blind_rank": None,
        }
        scenario_results.append(result)
    clean_query_matrix = np.stack([run["query_path"] for run in clean_runs])
    clean_selected_matrix = np.stack([run["selected_path"] for run in clean_runs])
    return {
        "task_id": task_id,
        "selector_family": family,
        "budget": context["budget"],
        "pool_size": int(context["pools"].shape[1]),
        "core_count": context["core_count"],
        "clean_run": {
            "query_matrix_sha256": common.digest_array(clean_query_matrix),
            "selected_matrix_sha256": common.digest_array(clean_selected_matrix),
            "mean_proxy_cumulative_regret": float(np.mean([run["cumulative_regret"] for run in clean_runs])),
            "minimum_disagreement_items": int(min(run["disagreement_items"] for run in clean_runs)),
        },
        "scenarios": scenario_results,
        "runtime_seconds": time.perf_counter() - started,
    }


def seal_risk(tasks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    selection: dict[str, Any] = {}
    for family in ("model_selector", "select_llm"):
        rows: list[dict[str, Any]] = []
        for task in tasks.values():
            if task["selector_family"] == family:
                rows.extend(task["scenarios"])
        if len(rows) != 210:
            raise AssertionError(f"{family}: expected 210 nonclean rows, found {len(rows)}")
        ordered = sorted(
            rows,
            key=lambda row: (float(row["scores"]["erfa"]), row["scenario_sha256"]),
        )
        high_count = len(ordered) // 2
        high = ordered[-high_count:]
        high_ids = {row["scenario_id"] for row in high}
        for rank, row in enumerate(ordered, start=1):
            row["family_blind_rank"] = rank
            row["high_risk"] = row["scenario_id"] in high_ids
        selection[family] = {
            "units": len(ordered),
            "high_risk_count": high_count,
            "low_risk_count": len(ordered) - high_count,
            "high_risk_scenario_ids": [row["scenario_id"] for row in high],
            "high_risk_id_set_sha256": common.digest_text(*sorted(high_ids)),
            "boundary": {
                "highest_low_score": float(ordered[-high_count - 1]["scores"]["erfa"]),
                "highest_low_scenario_sha256": ordered[-high_count - 1]["scenario_sha256"],
                "lowest_high_score": float(ordered[-high_count]["scores"]["erfa"]),
                "lowest_high_scenario_sha256": ordered[-high_count]["scenario_sha256"],
                "tie_break": "ascending scenario SHA256; upper half is high risk",
            },
        }
    return selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    lock = verify_lock()
    manifest = read_json(SCENARIO_MANIFEST)
    if manifest["decision"] != "BLIND_SCENARIOS_READY":
        raise AssertionError("scenario manifest not ready")
    task_ids = list(manifest["tasks"])
    started = time.perf_counter()
    tasks: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(score_task, task_id): task_id for task_id in task_ids}
        for future in as_completed(futures):
            task_id = futures[future]
            task = future.result()
            tasks[task_id] = task
            print(json.dumps({
                "completed_task": task_id,
                "family": task["selector_family"],
                "runtime_seconds": round(task["runtime_seconds"], 3),
                "completed": len(tasks),
                "total": len(task_ids),
            }), flush=True)
    tasks = {task_id: tasks[task_id] for task_id in task_ids}
    risk = seal_risk(tasks)
    flat = [row for task in tasks.values() for row in task["scenarios"]]
    if len(flat) != 420 or sum(bool(row["high_risk"]) for row in flat) != 210:
        raise AssertionError("final Stage A row/risk count mismatch")
    risk_artifact = {
        "manifest_id": "STEP61A_SEALED_HIGH_RISK_IDS_V1",
        "protocol_sha256": sha256_file(PROTOCOL),
        "scenario_manifest_sha256": sha256_file(SCENARIO_MANIFEST),
        "execution_lock_sha256": sha256_file(EXECUTION_LOCK),
        "selection": risk,
        "total_high_risk": sum(row["high_risk_count"] for row in risk.values()),
        "total_low_risk": sum(row["low_risk_count"] for row in risk.values()),
        "outcomes_used": False,
    }
    RISK_OUT.write_text(
        json.dumps(risk_artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    result = {
        "manifest_id": "STEP61A_BLIND_SCORE_RESULTS_V1",
        "protocol_sha256": sha256_file(PROTOCOL),
        "blind_input_manifest_sha256": sha256_file(BLIND_MANIFEST),
        "scenario_manifest_sha256": sha256_file(SCENARIO_MANIFEST),
        "execution_lock_sha256": sha256_file(EXECUTION_LOCK),
        "risk_artifact_sha256": sha256_file(RISK_OUT),
        "common_self_test": common.self_test(),
        "counts": {
            "tasks": len(tasks),
            "nonclean_units": len(flat),
            "high_risk": sum(bool(row["high_risk"]) for row in flat),
            "low_risk": sum(not bool(row["high_risk"]) for row in flat),
            "paired_runs_per_unit": 32,
        },
        "risk_selection": risk,
        "tasks": tasks,
        "outcome_blindness": {
            "oracle_arrays_loaded": False,
            "HELM_per_instance_stats_downloaded": False,
            "candidate_accuracies_used": False,
            "only_response_derived_feedback_used": True,
            "integrity_deviation_disclosed": True,
        },
        "runtime_seconds": time.perf_counter() - started,
        "decision": "STAGE_A_SCORES_AND_RISK_IDS_SEALED",
        "next_permitted_action": "Hash-lock these Stage A outputs and the separately written Stage B runner before opening outcomes.",
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": result["decision"],
        "counts": result["counts"],
        "runtime_seconds": round(result["runtime_seconds"], 3),
        "output": str(OUT),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
