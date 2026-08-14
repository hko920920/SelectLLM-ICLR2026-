"""Stage B: open the sealed Step 87B holdout and run every frozen task once."""

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
    PREREG,
    ROOT,
    TASKS,
    alias_utilities,
    bh_adjust,
    bootstrap_ci,
    digest_json,
    file_sha256,
    load_preregistration,
    one_sided_sign_flip_pvalue,
    opaque_token,
    pools_from_seeds,
    quality_summary,
    validate_tokens_against_references,
)


RUNNER = Path(__file__).resolve()
VALIDATOR = ROOT / "validate_step87b_sealed_holdout.py"
LOCK = ROOT / "STEP87B_STAGEB_EXECUTION_LOCK_2026-08-12.json"
ATTACK = ROOT / "STEP87B_FROZEN_REFERENCE_FREE_ATTACK_2026-08-12.json"
SEALED = ROOT / "external_data" / "step87b_sealed" / "STEP87B_SEALED_HOLDOUT_OUTCOMES_2026-08-12.json"
OUT = ROOT / "STEP87B_REFERENCE_FREE_OPEN_ENDED_RESULTS_2026-08-12.json"
RAW = ROOT / "STEP87B_REFERENCE_FREE_OPEN_ENDED_RAW_2026-08-12.npz"
DEPENDENCIES = {
    "step87b_openended_common.py": ROOT / "step87b_openended_common.py",
    "run_step80_main_t2a_v2_transfer.py": ROOT / "run_step80_main_t2a_v2_transfer.py",
    "run_step54_matrix_free_realism_audit.py": ROOT / "run_step54_matrix_free_realism_audit.py",
    "run_step46_clone_robust_soft_weighting.py": ROOT / "run_step46_clone_robust_soft_weighting.py",
    "run_step50_natural_alias_audit.py": ROOT / "run_step50_natural_alias_audit.py",
    "run_step29_locked_phase_a.py": ROOT / "run_step29_locked_phase_a.py",
}
HOLDOUT_SEEDS = tuple(range(88400, 89400))
TOL = 1e-12


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    prereg = load_preregistration()
    lock = load_json(LOCK)
    if lock.get("manifest_id") != "STEP87B_STAGEB_EXECUTION_LOCK_V1":
        raise AssertionError("unexpected Step 87B Stage-B lock")
    observed_inputs = {
        "preregistration": file_sha256(PREREG),
        "attack": file_sha256(ATTACK),
        "sealed_holdout": file_sha256(SEALED),
    }
    observed_code = {
        "stage_b_runner": file_sha256(RUNNER),
        "independent_validator": file_sha256(VALIDATOR),
        **{name: file_sha256(path) for name, path in DEPENDENCIES.items()},
    }
    if lock.get("input_sha256") != observed_inputs:
        raise AssertionError({"locked_inputs": lock.get("input_sha256"), "observed": observed_inputs})
    if lock.get("code_sha256") != observed_code:
        raise AssertionError({"locked_code": lock.get("code_sha256"), "observed": observed_code})
    if lock.get("outcomes_executed_before_lock") is not False:
        raise AssertionError("Stage-B lock does not precede outcomes")
    return prereg, lock


def make_clean(data: dict[str, Any], models: list[str], task: str) -> dict[str, Any]:
    phase_a = v2.step54.step46.phase_a
    return phase_a.make_scenario(
        np.asarray(data["registry_responses"], dtype=object),
        np.asarray(data["registry_utilities"], dtype=np.float64),
        list(range(len(models))),
        models,
        f"step87b_holdout_clean_{task}",
    )


def reconstruct_alias_rows(
    task: str, parent: list[str], supports: list[list[int]]
) -> list[list[str]]:
    rows = []
    for alias, support in enumerate(supports):
        row = list(parent)
        token = opaque_token(task, alias)
        for index in support:
            row[int(index)] = token
        rows.append(row)
    return rows


def make_refined(
    clean: dict[str, Any],
    target: int,
    rows: list[list[str]],
    utilities: np.ndarray,
    models: list[str],
    task: str,
    selected: dict[str, Any],
) -> dict[str, Any]:
    phase_a = v2.step54.step46.phase_a
    return phase_a.make_scenario(
        np.concatenate([np.asarray(clean["responses"], dtype=object), np.asarray(rows, dtype=object)]),
        np.concatenate([np.asarray(clean["oracle"], dtype=np.float64), utilities]),
        list(np.asarray(clean["parents"], dtype=np.int64)) + [target] * ALIASES,
        list(clean["labels"])
        + [
            f"STEP87B-HOLDOUT:{task}:{models[target]}:{selected['policy']}:{selected['distance']}:{alias}"
            for alias in range(ALIASES)
        ],
        f"step87b_holdout_refined_{task}",
    )


def fixed_queries(pools: np.ndarray, budget: int, task_index: int) -> np.ndarray:
    rows = []
    for run_index, pool in enumerate(pools):
        rng = random.Random(487000 + task_index * 10000 + run_index)
        rows.append(rng.sample([int(value) for value in pool], budget))
    return np.asarray(rows, dtype=np.int64)


def method_arrays(run: dict[str, Any]) -> dict[str, np.ndarray]:
    return v2.method_arrays(v2.primary_view(run))


def vector_summary(values: np.ndarray, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci95": bootstrap_ci(values, seed),
        "one_sided_sign_flip_p": one_sided_sign_flip_pvalue(values, seed + 1),
        "positive_fraction": float(np.mean(values > TOL)),
        "negative_fraction": float(np.mean(values < -TOL)),
        "zero_fraction": float(np.mean(np.abs(values) <= TOL)),
    }


def array_digest(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    header = json.dumps(
        {"dtype": value.dtype.str, "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\x00" + value.tobytes()).hexdigest()


def main() -> None:
    prereg, lock = verify_lock()
    attack = load_json(ATTACK)
    if attack.get("status") != "ATTACK_FROZEN_BEFORE_SEALED_HOLDOUT_OUTCOMES":
        raise AssertionError("attack was not frozen before outcomes")
    sealed = load_json(SEALED)
    models = list(sealed["models"])
    if models != list(attack["models"]):
        raise AssertionError("attack/sealed model roster mismatch")
    if models != list(v2.step54.step46.phase_a.MODELS):
        raise AssertionError("model roster/backend mismatch")

    result: dict[str, Any] = {
        "manifest_id": "STEP87B_REFERENCE_FREE_OPEN_ENDED_RESULTS_V1",
        "preregistration_sha256": file_sha256(PREREG),
        "stage_b_lock_sha256": file_sha256(LOCK),
        "frozen_attack_sha256": file_sha256(ATTACK),
        "sealed_holdout_sha256": file_sha256(SEALED),
        "all_eight_frozen_scenarios_reported": True,
        "tasks": {},
    }
    raw: dict[str, np.ndarray] = {}
    active_p: dict[str, float] = {}
    did_p: dict[str, float] = {}

    for task_index, task in enumerate(TASKS):
        print(f"[{task}] sealed holdout", flush=True)
        data = sealed["tasks"][task]
        spec = attack["tasks"][task]
        selected = spec["selected"]
        target = int(selected["target_index"])
        if selected["target"] != models[target]:
            raise AssertionError(f"{task}: selected target identity mismatch")
        ids = list(data["ids"])
        if digest_json(ids) != spec["holdout_attack"]["ids_digest"]:
            raise AssertionError(f"{task}: holdout ID digest mismatch")
        responses = np.asarray(data["registry_responses"], dtype=object)
        utilities = np.asarray(data["registry_utilities"], dtype=np.float64)
        parent = responses[target].tolist()
        if digest_json(parent) != spec["holdout_attack"]["parent_response_hash"]:
            raise AssertionError(f"{task}: blind/sealed parent response mismatch")
        validate_tokens_against_references(task, data["references"])
        supports = [
            [int(value) for value in support]
            for support in spec["holdout_attack"]["supports"]
        ]
        rows = reconstruct_alias_rows(task, parent, supports)
        row_hashes = [digest_json(row) for row in rows]
        if row_hashes != spec["holdout_attack"]["alias_response_hashes"]:
            raise AssertionError(f"{task}: frozen alias response hash mismatch")
        distances = [sum(left != right for left, right in zip(row, parent)) for row in rows]
        if distances != [int(spec["holdout_attack"]["distance_queries"])] * ALIASES:
            raise AssertionError(f"{task}: realized distance mismatch")
        alias_scores = alias_utilities(utilities[target], supports)
        quality = quality_summary(utilities[target], alias_scores)
        clean = make_clean(data, models, task)
        refined = make_refined(clean, target, rows, alias_scores, models, task, selected)

        budget = int(selected["budget"])
        tau = float(selected["tau"])
        pool_size = min(250, max(50, int(np.floor(0.8 * len(ids)))))
        if pool_size != int(spec["pool_size"]):
            raise AssertionError(f"{task}: calibration/holdout pool-size mismatch")
        pools = pools_from_seeds(list(HOLDOUT_SEEDS), len(ids), pool_size)
        queries = fixed_queries(pools, budget, task_index)
        clean_active = method_arrays(
            v2.active_runs(
                clean,
                utilities,
                pools,
                HOLDOUT_SEEDS,
                budget,
                tau,
                task,
                v2.PRIMARY_QUERY_POLICY,
            )
        )
        refined_active = method_arrays(
            v2.active_runs(
                refined,
                utilities,
                pools,
                HOLDOUT_SEEDS,
                budget,
                tau,
                task,
                v2.PRIMARY_QUERY_POLICY,
            )
        )
        clean_fixed = method_arrays(
            v2.fixed_runs(clean, utilities, pools, HOLDOUT_SEEDS, queries, task)
        )
        refined_fixed = method_arrays(
            v2.fixed_runs(refined, utilities, pools, HOLDOUT_SEEDS, queries, task)
        )
        active_terminal = refined_active["terminal_root_regret"] - clean_active["terminal_root_regret"]
        fixed_terminal = refined_fixed["terminal_root_regret"] - clean_fixed["terminal_root_regret"]
        terminal_did = active_terminal - fixed_terminal
        active_cumulative = refined_active["cumulative_root_regret"] - clean_active["cumulative_root_regret"]
        fixed_cumulative = refined_fixed["cumulative_root_regret"] - clean_fixed["cumulative_root_regret"]
        cumulative_did = active_cumulative - fixed_cumulative
        stats = {
            "active_terminal_root_delta": vector_summary(active_terminal, 287000 + task_index * 20),
            "fixed_terminal_root_delta": vector_summary(fixed_terminal, 287002 + task_index * 20),
            "terminal_root_did": vector_summary(terminal_did, 287004 + task_index * 20),
            "active_cumulative_root_delta": vector_summary(active_cumulative, 287006 + task_index * 20),
            "fixed_cumulative_root_delta": vector_summary(fixed_cumulative, 287008 + task_index * 20),
            "cumulative_root_did": vector_summary(cumulative_did, 287010 + task_index * 20),
        }
        active_p[task] = stats["active_terminal_root_delta"]["one_sided_sign_flip_p"]
        did_p[task] = stats["terminal_root_did"]["one_sided_sign_flip_p"]
        arrays_to_save = {
            "pools": pools,
            "fixed_queries": queries,
            "clean_active_queries": clean_active["queried"],
            "refined_active_queries": refined_active["queried"],
            "clean_active_root_path": clean_active["selected_root_path"],
            "refined_active_root_path": refined_active["selected_root_path"],
            "clean_fixed_root_path": clean_fixed["selected_root_path"],
            "refined_fixed_root_path": refined_fixed["selected_root_path"],
            "active_terminal_delta": active_terminal,
            "fixed_terminal_delta": fixed_terminal,
            "terminal_did": terminal_did,
            "active_cumulative_delta": active_cumulative,
            "fixed_cumulative_delta": fixed_cumulative,
            "cumulative_did": cumulative_did,
        }
        for name, value in arrays_to_save.items():
            raw[f"{task}__{name}"] = np.asarray(value)
        result["tasks"][task] = {
            "scenario": data["scenario"],
            "metric": data["metric"],
            "examples": len(ids),
            "pool_size": pool_size,
            "runs": len(HOLDOUT_SEEDS),
            "selected_from_calibration": selected,
            "attack_integrity": {
                "parent_response_hash_matches_blind_freeze": True,
                "four_distinct_nonparent_alias_hashes": len(set(row_hashes)) == ALIASES and digest_json(parent) not in set(row_hashes),
                "response_distance_queries_by_alias": distances,
                "response_distance_fraction_by_alias": [value / len(ids) for value in distances],
                "opaque_tokens_absent_from_all_references": True,
                "holdout_constructor_inputs": spec["holdout_attack"]["constructor_inputs"],
                "holdout_references_read_at_construction": spec["holdout_attack"]["references_read"],
                "holdout_peer_rows_read_at_construction": spec["holdout_attack"]["peer_response_rows_read"],
                "fixed_query_matrices_byte_identical": bool(np.array_equal(clean_fixed["queried"], refined_fixed["queried"])),
                "fixed_root_paths_byte_identical": bool(np.array_equal(clean_fixed["selected_root_path"], refined_fixed["selected_root_path"])),
            },
            "quality": quality,
            "path_effect": {
                "ordered_path_change_fraction": float(np.mean(np.any(clean_active["queried"] != refined_active["queried"], axis=1))),
                "mean_changed_query_positions": float(np.mean(np.sum(clean_active["queried"] != refined_active["queried"], axis=1))),
                "terminal_selected_root_change_fraction": float(np.mean(clean_active["selected_root_path"][:, -1] != refined_active["selected_root_path"][:, -1])),
            },
            "statistics": stats,
            "array_digests": {name: array_digest(value) for name, value in arrays_to_save.items()},
        }
        print(
            f"[{task}] active-terminal={stats['active_terminal_root_delta']['mean']:.6f} "
            f"DID={stats['terminal_root_did']['mean']:.6f}",
            flush=True,
        )

    active_q = bh_adjust(active_p)
    did_q = bh_adjust(did_p)
    passed = []
    for task in TASKS:
        row = result["tasks"][task]
        integrity = row["attack_integrity"]
        quality = row["quality"]
        statistics = row["statistics"]
        quality_pass = bool(
            quality["all_coordinatewise_nonimproving"]
            and quality["minimum_gap"] >= -0.010000000001
            and quality["maximum_gap"] <= 1e-12
            and max(integrity["response_distance_fraction_by_alias"]) <= 0.010000000001
        )
        integrity_pass = bool(
            integrity["parent_response_hash_matches_blind_freeze"]
            and integrity["four_distinct_nonparent_alias_hashes"]
            and integrity["opaque_tokens_absent_from_all_references"]
            and integrity["holdout_references_read_at_construction"] == 0
            and integrity["holdout_peer_rows_read_at_construction"] == 0
            and integrity["fixed_query_matrices_byte_identical"]
        )
        active_pass = bool(
            statistics["active_terminal_root_delta"]["ci95"][0] > 0.0
            and active_q[task] < 0.05
        )
        did_pass = bool(
            statistics["terminal_root_did"]["ci95"][0] > 0.0
            and did_q[task] < 0.05
        )
        row["multiplicity_adjustment"] = {
            "active_terminal_bh_q": active_q[task],
            "terminal_did_bh_q": did_q[task],
        }
        row["gate"] = {
            "quality": quality_pass,
            "integrity": integrity_pass,
            "active_terminal_root_harm": active_pass,
            "adaptive_terminal_excess": did_pass,
            "exact_fixed_query_mediation": integrity["fixed_root_paths_byte_identical"],
            "task_pass": quality_pass and integrity_pass and active_pass and did_pass,
        }
        if row["gate"]["task_pass"]:
            passed.append(task)

    global_separation = bool(
        attack["sealed_holdout_outcomes_read"] is False
        and attack["holdout_peer_responses_read"] is False
        and attack["holdout_references_or_utilities_read"] is False
        and set(result["tasks"]) == set(TASKS)
    )
    if not global_separation or len(passed) == 0:
        decision = "NO_TERMINAL_PROMOTION"
    elif len(passed) == 1:
        decision = "NARROW_TERMINAL_BRIDGE"
    else:
        decision = "PROMOTE_REFERENCE_FREE_TERMINAL_BRIDGE"
    result["decision"] = {
        "global_separation_and_completeness": global_separation,
        "passing_scenarios": passed,
        "passing_count": len(passed),
        "status": decision,
    }
    np.savez_compressed(RAW, **raw)
    result["raw_npz"] = {"path": RAW.name, "sha256": file_sha256(RAW), "arrays": len(raw)}
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"STEP87B_STAGEB_COMPLETE status={decision} pass={len(passed)}/8", flush=True)
    print(f"results_sha256={file_sha256(OUT)} raw_sha256={file_sha256(RAW)}", flush=True)


if __name__ == "__main__":
    main()

