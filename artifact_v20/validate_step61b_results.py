"""Independent post-outcome validator for the locked Step 61B adjudication.

This file deliberately does not import the Stage B runner.  It reimplements the
reported metrics, resampling procedures, gates, and artifact-integrity checks
from the frozen JSON outputs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "STEP61B_TRUE_EFFECT_RESULTS_2026-08-10.json"
PROTOCOL = ROOT / "STEP61_ERFA_FRESH_CROSS_SELECTOR_PREREGISTRATION_2026-08-10.json"
SCENARIOS = ROOT / "STEP61A_BLIND_SCENARIO_MANIFEST_2026-08-10.json"
RISK = ROOT / "STEP61A_SEALED_HIGH_RISK_IDS_2026-08-10.json"
STAGE_A = ROOT / "STEP61A_BLIND_SCORE_RESULTS_2026-08-10.json"
STAGE_A_LOCK = ROOT / "STEP61A_BLIND_OUTPUT_LOCK_2026-08-10.json"
EXECUTION_LOCK = ROOT / "STEP61B_OUTCOME_EXECUTION_LOCK_2026-08-10.json"
OUTCOME_MANIFEST = ROOT / "STEP61B_OUTCOME_SOURCE_MANIFEST_2026-08-10.json"
OUT = ROOT / "STEP61B_INDEPENDENT_VALIDATION_2026-08-10.json"

BASELINES = (
    "count_distance",
    "first_query_change_fraction",
    "path_hamming",
    "path_jaccard",
    "static_mass_rank_displacement",
)
MATERIALITY = 0.01
PERMUTATIONS = 10_000
BOOTSTRAPS = 10_000


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return bool(abs(float(left) - float(right)) <= tolerance)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata(np.asarray(x, dtype=np.float64), method="average")
    ry = rankdata(np.asarray(y, dtype=np.float64), method="average")
    if np.allclose(rx, rx[0]) or np.allclose(ry, ry[0]):
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive = int(np.sum(labels == 1))
    negative = int(np.sum(labels == 0))
    if positive == 0 or negative == 0:
        return 0.5
    ranks = rankdata(scores, method="average")
    rank_sum = float(np.sum(ranks[labels == 1]))
    return float(
        (rank_sum - positive * (positive + 1) / 2) / (positive * negative)
    )


def metric(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    score = np.asarray([row["blind_scores"][key] for row in rows], dtype=np.float64)
    effect = np.asarray([row["active"]["absolute_effect"] for row in rows], dtype=np.float64)
    label = np.asarray([row["active"]["material"] for row in rows], dtype=np.int64)
    return {
        "spearman_absolute_effect": safe_spearman(score, effect),
        "materiality_auroc": binary_auc(label, score),
    }


def task_block_permutation(rows: list[dict[str, Any]], observed: float) -> float:
    if observed <= 0:
        return 1.0
    task_ids = list(dict.fromkeys(row["task_id"] for row in rows))
    blocks = {
        task: np.asarray(
            [index for index, row in enumerate(rows) if row["task_id"] == task],
            dtype=np.int64,
        )
        for task in task_ids
    }
    rx = rankdata(
        np.asarray([row["blind_scores"]["erfa"] for row in rows]), method="average"
    )
    ry = rankdata(
        np.asarray([row["active"]["absolute_effect"] for row in rows]),
        method="average",
    )
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denominator = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    families = {
        family: [
            task
            for task in task_ids
            if next(row for row in rows if row["task_id"] == task)["selector_family"]
            == family
        ]
        for family in ("model_selector", "select_llm")
    }
    rng = np.random.default_rng(61200)
    extreme = 0
    for _ in range(PERMUTATIONS):
        permuted = np.empty_like(ry)
        for family_tasks in families.values():
            source_tasks = list(rng.permutation(family_tasks))
            for destination, source in zip(family_tasks, source_tasks):
                permuted[blocks[destination]] = ry[blocks[source]]
        null_value = float(np.dot(rx, permuted) / denominator)
        extreme += int(null_value >= observed - 1e-15)
    return float((extreme + 1) / (PERMUTATIONS + 1))


def task_bootstrap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    task_ids = list(dict.fromkeys(row["task_id"] for row in rows))
    task_indices = {
        task: np.asarray(
            [index for index, row in enumerate(rows) if row["task_id"] == task],
            dtype=np.int64,
        )
        for task in task_ids
    }
    labels = np.asarray([row["active"]["material"] for row in rows], dtype=np.int64)
    scores = {
        key: np.asarray([row["blind_scores"][key] for row in rows], dtype=np.float64)
        for key in ("erfa", *BASELINES)
    }
    rng = np.random.default_rng(61300)
    differences = np.zeros(BOOTSTRAPS, dtype=np.float64)
    for repetition in range(BOOTSTRAPS):
        sampled = rng.integers(0, len(task_ids), size=len(task_ids))
        indices = np.concatenate(
            [task_indices[task_ids[int(value)]] for value in sampled]
        )
        label = labels[indices]
        if len(np.unique(label)) < 2:
            continue
        erfa_auc = binary_auc(label, scores["erfa"][indices])
        best_auc = max(binary_auc(label, scores[key][indices]) for key in BASELINES)
        differences[repetition] = erfa_auc - best_auc
    quantile = np.quantile(differences, [0.025, 0.5, 0.975])
    return {
        "mean_difference": float(np.mean(differences)),
        "ci95": [float(quantile[0]), float(quantile[2])],
        "median_difference": float(quantile[1]),
        "nonpositive_fraction": float(np.mean(differences <= 0)),
    }


def main() -> None:
    result = read_json(RESULT)
    risk = read_json(RISK)
    outcome_manifest = read_json(OUTCOME_MANIFEST)
    execution_lock = read_json(EXECUTION_LOCK)
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any = None) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    # Artifact chain and the pre-outcome code lock.
    expected_result_hashes = {
        "protocol_sha256": sha256_file(PROTOCOL),
        "scenario_manifest_sha256": sha256_file(SCENARIOS),
        "stage_a_results_sha256": sha256_file(STAGE_A),
        "stage_a_risk_sha256": sha256_file(RISK),
        "stage_a_output_lock_sha256": sha256_file(STAGE_A_LOCK),
        "execution_lock_sha256": sha256_file(EXECUTION_LOCK),
        "outcome_manifest_sha256": sha256_file(OUTCOME_MANIFEST),
    }
    for field, observed in expected_result_hashes.items():
        check(f"result_hash::{field}", result[field] == observed, observed)
    bound_paths = {
        "protocol": PROTOCOL,
        "scenario_manifest": SCENARIOS,
        "stage_a_results": STAGE_A,
        "stage_a_risk": RISK,
        "stage_a_output_lock": STAGE_A_LOCK,
        "stage_a_common": ROOT / "step61a_blind_common.py",
        "stage_a_runner": ROOT / "run_step61a_blind_scores.py",
        "stage_b_runner": ROOT / "run_step61b_outcomes.py",
        "integrity_deviation": ROOT
        / "STEP61A_PRE_SCORE_INTEGRITY_DEVIATION_2026-08-10.md",
    }
    check(
        "execution_lock_bound_files",
        execution_lock["bound_sha256"]
        == {key: sha256_file(path) for key, path in bound_paths.items()},
    )
    check(
        "outcome_execution_was_permitted",
        execution_lock["outcomes_permitted_after_verification"] is True,
    )

    # Outcome package integrity and shape checks.
    package_entries = list(outcome_manifest["classifier_tasks"].values()) + list(
        outcome_manifest["helm_math_tasks"].values()
    )
    check("outcome_task_count", len(package_entries) == 14, len(package_entries))
    check("outcome_alignment_passed", outcome_manifest["alignment_passed"] is True)
    check(
        "raw_HELM_payloads_not_persisted",
        outcome_manifest["raw_HELM_payloads_persisted"] is False,
    )
    for entry in package_entries:
        package = ROOT / entry["outcome_package"]["path"]
        check(
            f"package_hash::{entry['task_id']}",
            sha256_file(package) == entry["outcome_package"]["sha256"],
        )
        check(
            f"package_size::{entry['task_id']}",
            package.stat().st_size == entry["outcome_package"]["bytes"],
        )
        with np.load(package, allow_pickle=False) as data:
            check(
                f"package_shape::{entry['task_id']}",
                list(data["candidate_rewards"].shape) == entry["shape"],
                list(data["candidate_rewards"].shape),
            )
            check(
                f"package_finite::{entry['task_id']}",
                bool(np.all(np.isfinite(data["candidate_rewards"]))),
            )
            check(
                f"package_identity::{entry['task_id']}",
                str(data["task_id"].item()) == entry["task_id"],
            )

    # Row-level reconstruction, sealed-risk membership, and selective-policy logic.
    tasks = result["tasks"]
    rows = [row for task in tasks.values() for row in task["rows"]]
    check("result_task_count", len(tasks) == 14, len(tasks))
    check("result_cell_count", len(rows) == 420, len(rows))
    check("unique_scenario_ids", len({row["scenario_id"] for row in rows}) == 420)
    check(
        "selector_family_balance",
        {
            family: sum(row["selector_family"] == family for row in rows)
            for family in ("model_selector", "select_llm")
        }
        == {"model_selector": 210, "select_llm": 210},
    )
    row_failures: list[str] = []
    risk_sets = {
        family: set(value["high_risk_scenario_ids"])
        for family, value in risk["selection"].items()
    }
    for task_id, task in tasks.items():
        task_rows = task["rows"]
        if len(task_rows) != 30:
            row_failures.append(f"{task_id}:row_count")
        kinds = {kind: sum(row["kind"] == kind for row in task_rows) for kind in ("exact", "near", "diverse")}
        if kinds != {"exact": 12, "near": 16, "diverse": 2}:
            row_failures.append(f"{task_id}:kind_counts={kinds}")
        for row in task_rows:
            sid = row["scenario_id"]
            family = row["selector_family"]
            budget = int(task["budget"])
            active_pair = np.asarray(row["active"]["paired_delta_cumulative_regret"], dtype=np.float64)
            fixed_pair = np.asarray(row["fixed_fallback"]["paired_delta_cumulative_regret"], dtype=np.float64)
            active_delta = float(np.mean(active_pair) / budget)
            fixed_delta = float(np.mean(fixed_pair) / budget)
            if len(active_pair) != 32 or len(fixed_pair) != 32:
                row_failures.append(f"{sid}:paired_length")
            if not close(active_delta, row["active"]["delta_normalized_cumulative_regret"]):
                row_failures.append(f"{sid}:active_delta")
            if not close(fixed_delta, row["fixed_fallback"]["delta_normalized_cumulative_regret"]):
                row_failures.append(f"{sid}:fixed_delta")
            if not close(abs(active_delta), row["active"]["absolute_effect"]):
                row_failures.append(f"{sid}:absolute_effect")
            if bool(abs(active_delta) >= MATERIALITY) != bool(row["active"]["material"]):
                row_failures.append(f"{sid}:materiality")
            sealed_high = sid in risk_sets[family]
            if sealed_high != bool(row["high_risk"]):
                row_failures.append(f"{sid}:sealed_risk")
            selected_pair = fixed_pair if sealed_high else active_pair
            selected_delta = fixed_delta if sealed_high else active_delta
            if not np.allclose(
                selected_pair,
                np.asarray(row["selective_policy"]["paired_delta_cumulative_regret"]),
                atol=1e-12,
                rtol=0,
            ):
                row_failures.append(f"{sid}:selective_pair")
            if not close(
                selected_delta,
                row["selective_policy"]["delta_normalized_cumulative_regret"],
            ):
                row_failures.append(f"{sid}:selective_delta")
            if bool(row["selective_policy"]["used_fallback"]) != sealed_high:
                row_failures.append(f"{sid}:fallback_flag")
            decomposition = row["decomposition"]
            if not close(
                decomposition["direct_registry_mean_normalized"]
                + decomposition["adaptive_path_mean_normalized"],
                active_delta,
                1e-10,
            ):
                row_failures.append(f"{sid}:decomposition")
    check("all_row_reconstructions", not row_failures, row_failures[:20])

    # Independent metric and resampling recomputation.
    adjudication = result["adjudication"]
    erfa = metric(rows, "erfa")
    permutation_p = task_block_permutation(rows, erfa["spearman_absolute_effect"])
    check(
        "erfa_spearman_recomputed",
        close(erfa["spearman_absolute_effect"], adjudication["erfa"]["spearman_absolute_effect"]),
        erfa["spearman_absolute_effect"],
    )
    check(
        "erfa_auroc_recomputed",
        close(erfa["materiality_auroc"], adjudication["erfa"]["materiality_auroc"]),
        erfa["materiality_auroc"],
    )
    check(
        "task_block_permutation_recomputed",
        close(permutation_p, adjudication["erfa"]["task_block_permutation_p"]),
        permutation_p,
    )
    labels = np.asarray([row["active"]["material"] for row in rows], dtype=bool)
    flags = np.asarray([row["high_risk"] for row in rows], dtype=bool)
    confusion = {
        "tp": int(np.sum(labels & flags)),
        "fn": int(np.sum(labels & ~flags)),
        "tn": int(np.sum(~labels & ~flags)),
        "fp": int(np.sum(~labels & flags)),
    }
    confusion["sensitivity"] = confusion["tp"] / (confusion["tp"] + confusion["fn"])
    confusion["specificity"] = confusion["tn"] / (confusion["tn"] + confusion["fp"])
    check(
        "operating_point_recomputed",
        all(
            close(confusion[key], adjudication["erfa"]["operating_point"][key])
            for key in confusion
        ),
        confusion,
    )
    recomputed_baselines = {key: metric(rows, key) for key in BASELINES}
    check(
        "all_baselines_recomputed",
        all(
            close(value[field], adjudication["baselines"][key][field])
            for key, value in recomputed_baselines.items()
            for field in ("spearman_absolute_effect", "materiality_auroc")
        ),
        recomputed_baselines,
    )
    family_diagnostics: dict[str, Any] = {}
    family_match = True
    for family in ("model_selector", "select_llm"):
        subset = [row for row in rows if row["selector_family"] == family]
        family_erfa = metric(subset, "erfa")
        family_baselines = {key: metric(subset, key) for key in BASELINES}
        best_key = max(BASELINES, key=lambda key: family_baselines[key]["materiality_auroc"])
        reported = adjudication["family"][family]
        family_match &= close(
            family_erfa["materiality_auroc"], reported["erfa"]["materiality_auroc"]
        )
        family_match &= best_key == reported["best_baseline"]
        family_diagnostics[family] = {
            "material_units": int(sum(row["active"]["material"] for row in subset)),
            "erfa": family_erfa,
            "best_baseline": best_key,
            "best_baseline_auroc": family_baselines[best_key]["materiality_auroc"],
        }
    check("cross_family_metrics_recomputed", family_match, family_diagnostics)
    bootstrap = task_bootstrap(rows)
    reported_bootstrap = adjudication["adaptive_value_task_bootstrap"]
    check(
        "task_bootstrap_recomputed",
        close(bootstrap["mean_difference"], reported_bootstrap["mean_difference"])
        and all(close(x, y) for x, y in zip(bootstrap["ci95"], reported_bootstrap["ci95"]))
        and close(bootstrap["median_difference"], reported_bootstrap["median_difference"])
        and close(bootstrap["nonpositive_fraction"], reported_bootstrap["nonpositive_fraction"]),
        bootstrap,
    )

    attack = [row for row in rows if row["kind"] in {"exact", "near"}]
    active_harm = float(
        sum(max(row["active"]["delta_normalized_cumulative_regret"], 0.0) for row in attack)
    )
    policy_harm = float(
        sum(max(row["selective_policy"]["delta_normalized_cumulative_regret"], 0.0) for row in attack)
    )
    reduction = (active_harm - policy_harm) / active_harm
    active_coverage = float(np.mean(~flags))
    diverse = [row for row in rows if row["kind"] == "diverse"]
    diverse_low = float(np.mean([not row["high_risk"] for row in diverse]))
    check(
        "fallback_recomputed",
        close(reduction, adjudication["selective_fallback"]["harmful_excess_reduction_fraction"])
        and close(active_coverage, adjudication["selective_fallback"]["active_coverage"]),
        {"reduction": reduction, "active_coverage": active_coverage},
    )
    check(
        "nonvacuity_recomputed",
        close(diverse_low, adjudication["nonvacuity"]["diverse_low_risk_fraction"]),
        diverse_low,
    )

    family_better = all(
        adjudication["family"][family]["erfa_minus_best_baseline_auroc"] > 0
        for family in ("model_selector", "select_llm")
    )
    gates = {
        "G1_magnitude": erfa["spearman_absolute_effect"] >= 0.60
        and permutation_p < 0.01,
        "G2_materiality": erfa["materiality_auroc"] >= 0.75
        and confusion["sensitivity"] >= 0.65
        and confusion["specificity"] >= 0.65,
        "G3_adaptive_value": bootstrap["ci95"][0] > 0,
        "G4_cross_selector": family_better,
        "G5_selective_fallback": reduction >= 0.50 and active_coverage >= 0.50,
        "G6_nonvacuity_control": diverse_low >= 1 / 3
        and erfa["materiality_auroc"]
        > recomputed_baselines["count_distance"]["materiality_auroc"],
    }
    if all(gates.values()):
        decision = "METHOD_PROMOTION_GO"
    elif gates["G1_magnitude"] and gates["G2_materiality"]:
        decision = "DIAGNOSTIC_ONLY_NO_GO"
    else:
        decision = "STOP_FRAGILITY_PIVOT"
    check("all_gates_recomputed", gates == adjudication["gates"], gates)
    check(
        "decision_recomputed",
        decision == adjudication["decision"] == result["decision"],
        decision,
    )

    failures = [item for item in checks if not item["passed"]]
    validation = {
        "manifest_id": "STEP61B_INDEPENDENT_VALIDATION_V1",
        "validator_independent_of_stage_b_runner_imports": True,
        "result_sha256": sha256_file(RESULT),
        "outcome_manifest_sha256": sha256_file(OUTCOME_MANIFEST),
        "execution_lock_sha256": sha256_file(EXECUTION_LOCK),
        "checks_passed": len(checks) - len(failures),
        "checks_total": len(checks),
        "failures": failures,
        "recomputed": {
            "erfa": {**erfa, "task_block_permutation_p": permutation_p},
            "operating_point": confusion,
            "baselines": recomputed_baselines,
            "family": family_diagnostics,
            "bootstrap": bootstrap,
            "fallback": {
                "harmful_excess_reduction_fraction": reduction,
                "active_coverage": active_coverage,
            },
            "diverse_low_risk_fraction": diverse_low,
            "gates": gates,
            "decision": decision,
        },
        "verdict": "VALIDATED" if not failures else "VALIDATION_FAILED",
    }
    OUT.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "verdict": validation["verdict"],
        "checks_passed": validation["checks_passed"],
        "checks_total": validation["checks_total"],
        "decision": decision,
        "family_material_units": {
            key: value["material_units"] for key, value in family_diagnostics.items()
        },
    }, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
