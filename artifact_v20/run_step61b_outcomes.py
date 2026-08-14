"""Locked Stage B outcome execution and preregistered ERFA gate adjudication."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
import urllib.parse
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

import audit_step61_freshness_feasibility as stage0
import prepare_step61a_blind_scenarios as scenario_prep
import run_step61a_blind_scores as stage_a
import step61a_blind_common as common


ROOT = stage0.ROOT
PROTOCOL = ROOT / "STEP61_ERFA_FRESH_CROSS_SELECTOR_PREREGISTRATION_2026-08-10.json"
SCENARIO_MANIFEST = ROOT / "STEP61A_BLIND_SCENARIO_MANIFEST_2026-08-10.json"
STAGE_A_RESULTS = ROOT / "STEP61A_BLIND_SCORE_RESULTS_2026-08-10.json"
STAGE_A_RISK = ROOT / "STEP61A_SEALED_HIGH_RISK_IDS_2026-08-10.json"
STAGE_A_OUTPUT_LOCK = ROOT / "STEP61A_BLIND_OUTPUT_LOCK_2026-08-10.json"
EXECUTION_LOCK = ROOT / "STEP61B_OUTCOME_EXECUTION_LOCK_2026-08-10.json"
RUNNER = ROOT / "run_step61b_outcomes.py"
OUTCOME_DIR = ROOT / "STEP61_OUTCOME_PACKAGES"
OUTCOME_MANIFEST = ROOT / "STEP61B_OUTCOME_SOURCE_MANIFEST_2026-08-10.json"
OUT = ROOT / "STEP61B_TRUE_EFFECT_RESULTS_2026-08-10.json"

HELM_BASE = (
    "https://storage.googleapis.com/crfm-helm-public/"
    "lite/benchmark_output/runs/v1.0.0/"
)
MATERIALITY = 0.01
PERMUTATIONS = 10_000
BOOTSTRAPS = 10_000
BASELINE_KEYS = (
    "count_distance",
    "first_query_change_fraction",
    "path_hamming",
    "path_jaccard",
    "static_mass_rank_displacement",
)


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
    if lock["manifest_id"] != "STEP61B_OUTCOME_EXECUTION_LOCK_V1":
        raise AssertionError("unexpected Stage B execution lock")
    paths = {
        "protocol": PROTOCOL,
        "scenario_manifest": SCENARIO_MANIFEST,
        "stage_a_results": STAGE_A_RESULTS,
        "stage_a_risk": STAGE_A_RISK,
        "stage_a_output_lock": STAGE_A_OUTPUT_LOCK,
        "stage_a_common": ROOT / "step61a_blind_common.py",
        "stage_a_runner": ROOT / "run_step61a_blind_scores.py",
        "stage_b_runner": RUNNER,
        "integrity_deviation": ROOT / "STEP61A_PRE_SCORE_INTEGRITY_DEVIATION_2026-08-10.md",
    }
    observed = {name: sha256_file(path) for name, path in paths.items()}
    if observed != lock["bound_sha256"]:
        raise AssertionError({"locked": lock["bound_sha256"], "observed": observed})
    if lock["outcomes_permitted_after_verification"] is not True:
        raise AssertionError("Stage B lock does not permit outcome execution")
    return lock


def helm_url(relative: str) -> str:
    return HELM_BASE + urllib.parse.quote(relative, safe="/:,@=-._")


def stat_name(stat: dict[str, Any]) -> str:
    value = stat.get("name")
    if isinstance(value, dict):
        return str(value.get("name", ""))
    return str(value)


def fetch_math_metric(
    scenario: str, model: str
) -> tuple[str, dict[str, float], dict[str, Any]]:
    relative = f"{scenario},model={model}/per_instance_stats.json"
    url = helm_url(relative)
    with urllib.request.urlopen(url, timeout=90) as response:
        payload = response.read()
    rows = json.loads(payload)
    values: dict[str, float] = {}
    for row in rows:
        key = f"{row['instance_id']}\x1f{int(row['train_trial_index'])}"
        matches = [
            stat
            for stat in row["stats"]
            if stat_name(stat) == "math_equiv_chain_of_thought"
        ]
        if len(matches) != 1:
            raise AssertionError(f"{scenario}:{model}:{key}: expected one MATH metric")
        value = float(matches[0]["mean"])
        if value not in (0.0, 1.0):
            raise AssertionError(f"{scenario}:{model}:{key}: nonbinary MATH metric {value}")
        if key in values:
            raise AssertionError(f"{scenario}:{model}: duplicate key")
        values[key] = value
    return model, values, {
        "url": url,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "rows": len(rows),
        "metric": "math_equiv_chain_of_thought",
        "raw_payload_persisted": False,
    }


def prepare_classifier_outcome(task: str, task_manifest: dict[str, Any]) -> dict[str, Any]:
    blind_path = ROOT / task_manifest["blind_package"]["path"]
    with np.load(blind_path, allow_pickle=False) as blind:
        responses = np.asarray(blind["responses"])
        candidate_ids = blind["candidate_ids"]
        instance_ids = blind["instance_ids"]
    source = ROOT / "external" / "model-selector" / "resources" / "datasets" / task / "oracle.npy"
    labels_raw = np.load(source, allow_pickle=False)
    labels = np.asarray(labels_raw).reshape(-1)
    if len(labels) != responses.shape[0]:
        raise AssertionError(f"{task}: outcome length mismatch")
    if not np.all(np.isfinite(labels)) or not np.array_equal(labels, np.rint(labels)):
        raise AssertionError(f"{task}: invalid classifier outcome values")
    labels = np.asarray(np.rint(labels), dtype=np.int32)
    rewards = (responses == labels[:, None]).astype(np.float32)
    output = OUTCOME_DIR / f"{task}.npz"
    np.savez_compressed(
        output,
        candidate_rewards=rewards,
        reference_labels=labels,
        candidate_ids=candidate_ids,
        instance_ids=instance_ids,
        task_id=np.asarray(task),
        selector_family=np.asarray("model_selector"),
    )
    return {
        "task_id": task,
        "selector_family": "model_selector",
        "shape": list(rewards.shape),
        "source": {
            "path": str(source.relative_to(ROOT)),
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        },
        "outcome_package": {
            "path": str(output.relative_to(ROOT)),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
        "alignment": {
            "instances": len(labels),
            "candidates": responses.shape[1],
            "exact_length_match": True,
        },
    }


def prepare_math_outcome(subject: str, task_manifest: dict[str, Any]) -> dict[str, Any]:
    scenario = stage0.math_scenario(subject)
    blind_path = ROOT / task_manifest["blind_package"]["path"]
    with np.load(blind_path, allow_pickle=False) as blind:
        candidate_ids = [str(value) for value in blind["candidate_ids"]]
        instance_ids = [str(value) for value in blind["instance_ids"]]
    if candidate_ids != stage0.MODELS:
        raise AssertionError(f"{scenario}: candidate roster mismatch")
    fetched: dict[str, dict[str, float]] = {}
    sources: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {
            executor.submit(fetch_math_metric, scenario, model): model
            for model in candidate_ids
        }
        for future in as_completed(futures):
            model, values, source = future.result()
            fetched[model] = values
            sources[model] = source
    expected_keys = set(instance_ids)
    for model in candidate_ids:
        if set(fetched[model]) != expected_keys:
            missing = sorted(expected_keys - set(fetched[model]))
            extra = sorted(set(fetched[model]) - expected_keys)
            raise AssertionError(
                f"{scenario}:{model}: alignment mismatch missing={len(missing)} extra={len(extra)}"
            )
    rewards = np.asarray(
        [[fetched[model][key] for model in candidate_ids] for key in instance_ids],
        dtype=np.float32,
    )
    output = OUTCOME_DIR / f"math_{subject}.npz"
    np.savez_compressed(
        output,
        candidate_rewards=rewards,
        candidate_ids=np.asarray(candidate_ids),
        instance_ids=np.asarray(instance_ids),
        task_id=np.asarray(scenario),
        selector_family=np.asarray("select_llm"),
    )
    return {
        "task_id": scenario,
        "subject": subject,
        "selector_family": "select_llm",
        "shape": list(rewards.shape),
        "sources": {model: sources[model] for model in candidate_ids},
        "outcome_package": {
            "path": str(output.relative_to(ROOT)),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
        "alignment": {
            "instances": len(instance_ids),
            "candidates": len(candidate_ids),
            "exact_key_set_match_all_models": True,
        },
    }


def prepare_outcomes() -> dict[str, Any]:
    OUTCOME_DIR.mkdir(parents=True, exist_ok=True)
    scenario_manifest = read_json(SCENARIO_MANIFEST)
    classifier = {
        task: prepare_classifier_outcome(task, scenario_manifest["tasks"][task])
        for task in stage0.MODEL_TASKS
    }
    math_tasks = {
        subject: prepare_math_outcome(
            subject, scenario_manifest["tasks"][stage0.math_scenario(subject)]
        )
        for subject in stage0.MATH_SUBJECTS
    }
    result = {
        "manifest_id": "STEP61B_OUTCOME_SOURCE_MANIFEST_V1",
        "execution_lock_sha256": sha256_file(EXECUTION_LOCK),
        "classifier_tasks": classifier,
        "helm_math_tasks": math_tasks,
        "counts": {
            "classifier_tasks": len(classifier),
            "helm_math_tasks": len(math_tasks),
            "total_tasks": len(classifier) + len(math_tasks),
        },
        "alignment_passed": True,
        "raw_HELM_payloads_persisted": False,
    }
    OUTCOME_MANIFEST.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def outcome_package_path(task_id: str, outcome_manifest: dict[str, Any]) -> Path:
    if task_id in outcome_manifest["classifier_tasks"]:
        row = outcome_manifest["classifier_tasks"][task_id]
    else:
        subject = task_id.split("subject=", 1)[1].split(",", 1)[0]
        row = outcome_manifest["helm_math_tasks"][subject]
    return ROOT / row["outcome_package"]["path"]


def load_outcome_context(task_id: str) -> dict[str, Any]:
    scenario_manifest = read_json(SCENARIO_MANIFEST)
    outcome_manifest = read_json(OUTCOME_MANIFEST)
    task_manifest = scenario_manifest["tasks"][task_id]
    blind_context = stage_a.load_context(task_id, task_manifest)
    package_path = outcome_package_path(task_id, outcome_manifest)
    with np.load(package_path, allow_pickle=False) as package:
        full_rewards = np.asarray(package["candidate_rewards"], dtype=np.float32)
        labels = (
            np.asarray(package["reference_labels"], dtype=np.int32)
            if "reference_labels" in package.files
            else None
        )
        observed_task = str(package["task_id"].item())
    if observed_task != task_id or full_rewards.shape != blind_context["full_responses"].shape:
        raise AssertionError(f"{task_id}: outcome package identity/shape mismatch")
    core_indices = [int(value) for value in task_manifest["candidate_partition"]["core_original_indices"]]
    clean_true = full_rewards[:, core_indices]
    return {
        **blind_context,
        "task_manifest": task_manifest,
        "full_true_rewards": full_rewards,
        "clean_true_rewards": clean_true,
        "reference_labels": labels,
    }


def llm_response_reward_maps(
    responses: np.ndarray, rewards: np.ndarray
) -> list[dict[str, float]]:
    maps: list[dict[str, float]] = []
    for item in range(responses.shape[0]):
        mapping: dict[str, float] = {}
        for response, reward in zip(responses[item], rewards[item]):
            key = str(response)
            value = float(reward)
            if key in mapping and not math.isclose(mapping[key], value, abs_tol=1e-12):
                raise AssertionError(
                    f"identical canonical response has inconsistent HELM metric at item {item}"
                )
            mapping[key] = value
        maps.append(mapping)
    return maps


def scenario_true_rewards(
    context: dict[str, Any], scenario: dict[str, Any], llm_maps: list[dict[str, float]] | None
) -> np.ndarray:
    responses = scenario["responses"]
    if context["family"] == "model_selector":
        labels = context["reference_labels"]
        if labels is None:
            raise AssertionError("classifier labels unavailable")
        return (responses == labels[:, None]).astype(np.float32)
    assert llm_maps is not None
    core_count = context["core_count"]
    additions = responses[:, core_count:]
    added_rewards = np.empty(additions.shape, dtype=np.float32)
    for item in range(additions.shape[0]):
        for candidate in range(additions.shape[1]):
            response = str(additions[item, candidate])
            if response not in llm_maps[item]:
                raise AssertionError("near/diverse response absent from frozen LLM source roster")
            added_rewards[item, candidate] = llm_maps[item][response]
    return np.concatenate([context["clean_true_rewards"], added_rewards], axis=1)


def fixed_query_path(seed: int, pool: np.ndarray, budget: int) -> np.ndarray:
    return np.asarray(
        random.Random(seed + scenario_prep.FALLBACK_SEED_OFFSET).sample(
            pool.tolist(), budget
        ),
        dtype=np.int64,
    )


def run_fixed_queries(
    feedback: np.ndarray,
    core_count: int,
    pool: np.ndarray,
    queries: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    entries = feedback.shape[1]
    selection_rng = np.random.Generator(
        np.random.PCG64(seed + common.SELECTION_RNG_OFFSET)
    )
    pool_quality = np.asarray(feedback[pool], dtype=np.float64).mean(axis=0)
    best_core = float(np.max(pool_quality[:core_count]))
    cumulative = np.zeros(entries, dtype=np.float64)
    selected = np.full(len(queries), -1, dtype=np.int64)
    regret = np.zeros(len(queries), dtype=np.float64)
    for step, query in enumerate(queries):
        cumulative += feedback[int(query)]
        maximum = float(np.max(cumulative))
        tied = np.flatnonzero(cumulative == maximum)
        choice = int(tied[int(selection_rng.integers(0, len(tied)))])
        selected[step] = choice
        regret[step] = best_core - float(pool_quality[choice])
    return {
        "query_path": np.asarray(queries, dtype=np.int64),
        "selected_path": selected,
        "regret_path": regret,
        "cumulative_regret": float(np.sum(regret)),
        "final_regret": float(regret[-1]),
    }


def run_true_active(
    task_id: str,
    context: dict[str, Any],
    codes: np.ndarray,
    rewards: np.ndarray,
) -> list[dict[str, Any]]:
    return stage_a.run_condition(
        task_id,
        context["family"],
        codes,
        rewards,
        context["core_count"],
        context["pools"],
        context["seeds"],
        context["budget"],
        context["class_count"],
    )


def score_outcome_task(task_id: str) -> dict[str, Any]:
    started = time.perf_counter()
    context = load_outcome_context(task_id)
    stage_a_results = read_json(STAGE_A_RESULTS)
    stage_a_rows = {
        row["scenario_id"]: row for row in stage_a_results["tasks"][task_id]["scenarios"]
    }
    clean_codes = context["clean_codes"]
    clean_true = context["clean_true_rewards"]
    clean_active = run_true_active(task_id, context, clean_codes, clean_true)
    fallback_paths = [
        fixed_query_path(seed, pool, context["budget"])
        for seed, pool in zip(context["seeds"], context["pools"])
    ]
    expected_fallback = common.digest_array(np.stack(fallback_paths))
    if expected_fallback != context["task_manifest"]["paired_design"]["fallback_query_matrix_digest"]:
        raise AssertionError(f"{task_id}: fallback path digest mismatch")
    clean_fallback = [
        run_fixed_queries(clean_true, context["core_count"], pool, path, seed)
        for seed, pool, path in zip(context["seeds"], context["pools"], fallback_paths)
    ]
    llm_maps = (
        llm_response_reward_maps(context["full_responses"], context["full_true_rewards"])
        if context["family"] == "select_llm"
        else None
    )
    rows: list[dict[str, Any]] = []
    max_decomposition_error = 0.0
    for slot, scenario in enumerate(context["scenarios"]):
        true_rewards = scenario_true_rewards(context, scenario, llm_maps)
        if context["family"] == "model_selector":
            codes = np.asarray(scenario["responses"], dtype=np.int32)
        else:
            codes, _ = common.encode_responses(scenario["responses"])
        proposed_active = run_true_active(task_id, context, codes, true_rewards)
        proposed_fallback = [
            run_fixed_queries(true_rewards, context["core_count"], pool, path, seed)
            for seed, pool, path in zip(
                context["seeds"], context["pools"], fallback_paths
            )
        ]
        proposed_hybrid = [
            run_fixed_queries(
                true_rewards,
                context["core_count"],
                pool,
                clean_run["query_path"],
                seed,
            )
            for seed, pool, clean_run in zip(
                context["seeds"], context["pools"], clean_active
            )
        ]
        active_delta = np.asarray(
            [b["cumulative_regret"] - a["cumulative_regret"] for a, b in zip(clean_active, proposed_active)],
            dtype=np.float64,
        )
        fallback_delta = np.asarray(
            [b["cumulative_regret"] - a["cumulative_regret"] for a, b in zip(clean_fallback, proposed_fallback)],
            dtype=np.float64,
        )
        direct_delta = np.asarray(
            [b["cumulative_regret"] - a["cumulative_regret"] for a, b in zip(clean_active, proposed_hybrid)],
            dtype=np.float64,
        )
        path_delta = active_delta - direct_delta
        reconstruction = active_delta - direct_delta - path_delta
        max_decomposition_error = max(
            max_decomposition_error, float(np.max(np.abs(reconstruction)))
        )
        stage_a_row = stage_a_rows[scenario["scenario_id"]]
        high_risk = bool(stage_a_row["high_risk"])
        policy_delta = fallback_delta if high_risk else active_delta
        budget = context["budget"]
        active_effect = float(np.mean(active_delta) / budget)
        fallback_effect = float(np.mean(fallback_delta) / budget)
        policy_effect = float(np.mean(policy_delta) / budget)
        core_count = context["core_count"]
        added_quality = np.mean(true_rewards[:, core_count:], axis=0)
        quality: dict[str, Any] = {
            "added_mean_accuracy": [float(value) for value in added_quality],
        }
        if scenario["kind"] in {"exact", "near"}:
            parent = int(scenario["root_local"])
            parent_quality = float(np.mean(clean_true[:, parent]))
            gaps = [float(value - parent_quality) for value in added_quality]
            quality.update(
                {
                    "parent_mean_accuracy": parent_quality,
                    "added_minus_parent_gaps": gaps,
                    "all_within_one_point": bool(
                        all(abs(value) <= 0.01 + 1e-12 for value in gaps)
                    ),
                }
            )
        addition_exact = [distance <= 1e-15 for distance in scenario["nearest_core_distances"]]
        incoming_exact_quotient_effect = (
            0.0 if all(addition_exact) else active_effect
        )
        lineage_effect = (
            0.0 if scenario["kind"] in {"exact", "near"} else active_effect
        )
        query_hamming = np.asarray(
            [
                np.mean(a["query_path"] != b["query_path"])
                for a, b in zip(clean_active, proposed_active)
            ],
            dtype=np.float64,
        )
        rows.append(
            {
                "scenario_slot": slot,
                "task_id": task_id,
                "selector_family": context["family"],
                "scenario_id": scenario["scenario_id"],
                "scenario_sha256": scenario["scenario_sha256"],
                "kind": scenario["kind"],
                "aliases": int(scenario["aliases"]),
                "nominal_rho": scenario.get("nominal_rho"),
                "high_risk": high_risk,
                "family_blind_rank": int(stage_a_row["family_blind_rank"]),
                "blind_scores": stage_a_row["scores"],
                "active": {
                    "paired_delta_cumulative_regret": [float(value) for value in active_delta],
                    "delta_normalized_cumulative_regret": active_effect,
                    "absolute_effect": abs(active_effect),
                    "material": abs(active_effect) >= MATERIALITY,
                    "mean_true_query_hamming": float(np.mean(query_hamming)),
                },
                "fixed_fallback": {
                    "paired_delta_cumulative_regret": [float(value) for value in fallback_delta],
                    "delta_normalized_cumulative_regret": fallback_effect,
                },
                "selective_policy": {
                    "used_fallback": high_risk,
                    "paired_delta_cumulative_regret": [float(value) for value in policy_delta],
                    "delta_normalized_cumulative_regret": policy_effect,
                },
                "decomposition": {
                    "direct_registry_mean_normalized": float(np.mean(direct_delta) / budget),
                    "adaptive_path_mean_normalized": float(np.mean(path_delta) / budget),
                    "max_abs_reconstruction_error": float(np.max(np.abs(reconstruction))),
                },
                "quality": quality,
                "comparators": {
                    "incoming_exact_response_quotient_effect": incoming_exact_quotient_effect,
                    "authenticated_lineage_collapse_effect": lineage_effect,
                    "comparator_scope": "Incoming additions only; the unchanged clean core is not repartitioned.",
                },
            }
        )
    return {
        "task_id": task_id,
        "selector_family": context["family"],
        "budget": context["budget"],
        "pool_size": int(context["pools"].shape[1]),
        "rows": rows,
        "max_abs_decomposition_error": max_decomposition_error,
        "runtime_seconds": time.perf_counter() - started,
    }


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rx = rankdata(x, method="average")
    ry = rankdata(y, method="average")
    if np.allclose(rx, rx[0]) or np.allclose(ry, ry[0]):
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return 0.5
    ranks = rankdata(scores, method="average")
    rank_sum = float(np.sum(ranks[labels == 1]))
    return float(
        (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    )


def classification_at_lock(labels: np.ndarray, flags: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=bool)
    flags = np.asarray(flags, dtype=bool)
    tp = int(np.sum(labels & flags))
    fn = int(np.sum(labels & ~flags))
    tn = int(np.sum(~labels & ~flags))
    fp = int(np.sum(~labels & flags))
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


def block_permutation_p(rows: list[dict[str, Any]], observed: float) -> float:
    if observed <= 0:
        return 1.0
    task_ids = list(dict.fromkeys(row["task_id"] for row in rows))
    if len(task_ids) != 14:
        raise AssertionError("task-block permutation requires 14 tasks")
    blocks = {
        task: [index for index, row in enumerate(rows) if row["task_id"] == task]
        for task in task_ids
    }
    if any(len(indices) != 30 for indices in blocks.values()):
        raise AssertionError("task-block permutation requires 30 cells per task")
    x = np.asarray([row["blind_scores"]["erfa"] for row in rows], dtype=np.float64)
    y = np.asarray([row["active"]["absolute_effect"] for row in rows], dtype=np.float64)
    rx = rankdata(x, method="average")
    ry = rankdata(y, method="average")
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denominator = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    if denominator == 0:
        return 1.0
    families = {
        family: [task for task in task_ids if next(row for row in rows if row["task_id"] == task)["selector_family"] == family]
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
        null = float(np.dot(rx, permuted) / denominator)
        extreme += int(null >= observed - 1e-15)
    return float((extreme + 1) / (PERMUTATIONS + 1))


def metric_summary(rows: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    scores = np.asarray([row["blind_scores"][score_key] for row in rows], dtype=np.float64)
    effects = np.asarray([row["active"]["absolute_effect"] for row in rows], dtype=np.float64)
    labels = np.asarray([row["active"]["material"] for row in rows], dtype=np.int64)
    return {
        "score": score_key,
        "units": len(rows),
        "spearman_absolute_effect": safe_spearman(scores, effects),
        "materiality_auroc": binary_auc(labels, scores),
    }


def bootstrap_adaptive_value(rows: list[dict[str, Any]]) -> dict[str, Any]:
    task_ids = list(dict.fromkeys(row["task_id"] for row in rows))
    task_indices = {
        task: np.asarray([index for index, row in enumerate(rows) if row["task_id"] == task], dtype=np.int64)
        for task in task_ids
    }
    labels = np.asarray([row["active"]["material"] for row in rows], dtype=np.int64)
    scores = {
        "erfa": np.asarray([row["blind_scores"]["erfa"] for row in rows], dtype=np.float64),
        **{
            key: np.asarray([row["blind_scores"][key] for row in rows], dtype=np.float64)
            for key in BASELINE_KEYS
        },
    }
    rng = np.random.default_rng(61300)
    differences = np.zeros(BOOTSTRAPS, dtype=np.float64)
    for repetition in range(BOOTSTRAPS):
        sampled = rng.integers(0, len(task_ids), size=len(task_ids))
        indices = np.concatenate([task_indices[task_ids[int(value)]] for value in sampled])
        y = labels[indices]
        if len(np.unique(y)) < 2:
            differences[repetition] = 0.0
            continue
        erfa_auc = binary_auc(y, scores["erfa"][indices])
        best = max(binary_auc(y, scores[key][indices]) for key in BASELINE_KEYS)
        differences[repetition] = erfa_auc - best
    quantiles = np.quantile(differences, [0.025, 0.5, 0.975])
    return {
        "repetitions": BOOTSTRAPS,
        "mean_difference": float(np.mean(differences)),
        "ci95": [float(quantiles[0]), float(quantiles[2])],
        "median_difference": float(quantiles[1]),
        "nonpositive_fraction": float(np.mean(differences <= 0)),
    }


def adjudicate(tasks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = [row for task in tasks.values() for row in task["rows"]]
    if len(rows) != 420:
        raise AssertionError("adjudication requires 420 rows")
    effects = np.asarray([row["active"]["absolute_effect"] for row in rows])
    material = np.asarray([row["active"]["material"] for row in rows], dtype=np.int64)
    flags = np.asarray([row["high_risk"] for row in rows], dtype=np.int64)
    erfa = metric_summary(rows, "erfa")
    erfa["task_block_permutation_p"] = block_permutation_p(
        rows, erfa["spearman_absolute_effect"]
    )
    erfa["operating_point"] = classification_at_lock(material, flags)
    baselines = {key: metric_summary(rows, key) for key in BASELINE_KEYS}
    family: dict[str, Any] = {}
    for family_name in ("model_selector", "select_llm"):
        subset = [row for row in rows if row["selector_family"] == family_name]
        erfa_family = metric_summary(subset, "erfa")
        baseline_family = {key: metric_summary(subset, key) for key in BASELINE_KEYS}
        best_key = max(
            BASELINE_KEYS,
            key=lambda key: baseline_family[key]["materiality_auroc"],
        )
        family[family_name] = {
            "erfa": erfa_family,
            "baselines": baseline_family,
            "best_baseline": best_key,
            "best_baseline_auroc": baseline_family[best_key]["materiality_auroc"],
            "erfa_minus_best_baseline_auroc": erfa_family["materiality_auroc"]
            - baseline_family[best_key]["materiality_auroc"],
        }
    bootstrap = bootstrap_adaptive_value(rows)
    attack = [row for row in rows if row["kind"] in {"exact", "near"}]
    active_harm = float(
        sum(max(row["active"]["delta_normalized_cumulative_regret"], 0.0) for row in attack)
    )
    policy_harm = float(
        sum(max(row["selective_policy"]["delta_normalized_cumulative_regret"], 0.0) for row in attack)
    )
    reduction = (active_harm - policy_harm) / active_harm if active_harm > 0 else 0.0
    active_coverage = float(np.mean(~flags.astype(bool)))
    diverse = [row for row in rows if row["kind"] == "diverse"]
    diverse_low_fraction = float(np.mean([not row["high_risk"] for row in diverse]))
    best_overall_key = max(
        BASELINE_KEYS, key=lambda key: baselines[key]["materiality_auroc"]
    )
    gates = {
        "G1_magnitude": bool(
            erfa["spearman_absolute_effect"] >= 0.60
            and erfa["task_block_permutation_p"] < 0.01
        ),
        "G2_materiality": bool(
            erfa["materiality_auroc"] >= 0.75
            and erfa["operating_point"]["sensitivity"] >= 0.65
            and erfa["operating_point"]["specificity"] >= 0.65
        ),
        "G3_adaptive_value": bool(bootstrap["ci95"][0] > 0.0),
        "G4_cross_selector": bool(
            all(row["erfa_minus_best_baseline_auroc"] > 0.0 for row in family.values())
        ),
        "G5_selective_fallback": bool(
            reduction >= 0.50 and active_coverage >= 0.50
        ),
        "G6_nonvacuity_control": bool(
            diverse_low_fraction >= 1.0 / 3.0
            and erfa["materiality_auroc"] > baselines["count_distance"]["materiality_auroc"]
        ),
    }
    if all(gates.values()):
        decision = "METHOD_PROMOTION_GO"
    elif gates["G1_magnitude"] and gates["G2_materiality"]:
        decision = "DIAGNOSTIC_ONLY_NO_GO"
    else:
        decision = "STOP_FRAGILITY_PIVOT"
    return {
        "units": len(rows),
        "material_units": int(np.sum(material)),
        "nonmaterial_units": int(len(material) - np.sum(material)),
        "effect_distribution": {
            "minimum": float(np.min(effects)),
            "median": float(np.median(effects)),
            "maximum": float(np.max(effects)),
        },
        "erfa": erfa,
        "baselines": baselines,
        "best_overall_baseline": best_overall_key,
        "family": family,
        "adaptive_value_task_bootstrap": bootstrap,
        "selective_fallback": {
            "active_harmful_exact_near_excess": active_harm,
            "policy_harmful_exact_near_excess": policy_harm,
            "harmful_excess_reduction_fraction": float(reduction),
            "active_coverage": active_coverage,
        },
        "nonvacuity": {
            "diverse_controls": len(diverse),
            "diverse_low_risk": int(sum(not row["high_risk"] for row in diverse)),
            "diverse_low_risk_fraction": diverse_low_fraction,
        },
        "gates": gates,
        "gates_passed": int(sum(gates.values())),
        "decision": decision,
    }


def self_test() -> dict[str, Any]:
    labels = np.asarray([0, 0, 1, 1])
    perfect = binary_auc(labels, np.asarray([0.0, 0.1, 0.8, 0.9]))
    reverse = binary_auc(labels, np.asarray([0.9, 0.8, 0.1, 0.0]))
    tied = binary_auc(labels, np.ones(4))
    spear = safe_spearman(np.arange(10), np.arange(10) ** 2)
    operating = classification_at_lock(labels, np.asarray([0, 0, 1, 1]))
    if not math.isclose(perfect, 1.0) or not math.isclose(reverse, 0.0):
        raise AssertionError("AUROC self-test failed")
    if not math.isclose(tied, 0.5) or not math.isclose(spear, 1.0):
        raise AssertionError("rank self-test failed")
    if operating["sensitivity"] != 1.0 or operating["specificity"] != 1.0:
        raise AssertionError("operating-point self-test failed")
    feedback = np.asarray(
        [[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.float32
    )
    fixed = run_fixed_queries(
        feedback, 2, np.arange(4), np.asarray([0, 2, 1]), 61000
    )
    if len(fixed["query_path"]) != 3 or not np.isfinite(fixed["cumulative_regret"]):
        raise AssertionError("fixed-query self-test failed")
    return {
        "perfect_auc": perfect,
        "reverse_auc": reverse,
        "tied_auc": tied,
        "monotone_spearman": spear,
        "operating_point": operating,
        "fixed_query_steps": len(fixed["query_path"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return
    lock = verify_lock()
    started = time.perf_counter()
    outcome_manifest = prepare_outcomes()
    print(json.dumps({"outcome_alignment": "PASS", "tasks": outcome_manifest["counts"]["total_tasks"]}), flush=True)
    scenario_manifest = read_json(SCENARIO_MANIFEST)
    task_ids = list(scenario_manifest["tasks"])
    tasks: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(score_outcome_task, task_id): task_id for task_id in task_ids}
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
    adjudication = adjudicate(tasks)
    result = {
        "manifest_id": "STEP61B_TRUE_EFFECT_RESULTS_V1",
        "protocol_sha256": sha256_file(PROTOCOL),
        "scenario_manifest_sha256": sha256_file(SCENARIO_MANIFEST),
        "stage_a_results_sha256": sha256_file(STAGE_A_RESULTS),
        "stage_a_risk_sha256": sha256_file(STAGE_A_RISK),
        "stage_a_output_lock_sha256": sha256_file(STAGE_A_OUTPUT_LOCK),
        "execution_lock_sha256": sha256_file(EXECUTION_LOCK),
        "outcome_manifest_sha256": sha256_file(OUTCOME_MANIFEST),
        "self_test": self_test(),
        "tasks": tasks,
        "adjudication": adjudication,
        "integrity": {
            "all_420_cells_retained": sum(len(task["rows"]) for task in tasks.values()) == 420,
            "stage_a_risk_ids_unchanged": True,
            "limited_pre_score_aggregate_accuracy_exposure_disclosed": True,
            "no_post_outcome_protocol_change": True,
        },
        "runtime_seconds": time.perf_counter() - started,
        "decision": adjudication["decision"],
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": result["decision"],
        "gates": adjudication["gates"],
        "erfa": adjudication["erfa"],
        "runtime_seconds": round(result["runtime_seconds"], 3),
        "output": str(OUT),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
