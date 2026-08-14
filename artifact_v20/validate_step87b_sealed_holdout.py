"""Independent reconstruction of Step 87B paths, effects, statistics, and gates."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "STEP87B_REFERENCE_FREE_OPEN_ENDED_PREREGISTRATION_2026-08-12.json"
LOCK = ROOT / "STEP87B_STAGEB_EXECUTION_LOCK_2026-08-12.json"
ATTACK = ROOT / "STEP87B_FROZEN_REFERENCE_FREE_ATTACK_2026-08-12.json"
SEALED = ROOT / "external_data" / "step87b_sealed" / "STEP87B_SEALED_HOLDOUT_OUTCOMES_2026-08-12.json"
RESULTS = ROOT / "STEP87B_REFERENCE_FREE_OPEN_ENDED_RESULTS_2026-08-12.json"
RAW = ROOT / "STEP87B_REFERENCE_FREE_OPEN_ENDED_RAW_2026-08-12.npz"
OUT = ROOT / "STEP87B_INDEPENDENT_VALIDATION_2026-08-12.json"
RUNNER = ROOT / "run_step87b_sealed_holdout.py"
DEPENDENCIES = {
    "step87b_openended_common.py": ROOT / "step87b_openended_common.py",
    "run_step80_main_t2a_v2_transfer.py": ROOT / "run_step80_main_t2a_v2_transfer.py",
    "run_step54_matrix_free_realism_audit.py": ROOT / "run_step54_matrix_free_realism_audit.py",
    "run_step46_clone_robust_soft_weighting.py": ROOT / "run_step46_clone_robust_soft_weighting.py",
    "run_step50_natural_alias_audit.py": ROOT / "run_step50_natural_alias_audit.py",
    "run_step29_locked_phase_a.py": ROOT / "run_step29_locked_phase_a.py",
}
TASKS = (
    "narrativeqa",
    "naturalqa_closed",
    "naturalqa_open",
    "wmt_cs_en",
    "wmt_de_en",
    "wmt_fr_en",
    "wmt_hi_en",
    "wmt_ru_en",
)
ALIASES = 4
SEEDS = tuple(range(88400, 89400))
TOL = 1e-12


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def digest_json(value: Any) -> str:
    return hashlib.sha256(compact_json(value)).hexdigest()


def opaque_token(task: str, alias: int) -> str:
    task_digest = hashlib.sha256(task.encode("utf-8")).hexdigest()[:16].upper()
    return f"ZZZQXVSTEP87B{task_digest}ALIAS{alias}ABSTAIN"


def response_vector_hash(row: np.ndarray) -> str:
    return hashlib.sha256(
        json.dumps(list(row), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def canonicalize(
    responses: np.ndarray, utilities: np.ndarray, parents: np.ndarray, labels: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hashes = [response_vector_hash(row) for row in responses]
    order = sorted(
        range(len(labels)),
        key=lambda index: (
            hashes[index],
            int(parents[index]),
            hashlib.sha256(labels[index].encode("utf-8")).hexdigest(),
            labels[index],
        ),
    )
    responses = responses[order]
    utilities = utilities[order]
    parents = parents[order]
    queries, entries = responses.shape[1], responses.shape[0]
    codes = np.zeros((queries, entries), dtype=np.int16)
    for query in range(queries):
        values = sorted(
            set(str(value) for value in responses[:, query]),
            key=lambda value: (hashlib.sha256(value.encode("utf-8")).hexdigest(), value.encode("utf-8")),
        )
        mapping = {value: index for index, value in enumerate(values)}
        for entry in range(entries):
            codes[query, entry] = mapping[str(responses[entry, query])]
    return codes, utilities, parents


def independent_active(
    responses: np.ndarray,
    utilities: np.ndarray,
    parents: np.ndarray,
    labels: list[str],
    original: np.ndarray,
    pools: np.ndarray,
    budget: int,
    tau: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    codes, utilities, parents = canonicalize(responses, utilities, parents, labels)
    runs = len(pools)
    queries_out = np.full((runs, budget), -1, dtype=np.int64)
    roots = np.full((runs, budget), -1, dtype=np.int64)
    regrets = np.zeros((runs, budget), dtype=np.float64)
    entries = len(parents)
    for run_index, pool in enumerate(pools):
        original_scores = original[:, pool].mean(axis=1)
        best_original = float(original_scores.max())
        pool_codes = codes[pool]
        flat_bins = (
            np.arange(len(pool), dtype=np.int64)[:, None] * entries + pool_codes
        ).ravel()
        active = np.ones(len(pool), dtype=bool)
        cumulative = np.zeros(entries, dtype=np.float64)
        for step in range(budget):
            shifted = cumulative / tau
            shifted -= shifted.max()
            exponentials = np.exp(shifted)
            posterior = exponentials / math.fsum(float(value) for value in exponentials)
            masses = np.bincount(
                flat_bins,
                weights=np.tile(posterior, len(pool)),
                minlength=len(pool) * entries,
            ).reshape(len(pool), entries)
            ordered = np.sort(masses, axis=1)
            acquisition = np.sum(ordered * ordered, axis=1, dtype=np.float64)
            acquisition[~active] = np.inf
            minimum = float(acquisition.min())
            tied_positions = np.flatnonzero(acquisition == minimum)
            selected_query = min(int(pool[int(position)]) for position in tied_positions)
            position = int(np.flatnonzero(pool == selected_query)[0])
            active[position] = False
            queries_out[run_index, step] = selected_query
            cumulative += utilities[:, selected_query]
            tied_entries = np.flatnonzero(cumulative == float(cumulative.max()))
            root = min(int(parents[int(entry)]) for entry in tied_entries)
            roots[run_index, step] = root
            regrets[run_index, step] = best_original - original_scores[root]
    return queries_out, roots, regrets


def independent_fixed(
    utilities: np.ndarray,
    parents: np.ndarray,
    labels: list[str],
    responses: np.ndarray,
    original: np.ndarray,
    pools: np.ndarray,
    queries: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    _, utilities, parents = canonicalize(responses, utilities, parents, labels)
    roots = np.full_like(queries, -1)
    regrets = np.zeros(queries.shape, dtype=np.float64)
    for run_index, pool in enumerate(pools):
        original_scores = original[:, pool].mean(axis=1)
        best = float(original_scores.max())
        cumulative = np.zeros(len(parents), dtype=np.float64)
        for step, query in enumerate(queries[run_index]):
            cumulative += utilities[:, int(query)]
            tied = np.flatnonzero(cumulative == float(cumulative.max()))
            root = min(int(parents[int(entry)]) for entry in tied)
            roots[run_index, step] = root
            regrets[run_index, step] = best - original_scores[root]
    return roots, regrets


def pools_from_seeds(queries: int, pool_size: int) -> np.ndarray:
    return np.asarray(
        [sorted(random.Random(seed).sample(range(queries), pool_size)) for seed in SEEDS],
        dtype=np.int64,
    )


def fixed_queries(pools: np.ndarray, budget: int, task_index: int) -> np.ndarray:
    rows = []
    for run_index, pool in enumerate(pools):
        rng = random.Random(487000 + task_index * 10000 + run_index)
        rows.append(rng.sample([int(value) for value in pool], budget))
    return np.asarray(rows, dtype=np.int64)


def bootstrap_ci(values: np.ndarray, seed: int, repetitions: int = 10000) -> list[float]:
    if np.allclose(values, values[0]):
        return [float(values[0]), float(values[0])]
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    cursor = 0
    while cursor < repetitions:
        count = min(500, repetitions - cursor)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[cursor : cursor + count] = values[indices].mean(axis=1)
        cursor += count
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def sign_flip(values: np.ndarray, seed: int, repetitions: int = 10000) -> float:
    if np.allclose(values, 0.0):
        return 1.0
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    extreme = 0
    cursor = 0
    while cursor < repetitions:
        count = min(500, repetitions - cursor)
        signs = rng.integers(0, 2, size=(count, len(values)), dtype=np.int8) * 2 - 1
        estimates = (signs * values[None, :]).mean(axis=1)
        extreme += int(np.sum(estimates >= observed - 1e-15))
        cursor += count
    return float((extreme + 1.0) / (repetitions + 1.0))


def bh_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    count = len(ordered)
    output: dict[str, float] = {}
    running = 1.0
    for reverse_rank, name in enumerate(reversed(ordered), start=1):
        rank = count - reverse_rank + 1
        candidate = min(1.0, float(pvalues[name]) * count / rank)
        running = min(running, candidate)
        output[name] = float(running)
    return output


def summary(values: np.ndarray, seed: int) -> dict[str, Any]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci95": bootstrap_ci(values, seed),
        "one_sided_sign_flip_p": sign_flip(values, seed + 1),
        "positive_fraction": float(np.mean(values > TOL)),
        "negative_fraction": float(np.mean(values < -TOL)),
        "zero_fraction": float(np.mean(np.abs(values) <= TOL)),
    }


def close_record(left: dict[str, Any], right: dict[str, Any]) -> bool:
    for key in ("mean", "median", "positive_fraction", "negative_fraction", "zero_fraction", "one_sided_sign_flip_p"):
        if not math.isclose(float(left[key]), float(right[key]), rel_tol=0.0, abs_tol=1e-12):
            return False
    return bool(np.allclose(left["ci95"], right["ci95"], rtol=0.0, atol=1e-12))


def close_quality(left: dict[str, Any], right: dict[str, Any]) -> bool:
    scalar_keys = ("parent_mean_utility", "minimum_gap", "maximum_gap")
    if any(
        not math.isclose(float(left[key]), float(right[key]), rel_tol=0.0, abs_tol=1e-12)
        for key in scalar_keys
    ):
        return False
    if not np.allclose(
        left["alias_mean_utilities"], right["alias_mean_utilities"], rtol=0.0, atol=1e-12
    ):
        return False
    if not np.allclose(
        left["alias_utility_gaps"], right["alias_utility_gaps"], rtol=0.0, atol=1e-12
    ):
        return False
    return bool(
        left["all_coordinatewise_nonimproving"]
        == right["all_coordinatewise_nonimproving"]
    )


def main() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    attack = json.loads(ATTACK.read_text(encoding="utf-8"))
    sealed = json.loads(SEALED.read_text(encoding="utf-8"))
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    raw = np.load(RAW, allow_pickle=False)
    lock_checks = {
        "manifest": lock.get("manifest_id") == "STEP87B_STAGEB_EXECUTION_LOCK_V1",
        "predates_outcomes": lock.get("outcomes_executed_before_lock") is False,
        "inputs": lock.get("input_sha256")
        == {
            "preregistration": file_sha256(PREREG),
            "attack": file_sha256(ATTACK),
            "sealed_holdout": file_sha256(SEALED),
        },
        "code": lock.get("code_sha256")
        == {
            "stage_b_runner": file_sha256(RUNNER),
            "independent_validator": file_sha256(Path(__file__).resolve()),
            **{name: file_sha256(path) for name, path in DEPENDENCIES.items()},
        },
    }
    models = list(sealed["models"])
    checks: dict[str, Any] = {}
    active_p: dict[str, float] = {}
    did_p: dict[str, float] = {}
    gate_material: dict[str, dict[str, Any]] = {}
    all_pass = all(lock_checks.values())
    for task_index, task in enumerate(TASKS):
        data = sealed["tasks"][task]
        spec = attack["tasks"][task]
        selected = spec["selected"]
        target = int(selected["target_index"])
        original_responses = np.asarray(data["registry_responses"], dtype=object)
        original_utilities = np.asarray(data["registry_utilities"], dtype=np.float64)
        supports = [[int(value) for value in row] for row in spec["holdout_attack"]["supports"]]
        alias_rows = []
        alias_scores = []
        for alias, support in enumerate(supports):
            response = original_responses[target].copy()
            utility = original_utilities[target].copy()
            for index in support:
                response[index] = opaque_token(task, alias)
                utility[index] = 0.0
            alias_rows.append(response)
            alias_scores.append(utility)
        alias_scores_array = np.asarray(alias_scores, dtype=np.float64)
        parent_utility = original_utilities[target]
        quality_gaps = alias_scores_array.mean(axis=1) - parent_utility.mean()
        quality_observed = {
            "parent_mean_utility": float(parent_utility.mean()),
            "alias_mean_utilities": [
                float(value) for value in alias_scores_array.mean(axis=1)
            ],
            "alias_utility_gaps": [float(value) for value in quality_gaps],
            "minimum_gap": float(quality_gaps.min()),
            "maximum_gap": float(quality_gaps.max()),
            "all_coordinatewise_nonimproving": bool(
                np.all(alias_scores_array <= parent_utility[None, :] + TOL)
            ),
        }
        reference_tokens = {
            token
            for references in data["references"]
            for reference in references
            for token in re.findall(r"[a-z0-9]+", str(reference).casefold())
        }
        opaque_absent = all(
            opaque_token(task, alias).casefold() not in reference_tokens
            for alias in range(ALIASES)
        )
        refined_responses = np.concatenate([original_responses, np.asarray(alias_rows, dtype=object)])
        refined_utilities = np.concatenate([original_utilities, alias_scores_array])
        clean_parents = np.arange(len(models), dtype=np.int64)
        refined_parents = np.concatenate([clean_parents, np.full(ALIASES, target, dtype=np.int64)])
        clean_labels = models
        refined_labels = models + [
            f"STEP87B-HOLDOUT:{task}:{models[target]}:{selected['policy']}:{selected['distance']}:{alias}"
            for alias in range(ALIASES)
        ]
        pools = pools_from_seeds(original_responses.shape[1], int(spec["pool_size"]))
        queries = fixed_queries(pools, int(selected["budget"]), task_index)
        clean_q, clean_roots, clean_regret = independent_active(
            original_responses,
            original_utilities,
            clean_parents,
            clean_labels,
            original_utilities,
            pools,
            int(selected["budget"]),
            float(selected["tau"]),
        )
        refined_q, refined_roots, refined_regret = independent_active(
            refined_responses,
            refined_utilities,
            refined_parents,
            refined_labels,
            original_utilities,
            pools,
            int(selected["budget"]),
            float(selected["tau"]),
        )
        clean_fixed_roots, clean_fixed_regret = independent_fixed(
            original_utilities,
            clean_parents,
            clean_labels,
            original_responses,
            original_utilities,
            pools,
            queries,
        )
        refined_fixed_roots, refined_fixed_regret = independent_fixed(
            refined_utilities,
            refined_parents,
            refined_labels,
            refined_responses,
            original_utilities,
            pools,
            queries,
        )
        vectors = {
            "active_terminal_delta": refined_regret[:, -1] - clean_regret[:, -1],
            "fixed_terminal_delta": refined_fixed_regret[:, -1] - clean_fixed_regret[:, -1],
            "terminal_did": (refined_regret[:, -1] - clean_regret[:, -1]) - (refined_fixed_regret[:, -1] - clean_fixed_regret[:, -1]),
            "active_cumulative_delta": refined_regret.sum(axis=1) - clean_regret.sum(axis=1),
            "fixed_cumulative_delta": refined_fixed_regret.sum(axis=1) - clean_fixed_regret.sum(axis=1),
        }
        vectors["cumulative_did"] = vectors["active_cumulative_delta"] - vectors["fixed_cumulative_delta"]
        array_checks = {
            "pools": np.array_equal(pools, raw[f"{task}__pools"]),
            "fixed_queries": np.array_equal(queries, raw[f"{task}__fixed_queries"]),
            "clean_active_queries": np.array_equal(clean_q, raw[f"{task}__clean_active_queries"]),
            "refined_active_queries": np.array_equal(refined_q, raw[f"{task}__refined_active_queries"]),
            "clean_active_root_path": np.array_equal(clean_roots, raw[f"{task}__clean_active_root_path"]),
            "refined_active_root_path": np.array_equal(refined_roots, raw[f"{task}__refined_active_root_path"]),
            "clean_fixed_root_path": np.array_equal(clean_fixed_roots, raw[f"{task}__clean_fixed_root_path"]),
            "refined_fixed_root_path": np.array_equal(refined_fixed_roots, raw[f"{task}__refined_fixed_root_path"]),
            **{
                name: bool(np.allclose(value, raw[f"{task}__{name}"], rtol=0.0, atol=1e-12))
                for name, value in vectors.items()
            },
        }
        stat_map = {
            "active_terminal_delta": ("active_terminal_root_delta", 287000 + task_index * 20),
            "fixed_terminal_delta": ("fixed_terminal_root_delta", 287002 + task_index * 20),
            "terminal_did": ("terminal_root_did", 287004 + task_index * 20),
            "active_cumulative_delta": ("active_cumulative_root_delta", 287006 + task_index * 20),
            "fixed_cumulative_delta": ("fixed_cumulative_root_delta", 287008 + task_index * 20),
            "cumulative_did": ("cumulative_root_did", 287010 + task_index * 20),
        }
        statistic_checks = {}
        observed_statistics: dict[str, dict[str, Any]] = {}
        for vector_name, (result_name, seed) in stat_map.items():
            observed = summary(vectors[vector_name], seed)
            observed_statistics[result_name] = observed
            statistic_checks[result_name] = close_record(
                observed, results["tasks"][task]["statistics"][result_name]
            )
        active_p[task] = observed_statistics["active_terminal_root_delta"][
            "one_sided_sign_flip_p"
        ]
        did_p[task] = observed_statistics["terminal_root_did"][
            "one_sided_sign_flip_p"
        ]
        parent_hash_matches = (
            digest_json(original_responses[target].tolist())
            == spec["holdout_attack"]["parent_response_hash"]
        )
        alias_hashes = [digest_json(row.tolist()) for row in alias_rows]
        aliases_distinct = (
            len(set(alias_hashes)) == ALIASES
            and digest_json(original_responses[target].tolist()) not in set(alias_hashes)
        )
        distances = [
            int(np.sum(row != original_responses[target])) for row in alias_rows
        ]
        quality_pass = bool(
            quality_observed["all_coordinatewise_nonimproving"]
            and quality_observed["minimum_gap"] >= -0.010000000001
            and quality_observed["maximum_gap"] <= TOL
            and max(value / original_responses.shape[1] for value in distances)
            <= 0.010000000001
        )
        integrity_pass = bool(
            parent_hash_matches
            and aliases_distinct
            and opaque_absent
            and spec["holdout_attack"]["references_read"] == 0
            and spec["holdout_attack"]["peer_response_rows_read"] == 0
        )
        gate_material[task] = {
            "quality_pass": quality_pass,
            "integrity_pass": integrity_pass,
            "fixed_root_paths_byte_identical": bool(
                np.array_equal(clean_fixed_roots, refined_fixed_roots)
            ),
            "statistics": observed_statistics,
        }
        quality_match = close_quality(
            quality_observed, results["tasks"][task]["quality"]
        )
        passed = (
            all(array_checks.values())
            and all(statistic_checks.values())
            and quality_match
        )
        checks[task] = {
            "array_checks": array_checks,
            "statistic_checks": statistic_checks,
            "quality_match": quality_match,
            "pass": passed,
        }
        all_pass = all_pass and passed
        print(f"[{task}] {'PASS' if passed else 'FAIL'}", flush=True)

    active_q = bh_adjust(active_p)
    did_q = bh_adjust(did_p)
    passing_tasks: list[str] = []
    for task in TASKS:
        material = gate_material[task]
        active_pass = bool(
            material["statistics"]["active_terminal_root_delta"]["ci95"][0] > 0.0
            and active_q[task] < 0.05
        )
        did_pass = bool(
            material["statistics"]["terminal_root_did"]["ci95"][0] > 0.0
            and did_q[task] < 0.05
        )
        observed_gate = {
            "quality": material["quality_pass"],
            "integrity": material["integrity_pass"],
            "active_terminal_root_harm": active_pass,
            "adaptive_terminal_excess": did_pass,
            "exact_fixed_query_mediation": material[
                "fixed_root_paths_byte_identical"
            ],
            "task_pass": bool(
                material["quality_pass"]
                and material["integrity_pass"]
                and active_pass
                and did_pass
            ),
        }
        observed_adjustment = {
            "active_terminal_bh_q": active_q[task],
            "terminal_did_bh_q": did_q[task],
        }
        reported_adjustment = results["tasks"][task]["multiplicity_adjustment"]
        adjustment_match = all(
            math.isclose(
                float(observed_adjustment[key]),
                float(reported_adjustment[key]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for key in observed_adjustment
        )
        gate_match = observed_gate == results["tasks"][task]["gate"]
        checks[task]["multiplicity_adjustment_match"] = adjustment_match
        checks[task]["gate_match"] = gate_match
        checks[task]["pass"] = bool(
            checks[task]["pass"] and adjustment_match and gate_match
        )
        all_pass = all_pass and checks[task]["pass"]
        if observed_gate["task_pass"]:
            passing_tasks.append(task)

    global_separation = bool(
        attack["sealed_holdout_outcomes_read"] is False
        and attack["holdout_peer_responses_read"] is False
        and attack["holdout_references_or_utilities_read"] is False
        and set(results["tasks"]) == set(TASKS)
    )
    if not global_separation or len(passing_tasks) == 0:
        decision_status = "NO_TERMINAL_PROMOTION"
    elif len(passing_tasks) == 1:
        decision_status = "NARROW_TERMINAL_BRIDGE"
    else:
        decision_status = "PROMOTE_REFERENCE_FREE_TERMINAL_BRIDGE"
    observed_decision = {
        "global_separation_and_completeness": global_separation,
        "passing_scenarios": passing_tasks,
        "passing_count": len(passing_tasks),
        "status": decision_status,
    }
    decision_match = observed_decision == results["decision"]
    all_pass = all_pass and decision_match
    report = {
        "manifest_id": "STEP87B_INDEPENDENT_VALIDATION_V1",
        "stage_b_lock_sha256": file_sha256(LOCK),
        "attack_sha256": file_sha256(ATTACK),
        "sealed_sha256": file_sha256(SEALED),
        "results_sha256": file_sha256(RESULTS),
        "raw_sha256": file_sha256(RAW),
        "independent_backend": "Separate canonicalization, acquisition loop, fixed-query loop, and statistical reconstruction; no import from the experimental runner or common module.",
        "stage_b_lock_checks": lock_checks,
        "tasks": checks,
        "decision_match": decision_match,
        "independently_reconstructed_decision": observed_decision,
        "status": "PASS_STEP87B_INDEPENDENT_RECONSTRUCTION" if all_pass else "FAIL_STEP87B_INDEPENDENT_RECONSTRUCTION",
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not all_pass:
        raise AssertionError("Step 87B independent validation failed")
    print(report["status"], flush=True)


if __name__ == "__main__":
    main()
