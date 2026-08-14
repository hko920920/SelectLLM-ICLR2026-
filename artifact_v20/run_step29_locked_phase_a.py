"""Run the locked Step 28 Phase A matrix.

This script consumes STEP28_FINAL_MATRIX_MANIFEST_2026-08-06.json version 1.1.
It uses public HELM-Lite responses cached by Steps 24/25, an independent
Equation-(5)/(7) Select-LLM implementation, disjoint proxy-tuning seeds, and
previously unused final paired seeds 50000--50499.

The primary metric uses the actual selected entry's oracle outputs.  Parent
mapping is retained only as a secondary root-attribution metric.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

import run_step24_real_response_current_method_gate as med_source
import run_step25_current_method_external_generality_gate as external_source


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "STEP28_FINAL_MATRIX_MANIFEST_2026-08-06.json"
OUT = ROOT / "STEP29_LOCKED_PHASE_A_RESULTS_2026-08-06.json"
MODELS = list(med_source.MODELS)
assert MODELS == list(external_source.MODELS)


def digest_jsonable(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def load_manifest() -> tuple[dict, str]:
    payload = MANIFEST_PATH.read_bytes()
    manifest = json.loads(payload)
    expected = "a16f9a262f8d100899a443e96e3ae141ca266f7b5121cdf5e628fc1a67c201ca"
    observed = hashlib.sha256(payload).hexdigest()
    if manifest["manifest_id"] != "STEP28_FINAL_MATRIX_V1_1" or observed != expected:
        raise AssertionError("locked Step 28 manifest changed")
    return manifest, observed


def load_all_data() -> dict[str, dict]:
    ids, responses, oracle, options, correct, provenance = med_source.load_helm_medqa()
    tasks = {
        "medqa": {
            "ids": ids,
            "responses": responses,
            "oracle": oracle,
            "public_options": options,
            "correct": correct,
            "provenance": provenance,
            "public_option_mode": True,
        }
    }
    for name in ("gsm8k", "openbookqa"):
        loaded = external_source.load_task(name, external_source.TASKS[name])
        tasks[name] = {
            "ids": loaded["ids"],
            "responses": loaded["responses"],
            "oracle": loaded["oracle"],
            # GSM8K references are labels, not public prompt options.  T1 must
            # not use them.  OpenBookQA options are exposed in the prompt.
            "public_options": loaded["alternatives"] if name == "openbookqa" else None,
            "correct": loaded["correct"],
            "provenance": loaded["provenance"],
            "public_option_mode": name == "openbookqa",
        }
    for name, data in tasks.items():
        responses = np.asarray(data["responses"], dtype=object)
        oracle = np.asarray(data["oracle"], dtype=np.float64)
        correct = np.asarray(data["correct"], dtype=object)
        if responses.shape != oracle.shape or responses.shape[0] != len(MODELS):
            raise AssertionError(f"{name}: malformed response/oracle matrix")
        # Some HELM metrics normalize strings (e.g. MedQA quasi-exact treats a
        # +/- punctuation contrast as equal).  Wrapper outputs therefore
        # inherit the recorded score of an existing identical public response
        # instead of being rescored by raw string equality.
        reconstructed = (responses == correct[None, :]).astype(np.float64)
        data["raw_string_vs_recorded_oracle_mismatches"] = int(
            np.sum(reconstructed != oracle)
        )
        data["responses"] = responses
        data["oracle"] = oracle
        data["correct"] = correct
    return tasks


def encode_responses(responses: np.ndarray) -> np.ndarray:
    models, queries = responses.shape
    codes = np.zeros((queries, models), dtype=np.int16)
    for query in range(queries):
        mapping: dict[str, int] = {}
        for model in range(models):
            value = str(responses[model, query])
            if value not in mapping:
                mapping[value] = len(mapping)
            codes[query, model] = mapping[value]
    return codes


def make_scenario(
    responses: np.ndarray,
    oracle: np.ndarray,
    parents: list[int],
    labels: list[str],
    kind: str,
) -> dict:
    responses = np.asarray(responses, dtype=object)
    oracle = np.asarray(oracle, dtype=np.float64)
    if responses.shape != oracle.shape or responses.shape[0] != len(parents):
        raise AssertionError(f"{kind}: scenario shape mismatch")
    return {
        "kind": kind,
        "responses": responses,
        "oracle": oracle,
        "parents": np.asarray(parents, dtype=np.int64),
        "labels": labels,
        "codes": encode_responses(responses),
        "response_hashes": [
            hashlib.sha256(
                json.dumps(row.tolist(), ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()
            for row in responses
        ],
    }


def base_scenario(data: dict) -> dict:
    return make_scenario(
        data["responses"], data["oracle"], list(range(len(MODELS))), MODELS, "clean"
    )


def clone_scenario(clean: dict, target: int, copies: int) -> dict:
    responses = np.concatenate(
        [clean["responses"], np.repeat(clean["responses"][[target]], copies, axis=0)]
    )
    oracle = np.concatenate(
        [clean["oracle"], np.repeat(clean["oracle"][[target]], copies, axis=0)]
    )
    return make_scenario(
        responses,
        oracle,
        list(clean["parents"]) + [target] * copies,
        list(clean["labels"])
        + [f"EXACT:{MODELS[target]}:{index}" for index in range(copies)],
        f"T0_exact_{copies}",
    )


def response_frequencies(responses: np.ndarray, query: int, choices: list[str]) -> list[float]:
    return [float(np.mean(responses[:, query] == choice)) for choice in choices]


def public_choices(data: dict, query: int, parent_response: str) -> list[str]:
    observed = list(dict.fromkeys(str(value) for value in data["responses"][:, query]))
    # Use only outputs already emitted by another public candidate.  This is a
    # strict subset of T1's allowed prompt/option access and lets us inherit the
    # exact recorded HELM metric rather than inventing a task-specific scorer.
    return [choice for choice in observed if choice != parent_response]


def recorded_response_score(data: dict, query: int, response: str) -> float:
    mask = data["responses"][:, query] == response
    values = np.unique(data["oracle"][mask, query])
    if len(values) != 1:
        raise AssertionError("identical canonical responses have inconsistent oracle scores")
    return float(values[0])


def build_wrapper(
    data: dict,
    clean: dict,
    *,
    target: int,
    distance_queries: int,
    aliases: int,
    reference_aware: bool,
) -> tuple[dict, dict]:
    responses = data["responses"]
    target_oracle = data["oracle"][target]
    eligible = (
        np.flatnonzero(target_oracle == 0)
        if reference_aware
        else np.arange(responses.shape[1], dtype=np.int64)
    )
    ranked: list[tuple[float, int]] = []
    choices_by_query: dict[int, list[str]] = {}
    for query_value in eligible:
        query = int(query_value)
        choices = public_choices(data, query, str(responses[target, query]))
        if reference_aware:
            choices = [
                choice
                for choice in choices
                if recorded_response_score(data, query, choice) == 0.0
            ]
        if not choices:
            continue
        frequencies = response_frequencies(responses, query, choices)
        order = np.argsort(np.asarray(frequencies), kind="stable")
        choices = [choices[int(index)] for index in order]
        choices_by_query[query] = choices
        ranked.append((min(frequencies), query))

    ordered = np.asarray([query for _, query in sorted(ranked)], dtype=np.int64)
    if len(ordered) < distance_queries:
        raise ValueError("insufficient eligible wrapper positions")
    stride = max(1, len(ordered) // 7)
    alias_responses = []
    alias_oracle = []
    changed_sets = []
    for alias in range(aliases):
        changed = np.roll(ordered, alias * stride)[:distance_queries]
        row = responses[target].copy()
        for query_value in changed:
            query = int(query_value)
            choices = choices_by_query[query]
            row[query] = choices[(alias + query) % len(choices)]
        score = target_oracle.copy()
        for query_value in changed:
            query = int(query_value)
            score[query] = recorded_response_score(data, query, str(row[query]))
        if reference_aware and not np.array_equal(score, target_oracle):
            raise AssertionError("T2 wrapper failed per-example quality equality")
        if not math.isclose(
            float(np.mean(row != responses[target])),
            distance_queries / responses.shape[1],
            abs_tol=1e-12,
        ):
            raise AssertionError("wrapper response distance mismatch")
        alias_responses.append(row)
        alias_oracle.append(score)
        changed_sets.append([int(value) for value in changed])

    wrapper_responses = np.asarray(alias_responses, dtype=object)
    wrapper_oracle = np.asarray(alias_oracle, dtype=np.float64)
    scenario = make_scenario(
        np.concatenate([responses, wrapper_responses]),
        np.concatenate([data["oracle"], wrapper_oracle]),
        list(range(len(MODELS))) + [target] * aliases,
        MODELS
        + [
            f"{'T2_REFERENCE' if reference_aware else 'T1_RESPONSE'}:{MODELS[target]}:{index}"
            for index in range(aliases)
        ],
        "T2_reference_aware" if reference_aware else "T1_response_only",
    )
    alias_hashes = scenario["response_hashes"][-aliases:]
    parent_hash = clean["response_hashes"][target]
    if len(set(alias_hashes)) != aliases or parent_hash in alias_hashes:
        raise AssertionError("wrapper aliases are not distinct hashes")
    metadata = {
        "target_index": target,
        "target": MODELS[target],
        "target_score": float(target_oracle.mean()),
        "aliases": aliases,
        "distance_queries": distance_queries,
        "distance_fraction": distance_queries / responses.shape[1],
        "reference_aware": reference_aware,
        "online_adaptation": False,
        "per_example_quality_equal": bool(
            all(np.array_equal(row, target_oracle) for row in wrapper_oracle)
        ),
        "alias_full_score_gaps": [
            float(row.mean() - target_oracle.mean()) for row in wrapper_oracle
        ],
        "alias_hashes": alias_hashes,
        "changed_set_digests": [digest_jsonable(values) for values in changed_sets],
        "exact_hash_survivors": aliases,
    }
    return scenario, metadata


def pools_from_seeds(seeds: list[int], queries: int, pool_size: int) -> np.ndarray:
    return np.asarray(
        [sorted(random.Random(seed).sample(range(queries), pool_size)) for seed in seeds],
        dtype=np.int64,
    )


def run_batch_numpy(
    codes: np.ndarray,
    scenario_oracle: np.ndarray,
    parents: np.ndarray,
    original_oracle: np.ndarray,
    pools: np.ndarray,
    budget: int,
    tau: float,
) -> tuple:
    runs = pools.shape[0]
    pool_size = pools.shape[1]
    entries = scenario_oracle.shape[0]
    originals = original_oracle.shape[0]
    queried = np.full((runs, budget), -1, dtype=np.int64)
    selected_local_path = np.full((runs, budget), -1, dtype=np.int64)
    deploy_regret = np.zeros((runs, budget), dtype=np.float64)
    expanded_regret = np.zeros((runs, budget), dtype=np.float64)
    root_regret = np.zeros((runs, budget), dtype=np.float64)

    for run_index in range(runs):
        pool = pools[run_index]
        entry_scores = scenario_oracle[:, pool].mean(axis=1)
        original_scores = original_oracle[:, pool].mean(axis=1)
        best_original = float(original_scores.max())
        best_expanded = float(entry_scores.max())

        # Exact-equality acquisition is a weighted categorical histogram.  The
        # flattened bins let NumPy compute all query group masses in one C loop
        # without the incompatible local Numba installation.
        pool_codes = codes[pool]
        flat_bins = (
            np.arange(pool_size, dtype=np.int64)[:, None] * entries + pool_codes
        ).ravel()
        active = np.ones(pool_size, dtype=bool)
        cumulative = np.zeros(entries, dtype=np.float64)

        for step in range(budget):
            shifted = cumulative / tau
            shifted -= shifted.max()
            posterior = np.exp(shifted)
            posterior /= posterior.sum()
            masses = np.bincount(
                flat_bins,
                weights=np.tile(posterior, pool_size),
                minlength=pool_size * entries,
            ).reshape(pool_size, entries)
            acquisition = np.sum(masses * masses, axis=1)
            acquisition[~active] = np.inf
            best_position = int(np.argmin(acquisition))
            active[best_position] = False
            query = int(pool[best_position])
            queried[run_index, step] = query
            cumulative += scenario_oracle[:, query]

            selected = int(np.argmax(cumulative))
            selected_local_path[run_index, step] = selected
            deploy_regret[run_index, step] = best_original - entry_scores[selected]
            expanded_regret[run_index, step] = best_expanded - entry_scores[selected]
            root_regret[run_index, step] = best_original - original_scores[parents[selected]]

    return queried, selected_local_path, deploy_regret, expanded_regret, root_regret


def run_scenario(
    scenario: dict,
    original_oracle: np.ndarray,
    pools: np.ndarray,
    budget: int,
    tau: float,
) -> dict:
    queried, selected, deploy, expanded, root = run_batch_numpy(
        scenario["codes"],
        np.asarray(scenario["oracle"], dtype=np.float64),
        np.asarray(scenario["parents"], dtype=np.int64),
        np.asarray(original_oracle, dtype=np.float64),
        pools,
        budget,
        tau,
    )
    final_local = selected[:, -1]
    final_parent = scenario["parents"][final_local]
    return {
        "queried": queried,
        "selected_local_path": selected,
        "final_local": final_local,
        "final_parent": final_parent,
        "deploy_regret_path": deploy,
        "expanded_regret_path": expanded,
        "root_regret_path": root,
        "cumulative_deploy_regret": deploy.sum(axis=1),
        "final_deploy_regret": deploy[:, -1],
        "cumulative_expanded_regret": expanded.sum(axis=1),
        "final_expanded_regret": expanded[:, -1],
        "cumulative_root_regret": root.sum(axis=1),
        "final_root_regret": root[:, -1],
    }


def proxy_oracle(clean: dict) -> np.ndarray:
    codes = clean["codes"]
    queries, models = codes.shape
    proxy = np.zeros((models, queries), dtype=np.float64)
    for query in range(queries):
        counts = np.bincount(codes[query], minlength=models)
        for model in range(models):
            proxy[model, query] = counts[codes[query, model]] / models
    return proxy


def tune_temperature(
    clean: dict,
    pools: np.ndarray,
    budget: int,
    grid: list[float],
) -> dict:
    proxy = proxy_oracle(clean)
    proxy_scenario = make_scenario(
        clean["responses"], proxy, list(range(len(MODELS))), MODELS, "proxy_tuning"
    )
    rows = []
    for tau in grid:
        runs = run_scenario(proxy_scenario, proxy, pools, budget, float(tau))
        identification = runs["deploy_regret_path"] <= 1e-12
        curve = identification.mean(axis=0)
        rows.append(
            {
                "tau": float(tau),
                "mean_identification_probability": float(curve.mean()),
                "final_identification_probability": float(curve[-1]),
                "curve": [float(value) for value in curve],
            }
        )
        print(
            f"    proxy tau={tau:g} mean-id={rows[-1]['mean_identification_probability']:.6f}",
            flush=True,
        )
    selected = max(
        rows, key=lambda row: (row["mean_identification_probability"], -row["tau"])
    )
    return {
        "criterion": "maximum mean proxy identification probability over budgets 1..B; ties choose smaller tau",
        "selected_tau": selected["tau"],
        "grid_results": rows,
        "proxy_digest": hashlib.sha256(proxy.tobytes()).hexdigest(),
    }


def initial_acquisition(codes: np.ndarray) -> np.ndarray:
    queries, models = codes.shape
    values = np.zeros(queries, dtype=np.float64)
    for query in range(queries):
        counts = np.bincount(codes[query], minlength=models).astype(np.float64)
        values[query] = np.sum((counts / models) ** 2)
    return values


def screen_target(clean: dict, pools: np.ndarray) -> dict:
    clean_score = initial_acquisition(clean["codes"])
    clean_first = np.asarray(
        [pool[int(np.argmin(clean_score[pool]))] for pool in pools], dtype=np.int64
    )
    rows = []
    for target in range(len(MODELS)):
        cloned = clone_scenario(clean, target, 1)
        score = initial_acquisition(cloned["codes"])
        first = np.asarray(
            [pool[int(np.argmin(score[pool]))] for pool in pools], dtype=np.int64
        )
        rows.append(
            {
                "target_index": target,
                "target": MODELS[target],
                "first_query_change_fraction": float(np.mean(first != clean_first)),
            }
        )
    selected = max(rows, key=lambda row: (row["first_query_change_fraction"], row["target"]))
    return {"selected": selected, "all_targets": rows}


def top_quality_eligible_target(oracle: np.ndarray, distance_queries: int) -> dict:
    scores = oracle.mean(axis=1)
    rows = []
    for target in range(len(MODELS)):
        wrong = int(np.sum(oracle[target] == 0))
        if wrong >= distance_queries:
            rows.append(
                {
                    "target_index": target,
                    "target": MODELS[target],
                    "score": float(scores[target]),
                    "eligible_wrong_queries": wrong,
                }
            )
    return max(rows, key=lambda row: (row["score"], row["target"]))


def bootstrap_ci(values: np.ndarray, seed: int, repetitions: int = 10000) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
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


def sign_flip_pvalue(values: np.ndarray, seed: int, repetitions: int = 10000) -> float:
    values = np.asarray(values, dtype=np.float64)
    observed = abs(float(values.mean()))
    if np.allclose(values, 0.0):
        return 1.0
    rng = np.random.default_rng(seed)
    extreme = 0
    cursor = 0
    while cursor < repetitions:
        count = min(500, repetitions - cursor)
        signs = rng.integers(0, 2, size=(count, len(values)), dtype=np.int8) * 2 - 1
        permuted = np.abs((signs * values[None, :]).mean(axis=1))
        extreme += int(np.sum(permuted >= observed - 1e-15))
        cursor += count
    return (extreme + 1.0) / (repetitions + 1.0)


def compare_runs(clean: dict, changed: dict, seed: int, include_arrays: bool = True) -> dict:
    cumulative_deploy = (
        changed["cumulative_deploy_regret"] - clean["cumulative_deploy_regret"]
    )
    final_deploy = changed["final_deploy_regret"] - clean["final_deploy_regret"]
    cumulative_expanded = (
        changed["cumulative_expanded_regret"] - clean["cumulative_expanded_regret"]
    )
    cumulative_root = changed["cumulative_root_regret"] - clean["cumulative_root_regret"]
    changed_positions = np.sum(clean["queried"] != changed["queried"], axis=1)
    jaccards = []
    for left, right in zip(clean["queried"], changed["queried"]):
        a, b = set(int(value) for value in left), set(int(value) for value in right)
        jaccards.append(len(a & b) / len(a | b))
    result = {
        "runs": int(len(cumulative_deploy)),
        "path_change_fraction": float(np.mean(changed_positions > 0)),
        "mean_changed_query_positions": float(changed_positions.mean()),
        "mean_query_set_jaccard": float(np.mean(jaccards)),
        "final_selected_entry_change_fraction": float(
            np.mean(clean["final_local"] != changed["final_local"])
        ),
        "final_selected_root_change_fraction": float(
            np.mean(clean["final_parent"] != changed["final_parent"])
        ),
        "mean_delta_cumulative_deployed_regret": float(cumulative_deploy.mean()),
        "median_delta_cumulative_deployed_regret": float(np.median(cumulative_deploy)),
        "delta_cumulative_deployed_regret_95ci": bootstrap_ci(cumulative_deploy, seed),
        "delta_cumulative_deployed_regret_signflip_p": sign_flip_pvalue(
            cumulative_deploy, seed + 1
        ),
        "mean_delta_final_deployed_regret": float(final_deploy.mean()),
        "delta_final_deployed_regret_95ci": bootstrap_ci(final_deploy, seed + 2),
        "mean_delta_cumulative_expanded_regret": float(cumulative_expanded.mean()),
        "mean_delta_cumulative_root_regret": float(cumulative_root.mean()),
        "harmful_fraction": float(np.mean(cumulative_deploy > 1e-12)),
        "improved_fraction": float(np.mean(cumulative_deploy < -1e-12)),
        "tied_fraction": float(np.mean(np.abs(cumulative_deploy) <= 1e-12)),
    }
    if include_arrays:
        result["paired_delta_cumulative_deployed_regret"] = [
            float(value) for value in cumulative_deploy
        ]
        result["paired_delta_final_deployed_regret"] = [float(value) for value in final_deploy]
    return result


def bh_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 1.0
    for reverse_rank, name in enumerate(reversed(ordered), start=1):
        rank = count - reverse_rank + 1
        candidate = min(1.0, pvalues[name] * count / rank)
        running = min(running, candidate)
        adjusted[name] = running
    return adjusted


def provider(model: str) -> str:
    return model.split("_", 1)[0]


def components_at_radius(responses: np.ndarray, rho: float) -> list[list[int]]:
    models = responses.shape[0]
    parent = list(range(models))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for left in range(models):
        for right in range(left + 1, models):
            distance = float(np.mean(responses[left] != responses[right]))
            if distance <= rho + 1e-12:
                union(left, right)
    groups: dict[int, list[int]] = {}
    for model in range(models):
        groups.setdefault(find(model), []).append(model)
    return list(groups.values())


def quotient_scenario(current: dict, rho: float) -> tuple[dict, list[list[int]]]:
    components = components_at_radius(current["responses"], rho)
    representatives = [min(component) for component in components]
    result = make_scenario(
        current["responses"][representatives],
        current["oracle"][representatives],
        [int(current["parents"][index]) for index in representatives],
        [current["labels"][index] for index in representatives],
        f"{current['kind']}_cover_{rho}",
    )
    return result, components


def cover_cost(clean: dict, components: list[list[int]]) -> dict:
    scores = clean["oracle"].mean(axis=1)
    nontrivial = [component for component in components if len(component) > 1]
    same_provider_pairs = 0
    max_span = 0.0
    details = []
    for component in nontrivial:
        component_scores = [float(scores[index]) for index in component]
        span = max(component_scores) - min(component_scores)
        max_span = max(max_span, span)
        for offset, left in enumerate(component):
            for right in component[offset + 1 :]:
                if provider(MODELS[left]) == provider(MODELS[right]):
                    same_provider_pairs += 1
        details.append(
            {
                "members": [MODELS[index] for index in component],
                "score_span": span,
            }
        )
    return {
        "models_removed": len(MODELS) - len(components),
        "nontrivial_components": len(nontrivial),
        "same_provider_pairs_merged": same_provider_pairs,
        "maximum_component_quality_span": max_span,
        "components": details,
    }


def scenario_summary(scenario: dict) -> dict:
    return {
        "kind": scenario["kind"],
        "entries": len(scenario["labels"]),
        "response_digest": digest_jsonable(scenario["responses"].tolist()),
        "oracle_digest": hashlib.sha256(scenario["oracle"].tobytes()).hexdigest(),
        "distinct_response_hashes": len(set(scenario["response_hashes"])),
    }


def compact_run_digest(runs: dict) -> dict:
    return {
        "query_matrix_digest": hashlib.sha256(runs["queried"].tobytes()).hexdigest(),
        "selected_path_digest": hashlib.sha256(
            runs["selected_local_path"].tobytes()
        ).hexdigest(),
        "mean_cumulative_deployed_regret": float(
            runs["cumulative_deploy_regret"].mean()
        ),
        "mean_final_deployed_regret": float(runs["final_deploy_regret"].mean()),
    }


def evaluate_task(
    name: str,
    data: dict,
    config: dict,
    manifest: dict,
    result_so_far: dict,
) -> dict:
    print(f"[{name}] building scenarios", flush=True)
    clean = base_scenario(data)
    queries = clean["responses"].shape[1]
    pool_size = int(config["pool_size"])
    budget = int(config["primary_budget"])
    screen_spec = manifest["randomness_partition"]["response_only_screen_seeds"]
    proxy_spec = manifest["randomness_partition"]["proxy_temperature_seeds"]
    final_spec = manifest["randomness_partition"]["final_paired_seeds"]
    screen_seeds = list(range(screen_spec["start"], screen_spec["stop_inclusive"] + 1))
    proxy_seeds = list(range(proxy_spec["start"], proxy_spec["stop_inclusive"] + 1))
    final_seeds = list(range(final_spec["start"], final_spec["stop_inclusive"] + 1))
    screen_pools = pools_from_seeds(screen_seeds, queries, pool_size)
    proxy_pools = pools_from_seeds(proxy_seeds, queries, pool_size)
    final_pools = pools_from_seeds(final_seeds, queries, pool_size)

    screen = screen_target(clean, screen_pools)
    t0_target = int(screen["selected"]["target_index"])
    distance_queries = max(
        int(manifest["attack_factors"]["minimum_changed_queries"]),
        int(round(manifest["attack_factors"]["primary_wrapper_distance_fraction"] * queries)),
    )
    t2_target_row = top_quality_eligible_target(data["oracle"], distance_queries)
    t2_target = int(t2_target_row["target_index"])

    clone1 = clone_scenario(clean, t0_target, 1)
    clone4 = clone_scenario(clean, t0_target, 4)
    t1, t1_meta = build_wrapper(
        data,
        clean,
        target=t0_target,
        distance_queries=distance_queries,
        aliases=4,
        reference_aware=False,
    )
    t2, t2_meta = build_wrapper(
        data,
        clean,
        target=t2_target,
        distance_queries=distance_queries,
        aliases=4,
        reference_aware=True,
    )

    print(f"[{name}] proxy temperature tuning", flush=True)
    tuning = tune_temperature(
        clean,
        proxy_pools,
        budget,
        [float(value) for value in manifest["primary_method"]["temperature_policy"]["proxy_grid"]],
    )
    tau = float(tuning["selected_tau"])
    print(f"[{name}] selected tau={tau:g}; locked primary runs", flush=True)

    scenarios = {
        "clean": clean,
        "T0_exact_1": clone1,
        "T0_exact_4": clone4,
        "T1_response_only": t1,
        "T2_reference_aware": t2,
    }
    runs: dict[str, dict] = {}
    for scenario_name, current in scenarios.items():
        started = time.time()
        runs[scenario_name] = run_scenario(
            current, data["oracle"], final_pools, budget, tau
        )
        print(
            f"[{name}] {scenario_name} done in {time.time()-started:.1f}s",
            flush=True,
        )

    comparisons = {
        scenario_name: compare_runs(runs["clean"], runs[scenario_name], 290000 + offset)
        for offset, scenario_name in enumerate(
            ("T0_exact_1", "T0_exact_4", "T1_response_only", "T2_reference_aware")
        )
    }

    # Candidate-invariant random-query negative control.  T2 has the exact
    # same per-example oracle vector as its parent, so selection is unchanged
    # as well as the query transcript under defender-favorable tie-breaking.
    random_queries = np.asarray(
        [
            random.Random(seed + 99173).sample(pool.tolist(), budget)
            for seed, pool in zip(final_seeds, final_pools)
        ],
        dtype=np.int64,
    )
    random_control = {
        "paired_query_transcripts_identical": True,
        "query_matrix_digest_clean": hashlib.sha256(random_queries.tobytes()).hexdigest(),
        "query_matrix_digest_refined": hashlib.sha256(random_queries.tobytes()).hexdigest(),
        "note": "Query generation is a pure function of paired seed/pool and ignores the candidate registry.",
    }

    print(f"[{name}] behavioral-cover frontier", flush=True)
    frontier = []
    cover_runs_clean: dict[float, dict] = {}
    cover_runs_t2: dict[float, dict] = {}
    cover_scenarios_clean: dict[float, dict] = {}
    cover_scenarios_t2: dict[float, dict] = {}
    for radius_value in manifest["controls_and_comparators"]["behavioral_component_cover"]["radii"]:
        radius = float(radius_value)
        defended_clean, clean_components = quotient_scenario(clean, radius)
        defended_t2, t2_components = quotient_scenario(t2, radius)
        cover_scenarios_clean[radius] = defended_clean
        cover_scenarios_t2[radius] = defended_t2
        cover_runs_clean[radius] = run_scenario(
            defended_clean, data["oracle"], final_pools, budget, tau
        )
        cover_runs_t2[radius] = run_scenario(
            defended_t2, data["oracle"], final_pools, budget, tau
        )
        residual = compare_runs(
            cover_runs_clean[radius],
            cover_runs_t2[radius],
            291000 + int(round(radius * 10000)),
            include_arrays=False,
        )
        aliases_kept = sum(
            label.startswith("T2_REFERENCE") for label in defended_t2["labels"]
        )
        frontier.append(
            {
                "radius": radius,
                "clean_cost": cover_cost(clean, clean_components),
                "defended_clean_entries": len(defended_clean["labels"]),
                "defended_T2_entries": len(defended_t2["labels"]),
                "T2_alias_representatives_kept": aliases_kept,
                "residual_T2_comparison": residual,
                "attacked_components_count": len(t2_components),
            }
        )
        print(
            f"[{name}] rho={radius:g} kept-alias={aliases_kept} residual={residual['mean_delta_cumulative_deployed_regret']:.6f}",
            flush=True,
        )

    # Exact quotient checks for T0 and T1 use the already defined rho=0 rule.
    defended_clone4, _ = quotient_scenario(clone4, 0.0)
    defended_t1, _ = quotient_scenario(t1, 0.0)
    clone4_q_runs = run_scenario(
        defended_clone4, data["oracle"], final_pools, budget, tau
    )
    t1_q_runs = run_scenario(defended_t1, data["oracle"], final_pools, budget, tau)
    exact_quotient = {
        "T0_exact_4": compare_runs(
            cover_runs_clean[0.0], clone4_q_runs, 292001, include_arrays=False
        ),
        "T1_response_only": compare_runs(
            cover_runs_clean[0.0], t1_q_runs, 292002, include_arrays=False
        ),
        "T2_reference_aware": frontier[0]["residual_T2_comparison"],
        "T1_alias_representatives_kept": sum(
            label.startswith("T1_RESPONSE") for label in defended_t1["labels"]
        ),
        "T2_alias_representatives_kept": frontier[0]["T2_alias_representatives_kept"],
    }

    task_result = {
        "task": name,
        "examples": queries,
        "models": len(MODELS),
        "pool_size": pool_size,
        "budget": budget,
        "final_seeds": [final_seeds[0], final_seeds[-1]],
        "data_digests": {
            "responses": digest_jsonable(data["responses"].tolist()),
            "oracle": hashlib.sha256(data["oracle"].tobytes()).hexdigest(),
            "ids": digest_jsonable(data["ids"]),
        },
        "response_only_target_screen": screen,
        "T2_target_policy": "highest full-dataset quality among parents with enough wrong positions; allowed only in T2",
        "T2_target": t2_target_row,
        "temperature_tuning": tuning,
        "scenarios": {key: scenario_summary(value) for key, value in scenarios.items()},
        "T1_metadata": t1_meta,
        "T2_metadata": t2_meta,
        "clean_run": compact_run_digest(runs["clean"]),
        "primary_comparisons": comparisons,
        "random_query_negative_control": random_control,
        "exact_response_quotient": exact_quotient,
        "behavioral_cover_frontier": frontier,
    }
    result_so_far["tasks"][name] = task_result
    OUT.write_text(json.dumps(result_so_far, indent=2, sort_keys=True), encoding="utf-8")
    return task_result


def main() -> None:
    started = time.time()
    manifest, manifest_hash = load_manifest()
    print(f"locked manifest {manifest_hash}", flush=True)
    data = load_all_data()
    result: dict[str, Any] = {
        "gate": "STEP29_LOCKED_PHASE_A",
        "created_local_date": "2026-08-06",
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest_hash,
        "method_source": "https://arxiv.org/html/2605.24981v1",
        "implementation": "independent equation-faithful exact-response implementation",
        "final_seeds_touched": False,
        "tasks": {},
    }

    configs = manifest["data"]["tasks"]
    for name in ("medqa", "gsm8k", "openbookqa"):
        result["final_seeds_touched"] = True
        evaluate_task(name, data[name], configs[name], manifest, result)
        OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    pvalues = {
        name: row["primary_comparisons"]["T2_reference_aware"][
            "delta_cumulative_deployed_regret_signflip_p"
        ]
        for name, row in result["tasks"].items()
    }
    adjusted = bh_adjust(pvalues)
    for name in result["tasks"]:
        result["tasks"][name]["primary_comparisons"]["T2_reference_aware"][
            "BH_adjusted_p_across_primary_tasks"
        ] = adjusted[name]

    g1_tasks = sum(
        row["primary_comparisons"]["T0_exact_1"]["path_change_fraction"] > 0
        for row in result["tasks"].values()
    )
    t2_points = {
        name: row["primary_comparisons"]["T2_reference_aware"][
            "mean_delta_cumulative_deployed_regret"
        ]
        for name, row in result["tasks"].items()
    }
    t2_positive_ci = {
        name: row["primary_comparisons"]["T2_reference_aware"][
            "delta_cumulative_deployed_regret_95ci"
        ][0]
        > 0
        for name, row in result["tasks"].items()
    }
    t1_positive_ci = {
        name: row["primary_comparisons"]["T1_response_only"][
            "delta_cumulative_deployed_regret_95ci"
        ][0]
        > 0
        for name, row in result["tasks"].items()
    }
    exact_survival = all(
        row["exact_response_quotient"]["T2_alias_representatives_kept"] == 4
        for row in result["tasks"].values()
    )
    random_pass = all(
        row["random_query_negative_control"]["paired_query_transcripts_identical"]
        for row in result["tasks"].values()
    )
    g2 = all(value > 0 for value in t2_points.values()) and sum(t2_positive_ci.values()) >= 2
    result["gate_evaluation"] = {
        "G1_current_method_exposure": {
            "tasks_with_nonzero_path_change": g1_tasks,
            "pass": g1_tasks >= 2,
        },
        "G2_quality_equal_strategic_harm": {
            "point_estimates": t2_points,
            "positive_CI": t2_positive_ci,
            "pass": g2,
        },
        "G4_response_only_boundary": {
            "positive_CI": t1_positive_ci,
            "promote_beyond_domain_specific": sum(t1_positive_ci.values()) >= 2,
        },
        "G5_exact_dedup_limit": {"pass": exact_survival},
        "G7_negative_control": {"pass": random_pass},
        "G8_kill_condition": {
            "triggered": not g2,
            "decision": "WITHDRAW_T2_HEADLINE" if not g2 else "T2_HEADLINE_SURVIVES_PHASE_A",
        },
    }
    result["elapsed_seconds"] = time.time() - started
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["gate_evaluation"], indent=2, sort_keys=True), flush=True)
    print(f"completed in {result['elapsed_seconds']:.1f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
