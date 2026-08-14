"""Step 46: result-blind clone-robust soft-weighting audit.

This runner implements the exact breakpoint integral in Algorithm 1 of
Berriaud and Wattenhofer (arXiv:2602.24024v1) with the three explicit graph
weighting rules preregistered in Step 46.  The resulting distribution replaces
the uniform candidate prior in the locked Select-LLM instantiation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

import run_step29_locked_phase_a as phase_a
import run_step31_phase_b2_grid as step31


ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "STEP46_CLONE_ROBUST_SOFT_WEIGHTING_PREREGISTRATION_2026-08-08.json"
OUT = ROOT / "STEP46_CLONE_ROBUST_SOFT_WEIGHTING_RESULTS_2026-08-08.json"

INPUTS = {
    "step28": ROOT / "STEP28_FINAL_MATRIX_MANIFEST_2026-08-06.json",
    "step29": ROOT / "STEP29_LOCKED_PHASE_A_RESULTS_2026-08-06.json",
    "step30_selection": ROOT / "STEP30_PHASE_B1_TARGET_SELECTION_LOCK_2026-08-06.json",
    "step30_b1": ROOT / "STEP30_PHASE_B1_ALL_TARGET_RESULTS_2026-08-06.json",
    "step31_prereg": ROOT / "STEP31_PHASE_B2_PREREGISTRATION_2026-08-06.json",
    "step31_results": ROOT / "STEP31_PHASE_B2_RESULTS_2026-08-06.json",
    "step46_prereg": PREREG,
}
EXPECTED = {
    "step28": "a16f9a262f8d100899a443e96e3ae141ca266f7b5121cdf5e628fc1a67c201ca",
    "step29": "9a075b143db6a8f11a42dbbbc313a9262a4547249fe81d39caa3f1c7a1540aaa",
    "step30_selection": "98af2c94f8cd3f831ff83fd78fbf6c6755f0fda49460d4c16778b57be1974c7d",
    "step30_b1": "53f45f99edf2116a2e2f871fd7faac425e8ec0d38aa2944fb6e867c3420760bd",
    "step31_prereg": "a757631ef6f43ed90fd86c40f042a8ab9259e6008704fe460e04b8bc612b2af0",
    "step31_results": "67249d225606a2bfc9ef5280e8430a37252c4f5dd7d047be9a90ccfed3eee15b",
    "step46_prereg": "9cdfaeeab69bb4e516e99f452fd932bd34323d3ed3a225c5e2833e8a1544b8e2",
}
TASKS = ("medqa", "gsm8k", "openbookqa")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure_legacy_paths() -> None:
    """Point moved legacy modules at the current audit directory."""
    phase_a.ROOT = ROOT
    phase_a.MANIFEST_PATH = INPUTS["step28"]
    phase_a.OUT = INPUTS["step29"]
    phase_a.med_source.ROOT = ROOT
    phase_a.med_source.CACHE = ROOT / "external_data" / "helm_lite_medqa_v1_0_0_raw"
    phase_a.external_source.ROOT = ROOT
    phase_a.external_source.CACHE = ROOT / "external_data" / "helm_lite_step25_raw"

    step31.ROOT = ROOT
    step31.PREREG = INPUTS["step31_prereg"]
    step31.STEP28 = INPUTS["step28"]
    step31.PHASE_A = INPUTS["step29"]
    step31.SELECTION = INPUTS["step30_selection"]
    step31.PHASE_B1 = INPUTS["step30_b1"]
    step31.OUT = INPUTS["step31_results"]


def verify_inputs() -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    observed = {name: file_sha256(path) for name, path in INPUTS.items()}
    if observed != EXPECTED:
        raise AssertionError({"expected": EXPECTED, "observed": observed})
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg["manifest_id"] != "STEP46_CLONE_ROBUST_SOFT_WEIGHTING_V1":
        raise AssertionError("unexpected Step 46 manifest")
    step31_result = json.loads(INPUTS["step31_results"].read_text(encoding="utf-8"))
    return observed, prereg, step31_result


def pairwise_hamming(responses: np.ndarray) -> np.ndarray:
    responses = np.asarray(responses, dtype=object)
    n = responses.shape[0]
    distances = np.zeros((n, n), dtype=np.float64)
    for left in range(n):
        for right in range(left + 1, n):
            value = float(np.mean(responses[left] != responses[right]))
            distances[left, right] = value
            distances[right, left] = value
    return distances


def graph_from_distances(distances: np.ndarray, radius: float) -> np.ndarray:
    adjacency = distances <= float(radius) + 1e-12
    np.fill_diagonal(adjacency, True)
    return adjacency


def closed_neighborhood_classes(adjacency: np.ndarray) -> list[list[int]]:
    groups: dict[bytes, list[int]] = {}
    for node, row in enumerate(np.asarray(adjacency, dtype=np.bool_)):
        groups.setdefault(row.tobytes(), []).append(node)
    return sorted((sorted(group) for group in groups.values()), key=lambda x: x[0])


def maximal_cliques(adjacency: np.ndarray) -> list[list[int]]:
    n = adjacency.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(
        (left, right)
        for left in range(n)
        for right in range(left + 1, n)
        if bool(adjacency[left, right])
    )
    cliques = [sorted(int(node) for node in clique) for clique in nx.find_cliques(graph)]
    return sorted(cliques, key=lambda values: (values[0], len(values), values))


def graph_weights(
    adjacency: np.ndarray, rule: str
) -> tuple[np.ndarray, list[list[int]], list[list[int]]]:
    n = adjacency.shape[0]
    classes = closed_neighborhood_classes(adjacency)
    cliques = maximal_cliques(adjacency)
    weights = np.zeros(n, dtype=np.float64)

    if rule == "class_uniform":
        for group in classes:
            weights[group] = 1.0 / (len(classes) * len(group))
    elif rule == "mcca":
        for clique in cliques:
            contribution = 1.0 / (len(cliques) * len(clique))
            weights[clique] += contribution
    elif rule == "mccp":
        participation = np.zeros(n, dtype=np.float64)
        for clique in cliques:
            participation[clique] += 1.0
        for clique in cliques:
            denominator = float(np.sum(1.0 / participation[clique]))
            for node in clique:
                weights[node] += 1.0 / (
                    len(cliques) * participation[node] * denominator
                )
    else:
        raise ValueError(rule)

    if np.any(weights <= 0) or not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-10):
        raise AssertionError({"rule": rule, "weights": weights.tolist()})
    return weights, classes, cliques


def group_span(groups: list[list[int]], scores: np.ndarray) -> float:
    return max(
        (float(np.max(scores[group]) - np.min(scores[group])) for group in groups),
        default=0.0,
    )


def integrated_weights(
    distances: np.ndarray,
    alpha: float,
    rule: str,
    scores: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if alpha <= 0:
        raise ValueError("the preregistered source construction requires alpha > 0")
    upper = distances[np.triu_indices(distances.shape[0], 1)]
    internal = sorted(
        set(float(value) for value in upper if 1e-15 < value < alpha - 1e-12)
    )
    boundaries = [0.0, *internal, float(alpha)]
    integrated = np.zeros(distances.shape[0], dtype=np.float64)
    segments: list[dict[str, Any]] = []
    class_high_gap_exposure = 0.0
    clique_high_gap_exposure = 0.0
    expected_class_span = 0.0
    expected_clique_span = 0.0
    maximum_class_span = 0.0
    maximum_clique_span = 0.0

    for left, right in zip(boundaries[:-1], boundaries[1:]):
        share = (right - left) / alpha
        if share <= 0:
            continue
        adjacency = graph_from_distances(distances, left)
        weights, classes, cliques = graph_weights(adjacency, rule)
        integrated += share * weights
        class_span = group_span(classes, scores) if scores is not None else 0.0
        clique_span = group_span(cliques, scores) if scores is not None else 0.0
        maximum_class_span = max(maximum_class_span, class_span)
        maximum_clique_span = max(maximum_clique_span, clique_span)
        expected_class_span += share * class_span
        expected_clique_span += share * clique_span
        class_high_gap_exposure += share * float(class_span >= 0.05 - 1e-12)
        clique_high_gap_exposure += share * float(clique_span >= 0.05 - 1e-12)
        segments.append(
            {
                "left": float(left),
                "right": float(right),
                "measure_fraction": float(share),
                "equivalence_classes": len(classes),
                "maximal_cliques": len(cliques),
                "maximum_equivalence_class_quality_span": class_span,
                "maximum_clique_quality_span": clique_span,
            }
        )

    integrated /= integrated.sum()
    if np.any(integrated <= 0) or not math.isclose(float(integrated.sum()), 1.0, abs_tol=1e-10):
        raise AssertionError("invalid integrated weights")
    entropy = float(-np.sum(integrated * np.log(integrated)))
    diagnostic = {
        "alpha": float(alpha),
        "rule": rule,
        "breakpoint_segments": len(segments),
        "minimum_weight": float(integrated.min()),
        "maximum_weight": float(integrated.max()),
        "entropy_nats": entropy,
        "perplexity": float(math.exp(entropy)),
        "inverse_simpson_effective_candidates": float(1.0 / np.sum(integrated**2)),
        "maximum_equivalence_class_quality_span": maximum_class_span,
        "expected_equivalence_class_quality_span_over_radius": expected_class_span,
        "radius_fraction_with_equivalence_span_at_least_5pp": class_high_gap_exposure,
        "maximum_clique_quality_span": maximum_clique_span,
        "expected_clique_quality_span_over_radius": expected_clique_span,
        "radius_fraction_with_clique_span_at_least_5pp": clique_high_gap_exposure,
        "segments": segments,
    }
    return integrated, diagnostic


def collapse_prior(prior: np.ndarray, parents: np.ndarray, originals: int) -> np.ndarray:
    collapsed = np.zeros(originals, dtype=np.float64)
    for value, parent in zip(prior, parents):
        collapsed[int(parent)] += float(value)
    if not math.isclose(float(collapsed.sum()), 1.0, abs_tol=1e-10):
        raise AssertionError("collapsed prior does not normalize")
    return collapsed


def run_batch_weighted(
    codes: np.ndarray,
    scenario_oracle: np.ndarray,
    parents: np.ndarray,
    original_oracle: np.ndarray,
    pools: np.ndarray,
    budget: int,
    tau: float,
    prior: np.ndarray,
) -> tuple[np.ndarray, ...]:
    runs = pools.shape[0]
    pool_size = pools.shape[1]
    entries = scenario_oracle.shape[0]
    prior = np.asarray(prior, dtype=np.float64)
    if prior.shape != (entries,) or np.any(prior <= 0):
        raise AssertionError("prior shape or positivity failure")
    prior = prior / prior.sum()
    log_prior = np.log(prior)
    log_prior -= log_prior.max()

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
        pool_codes = codes[pool]
        flat_bins = (
            np.arange(pool_size, dtype=np.int64)[:, None] * entries + pool_codes
        ).ravel()
        active = np.ones(pool_size, dtype=bool)
        cumulative = np.zeros(entries, dtype=np.float64)

        for step in range(budget):
            shifted = log_prior + cumulative / tau
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

            selected = int(np.argmax(log_prior + cumulative / tau))
            selected_local_path[run_index, step] = selected
            deploy_regret[run_index, step] = best_original - entry_scores[selected]
            expanded_regret[run_index, step] = best_expanded - entry_scores[selected]
            root_regret[run_index, step] = (
                best_original - original_scores[int(parents[selected])]
            )

    return queried, selected_local_path, deploy_regret, expanded_regret, root_regret


def run_weighted_scenario(
    scenario: dict,
    original_oracle: np.ndarray,
    pools: np.ndarray,
    budget: int,
    tau: float,
    prior: np.ndarray,
) -> dict[str, np.ndarray]:
    queried, selected, deploy, expanded, root = run_batch_weighted(
        scenario["codes"],
        np.asarray(scenario["oracle"], dtype=np.float64),
        np.asarray(scenario["parents"], dtype=np.int64),
        np.asarray(original_oracle, dtype=np.float64),
        pools,
        budget,
        tau,
        prior,
    )
    final_local = selected[:, -1]
    return {
        "queried": queried,
        "selected_local_path": selected,
        "final_local": final_local,
        "final_parent": scenario["parents"][final_local],
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


def interval_contains_zero(interval: list[float]) -> bool:
    return float(interval[0]) <= 0.0 <= float(interval[1])


def step31_cell(step31_result: dict, task: str, cell_id: str) -> dict:
    task_result = step31_result["tasks"][task]
    return next(row for row in task_result["grid_cells"] if row["cell_id"] == cell_id)


def cell_seed(task_index: int, rule_index: int, alpha_index: int, offset: int) -> int:
    return 460000 + task_index * 10000 + rule_index * 1000 + alpha_index * 20 + offset


def evaluate_task(task: str) -> dict[str, Any]:
    configure_legacy_paths()
    observed, prereg, step31_result = verify_inputs()
    locks = step31.verify_locks()
    context = step31.task_context(task, locks)
    data = context["data"]
    clean = context["clean"]
    target = int(context["target_index"])
    tau = float(context["selected_tau"])
    budget = int(context["max_budget"])
    pools = context["pools"]
    task_index = TASKS.index(task)
    expected_target = prereg["locked_inputs"]["targets"][task]
    if phase_a.MODELS[target] != expected_target:
        raise AssertionError("target drift")

    print(f"[{task}] reconstructing locked attacks", flush=True)
    attacks: dict[float, tuple[dict, dict]] = {}
    for nominal in (0.05, 0.1):
        distance_queries = max(1, int(round(nominal * context["examples"])))
        try:
            scenario, metadata = phase_a.build_wrapper(
                data,
                clean,
                target=target,
                distance_queries=distance_queries,
                aliases=4,
                reference_aware=True,
            )
        except ValueError:
            if not (task == "gsm8k" and math.isclose(nominal, 0.1)):
                raise
            continue
        if not metadata["per_example_quality_equal"]:
            raise AssertionError("quality invariance failure")
        if any(abs(float(value)) > 1e-12 for value in metadata["alias_full_score_gaps"]):
            raise AssertionError("alias score drift")
        attacks[nominal] = (scenario, metadata)

    uniform_clean_prior = np.full(len(clean["parents"]), 1.0 / len(clean["parents"]))
    uniform_clean = run_weighted_scenario(
        clean, data["oracle"], pools, budget, tau, uniform_clean_prior
    )
    original_clean = phase_a.run_scenario(clean, data["oracle"], pools, budget, tau)
    parity_fields = (
        "queried",
        "selected_local_path",
        "deploy_regret_path",
        "expanded_regret_path",
        "root_regret_path",
    )
    parity = {field: bool(np.array_equal(uniform_clean[field], original_clean[field])) for field in parity_fields}
    if not all(parity.values()):
        raise AssertionError({"uniform_runner_parity": parity})

    uniform_attacks: dict[float, dict] = {}
    undefended: dict[str, Any] = {}
    for attack_index, (nominal, (scenario, metadata)) in enumerate(attacks.items()):
        prior = np.full(len(scenario["parents"]), 1.0 / len(scenario["parents"]))
        runs = run_weighted_scenario(scenario, data["oracle"], pools, budget, tau, prior)
        uniform_attacks[nominal] = runs
        comparison = phase_a.compare_runs(
            uniform_clean, runs, 459000 + task_index * 10 + attack_index, include_arrays=True
        )
        locked_id = f"d={nominal:g}|a=4|tau={tau:g}|B={budget}"
        locked = step31_cell(step31_result, task, locked_id)
        locked_parity = bool(
            comparison["paired_delta_cumulative_deployed_regret"]
            == locked["paired_delta_cumulative_deployed_regret"]
            and comparison["paired_delta_final_deployed_regret"]
            == locked["paired_delta_final_deployed_regret"]
        )
        if not locked_parity:
            raise AssertionError(f"{task} {locked_id}: Step 31 parity failure")
        undefended[f"{nominal:g}"] = {
            "nominal_distance": nominal,
            "realized_distance": metadata["distance_fraction"],
            "comparison": comparison,
            "step31_paired_array_parity": locked_parity,
        }

    clean_distances = pairwise_hamming(clean["responses"])
    clean_scores = np.asarray(clean["oracle"], dtype=np.float64).mean(axis=1)
    rules = list(prereg["grid"]["graph_rules"])
    alphas = [float(value) for value in prereg["grid"]["alpha_values"]]
    configs: list[dict[str, Any]] = []

    for rule_index, rule in enumerate(rules):
        for alpha_index, alpha in enumerate(alphas):
            clean_prior, clean_weight_info = integrated_weights(
                clean_distances, alpha, rule, clean_scores
            )
            clean_runs = run_weighted_scenario(
                clean, data["oracle"], pools, budget, tau, clean_prior
            )
            clean_cost = phase_a.compare_runs(
                uniform_clean,
                clean_runs,
                cell_seed(task_index, rule_index, alpha_index, 1),
                include_arrays=True,
            )
            cheap_clean = bool(
                abs(float(clean_cost["mean_delta_cumulative_deployed_regret"])) <= 0.10
                and interval_contains_zero(clean_cost["delta_cumulative_deployed_regret_95ci"])
                and interval_contains_zero(clean_cost["delta_final_deployed_regret_95ci"])
            )
            cell: dict[str, Any] = {
                "config_id": f"{rule}|alpha={alpha:g}",
                "rule": rule,
                "alpha": alpha,
                "clean_prior": [float(value) for value in clean_prior],
                "clean_weight_diagnostics": clean_weight_info,
                "clean_cost": clean_cost,
                "cheap_clean": cheap_clean,
                "attacks": {},
            }

            for attack_offset, (nominal, (scenario, metadata)) in enumerate(attacks.items(), start=2):
                attack_distances = pairwise_hamming(scenario["responses"])
                attack_scores = np.asarray(scenario["oracle"], dtype=np.float64).mean(axis=1)
                attack_prior, attack_weight_info = integrated_weights(
                    attack_distances, alpha, rule, attack_scores
                )
                attack_runs = run_weighted_scenario(
                    scenario, data["oracle"], pools, budget, tau, attack_prior
                )
                residual = phase_a.compare_runs(
                    clean_runs,
                    attack_runs,
                    cell_seed(task_index, rule_index, alpha_index, attack_offset),
                    include_arrays=True,
                )
                undefended_mean = float(
                    undefended[f"{nominal:g}"]["comparison"]
                    ["mean_delta_cumulative_deployed_regret"]
                )
                residual_mean = float(residual["mean_delta_cumulative_deployed_regret"])
                ratio = abs(residual_mean) / max(abs(undefended_mean), 1e-15)
                suppressed = bool(
                    ratio <= 0.10 + 1e-12
                    and float(residual["path_change_fraction"]) <= 0.10 + 1e-12
                )
                collapsed = collapse_prior(
                    attack_prior, np.asarray(scenario["parents"]), len(clean_prior)
                )
                root_tv = float(0.5 * np.sum(np.abs(collapsed - clean_prior)))
                aliases = np.arange(len(clean_prior), len(attack_prior), dtype=np.int64)
                family = np.concatenate([np.asarray([target]), aliases])
                cell["attacks"][f"{nominal:g}"] = {
                    "nominal_distance": nominal,
                    "realized_distance": metadata["distance_fraction"],
                    "attack_prior": [float(value) for value in attack_prior],
                    "attack_weight_diagnostics": attack_weight_info,
                    "clean_target_prior_mass": float(clean_prior[target]),
                    "attacked_parent_plus_alias_mass": float(attack_prior[family].sum()),
                    "attacked_alias_only_mass": float(attack_prior[aliases].sum()),
                    "root_collapsed_prior_total_variation": root_tv,
                    "residual": residual,
                    "absolute_residual_to_undefended_ratio": ratio,
                    "attack_suppressed": suppressed,
                    "quality_invariance": True,
                }
            configs.append(cell)
            print(
                f"[{task}] {rule} alpha={alpha:g} cheap={cheap_clean} "
                + " ".join(
                    f"d={key}:ratio={value['absolute_residual_to_undefended_ratio']:.3f}"
                    for key, value in cell["attacks"].items()
                ),
                flush=True,
            )

    return {
        "task": task,
        "target_index": target,
        "target": phase_a.MODELS[target],
        "temperature": tau,
        "budget": budget,
        "runs": int(pools.shape[0]),
        "uniform_runner_parity": parity,
        "undefended": undefended,
        "configs": configs,
        "input_hashes": observed,
    }


def synthetic_tests() -> dict[str, Any]:
    responses = np.asarray(
        [
            ["a", "a", "a", "a", "a"],
            ["a", "a", "b", "a", "a"],
            ["z", "z", "z", "z", "z"],
            ["q", "q", "q", "q", "q"],
        ],
        dtype=object,
    )
    distances = pairwise_hamming(responses)
    scores = np.asarray([0.7, 0.6, 0.8, 0.5])
    permutation = np.asarray([2, 0, 3, 1], dtype=np.int64)
    inverse = np.argsort(permutation)
    tests: dict[str, Any] = {}

    for rule in ("class_uniform", "mcca", "mccp"):
        exact, _ = integrated_weights(distances, 0.6, rule, scores)
        permuted_distances = distances[np.ix_(permutation, permutation)]
        permuted_scores = scores[permutation]
        permuted, _ = integrated_weights(permuted_distances, 0.6, rule, permuted_scores)
        permutation_error = float(np.max(np.abs(exact - permuted[inverse])))

        with_twin = np.concatenate([responses, responses[[2]]], axis=0)
        twin_weights, _ = integrated_weights(
            pairwise_hamming(with_twin), 0.6, rule, None
        )
        twin_equal = bool(math.isclose(twin_weights[2], twin_weights[4], abs_tol=1e-12))

        far_base = np.asarray([["a"] * 10, ["z"] * 10], dtype=object)
        far_twin = np.concatenate([far_base, far_base[[0]]], axis=0)
        far_before, _ = integrated_weights(pairwise_hamming(far_base), 0.5, rule)
        far_after, _ = integrated_weights(pairwise_hamming(far_twin), 0.5, rule)
        far_locality_error = abs(float(far_before[1] - far_after[1]))

        grid = np.linspace(0.0, 0.6, 20001, endpoint=False) + 0.6 / 40002
        dense = np.mean(
            np.stack(
                [graph_weights(graph_from_distances(distances, r), rule)[0] for r in grid]
            ),
            axis=0,
        )
        quadrature_error = float(np.max(np.abs(exact - dense)))
        passed = bool(
            math.isclose(float(exact.sum()), 1.0, abs_tol=1e-12)
            and np.all(exact > 0)
            and permutation_error <= 1e-12
            and twin_equal
            and far_locality_error <= 1e-12
            and quadrature_error <= 1e-4
        )
        if not passed:
            raise AssertionError(
                {
                    "rule": rule,
                    "permutation_error": permutation_error,
                    "twin_equal": twin_equal,
                    "far_locality_error": far_locality_error,
                    "quadrature_error": quadrature_error,
                }
            )
        tests[rule] = {
            "passed": passed,
            "normalizes": True,
            "strictly_positive": True,
            "permutation_equivariance_max_error": permutation_error,
            "exact_twin_weights_equal": twin_equal,
            "far_component_locality_error": far_locality_error,
            "dense_quadrature_max_error": quadrature_error,
        }
    return tests


def aggregate_decision(tasks: dict[str, dict], prereg: dict) -> dict[str, Any]:
    rule_alpha_rows = []
    for rule in prereg["grid"]["graph_rules"]:
        for alpha_value in prereg["grid"]["alpha_values"]:
            alpha = float(alpha_value)
            suppressed = 0
            eligible = 0
            cheap = 0
            residual_ratios = []
            clean_abs = []
            for task in TASKS:
                cell = next(
                    row
                    for row in tasks[task]["configs"]
                    if row["rule"] == rule and math.isclose(float(row["alpha"]), alpha)
                )
                cheap += int(cell["cheap_clean"])
                clean_abs.append(
                    abs(float(cell["clean_cost"]["mean_delta_cumulative_deployed_regret"]))
                )
                for attack in cell["attacks"].values():
                    eligible += 1
                    suppressed += int(attack["attack_suppressed"])
                    residual_ratios.append(float(attack["absolute_residual_to_undefended_ratio"]))
            rule_alpha_rows.append(
                {
                    "config_id": f"{rule}|alpha={alpha:g}",
                    "rule": rule,
                    "alpha": alpha,
                    "suppressed_attack_cells": suppressed,
                    "eligible_attack_cells": eligible,
                    "cheap_clean_tasks": cheap,
                    "mean_absolute_residual_ratio": float(np.mean(residual_ratios)),
                    "maximum_absolute_residual_ratio": float(np.max(residual_ratios)),
                    "mean_absolute_clean_cumulative_cost": float(np.mean(clean_abs)),
                    "dominant": bool(suppressed == 5 and cheap == 3),
                    "partial": bool(suppressed >= 3 and cheap >= 2),
                }
            )

    dominant = [row for row in rule_alpha_rows if row["dominant"]]
    partial = [row for row in rule_alpha_rows if row["partial"]]
    if dominant:
        classification = "DOMINANT_SOFT_FIX"
    elif partial:
        classification = "PARTIAL_SOFT_FIX"
    else:
        classification = "TRADEOFF_OR_FAILURE"
    ranked = sorted(
        rule_alpha_rows,
        key=lambda row: (
            -row["suppressed_attack_cells"],
            -row["cheap_clean_tasks"],
            row["mean_absolute_residual_ratio"],
            row["mean_absolute_clean_cumulative_cost"],
        ),
    )
    return {
        "classification": classification,
        "dominant_configs": dominant,
        "partial_configs": partial,
        "all_configs": rule_alpha_rows,
        "ranked_configs": ranked,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    configure_legacy_paths()
    observed, prereg, _ = verify_inputs()
    tests = synthetic_tests()
    print("synthetic weighting tests passed", flush=True)
    if args.smoke:
        print(json.dumps(tests, indent=2, sort_keys=True))
        return

    started = time.time()
    task_results: dict[str, dict] = {}
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(TASKS))) as pool:
            futures = {pool.submit(evaluate_task, task): task for task in TASKS}
            for future in as_completed(futures):
                task = futures[future]
                task_results[task] = future.result()
                print(f"[{task}] complete", flush=True)
    else:
        for task in TASKS:
            task_results[task] = evaluate_task(task)

    ordered = {task: task_results[task] for task in TASKS}
    decision = aggregate_decision(ordered, prereg)
    output = {
        "gate": "STEP46_CLONE_ROBUST_SOFT_WEIGHTING",
        "created_local_date": "2026-08-08",
        "manifest_sha256": observed["step46_prereg"],
        "runner_sha256": file_sha256(Path(__file__)),
        "source_implementation": {
            "construction": "Algorithm 1 exact breakpoint integral with uniform density",
            "metric": prereg["metric"]["name"],
            "graph_rules": prereg["grid"]["graph_rules"],
            "nonclaim": prereg["adaptation_boundary"]["nonclaim"],
        },
        "input_hashes": observed,
        "synthetic_tests": tests,
        "tasks": ordered,
        "decision": decision,
        "runtime_seconds": time.time() - started,
    }
    OUT.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(f"decision={decision['classification']}", flush=True)
    print(f"output={OUT}", flush=True)


if __name__ == "__main__":
    main()
