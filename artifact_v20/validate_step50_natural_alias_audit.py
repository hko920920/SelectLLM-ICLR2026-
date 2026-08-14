"""Independent consistency and deterministic replay validator for Step 50."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np

import run_step50_natural_alias_audit as audit


EXPECTED = {
    "prereg": "4e5201cb67e8603378477510366b94847797905cf79d40a63fc43842fa2bedda",
    "runner": "a355dfe7e83e5348564f89248865d726af5f1c410dd24fd197bb8527d99261f0",
    "result": "14b77bd3a4193ea3e60ad2f300524fc37393ab250adfa59af016d7758b0f8ea4",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(left: float, right: float, tolerance: float = 1e-12) -> None:
    if not np.isclose(float(left), float(right), rtol=0.0, atol=tolerance):
        raise AssertionError((left, right))


def replay_task(
    task: str,
    context: dict,
    stored: dict,
    prereg: dict,
) -> None:
    phase_a = audit.step46.phase_a
    retained = int(prereg["natural_pair"]["retained_index"])
    added = int(prereg["natural_pair"]["added_index"])
    base, refined = audit.build_registries(context["clean"], retained, added)
    spec = prereg["evaluation"]["tasks"][task]
    temperatures = [float(v) for v in prereg["evaluation"]["temperature_grid"]]
    max_budget = max(int(v) for v in spec["robustness_budgets"])
    pools = context["pools"]
    cache = {}
    for tau in temperatures:
        cache[tau] = (
            phase_a.run_scenario(base, context["data"]["oracle"], pools, max_budget, tau),
            phase_a.run_scenario(
                refined, context["data"]["oracle"], pools, max_budget, tau
            ),
        )
    if len(stored["cells"]) != 7:
        raise AssertionError(f"{task}: expected seven unique cells")
    for identifier, cell in stored["cells"].items():
        tau = float(cell["temperature"])
        budget = int(cell["budget"])
        base_run = audit.truncate_runs(cache[tau][0], base, budget)
        refined_run = audit.truncate_runs(cache[tau][1], refined, budget)
        comparison = cell["comparison"]
        if audit.digest_array(base_run["queried"]) != comparison[
            "query_matrix_digest_base"
        ]:
            raise AssertionError(f"{task}/{identifier}: base query digest")
        if audit.digest_array(refined_run["queried"]) != comparison[
            "query_matrix_digest_refined"
        ]:
            raise AssertionError(f"{task}/{identifier}: refined query digest")
        base_roots = np.asarray(base["parents"])[base_run["selected_local_path"]]
        refined_roots = np.asarray(refined["parents"])[
            refined_run["selected_local_path"]
        ]
        if audit.digest_array(base_roots) != comparison[
            "root_selected_path_digest_base"
        ]:
            raise AssertionError(f"{task}/{identifier}: base root-path digest")
        if audit.digest_array(refined_roots) != comparison[
            "root_selected_path_digest_refined"
        ]:
            raise AssertionError(f"{task}/{identifier}: refined root-path digest")
        changed = np.sum(base_run["queried"] != refined_run["queried"], axis=1)
        close(comparison["path_change_fraction"], np.mean(changed > 0))
        close(comparison["mean_changed_query_positions"], changed.mean())
        delta = (
            refined_run["cumulative_root_regret"]
            - base_run["cumulative_root_regret"]
        )
        stored_delta = np.asarray(
            comparison["paired_delta_cumulative_root_regret"], dtype=np.float64
        )
        if not np.array_equal(delta, stored_delta):
            raise AssertionError(f"{task}/{identifier}: paired cumulative deltas")
        close(comparison["mean_delta_cumulative_root_regret"], delta.mean())
        final_delta = refined_run["final_root_regret"] - base_run["final_root_regret"]
        if not np.array_equal(
            final_delta,
            np.asarray(comparison["paired_delta_final_root_regret"], dtype=np.float64),
        ):
            raise AssertionError(f"{task}/{identifier}: paired final deltas")
        if not comparison["root_and_entry_deltas_exactly_equal"]:
            raise AssertionError(f"{task}/{identifier}: root/entry equality")

    primary_budget = int(spec["primary_budget"])
    seeds = list(range(50000, 50500))
    random_queries = np.asarray(
        [
            random.Random(seed + 99173).sample(pool.tolist(), primary_budget)
            for seed, pool in zip(seeds, pools)
        ],
        dtype=np.int64,
    )
    fixed_base = audit.run_fixed_queries(
        base, context["data"]["oracle"], pools, random_queries
    )
    fixed_refined = audit.run_fixed_queries(
        refined, context["data"]["oracle"], pools, random_queries
    )
    if not np.array_equal(fixed_base["queried"], fixed_refined["queried"]):
        raise AssertionError(f"{task}: random query matrices differ")
    base_roots = np.asarray(base["parents"])[fixed_base["selected_local_path"]]
    refined_roots = np.asarray(refined["parents"])[fixed_refined["selected_local_path"]]
    if not np.array_equal(base_roots, refined_roots):
        raise AssertionError(f"{task}: random selected-root paths differ")
    if not np.array_equal(
        fixed_base["cumulative_root_regret"],
        fixed_refined["cumulative_root_regret"],
    ):
        raise AssertionError(f"{task}: random cumulative regrets differ")
    if not stored["random_query_control"]["pass"]:
        raise AssertionError(f"{task}: stored random control failed")


def main() -> None:
    paths = {
        "prereg": audit.PREREG,
        "runner": audit.RUNNER,
        "result": audit.OUT,
    }
    observed = {name: sha(path) for name, path in paths.items()}
    if observed != EXPECTED:
        raise AssertionError({"expected": EXPECTED, "observed": observed})
    audit.configure_paths()
    prereg, _, _ = audit.verify_locks()
    result = json.loads(audit.OUT.read_text(encoding="utf-8"))
    if result["manifest_id"] != "STEP50_NATURAL_EXACT_ALIAS_RESULTS_V1":
        raise AssertionError("unexpected result manifest")
    legacy_locks = audit.step46.step31.verify_locks()
    contexts = {
        task: audit.step46.step31.task_context(task, legacy_locks)
        for task in audit.TASKS
    }
    pair_lists = audit.exact_pairs(contexts)
    intersection = set(tuple(row) for row in pair_lists[audit.TASKS[0]])
    for task in audit.TASKS[1:]:
        intersection &= set(tuple(row) for row in pair_lists[task])
    if intersection != {(9, 10)}:
        raise AssertionError(intersection)
    for task in audit.TASKS:
        replay_task(task, contexts[task], result["tasks"][task], prereg)

    copied_tasks = copy.deepcopy(result["tasks"])
    recomputed = audit.apply_multiplicity_and_classify(copied_tasks, True)
    for key in (
        "classification",
        "pair_valid",
        "random_query_controls_pass",
        "path_material_tasks",
        "outcome_material_tasks",
        "outcome_harm_tasks",
        "outcome_benefit_tasks",
    ):
        if recomputed[key] != result["decision"][key]:
            raise AssertionError((key, recomputed[key], result["decision"][key]))
    if result["decision"]["classification"] != "LIMITED_NATURAL_OUTCOME_ANCHOR":
        raise AssertionError("unexpected Step 50 classification")
    print("PASS_STEP50_NATURAL_ALIAS_AUDIT")
    print("classification =", result["decision"]["classification"])
    for row in result["decision"]["primary_rows"]:
        print(
            row["task"],
            "path=", row["path_change_fraction"],
            "changed=", row["mean_changed_query_positions"],
            "delta_CReg=", row["mean_delta_cumulative_root_regret"],
            "CI=", row["delta_cumulative_root_regret_95ci"],
            "q=", row["primary_family_bh_q"],
        )
    print("random_query_controls = exact zero in all tasks")


if __name__ == "__main__":
    main()
