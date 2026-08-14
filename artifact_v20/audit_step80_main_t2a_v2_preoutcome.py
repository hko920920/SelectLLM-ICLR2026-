"""Pre-outcome code and protocol audit for Step 80."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import run_step80_main_t2a_v2_transfer as runner


ROOT = Path(__file__).resolve().parent
AUDITOR = ROOT / "audit_step80_main_t2a_v2_preoutcome.py"
VALIDATOR = ROOT / "validate_step80_main_t2a_v2_transfer.py"
RESULT = ROOT / "STEP80_MAIN_T2A_V2_PREOUTCOME_AUDIT_RESULTS_2026-08-10.json"
RESERVED = (
    ROOT / "STEP80_MAIN_T2A_V2_TRANSFER_RESULTS_2026-08-10.json",
    ROOT / "STEP80_MAIN_T2A_V2_TRANSFER_RAW_2026-08-10.npz",
    ROOT / "STEP80_MAIN_T2A_V2_TRANSFER_VALIDATION_2026-08-10.json",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def call_attributes(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def toy_scenario(order: list[int] | None = None) -> dict[str, Any]:
    responses = np.asarray(
        [
            ["a", "x", "u", "m", "k", "q"],
            ["a", "y", "v", "m", "l", "r"],
            ["b", "y", "u", "n", "l", "q"],
            ["b", "x", "v", "n", "k", "r"],
        ],
        dtype=object,
    )
    oracle = np.asarray(
        [
            [1, 0, 1, 0, 1, 0],
            [1, 1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1],
        ],
        dtype=np.float64,
    )
    parents = np.asarray([0, 1, 2, 3], dtype=np.int64)
    labels = ["toy-z", "toy-a", "toy-m", "toy-b"]
    if order is None:
        order = list(range(4))
    return {
        "kind": "toy",
        "responses": responses[order],
        "oracle": oracle[order],
        "parents": parents[order],
        "labels": [labels[index] for index in order],
        "response_hashes": [runner.response_vector_hash(responses[index]) for index in order],
    }


def main() -> None:
    if RESULT.exists():
        raise FileExistsError("pre-outcome audit result already exists")
    if any(path.exists() for path in RESERVED):
        raise FileExistsError("scientific output exists before pre-outcome audit")
    prereg, frozen = runner.verify_frozen_inputs()
    runner_calls = call_attributes(runner.RUNNER)
    validator_imports = imported_modules(VALIDATOR)
    static_checks = {
        "runner_calls_public_reference_constructor": "build_public_reference_variant" in runner_calls,
        "runner_does_not_call_legacy_build_wrapper": "build_wrapper" not in runner_calls,
        "runner_uses_fsum": "fsum" in runner_calls,
        "runner_uses_numeric_sort": "sort" in runner_calls,
        "validator_does_not_import_runner": "run_step80_main_t2a_v2_transfer" not in validator_imports,
        "three_query_policies": len(runner.QUERY_POLICIES) == 3,
        "three_root_policies": len(runner.ROOT_POLICIES) == 3,
        "sixteen_permutations": runner.PERMUTATION_SEEDS == tuple(range(803000, 803016)),
        "five_hundred_frozen_runs": runner.PAIRED_SEEDS == tuple(range(91000, 91500)),
    }
    if not all(static_checks.values()):
        raise AssertionError(static_checks)

    base = toy_scenario()
    reference = runner.canonical_scenario(base, "toy-reference")
    permutation_checks: dict[str, bool] = {}
    for seed in range(32):
        order = list(np.random.default_rng(seed).permutation(4))
        observed = runner.canonical_scenario(toy_scenario(order), f"toy-{seed}")
        permutation_checks[str(seed)] = bool(
            observed["canonical_digest"] == reference["canonical_digest"]
            and np.array_equal(observed["codes"], reference["codes"])
            and np.array_equal(observed["parents"], reference["parents"])
            and np.array_equal(observed["oracle"], reference["oracle"])
        )
    if not all(permutation_checks.values()):
        raise AssertionError("synthetic canonical permutation failure")

    pool = np.asarray([9, 3, 7, 1], dtype=np.int64)
    tied = np.arange(4, dtype=np.int64)
    query_checks = {
        "minimum": runner.choose_query_position(
            runner.QUERY_POLICIES[0], tied, pool, "toy", 11, 0
        ) == 3,
        "maximum": runner.choose_query_position(
            runner.QUERY_POLICIES[1], tied, pool, "toy", 11, 0
        ) == 0,
        "hash_member": runner.choose_query_position(
            runner.QUERY_POLICIES[2], tied, pool, "toy", 11, 0
        ) in {0, 1, 2, 3},
    }
    parents = np.asarray([8, 2, 5, 2], dtype=np.int64)
    root_checks = {
        "minimum": runner.choose_root(
            runner.ROOT_POLICIES[0], tied, parents, "toy", 11, 0
        ) == 2,
        "maximum": runner.choose_root(
            runner.ROOT_POLICIES[1], tied, parents, "toy", 11, 0
        ) == 8,
        "hash_member": runner.choose_root(
            runner.ROOT_POLICIES[2], tied, parents, "toy", 11, 0
        ) in {2, 5, 8},
    }
    if not all(query_checks.values()) or not all(root_checks.values()):
        raise AssertionError({"query": query_checks, "root": root_checks})

    toy_pools = np.asarray([[0, 1, 2, 3, 4, 5]], dtype=np.int64)
    first = runner.active_runs(
        base,
        np.asarray(base["oracle"], dtype=np.float64),
        toy_pools,
        (123,),
        4,
        2.0,
        "toy",
        runner.PRIMARY_QUERY_POLICY,
    )
    second = runner.active_runs(
        toy_scenario([3, 1, 0, 2]),
        np.asarray(base["oracle"], dtype=np.float64),
        toy_pools,
        (123,),
        4,
        2.0,
        "toy",
        runner.PRIMARY_QUERY_POLICY,
    )
    synthetic_run_checks = {
        "query_path_exact": np.array_equal(first["queried"], second["queried"]),
        "acquisition_values_bitwise": np.array_equal(
            first["selected_acquisition_values"].view(np.uint64),
            second["selected_acquisition_values"].view(np.uint64),
        ),
        "canonical_digest_exact": first["canonical_digest"] == second["canonical_digest"],
        "all_root_paths_exact": all(
            np.array_equal(first["selected_root_paths"][policy], second["selected_root_paths"][policy])
            for policy in runner.ROOT_POLICIES
        ),
        "all_regret_paths_bitwise": all(
            np.array_equal(
                first["root_regret_paths"][policy].view(np.uint64),
                second["root_regret_paths"][policy].view(np.uint64),
            )
            for policy in runner.ROOT_POLICIES
        ),
    }
    if not all(synthetic_run_checks.values()):
        raise AssertionError(synthetic_run_checks)

    runner.step54.step46.configure_legacy_paths()
    all_data = runner.step54.step46.phase_a.load_all_data()
    historical = runner.load_json(runner.FROZEN_FILES["step54_results"])
    real_binding_checks: dict[str, Any] = {}
    for task_index, task in enumerate(runner.TASKS):
        context = runner.load_task_context(
            task,
            task_index,
            prereg["frozen_tasks"][task],
            {"all_data": all_data, "step54_results": historical},
        )
        real_binding_checks[task] = {
            "all_checks_pass": all(context["checks"].values()),
            "pool_digest": context["pool_digest"],
            "fixed_query_digest": context["fixed_query_digest"],
            "clean_canonical_digest": runner.canonical_scenario(
                context["clean"], f"audit_{task}_clean"
            )["canonical_digest"],
            "refined_canonical_digest": runner.canonical_scenario(
                context["refined"], f"audit_{task}_refined"
            )["canonical_digest"],
        }
    if not all(value["all_checks_pass"] for value in real_binding_checks.values()):
        raise AssertionError(real_binding_checks)

    code_hashes = {
        "runner": digest(runner.RUNNER),
        "validator": digest(VALIDATOR),
        "preoutcome_auditor": digest(AUDITOR),
    }
    payload = {
        "manifest_id": "STEP80_MAIN_T2A_V2_PREOUTCOME_AUDIT_V1",
        "status": "PASS_STEP80_PREOUTCOME_AUDIT",
        "outcomes_run": False,
        "reserved_outputs_absent": [str(path.name) for path in RESERVED],
        "frozen_input_sha256": frozen,
        "code_sha256": code_hashes,
        "static_checks": static_checks,
        "synthetic_canonical_permutations_passed": sum(permutation_checks.values()),
        "query_tie_checks": query_checks,
        "root_tie_checks": root_checks,
        "synthetic_run_checks": synthetic_run_checks,
        "real_condition_binding_checks": real_binding_checks,
        "scientific_outcome_statistics_computed": False,
    }
    RESULT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
