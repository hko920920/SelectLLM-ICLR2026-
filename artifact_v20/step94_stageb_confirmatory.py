from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np

import step94_common as common
from step94_executable_variant_endpoint import execute_registry_from_raw_texts


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / f"STEP94_STAGEA_FROZEN_CONFIG_{common.DATE}.json"
LEDGER_PATH = ROOT / f"STEP94_STAGEA_COMPLETE_LEDGER_{common.DATE}.json"
MANIFEST_PATH = ROOT / f"STEP94_STAGE0_MANIFEST_{common.DATE}.json"
LOCK_PATH = ROOT / f"STEP94_CONFIRMATORY_EXECUTION_LOCK_{common.DATE}.json"
RAW_PATH = ROOT / f"STEP94_CONFIRMATORY_RAW_{common.DATE}.npz"
RESULT_PATH = ROOT / f"STEP94_CONFIRMATORY_RESULTS_{common.DATE}.json"

CONFIRMATORY_SEEDS = tuple(range(941000, 943000))
BOOTSTRAP_SEED = 94900
SIGNFLIP_SEED = 94901


def source_sha256(function: object) -> str:
    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


def selected_manifest_row(manifest: dict[str, Any], task_key: str) -> dict[str, Any]:
    matches = [row for row in manifest["tasks"] if row["task_key"] == task_key]
    if len(matches) != 1:
        raise AssertionError({"task_key": task_key, "matches": len(matches)})
    return matches[0]


def create_execution_lock() -> dict[str, Any]:
    if LOCK_PATH.exists():
        raise RuntimeError(f"refusing to overwrite lock: {LOCK_PATH}")
    if RAW_PATH.exists() or RESULT_PATH.exists():
        raise RuntimeError("confirmatory output predates execution lock")
    for path in (CONFIG_PATH, LEDGER_PATH, MANIFEST_PATH):
        if not path.exists():
            raise FileNotFoundError(path)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if config["decision"] != "GO_TO_ONE_TIME_CONFIRMATORY_LOCK":
        raise RuntimeError(config["decision"])
    selected = config["selected_task"]
    task_key = str(selected["task_key"])
    row = selected_manifest_row(manifest, task_key)
    if selected["test_input_sha256"] != row["test_input_sha256"]:
        raise AssertionError("test-input binding mismatch")
    if selected["sealed_outcome_sha256"] != row["sealed_outcome_sha256"]:
        raise AssertionError("sealed-outcome binding mismatch")
    test_input = ROOT / selected["test_input_file"]
    sealed_outcome = ROOT / selected["sealed_outcome_file"]
    if common.sha256_path(test_input) != selected["test_input_sha256"]:
        raise AssertionError("test input changed")
    if common.sha256_path(sealed_outcome) != selected["sealed_outcome_sha256"]:
        raise AssertionError("sealed outcome changed")
    for relative, expected in config["all_model_sha256"].items():
        path = ROOT / relative
        if common.sha256_path(path) != expected:
            raise AssertionError({"model": relative, "hash": False})

    code_paths = [
        ROOT / "step93_common.py",
        ROOT / "step94_common.py",
        ROOT / "step94_executable_variant_endpoint.py",
        ROOT / "step94_stageb_confirmatory.py",
        ROOT / "lock_step94_confirmatory.py",
        ROOT / "validate_step94_independent.py",
    ]
    lock = {
        "lock_id": "STEP94_ONE_TIME_CONFIRMATORY_EXECUTION_LOCK_V1",
        "date": common.DATE,
        "authority_sha256": config["authority_sha256"],
        "stage0_manifest_sha256": common.sha256_path(MANIFEST_PATH),
        "stagea_ledger_sha256": common.sha256_path(LEDGER_PATH),
        "stagea_config_sha256": common.sha256_path(CONFIG_PATH),
        "selected_task": {
            "task_key": task_key,
            "dataset": selected["dataset"],
            "test_input_file": selected["test_input_file"],
            "test_input_sha256": selected["test_input_sha256"],
            "sealed_outcome_file": selected["sealed_outcome_file"],
            "sealed_outcome_sha256": selected["sealed_outcome_sha256"],
            "selected_condition": selected["selected_condition"],
            "integrated_variants": selected["integrated_variants"],
        },
        "model_sha256": config["all_model_sha256"],
        "code_sha256": {path.name: common.sha256_path(path) for path in code_paths},
        "source_assertions": {
            "raw_endpoint_signature": str(inspect.signature(execute_registry_from_raw_texts)),
            "raw_endpoint_source_sha256": source_sha256(execute_registry_from_raw_texts),
            "exact_similarity_source_sha256": source_sha256(common.exact_match_similarity),
            "active_source_sha256": source_sha256(common.run_active),
            "fixed_source_sha256": source_sha256(common.run_fixed),
            "bootstrap_source_sha256": source_sha256(common.bootstrap_interval),
            "signflip_source_sha256": source_sha256(common.signflip_pvalue),
        },
        "confirmatory": {
            "seeds": [CONFIRMATORY_SEEDS[0], CONFIRMATORY_SEEDS[-1]],
            "seed_count": len(CONFIRMATORY_SEEDS),
            "pool_size": common.POOL_SIZE,
            "budget": common.BUDGET,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "signflip_seed": SIGNFLIP_SEED,
            "bootstrap_repetitions": 10_000,
            "signflip_repetitions": 100_000,
        },
        "pre_outcome_assertions": {
            "raw_result_absent": not RAW_PATH.exists(),
            "summary_result_absent": not RESULT_PATH.exists(),
            "selected_one_task_only": True,
            "unselected_test_outcomes_not_authorized": True,
        },
    }
    common.json_dump(LOCK_PATH, lock)
    return lock


def main() -> None:
    for output in (RAW_PATH, RESULT_PATH):
        if output.exists():
            raise RuntimeError(f"refusing to overwrite confirmatory output: {output}")
    if not LOCK_PATH.exists():
        raise RuntimeError("execution lock must exist before confirmatory runner starts")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if lock["stagea_config_sha256"] != common.sha256_path(CONFIG_PATH):
        raise AssertionError("config changed after lock")
    if lock["stagea_ledger_sha256"] != common.sha256_path(LEDGER_PATH):
        raise AssertionError("ledger changed after lock")
    for relative, expected in lock["model_sha256"].items():
        if common.sha256_path(ROOT / relative) != expected:
            raise AssertionError({"model": relative, "post_lock_hash": False})
    for filename, expected in lock["code_sha256"].items():
        if common.sha256_path(ROOT / filename) != expected:
            raise AssertionError({"code": filename, "post_lock_hash": False})

    selected = config["selected_task"]
    condition = selected["selected_condition"]
    task = common.task_by_key(str(selected["task_key"]))
    if task.key != "20newsgroups":
        raise AssertionError({"frozen_selected_task": task.key})
    input_path = ROOT / selected["test_input_file"]
    sealed_path = ROOT / selected["sealed_outcome_file"]
    if common.sha256_path(input_path) != selected["test_input_sha256"]:
        raise AssertionError("input changed after lock")
    if common.sha256_path(sealed_path) != selected["sealed_outcome_sha256"]:
        raise AssertionError("sealed outcome changed after lock")

    inputs = json.loads(input_path.read_text(encoding="utf-8"))
    texts = [str(row["text"]) for row in inputs["rows"]]
    source_indices = np.asarray(
        [int(row["source_index"]) for row in inputs["rows"]], dtype=np.int64
    )
    roster = [int(value) for value in condition["roster_bank_root_ids"]]
    root_paths = [
        ROOT
        / "step94_models"
        / task.key
        / f"step94_{task.key}_root_{bank_root:02d}.safetensors"
        for bank_root in roster
    ]
    variant_paths = [
        ROOT / value["checkpoint"] for value in selected["integrated_variants"]
    ]
    endpoint = execute_registry_from_raw_texts(
        texts,
        root_paths,
        variant_paths,
        [float(value) for value in condition["thresholds"]],
        task.n_classes,
    )
    clean_codes = np.asarray(endpoint["clean_root_codes"], dtype=np.int64)
    alias_codes = np.asarray(endpoint["variant_codes"], dtype=np.int64)
    parent_predictions = np.asarray(
        endpoint["variant_parent_predictions"], dtype=np.int64
    )
    parent_position = int(condition["parent_position"])
    if not np.all(parent_predictions == clean_codes[:, [parent_position]]):
        raise AssertionError("integrated parent replay differs from clean parent")

    # This is the one authorized sealed-outcome opening.
    with np.load(sealed_path, allow_pickle=False) as sealed:
        sealed_indices = sealed["source_indices"].astype(np.int64)
        labels = sealed["labels"].astype(np.int64)
    if not np.array_equal(source_indices, sealed_indices):
        raise AssertionError("input/outcome source-index mismatch")
    if len(labels) != len(clean_codes):
        raise AssertionError("input/outcome length mismatch")

    parent_feedback = common.exact_match_feedback(
        clean_codes[:, [parent_position]], labels
    )
    alias_feedback = common.exact_match_feedback(alias_codes, labels)
    coordinate_nonimproving = bool(
        np.all(
            alias_feedback
            <= np.repeat(parent_feedback, common.N_ALIASES, axis=1)
        )
    )
    alias_losses = (
        np.mean(parent_feedback, axis=0)[0] - np.mean(alias_feedback, axis=0)
    ).astype(np.float64)

    clean_parents = np.arange(common.N_ROSTER_ROOTS, dtype=np.int64)
    refined_codes = np.column_stack([clean_codes, alias_codes]).astype(np.int64)
    refined_parents = np.concatenate(
        [
            clean_parents,
            np.full(common.N_ALIASES, parent_position, dtype=np.int64),
        ]
    )
    pools = common.sample_pools(
        len(labels), CONFIRMATORY_SEEDS, pool_size=common.POOL_SIZE
    )
    tau = float(condition["tau"])
    clean = common.run_active(
        clean_codes,
        labels,
        clean_parents,
        clean_codes,
        pools,
        CONFIRMATORY_SEEDS,
        common.BUDGET,
        tau,
    )
    refined = common.run_active(
        refined_codes,
        labels,
        refined_parents,
        clean_codes,
        pools,
        CONFIRMATORY_SEEDS,
        common.BUDGET,
        tau,
    )
    refined_fixed = common.run_fixed(
        refined_codes,
        labels,
        refined_parents,
        clean_codes,
        pools,
        clean.queries,
        CONFIRMATORY_SEEDS,
    )

    terminal = refined.terminal - clean.terminal
    cumulative = refined.cumulative - clean.cumulative
    fixed_terminal = refined_fixed.terminal - clean.terminal
    fixed_cumulative = refined_fixed.cumulative - clean.cumulative
    active_minus_fixed = terminal - fixed_terminal
    terminal_summary = common.effect_summary(
        terminal, BOOTSTRAP_SEED, SIGNFLIP_SEED
    )
    active_minus_fixed_summary = common.effect_summary(
        active_minus_fixed, BOOTSTRAP_SEED + 1, SIGNFLIP_SEED + 1
    )
    cumulative_summary = common.effect_summary(
        cumulative, BOOTSTRAP_SEED + 2, SIGNFLIP_SEED + 2
    )

    raw_arrays = {
        "source_indices": source_indices,
        "labels": labels.astype(np.int16),
        "clean_codes": clean_codes.astype(np.int16),
        "alias_codes": alias_codes.astype(np.int16),
        "variant_gate_scores": np.asarray(
            endpoint["variant_gate_scores"], dtype=np.float64
        ),
        "parent_predictions": parent_predictions.astype(np.int16),
        "pools": pools,
        "clean_terminal": clean.terminal,
        "refined_terminal": refined.terminal,
        "refined_fixed_terminal": refined_fixed.terminal,
        "clean_cumulative": clean.cumulative,
        "refined_cumulative": refined.cumulative,
        "refined_fixed_cumulative": refined_fixed.cumulative,
        "clean_queries": clean.queries,
        "refined_queries": refined.queries,
        "clean_roots": clean.roots,
        "refined_roots": refined.roots,
        "refined_fixed_roots": refined_fixed.roots,
        "terminal_delta": terminal,
        "cumulative_delta": cumulative,
        "fixed_terminal_delta": fixed_terminal,
        "fixed_cumulative_delta": fixed_cumulative,
        "active_minus_fixed_terminal_delta": active_minus_fixed,
    }
    np.savez_compressed(RAW_PATH, **raw_arrays)

    path_change_rate = float(
        np.mean(np.any(clean.queries != refined.queries, axis=1))
    )
    query_set_change_rate = float(
        np.mean(
            [
                set(left.tolist()) != set(right.tolist())
                for left, right in zip(clean.queries, refined.queries)
            ]
        )
    )
    fixed_root_exact = bool(np.array_equal(refined_fixed.roots, clean.roots))
    variant_hashes = [common.sha256_path(path) for path in variant_paths]
    gates = {
        "pre_outcome_integrity": True,
        "same_literal_similarity": True,
        "four_distinct_integrated_checkpoint_hashes": len(set(variant_hashes))
        == common.N_ALIASES,
        "raw_input_endpoint_signature_has_no_forbidden_input": all(
            token not in lock["source_assertions"]["raw_endpoint_signature"].lower()
            for token in (
                "label",
                "reference",
                "item_id",
                "lookup",
                "peer",
                "pool",
                "trajectory",
                "posterior",
                "selector",
            )
        ),
        "coordinate_wise_nonimproving": coordinate_nonimproving,
        "each_alias_loss_at_most_one_point": bool(np.max(alias_losses) <= 0.0100),
        "path_change_at_least_half": bool(path_change_rate >= 0.50),
        "terminal_mean_at_least_half_point": bool(terminal_summary["mean"] >= 0.005),
        "terminal_bootstrap_lower_positive": bool(terminal_summary["bootstrap_95"][0] > 0),
        "terminal_signflip_at_most_point05": bool(
            terminal_summary["one_sided_signflip_p"] <= 0.05
        ),
        "active_minus_fixed_mean_at_least_half_point": bool(
            active_minus_fixed_summary["mean"] >= 0.005
        ),
        "active_minus_fixed_bootstrap_lower_positive": bool(
            active_minus_fixed_summary["bootstrap_95"][0] > 0
        ),
        "cumulative_mean_positive": bool(cumulative_summary["mean"] > 0),
        "cumulative_bootstrap_lower_positive": bool(
            cumulative_summary["bootstrap_95"][0] > 0
        ),
        "fixed_terminal_exact_zero": bool(np.max(np.abs(fixed_terminal)) == 0),
        "fixed_cumulative_exact_zero": bool(np.max(np.abs(fixed_cumulative)) == 0),
        "fixed_root_history_bitwise_exact": fixed_root_exact,
    }
    decision = (
        "GO_FRESH_SOURCE_FAITHFUL_LEARNED_VARIANT_BRIDGE"
        if all(gates.values())
        else "NO_GO_RETAIN_FRESH_TASK_NEGATIVE"
    )
    results = {
        "result_id": "STEP94_FRESH_TASK_CONFIRMATORY_V1",
        "date": common.DATE,
        "decision": decision,
        "execution_lock_sha256": common.sha256_path(LOCK_PATH),
        "stagea_config_sha256": common.sha256_path(CONFIG_PATH),
        "raw_sha256": common.sha256_path(RAW_PATH),
        "selected_task": task.key,
        "dataset": selected["dataset"],
        "roster_bank_root_ids": roster,
        "parent_position": parent_position,
        "parent_bank_root_id": int(condition["parent_bank_root_id"]),
        "tau": tau,
        "thresholds": [float(value) for value in condition["thresholds"]],
        "endpoint": {
            "hidden_sha256": endpoint["hidden_sha256"],
            "integrated_variant_sha256": variant_hashes,
            "integrated_variant_parameter_counts": [
                common.tensor_audit(path)["parameter_count"] for path in variant_paths
            ],
            "parent_replay_exact": True,
            "forbidden_runtime_inputs": False,
        },
        "quality": {
            "parent_accuracy": float(np.mean(parent_feedback)),
            "alias_accuracies": np.mean(alias_feedback, axis=0).tolist(),
            "alias_losses": alias_losses.tolist(),
            "coordinate_wise_nonimproving": coordinate_nonimproving,
            "trigger_fractions": np.mean(alias_codes >= task.n_classes, axis=0).tolist(),
        },
        "mechanism": {
            "path_change_rate": path_change_rate,
            "query_set_change_rate": query_set_change_rate,
            "final_root_change_rate": float(
                np.mean(clean.roots[:, -1] != refined.roots[:, -1])
            ),
        },
        "effects": {
            "terminal": terminal_summary,
            "active_minus_fixed_terminal": active_minus_fixed_summary,
            "cumulative": cumulative_summary,
            "fixed_terminal_max_abs": float(np.max(np.abs(fixed_terminal))),
            "fixed_cumulative_max_abs": float(np.max(np.abs(fixed_cumulative))),
            "fixed_root_history_exact": fixed_root_exact,
        },
        "gates": gates,
        "failed_gates": [key for key, value in gates.items() if not value],
    }
    common.json_dump(RESULT_PATH, results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
