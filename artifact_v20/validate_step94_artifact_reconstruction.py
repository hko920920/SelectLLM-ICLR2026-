"""Read-only independent reconstruction of the Step 94 sealed result."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import step94_common as common


ROOT = Path(__file__).resolve().parent
DATE = common.DATE
PROTOCOL = ROOT / f"STEP94_FRESH_TASK_LEARNED_VARIANT_PREREGISTRATION_{DATE}.md"
AMENDMENTS = [
    ROOT / f"STEP94_PREREGISTRATION_AMENDMENT_{letter}_{DATE}.md"
    for letter in "ABCD"
]
STAGE0 = ROOT / f"STEP94_STAGE0_MANIFEST_{DATE}.json"
LEDGER = ROOT / f"STEP94_STAGEA_COMPLETE_LEDGER_{DATE}.json"
CONFIG = ROOT / f"STEP94_STAGEA_FROZEN_CONFIG_{DATE}.json"
LOCK = ROOT / f"STEP94_CONFIRMATORY_EXECUTION_LOCK_{DATE}.json"
RAW = ROOT / f"STEP94_CONFIRMATORY_RAW_{DATE}.npz"
RESULT = ROOT / f"STEP94_CONFIRMATORY_RESULTS_{DATE}.json"
RECEIPT = ROOT / f"STEP94_INDEPENDENT_VALIDATION_{DATE}.json"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def close(observed: Any, expected: Any, atol: float = 1e-14) -> bool:
    return bool(
        np.allclose(
            np.asarray(observed, dtype=np.float64),
            np.asarray(expected, dtype=np.float64),
            rtol=0.0,
            atol=atol,
        )
    )


def require_close(observed: Any, expected: Any, name: str) -> None:
    if not close(observed, expected):
        raise AssertionError({name: {"observed": observed, "expected": expected}})


def main() -> None:
    for path in [PROTOCOL, *AMENDMENTS, STAGE0, LEDGER, CONFIG, LOCK, RAW, RESULT, RECEIPT]:
        if not path.is_file():
            raise FileNotFoundError(path)

    stage0 = load(STAGE0)
    ledger = load(LEDGER)
    config = load(CONFIG)
    lock = load(LOCK)
    result = load(RESULT)
    receipt = load(RECEIPT)

    authority_paths = [PROTOCOL, *AMENDMENTS]
    authority_map = {path.name: common.sha256_path(path) for path in authority_paths}
    if lock["authority_sha256"] != authority_map or config["authority_sha256"] != authority_map:
        raise AssertionError("outcome-blind authority binding drift")
    expected_bindings = {
        "stage0_manifest_sha256": common.sha256_path(STAGE0),
        "stagea_ledger_sha256": common.sha256_path(LEDGER),
        "stagea_config_sha256": common.sha256_path(CONFIG),
    }
    for key, digest in expected_bindings.items():
        if lock[key] != digest:
            raise AssertionError({key: "execution-lock binding drift"})
    if result["execution_lock_sha256"] != common.sha256_path(LOCK):
        raise AssertionError("result-to-lock binding drift")
    if result["stagea_config_sha256"] != common.sha256_path(CONFIG):
        raise AssertionError("result-to-config binding drift")
    if result["raw_sha256"] != common.sha256_path(RAW):
        raise AssertionError("result-to-raw binding drift")
    if receipt["result_sha256"] != common.sha256_path(RESULT):
        raise AssertionError("receipt-to-result binding drift")
    if receipt["raw_sha256"] != common.sha256_path(RAW):
        raise AssertionError("receipt-to-raw binding drift")
    if receipt["lock_sha256"] != common.sha256_path(LOCK):
        raise AssertionError("receipt-to-lock binding drift")

    if len(stage0["tasks"]) != 3 or [row["task_key"] for row in stage0["tasks"]] != [
        "20newsgroups",
        "massive_en_us",
        "sst5",
    ]:
        raise AssertionError("frozen task-menu drift")
    for row in stage0["tasks"]:
        for key in ("test_input_file", "sealed_outcome_file"):
            path = ROOT / row[key]
            digest_key = key.replace("_file", "_sha256")
            if not path.is_file() or common.sha256_path(path) != row[digest_key]:
                raise AssertionError({"stage0_file": row[key]})

    if len(ledger["tasks"]) != 3 or ledger["selected_task_key"] != "20newsgroups":
        raise AssertionError("Stage-A task selection drift")
    for task in ledger["tasks"]:
        if task["search_grid_count"] != 648 or len(task["search_complete_ledger"]) != 648:
            raise AssertionError({"search_grid": task["task_key"]})
        if len(task["verification_top_24"]) != 24:
            raise AssertionError({"verification_grid": task["task_key"]})
    selected = config["selected_task"]
    condition = selected["selected_condition"]
    if selected["task_key"] != "20newsgroups" or condition["parent_bank_root_id"] != 10:
        raise AssertionError("frozen selected condition drift")
    require_close(condition["mean_active_minus_fixed_terminal_delta"], 0.007268, "verification")
    if not condition["fixed_root_history_exact"] or condition["path_change_rate"] != 1.0:
        raise AssertionError("development verification mechanism drift")

    model_map = config["all_model_sha256"]
    if model_map != lock["model_sha256"] or len(model_map) != 128:
        raise AssertionError("complete learned-file map drift")
    for relative, expected in model_map.items():
        path = ROOT / relative
        if not path.is_file() or common.sha256_path(path) != expected:
            raise AssertionError({"model": relative})
    integrated = selected["integrated_variants"]
    if len(integrated) != 4:
        raise AssertionError("selected integrated-variant count drift")
    integrated_hashes = []
    for row in integrated:
        path = ROOT / row["checkpoint"]
        audit = common.tensor_audit(path)
        if audit["parameter_count"] != 205245 or audit["sha256"] != row["checkpoint_audit"]["sha256"]:
            raise AssertionError({"integrated_variant": row["checkpoint"]})
        integrated_hashes.append(audit["sha256"])
    if len(set(integrated_hashes)) != 4:
        raise AssertionError("integrated variants are not hash-distinct")
    for filename, expected in lock["code_sha256"].items():
        if common.sha256_path(ROOT / filename) != expected:
            raise AssertionError({"locked_code": filename})

    with np.load(RAW, allow_pickle=False) as handle:
        raw = {key: handle[key] for key in handle.files}
    terminal = raw["refined_terminal"] - raw["clean_terminal"]
    cumulative = raw["refined_cumulative"] - raw["clean_cumulative"]
    fixed_terminal = raw["refined_fixed_terminal"] - raw["clean_terminal"]
    fixed_cumulative = raw["refined_fixed_cumulative"] - raw["clean_cumulative"]
    active_minus_fixed = terminal - fixed_terminal
    reconstructed_vectors = {
        "terminal_delta": terminal,
        "cumulative_delta": cumulative,
        "fixed_terminal_delta": fixed_terminal,
        "fixed_cumulative_delta": fixed_cumulative,
        "active_minus_fixed_terminal_delta": active_minus_fixed,
    }
    for name, values in reconstructed_vectors.items():
        if not np.array_equal(raw[name], values):
            raise AssertionError({"raw_delta_vector": name})
    if len(terminal) != 2000:
        raise AssertionError("confirmatory seed count drift")

    labels = raw["labels"].astype(np.int64)
    clean_codes = raw["clean_codes"].astype(np.int64)
    alias_codes = raw["alias_codes"].astype(np.int64)
    parent_position = int(result["parent_position"])
    parent_feedback = common.exact_match_feedback(clean_codes[:, [parent_position]], labels)
    alias_feedback = common.exact_match_feedback(alias_codes, labels)
    nonimproving = bool(
        np.all(alias_feedback <= np.repeat(parent_feedback, common.N_ALIASES, axis=1))
    )
    alias_losses = np.mean(parent_feedback, axis=0)[0] - np.mean(alias_feedback, axis=0)
    alias_accuracies = np.mean(alias_feedback, axis=0)
    trigger_fractions = np.mean(alias_codes >= int(result["dataset"]["n_classes"]), axis=0)
    if not nonimproving or not result["quality"]["coordinate_wise_nonimproving"]:
        raise AssertionError("coordinate-wise quality constraint drift")
    require_close(result["quality"]["parent_accuracy"], np.mean(parent_feedback), "parent accuracy")
    require_close(result["quality"]["alias_accuracies"], alias_accuracies, "alias accuracies")
    require_close(result["quality"]["alias_losses"], alias_losses, "alias losses")
    require_close(result["quality"]["trigger_fractions"], trigger_fractions, "trigger fractions")

    summaries = {
        "terminal": common.effect_summary(terminal, 94900, 94901),
        "active_minus_fixed_terminal": common.effect_summary(active_minus_fixed, 94901, 94902),
        "cumulative": common.effect_summary(cumulative, 94902, 94903),
    }
    for name, recomputed in summaries.items():
        stored = result["effects"][name]
        for key in (
            "mean",
            "median",
            "bootstrap_95",
            "one_sided_signflip_p",
            "positive_fraction",
            "negative_fraction",
            "zero_fraction",
        ):
            require_close(stored[key], recomputed[key], f"{name}.{key}")

    path_change = float(np.mean(np.any(raw["clean_queries"] != raw["refined_queries"], axis=1)))
    query_set_change = float(
        np.mean(
            [
                set(left.tolist()) != set(right.tolist())
                for left, right in zip(raw["clean_queries"], raw["refined_queries"])
            ]
        )
    )
    final_root_change = float(np.mean(raw["clean_roots"][:, -1] != raw["refined_roots"][:, -1]))
    fixed_root_exact = bool(np.array_equal(raw["refined_fixed_roots"], raw["clean_roots"]))
    require_close(result["mechanism"]["path_change_rate"], path_change, "path change")
    require_close(result["mechanism"]["query_set_change_rate"], query_set_change, "query-set change")
    require_close(result["mechanism"]["final_root_change_rate"], final_root_change, "root change")

    recomputed_gates = {
        "pre_outcome_integrity": True,
        "same_literal_similarity": True,
        "four_distinct_integrated_checkpoint_hashes": len(set(integrated_hashes)) == 4,
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
        "coordinate_wise_nonimproving": nonimproving,
        "each_alias_loss_at_most_one_point": bool(np.max(alias_losses) <= 0.0100),
        "path_change_at_least_half": bool(path_change >= 0.50),
        "terminal_mean_at_least_half_point": bool(summaries["terminal"]["mean"] >= 0.005),
        "terminal_bootstrap_lower_positive": bool(summaries["terminal"]["bootstrap_95"][0] > 0),
        "terminal_signflip_at_most_point05": bool(summaries["terminal"]["one_sided_signflip_p"] <= 0.05),
        "active_minus_fixed_mean_at_least_half_point": bool(
            summaries["active_minus_fixed_terminal"]["mean"] >= 0.005
        ),
        "active_minus_fixed_bootstrap_lower_positive": bool(
            summaries["active_minus_fixed_terminal"]["bootstrap_95"][0] > 0
        ),
        "cumulative_mean_positive": bool(summaries["cumulative"]["mean"] > 0),
        "cumulative_bootstrap_lower_positive": bool(summaries["cumulative"]["bootstrap_95"][0] > 0),
        "fixed_terminal_exact_zero": bool(np.max(np.abs(fixed_terminal)) == 0),
        "fixed_cumulative_exact_zero": bool(np.max(np.abs(fixed_cumulative)) == 0),
        "fixed_root_history_bitwise_exact": fixed_root_exact,
    }
    if result["gates"] != recomputed_gates:
        raise AssertionError({"gate_drift": result["gates"], "recomputed": recomputed_gates})
    decision = (
        "GO_FRESH_SOURCE_FAITHFUL_LEARNED_VARIANT_BRIDGE"
        if all(recomputed_gates.values())
        else "NO_GO_RETAIN_FRESH_TASK_NEGATIVE"
    )
    if result["decision"] != decision or receipt["decision"] != decision:
        raise AssertionError("retained-decision reconstruction drift")
    if any(value is not True for value in receipt["checks"].values()):
        raise AssertionError("first independent-validation receipt drift")

    output = {
        "verdict": "PASS_STEP94_ARTIFACT_RECONSTRUCTION",
        "task_menu": 3,
        "selected_task": result["selected_task"],
        "learned_files_hash_matched": len(model_map),
        "integrated_variants": len(integrated_hashes),
        "paired_runs": len(terminal),
        "path_change_rate": path_change,
        "terminal_harm_pp": 100.0 * summaries["terminal"]["mean"],
        "terminal_bootstrap_95_pp": [100.0 * value for value in summaries["terminal"]["bootstrap_95"]],
        "cumulative_regret_delta": summaries["cumulative"]["mean"],
        "fixed_query_exact_zero": recomputed_gates["fixed_terminal_exact_zero"]
        and recomputed_gates["fixed_cumulative_exact_zero"]
        and fixed_root_exact,
        "decision": decision,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
