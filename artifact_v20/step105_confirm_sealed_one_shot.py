from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import step105_common as common
import step105_executable_adapter_endpoint as endpoint


ROOT = Path(__file__).resolve().parent
LOCK = ROOT / f"STEP105_PREOUTCOME_LOCK_{common.DATE}.json"
LEDGER = ROOT / f"STEP105_PRIMARY_CONFIRMATORY_LEDGER_{common.DATE}.json"
ARRAYS = ROOT / f"STEP105_PRIMARY_CONFIRMATORY_ARRAYS_{common.DATE}.npz"


def hash_bindings(lock: dict[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for key in ("predata_lock", "protocol", "metadata", "stage0_manifest", "train_ledger"):
        file_key = f"{key}_file"
        if file_key in lock:
            checks[key] = common.sha256_path(ROOT / lock[file_key]) == lock[f"{key}_sha256"]
        elif key == "protocol":
            checks[key] = common.sha256_path(common.PROTOCOL) == lock["protocol_sha256"]
        elif key == "metadata":
            checks[key] = common.sha256_path(common.METADATA) == lock["metadata_sha256"]
    checks["development"] = common.sha256_path(ROOT / lock["development_file"]) == lock["development_sha256"]
    checks["test_input"] = common.sha256_path(ROOT / lock["test_input_file"]) == lock["test_input_sha256"]
    checks["sealed"] = common.sha256_path(ROOT / lock["sealed_file"]) == lock["sealed_sha256"]
    checks["prediction"] = common.sha256_path(ROOT / lock["prediction_file"]) == lock["prediction_sha256"]
    for name, expected in lock["adapter_sha256"].items():
        checks[f"adapter:{name}"] = common.sha256_path(ROOT / name) == expected
    for name, expected in lock["adapter_audit_sha256"].items():
        checks[f"adapter_audit:{name}"] = common.sha256_path(ROOT / name) == expected
    for name, expected in lock["code_sha256"].items():
        checks[f"code:{name}"] = common.sha256_path(ROOT / name) == expected
    return checks


def main() -> None:
    if any(path.exists() for path in (LEDGER, ARRAYS)):
        raise FileExistsError("Step 105 primary output already exists")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    bindings = hash_bindings(lock)
    if not all(bindings.values()):
        raise AssertionError({key: value for key, value in bindings.items() if not value})
    if any(lock[key] != 0 for key in ("target_selector_search_rows", "target_selector_verify_rows", "target_selector_grid_cells")):
        raise AssertionError("target-task tuning lock drift")
    source = inspect.getsource(endpoint.execute_aliases_from_raw_text).lower()
    forbidden = ["reference", "label", "item_id", "lookup", "peer", "pool", "posterior", "trajectory", "selector_state", "dataset_id"]
    forbidden_hits = [token for token in forbidden if token in source]
    if forbidden_hits:
        raise AssertionError({"forbidden_endpoint_source_tokens": forbidden_hits})

    package = np.load(ROOT / lock["prediction_file"], allow_pickle=False)
    roots = package["test_root_predictions"].astype(np.int64)
    aliases = package["test_aliases"].astype(np.int64)
    triggers = package["test_triggers"].astype(bool)
    scores = package["test_error_scores"].astype(np.float64)
    thresholds = package["thresholds"].astype(np.float64)
    sealed = np.load(ROOT / lock["sealed_file"], allow_pickle=False)
    labels = sealed["labels"].astype(np.int64)
    if roots.shape != (38_000, common.ROSTER_SIZE) or aliases.shape != (38_000, common.N_ALIASES):
        raise AssertionError({"roots": roots.shape, "aliases": aliases.shape})
    if labels.shape != (38_000,) or scores.shape != (38_000, common.N_ALIASES):
        raise AssertionError({"labels": labels.shape, "scores": scores.shape})
    reconstructed_aliases, reconstructed_triggers = common.make_alias_codes(roots[:, 0], scores, thresholds.tolist())
    if not np.array_equal(reconstructed_aliases, aliases) or not np.array_equal(reconstructed_triggers, triggers):
        raise AssertionError("pre-outcome alias package reconstruction failed")

    parent_feedback = common.exact_match_feedback(roots[:, 0, None], labels)
    alias_feedback = common.exact_match_feedback(aliases, labels)
    coordinate_nonimproving = bool(np.all(alias_feedback <= np.repeat(parent_feedback, common.N_ALIASES, axis=1)))
    parent_correct = roots[:, 0] == labels
    loss_counts = np.sum(triggers & parent_correct[:, None], axis=0).astype(np.int64)
    allowed_loss_count = math.floor(0.01 * len(labels) + 1e-12)
    trigger_counts = np.sum(triggers, axis=0).astype(np.int64)
    root_accuracies = np.mean(roots == labels[:, None], axis=0)
    summary, arrays = common.compute_effects(roots, aliases, labels)

    terminal = summary["terminal"]
    active_minus_fixed = summary["active_minus_fixed_terminal"]
    cumulative = summary["cumulative"]
    adapter_audits = [
        json.loads((ROOT / name).read_text(encoding="utf-8"))
        for name in lock["adapter_audit_sha256"]
    ]
    endpoint_parameters = list(inspect.signature(endpoint.execute_aliases_from_raw_text).parameters)
    gates = {
        "all_hash_bindings_match": all(bindings.values()),
        "no_target_selector_tuning": lock["target_selector_search_rows"] == 0
        and lock["target_selector_verify_rows"] == 0
        and lock["target_selector_grid_cells"] == 0,
        "four_distinct_adapters": len(set(lock["adapter_sha256"].values())) == common.N_ALIASES,
        "adapter_parameter_count": all(
            int(row["weight_audit"]["parameter_count"]) >= 10_000_000 for row in adapter_audits
        ),
        "endpoint_signature_and_source_independent": endpoint_parameters == [
            "texts", "parent_predictions", "adapter_paths", "thresholds"
        ] and not forbidden_hits,
        "same_literal_similarity": bool(lock["selector"]["same_literal_exact_match"]),
        "coordinate_wise_nonimproving": coordinate_nonimproving,
        "quality_loss_within_one_point": bool(np.all(loss_counts <= allowed_loss_count)),
        "path_change_at_least_half": summary["path_change_rate"] >= 0.50,
        "directionality": summary["parent_to_challenger_rate"] > summary["challenger_to_parent_rate"],
        "fixed_query_exact_zero": summary["max_abs_fixed_terminal_delta"] == 0
        and summary["max_abs_fixed_cumulative_delta"] == 0
        and summary["fixed_root_history_exact"],
        "terminal_mean_at_least_half_point": terminal["mean"] >= 0.005,
        "active_minus_fixed_mean_at_least_half_point": active_minus_fixed["mean"] >= 0.005,
        "terminal_inference": terminal["bootstrap_95"][0] > 0
        and terminal["one_sided_signflip_p"] <= 0.05,
        "active_minus_fixed_inference": active_minus_fixed["bootstrap_95"][0] > 0
        and active_minus_fixed["one_sided_signflip_p"] <= 0.05,
        "cumulative_inference": cumulative["mean"] > 0 and cumulative["bootstrap_95"][0] > 0,
    }
    failed = [key for key, value in gates.items() if not value]
    decision = (
        lock["success_label"] if not failed else lock["failure_label"]
    )
    np.savez_compressed(
        ARRAYS,
        root_predictions=roots,
        aliases=aliases,
        triggers=triggers.astype(np.uint8),
        thresholds=thresholds,
        labels=labels,
        loss_counts=loss_counts,
        trigger_counts=trigger_counts,
        root_accuracies=root_accuracies,
        **arrays,
    )
    ledger = {
        "schema": "step105.primary_confirmatory_one_shot.v1",
        "date": common.DATE,
        "decision": decision,
        "failed_gates": failed,
        "preoutcome_lock_file": LOCK.name,
        "preoutcome_lock_sha256": common.sha256_path(LOCK),
        "hash_bindings": bindings,
        "target_selector_search_rows": 0,
        "target_selector_verify_rows": 0,
        "target_selector_grid_cells": 0,
        "rows": len(labels),
        "paired_runs": len(common.TEST_SEEDS),
        "root_accuracies": root_accuracies.tolist(),
        "parent_minus_best_challenger": float(root_accuracies[0] - np.max(root_accuracies[1:])),
        "parent_error_prevalence": float(np.mean(~parent_correct)),
        "loss_counts": loss_counts.tolist(),
        "loss_fractions": (loss_counts / len(labels)).tolist(),
        "allowed_loss_count": allowed_loss_count,
        "trigger_counts": trigger_counts.tolist(),
        "trigger_fractions": (trigger_counts / len(labels)).tolist(),
        "coordinate_wise_nonimproving": coordinate_nonimproving,
        "effect": summary,
        "gates": gates,
        "endpoint_parameters": endpoint_parameters,
        "forbidden_endpoint_source_hits": forbidden_hits,
        "arrays_file": ARRAYS.name,
        "arrays_sha256": common.sha256_path(ARRAYS),
        "interpretation": {
            "prospective_cross_benchmark_one_shot": True,
            "target_task_selector_tuning": False,
            "test_reference_or_item_lookup": False,
            "same_similarity_acquisition_and_evidence": True,
            "offline_not_live_registry": True,
            "binary_sentiment_family_shared_with_imdb": True,
        },
    }
    common.json_dump(LEDGER, ledger)
    print(json.dumps({
        "decision": decision,
        "failed_gates": failed,
        "root_accuracies_pct": (100 * root_accuracies).tolist(),
        "loss_counts": loss_counts.tolist(),
        "trigger_counts": trigger_counts.tolist(),
        "terminal_harm_pp": 100 * terminal["mean"],
        "terminal_ci_pp": [100 * value for value in terminal["bootstrap_95"]],
        "active_minus_fixed_pp": 100 * active_minus_fixed["mean"],
        "cumulative_harm": cumulative["mean"],
        "path_change_rate": summary["path_change_rate"],
        "final_root_change_rate": summary["final_root_change_rate"],
        "fixed_terminal_max": summary["max_abs_fixed_terminal_delta"],
        "output": LEDGER.name,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
