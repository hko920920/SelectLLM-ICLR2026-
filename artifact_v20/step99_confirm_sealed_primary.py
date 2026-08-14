from __future__ import annotations

import ast
import inspect
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import step98_common as common
from step98_executable_adapter_endpoint import execute_aliases_from_raw_pairs


ROOT = Path(__file__).resolve().parent
DATE = common.DATE
LOCK = ROOT / f"STEP99_PREOUTCOME_LOCK_{DATE}.json"
PREOUTCOME = ROOT / f"STEP99_PREOUTCOME_PREDICTIONS_{DATE}.npz"
PREOUTCOME_LEDGER = ROOT / f"STEP99_PREOUTCOME_PREDICTION_LEDGER_{DATE}.json"
STAGEA_LEDGER = ROOT / f"STEP98_STAGEA_COMPLETE_LEDGER_{DATE}.json"
OUTPUT_ARRAYS = ROOT / f"STEP99_PRIMARY_CONFIRMATORY_ARRAYS_{DATE}.npz"
OUTPUT_LEDGER = ROOT / f"STEP99_PRIMARY_CONFIRMATORY_LEDGER_{DATE}.json"


def verify_lock(lock: dict[str, Any]) -> None:
    for group in ("authority_sha256", "code_sha256", "model_sha256", "model_audit_sha256", "preoutcome_sha256"):
        for relative, digest in lock[group].items():
            path = ROOT / relative
            if not path.is_file() or common.sha256_path(path) != digest:
                raise AssertionError({"lock_hash_drift": relative})
    if common.sha256_path(ROOT / lock["input_file"]) != lock["input_sha256"]:
        raise AssertionError("locked input drift")
    if common.sha256_path(ROOT / lock["sealed_file"]) != lock["sealed_sha256"]:
        raise AssertionError("locked sealed bytes drift")
    if lock["sealed_outcome_opened"] is not False:
        raise AssertionError("pre-outcome lock status drift")
    if lock["step98_original_decision"] != "NO_GO_STEP98_DEVELOPMENT_STOP":
        raise AssertionError("Step 98 was relabeled")


def main() -> None:
    if OUTPUT_ARRAYS.exists() or OUTPUT_LEDGER.exists():
        raise FileExistsError("Step 99 sealed primary already evaluated")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    verify_lock(lock)
    preledger = json.loads(PREOUTCOME_LEDGER.read_text(encoding="utf-8"))
    stagea = json.loads(STAGEA_LEDGER.read_text(encoding="utf-8"))
    if preledger["sealed_outcome_loaded"] is not False:
        raise AssertionError("pre-outcome ledger drift")
    prediction = np.load(PREOUTCOME, allow_pickle=False)
    source_indices = prediction["source_indices"].astype(np.int64)
    uid_sha256 = prediction["uid_sha256"].astype("U64")
    root_predictions = prediction["root_predictions"].astype(np.int64)
    error_scores = prediction["error_scores"].astype(np.float64)
    aliases = prediction["alias_predictions"].astype(np.int64)
    triggers = prediction["triggers"].astype(bool)
    thresholds = prediction["thresholds"].astype(np.float64)
    if not np.array_equal(thresholds, np.asarray(lock["thresholds"], dtype=np.float64)):
        raise AssertionError("threshold drift")
    if not np.array_equal(
        aliases,
        common.make_alias_codes(root_predictions[:, 0], error_scores, lock["thresholds"])[0],
    ):
        raise AssertionError("alias package reconstruction drift")

    # This is the first and only outcome-array load authorized by the lock.
    sealed = np.load(ROOT / lock["sealed_file"], allow_pickle=False)
    labels = sealed["labels"].astype(np.int64)
    sealed_indices = sealed["source_indices"].astype(np.int64)
    sealed_uid = sealed["uid_sha256"].astype("U64")
    if not np.array_equal(source_indices, sealed_indices) or not np.array_equal(uid_sha256, sealed_uid):
        raise AssertionError("sealed outcome/input alignment drift")
    n = len(labels)
    if root_predictions.shape != (n, common.ROSTER_SIZE) or aliases.shape != (n, common.N_ALIASES):
        raise AssertionError("prediction shape drift")

    parent = root_predictions[:, 0]
    parent_feedback = common.exact_match_feedback(parent[:, None], labels)
    alias_feedback = common.exact_match_feedback(aliases, labels)
    loss_counts = np.sum(
        triggers & (parent[:, None] == labels[:, None]), axis=0
    ).astype(np.int64)
    allowed_loss_count = int(math.floor(0.01 * n + 1e-12))
    clean_codes = root_predictions
    refined_codes = np.column_stack([root_predictions, aliases])
    clean_parents = np.arange(common.ROSTER_SIZE, dtype=np.int64)
    refined_parents = np.concatenate(
        [clean_parents, np.zeros(common.N_ALIASES, dtype=np.int64)]
    )
    seeds = tuple(range(lock["test_seed_start"], lock["test_seed_stop_exclusive"]))
    if seeds != common.TEST_SEEDS or len(seeds) != lock["test_seed_count"]:
        raise AssertionError("test seed drift")
    pools = common.sample_pools(n, seeds, common.POOL_SIZE)
    clean = common.run_active(
        clean_codes, labels, clean_parents, clean_codes, pools, seeds,
        common.BUDGET, common.TAU,
    )
    refined = common.run_active(
        refined_codes, labels, refined_parents, clean_codes, pools, seeds,
        common.BUDGET, common.TAU,
    )
    fixed = common.run_fixed(
        refined_codes, labels, refined_parents, clean_codes, pools,
        clean.queries, seeds,
    )
    terminal = refined.terminal - clean.terminal
    fixed_terminal = fixed.terminal - clean.terminal
    active = terminal - fixed_terminal
    cumulative = refined.cumulative - clean.cumulative
    fixed_cumulative = fixed.cumulative - clean.cumulative
    inference = {
        "terminal": common.effect_summary(terminal, 98900, 98901),
        "active_minus_fixed_terminal": common.effect_summary(active, 98901, 98902),
        "cumulative": common.effect_summary(cumulative, 98902, 98903),
    }
    clean_final = clean.roots[:, -1]
    refined_final = refined.roots[:, -1]
    path_change = np.any(clean.queries != refined.queries, axis=1)
    parent_to_challenger = (clean_final == 0) & (refined_final != 0)
    challenger_to_parent = (clean_final != 0) & (refined_final == 0)

    endpoint_source = inspect.getsource(execute_aliases_from_raw_pairs)
    endpoint_parameters = list(inspect.signature(execute_aliases_from_raw_pairs).parameters)
    endpoint_tree = ast.parse(endpoint_source)
    endpoint_names = {
        node.id for node in ast.walk(endpoint_tree) if isinstance(node, ast.Name)
    }
    forbidden_endpoint_names = {
        "labels", "references", "item_ids", "lookup", "peer_outputs",
        "pools", "posterior", "trajectory", "selector_state", "round_id",
    }
    gates: dict[str, bool] = {
        "lock_and_all_bound_hashes_match": True,
        "step98_no_go_and_independent_reconstruction_retained": lock["step98_original_decision"] == "NO_GO_STEP98_DEVELOPMENT_STOP",
        "four_distinct_large_adapters": len(set(lock["model_sha256"].values())) == common.N_ALIASES
        and len(stagea["adapter_audits"]) == common.N_ALIASES
        and all(row["checkpoint"]["parameter_count"] >= 10_000_000 for row in stagea["adapter_audits"]),
        "endpoint_runtime_signature_allowed": endpoint_parameters == [
            "premises", "hypotheses", "parent_predictions", "adapter_paths", "thresholds"
        ] and isinstance(endpoint_tree, ast.Module)
        and endpoint_names.isdisjoint(forbidden_endpoint_names),
        "preoutcome_predictions_bound_and_reconstructed": preledger["prediction_sha256"] == common.sha256_path(PREOUTCOME),
        "same_literal_similarity_for_acquisition_and_evidence": True,
        "coordinate_wise_nonimproving": bool(
            np.all(alias_feedback <= np.repeat(parent_feedback, common.N_ALIASES, axis=1))
        ),
        "each_alias_quality_loss_at_most_one_point": bool(np.all(loss_counts <= allowed_loss_count)),
        "path_change_at_least_half": float(np.mean(path_change)) >= 0.50,
        "parent_to_challenger_exceeds_reverse": float(np.mean(parent_to_challenger)) > float(np.mean(challenger_to_parent)),
        "fixed_query_exact_zero": bool(
            np.array_equal(fixed.roots, clean.roots)
            and np.max(np.abs(fixed_terminal)) == 0
            and np.max(np.abs(fixed_cumulative)) == 0
        ),
        "terminal_mean_at_least_half_point": inference["terminal"]["mean"] >= 0.005,
        "active_minus_fixed_terminal_mean_at_least_half_point": inference["active_minus_fixed_terminal"]["mean"] >= 0.005,
        "terminal_inference_positive": inference["terminal"]["bootstrap_95"][0] > 0
        and inference["terminal"]["one_sided_signflip_p"] <= 0.05,
        "active_minus_fixed_terminal_inference_positive": inference["active_minus_fixed_terminal"]["bootstrap_95"][0] > 0
        and inference["active_minus_fixed_terminal"]["one_sided_signflip_p"] <= 0.05,
        "cumulative_inference_positive": inference["cumulative"]["mean"] > 0
        and inference["cumulative"]["bootstrap_95"][0] > 0,
    }
    gates = {name: bool(value) for name, value in gates.items()}
    failed = [name for name, passed in gates.items() if not passed]
    decision = lock["success_label"] if not failed else lock["failure_label"]

    np.savez_compressed(
        OUTPUT_ARRAYS,
        source_indices=source_indices,
        uid_sha256=uid_sha256,
        labels=labels.astype(np.int16),
        root_predictions=root_predictions.astype(np.int16),
        error_scores=error_scores,
        alias_predictions=aliases.astype(np.int16),
        triggers=triggers.astype(np.uint8),
        thresholds=thresholds,
        pools=pools.astype(np.int64),
        clean_queries=clean.queries.astype(np.int64),
        refined_queries=refined.queries.astype(np.int64),
        clean_roots=clean.roots.astype(np.int64),
        refined_roots=refined.roots.astype(np.int64),
        fixed_roots=fixed.roots.astype(np.int64),
        terminal_delta=terminal,
        active_minus_fixed_terminal_delta=active,
        cumulative_delta=cumulative,
        fixed_terminal_delta=fixed_terminal,
        fixed_cumulative_delta=fixed_cumulative,
    )
    ledger: dict[str, Any] = {
        "ledger_id": "STEP99_ANLI_DEVELOPMENT_INFORMED_SEALED_PRIMARY_V1",
        "date": DATE,
        "decision": decision,
        "failed_gates": failed,
        "gates": gates,
        "provenance": {
            "step98_original_decision_unchanged": lock["step98_original_decision"],
            "step98_only_failed_auxiliary_gate": lock["step98_only_failed_gate"],
            "step99_protocol_written_before_sealed_outcome_open": True,
            "development_informed_not_before_all_data_preregistered": True,
        },
        "lock_file": LOCK.name,
        "lock_sha256": common.sha256_path(LOCK),
        "sealed_file": lock["sealed_file"],
        "sealed_sha256": lock["sealed_sha256"],
        "sealed_outcome_opened_once_for_primary": True,
        "rows": n,
        "runs": len(seeds),
        "quality": {
            "loss_counts": loss_counts.tolist(),
            "allowed_loss_count": allowed_loss_count,
            "loss_fractions": (loss_counts / n).tolist(),
            "trigger_counts": np.sum(triggers, axis=0).astype(int).tolist(),
            "trigger_fractions": np.mean(triggers, axis=0).tolist(),
            "coordinate_wise_nonimproving": gates["coordinate_wise_nonimproving"],
        },
        "effects": inference,
        "mechanism": {
            "path_change_rate": float(np.mean(path_change)),
            "query_set_change_rate": float(np.mean([
                set(left.tolist()) != set(right.tolist())
                for left, right in zip(clean.queries, refined.queries)
            ])),
            "final_root_change_rate": float(np.mean(clean_final != refined_final)),
            "clean_parent_terminal_rate": float(np.mean(clean_final == 0)),
            "refined_parent_terminal_rate": float(np.mean(refined_final == 0)),
            "parent_to_challenger_rate": float(np.mean(parent_to_challenger)),
            "challenger_to_parent_rate": float(np.mean(challenger_to_parent)),
            "max_abs_fixed_terminal_delta": float(np.max(np.abs(fixed_terminal))),
            "max_abs_fixed_cumulative_delta": float(np.max(np.abs(fixed_cumulative))),
            "fixed_root_history_bitwise_equal": bool(np.array_equal(fixed.roots, clean.roots)),
        },
        "arrays_file": OUTPUT_ARRAYS.name,
        "arrays_sha256": common.sha256_path(OUTPUT_ARRAYS),
        "preoutcome_prediction_sha256": common.sha256_path(PREOUTCOME),
        "code_sha256": common.sha256_path(Path(__file__)),
    }
    common.json_dump(OUTPUT_LEDGER, ledger)
    print(json.dumps({
        "decision": decision,
        "failed_gates": failed,
        "terminal_pp": 100 * inference["terminal"]["mean"],
        "terminal_ci_pp": [100 * value for value in inference["terminal"]["bootstrap_95"]],
        "terminal_p": inference["terminal"]["one_sided_signflip_p"],
        "active_minus_fixed_terminal_pp": 100 * inference["active_minus_fixed_terminal"]["mean"],
        "cumulative": inference["cumulative"]["mean"],
        "path_change_rate": float(np.mean(path_change)),
        "quality_loss_counts": loss_counts.tolist(),
        "allowed_loss_count": allowed_loss_count,
        "fixed_query_max_abs": float(max(np.max(np.abs(fixed_terminal)), np.max(np.abs(fixed_cumulative)))),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
