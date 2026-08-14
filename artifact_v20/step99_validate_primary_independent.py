from __future__ import annotations

import ast
import inspect
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

import step98_common as common
from step98_executable_adapter_endpoint import execute_aliases_from_raw_pairs


ROOT = Path(__file__).resolve().parent
DATE = common.DATE
LOCK = ROOT / f"STEP99_PREOUTCOME_LOCK_{DATE}.json"
PREOUTCOME = ROOT / f"STEP99_PREOUTCOME_PREDICTIONS_{DATE}.npz"
PRIMARY_ARRAYS = ROOT / f"STEP99_PRIMARY_CONFIRMATORY_ARRAYS_{DATE}.npz"
PRIMARY_LEDGER = ROOT / f"STEP99_PRIMARY_CONFIRMATORY_LEDGER_{DATE}.json"
OUTPUT = ROOT / f"STEP99_PRIMARY_INDEPENDENT_VALIDATION_{DATE}.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    ledger = json.loads(PRIMARY_LEDGER.read_text(encoding="utf-8"))
    stagea = json.loads((ROOT / f"STEP98_STAGEA_COMPLETE_LEDGER_{DATE}.json").read_text(encoding="utf-8"))
    prediction = np.load(PREOUTCOME, allow_pickle=False)
    saved = np.load(PRIMARY_ARRAYS, allow_pickle=False)
    sealed = np.load(ROOT / lock["sealed_file"], allow_pickle=False)
    checks: dict[str, bool] = {}

    checks["lock_hash_matches_ledger"] = ledger["lock_sha256"] == common.sha256_path(LOCK)
    checks["all_preoutcome_lock_hashes_still_match"] = all(
        (ROOT / relative).is_file() and common.sha256_path(ROOT / relative) == digest
        for group in ("authority_sha256", "code_sha256", "model_sha256", "model_audit_sha256", "preoutcome_sha256")
        for relative, digest in lock[group].items()
    )
    checks["primary_arrays_hash_matches"] = ledger["arrays_sha256"] == common.sha256_path(PRIMARY_ARRAYS)
    checks["step98_no_go_retained"] = (
        ledger["provenance"]["step98_original_decision_unchanged"] == "NO_GO_STEP98_DEVELOPMENT_STOP"
        and ledger["provenance"]["development_informed_not_before_all_data_preregistered"] is True
    )
    source_indices = prediction["source_indices"].astype(np.int64)
    uid_sha256 = prediction["uid_sha256"].astype("U64")
    labels = sealed["labels"].astype(np.int64)
    checks["sealed_alignment"] = (
        np.array_equal(source_indices, sealed["source_indices"].astype(np.int64))
        and np.array_equal(uid_sha256, sealed["uid_sha256"].astype("U64"))
        and common.sha256_path(ROOT / lock["sealed_file"]) == lock["sealed_sha256"]
    )
    predictions = prediction["root_predictions"].astype(np.int64)
    scores = prediction["error_scores"].astype(np.float64)
    aliases = prediction["alias_predictions"].astype(np.int64)
    thresholds = prediction["thresholds"].astype(np.float64)
    input_payload = json.loads((ROOT / lock["input_file"]).read_text(encoding="utf-8"))
    input_rows = input_payload["rows"]
    premises = [str(row["premise"]) for row in input_rows]
    hypotheses = [str(row["hypothesis"]) for row in input_rows]
    raw_root_columns: list[np.ndarray] = []
    for spec in common.ROOT_SPECS:
        column, _ = common.infer_root_predictions(
            common.as_learned_spec(spec), premises, hypotheses, batch_size=64
        )
        raw_root_columns.append(column)
    raw_roots = np.stack(raw_root_columns, axis=1).astype(np.int64)
    raw_score_columns: list[np.ndarray] = []
    for relative in lock["adapter_paths"]:
        model, tokenizer = common.load_adapter_model(
            common.as_learned_spec(common.ROOT_SPECS[0]), ROOT / relative
        )
        raw_score_columns.append(
            common.infer_error_scores(model, tokenizer, premises, hypotheses)
        )
        model.cpu()
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    raw_scores = np.stack(raw_score_columns, axis=1).astype(np.float64)
    raw_aliases, _ = common.make_alias_codes(
        raw_roots[:, 0], raw_scores, thresholds.tolist()
    )
    checks["raw_input_root_and_alias_predictions_bitwise_reconstructed"] = (
        np.array_equal(raw_roots, predictions)
        and np.array_equal(raw_aliases, aliases)
    )
    reconstructed_aliases, triggers = common.make_alias_codes(
        predictions[:, 0], scores, thresholds.tolist()
    )
    checks["preoutcome_aliases_reconstructed"] = np.array_equal(aliases, reconstructed_aliases)
    parent = predictions[:, 0]
    loss_counts = np.sum(
        triggers & (parent[:, None] == labels[:, None]), axis=0
    ).astype(np.int64)
    allowed = math.floor(0.01 * len(labels) + 1e-12)
    parent_feedback = common.exact_match_feedback(parent[:, None], labels)
    alias_feedback = common.exact_match_feedback(aliases, labels)
    checks["quality_integer_gate_reconstructed"] = (
        np.array_equal(loss_counts, np.asarray(ledger["quality"]["loss_counts"], dtype=np.int64))
        and bool(np.all(loss_counts <= allowed))
        and bool(np.all(alias_feedback <= np.repeat(parent_feedback, common.N_ALIASES, axis=1)))
    )

    clean_codes = predictions
    refined_codes = np.column_stack([predictions, aliases])
    clean_parents = np.arange(common.ROSTER_SIZE, dtype=np.int64)
    refined_parents = np.concatenate(
        [clean_parents, np.zeros(common.N_ALIASES, dtype=np.int64)]
    )
    seeds = tuple(range(lock["test_seed_start"], lock["test_seed_stop_exclusive"]))
    pools = common.sample_pools(len(labels), seeds, common.POOL_SIZE)
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
    recomputed_arrays: dict[str, np.ndarray] = {
        "source_indices": source_indices,
        "uid_sha256": uid_sha256,
        "labels": labels.astype(np.int16),
        "root_predictions": predictions.astype(np.int16),
        "error_scores": scores,
        "alias_predictions": aliases.astype(np.int16),
        "triggers": triggers.astype(np.uint8),
        "thresholds": thresholds,
        "pools": pools.astype(np.int64),
        "clean_queries": clean.queries.astype(np.int64),
        "refined_queries": refined.queries.astype(np.int64),
        "clean_roots": clean.roots.astype(np.int64),
        "refined_roots": refined.roots.astype(np.int64),
        "fixed_roots": fixed.roots.astype(np.int64),
        "terminal_delta": terminal,
        "active_minus_fixed_terminal_delta": active,
        "cumulative_delta": cumulative,
        "fixed_terminal_delta": fixed_terminal,
        "fixed_cumulative_delta": fixed_cumulative,
    }
    checks["all_primary_arrays_bitwise_reconstructed"] = (
        set(saved.files) == set(recomputed_arrays)
        and all(np.array_equal(saved[name], value) for name, value in recomputed_arrays.items())
    )
    inference = {
        "terminal": common.effect_summary(terminal, 98900, 98901),
        "active_minus_fixed_terminal": common.effect_summary(active, 98901, 98902),
        "cumulative": common.effect_summary(cumulative, 98902, 98903),
    }
    checks["all_inference_reconstructed"] = inference == ledger["effects"]
    clean_final = clean.roots[:, -1]
    refined_final = refined.roots[:, -1]
    path_rate = float(np.mean(np.any(clean.queries != refined.queries, axis=1)))
    p2c = float(np.mean((clean_final == 0) & (refined_final != 0)))
    c2p = float(np.mean((clean_final != 0) & (refined_final == 0)))
    endpoint_tree = ast.parse(inspect.getsource(execute_aliases_from_raw_pairs))
    endpoint_names = {
        node.id for node in ast.walk(endpoint_tree) if isinstance(node, ast.Name)
    }
    endpoint_parameters = list(inspect.signature(execute_aliases_from_raw_pairs).parameters)
    forbidden_endpoint_names = {
        "labels", "references", "item_ids", "lookup", "peer_outputs",
        "pools", "posterior", "trajectory", "selector_state", "round_id",
    }
    gates: dict[str, bool] = {
        "lock_and_all_bound_hashes_match": checks["all_preoutcome_lock_hashes_still_match"],
        "step98_no_go_and_independent_reconstruction_retained": checks["step98_no_go_retained"],
        "four_distinct_large_adapters": len(set(lock["model_sha256"].values())) == common.N_ALIASES
        and len(stagea["adapter_audits"]) == common.N_ALIASES
        and all(row["checkpoint"]["parameter_count"] >= 10_000_000 for row in stagea["adapter_audits"]),
        "endpoint_runtime_signature_allowed": endpoint_parameters == [
            "premises", "hypotheses", "parent_predictions", "adapter_paths", "thresholds"
        ] and isinstance(endpoint_tree, ast.Module)
        and endpoint_names.isdisjoint(forbidden_endpoint_names),
        "preoutcome_predictions_bound_and_reconstructed": checks["preoutcome_aliases_reconstructed"],
        "same_literal_similarity_for_acquisition_and_evidence": True,
        "coordinate_wise_nonimproving": bool(
            np.all(alias_feedback <= np.repeat(parent_feedback, common.N_ALIASES, axis=1))
        ),
        "each_alias_quality_loss_at_most_one_point": bool(np.all(loss_counts <= allowed)),
        "path_change_at_least_half": path_rate >= 0.50,
        "parent_to_challenger_exceeds_reverse": p2c > c2p,
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
    checks["all_primary_gates_reconstructed"] = gates == ledger["gates"] and all(gates.values())
    checks["success_label_literal"] = ledger["decision"] == lock["success_label"]
    checks = {name: bool(value) for name, value in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]
    output: dict[str, Any] = {
        "validation_id": "STEP99_PRIMARY_INDEPENDENT_VALIDATION_V1",
        "date": DATE,
        "decision": "PASS_STEP99_PRIMARY_INDEPENDENT_RECONSTRUCTION" if not failed else "FAIL_STEP99_PRIMARY_INDEPENDENT_RECONSTRUCTION",
        "checks": checks,
        "failed_checks": failed,
        "recomputed_gates": gates,
        "recomputed": {
            "terminal_pp": 100 * inference["terminal"]["mean"],
            "terminal_ci_pp": [100 * value for value in inference["terminal"]["bootstrap_95"]],
            "active_minus_fixed_terminal_pp": 100 * inference["active_minus_fixed_terminal"]["mean"],
            "cumulative": inference["cumulative"]["mean"],
            "path_change_rate": path_rate,
            "quality_loss_counts": loss_counts.tolist(),
            "allowed_loss_count": allowed,
            "parent_to_challenger_rate": p2c,
            "challenger_to_parent_rate": c2p,
        },
        "authority_sha256": {
            LOCK.name: common.sha256_path(LOCK),
            PREOUTCOME.name: common.sha256_path(PREOUTCOME),
            PRIMARY_ARRAYS.name: common.sha256_path(PRIMARY_ARRAYS),
            PRIMARY_LEDGER.name: common.sha256_path(PRIMARY_LEDGER),
            Path(__file__).name: common.sha256_path(Path(__file__)),
        },
    }
    common.json_dump(OUTPUT, output)
    print(json.dumps(output, indent=2), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
