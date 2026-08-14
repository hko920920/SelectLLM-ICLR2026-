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
ARRAYS = ROOT / f"STEP98_STAGEA_DEVELOPMENT_ARRAYS_{common.DATE}.npz"
LEDGER = ROOT / f"STEP98_STAGEA_COMPLETE_LEDGER_{common.DATE}.json"
CONFIG = ROOT / f"STEP98_STAGEA_FROZEN_CONFIG_{common.DATE}.json"
SELECTION = ROOT / f"STEP98_TRAIN_ONLY_ROUND_SELECTION_{common.DATE}.json"
OUTPUT = ROOT / f"STEP98_STAGEA_INDEPENDENT_VALIDATION_{common.DATE}.json"


def choose_round(selection: dict[str, Any]) -> str | None:
    eligible = [row for row in selection["rounds"].values() if row["eligible"]]
    if not eligible:
        return None
    order = {key: index for index, key in enumerate(common.ROUNDS)}
    return str(max(
        eligible,
        key=lambda row: (
            row["selection_primary"], row["selection_secondary"],
            row["selection_gap"], -order[row["round"]],
        ),
    )["round"])


def reproduce(
    predictions: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    thresholds: list[float],
    indices: np.ndarray,
    seeds: tuple[int, ...],
) -> dict[str, np.ndarray]:
    clean_codes = predictions[indices]
    references = labels[indices]
    parent = predictions[indices, 0]
    triggers = scores[indices] > np.asarray(thresholds)[None, :]
    aliases = np.repeat(parent[:, None], common.N_ALIASES, axis=1)
    for alias in range(common.N_ALIASES):
        aliases[triggers[:, alias], alias] = common.N_CLASSES + alias
    refined_codes = np.column_stack([clean_codes, aliases])
    clean_parents = np.arange(common.ROSTER_SIZE, dtype=np.int64)
    refined_parents = np.concatenate(
        [clean_parents, np.zeros(common.N_ALIASES, dtype=np.int64)]
    )
    pools = common.sample_pools(len(indices), seeds, common.POOL_SIZE)
    clean = common.run_active(
        clean_codes, references, clean_parents, clean_codes, pools, seeds,
        common.BUDGET, common.TAU,
    )
    refined = common.run_active(
        refined_codes, references, refined_parents, clean_codes, pools, seeds,
        common.BUDGET, common.TAU,
    )
    fixed = common.run_fixed(
        refined_codes, references, refined_parents, clean_codes, pools,
        clean.queries, seeds,
    )
    terminal = refined.terminal - clean.terminal
    fixed_terminal = fixed.terminal - clean.terminal
    return {
        "terminal": terminal,
        "active": terminal - fixed_terminal,
        "cumulative": refined.cumulative - clean.cumulative,
        "fixed_terminal": fixed_terminal,
        "fixed_cumulative": fixed.cumulative - clean.cumulative,
        "clean_queries": clean.queries,
        "refined_queries": refined.queries,
        "clean_roots": clean.roots,
        "refined_roots": refined.roots,
        "fixed_roots": fixed.roots,
        "pools": pools,
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    arrays = np.load(ARRAYS, allow_pickle=False)
    labels = arrays["labels"].astype(np.int64)
    predictions = arrays["root_predictions"].astype(np.int64)
    scores = arrays["error_scores"].astype(np.float64)
    thresholds = [float(value) for value in config["thresholds"]]

    checks: dict[str, bool] = {}
    checks["stagea_declared_no_go"] = ledger["decision"] == "NO_GO_STEP98_DEVELOPMENT_STOP"
    checks["only_failed_gate_is_clean_parent_frequency"] = ledger["failed_gates"] == [
        "clean_parent_terminal_rate_at_least_60pct"
    ]
    checks["round_selection_reconstructed"] = choose_round(selection) == selection["selected_round"] == "r3"
    checks["ledger_and_array_hashes_bound"] = (
        config["stagea_ledger_sha256"] == common.sha256_path(LEDGER)
        and config["development_arrays_sha256"] == common.sha256_path(ARRAYS)
    )
    checks["sealed_hash_bound_without_loading_outcome"] = bool(
        config["sealed_outcome_sha256"] and config["test_input_sha256"]
    )
    all_indices = np.concatenate([
        arrays[f"{name}_indices"] for name in common.TRAIN_QUOTAS_PER_LABEL
    ])
    checks["partitions_disjoint_and_complete"] = (
        len(all_indices) == len(labels)
        and len(np.unique(all_indices)) == len(labels)
        and np.array_equal(np.sort(all_indices), np.arange(len(labels)))
    )
    checks["root_and_score_shapes"] = (
        predictions.shape == (len(labels), common.ROSTER_SIZE)
        and scores.shape == (len(labels), common.N_ALIASES)
        and np.all(np.isfinite(scores))
    )
    checks["adapter_files_match_locked_hashes"] = all(
        (ROOT / relative).is_file()
        and common.sha256_path(ROOT / relative) == digest
        for relative, digest in config["adapter_sha256"].items()
    )
    source = inspect.getsource(execute_aliases_from_raw_pairs)
    tree = ast.parse(source)
    parameters = list(inspect.signature(execute_aliases_from_raw_pairs).parameters)
    forbidden = ("label", "reference", "item_id", "lookup", "peer", "pool", "posterior", "trajectory", "selector_state", "round_id")
    checks["endpoint_signature_raw_input_only"] = parameters == [
        "premises", "hypotheses", "parent_predictions", "adapter_paths", "thresholds"
    ] and not any(token in parameters for token in forbidden) and isinstance(tree, ast.Module)

    quality_checks: list[bool] = []
    for name in ("threshold_calibration", "threshold_safety", "selector_search", "selector_verify"):
        indices = arrays[f"{name}_indices"]
        parent = predictions[indices, 0]
        trigger = scores[indices] > np.asarray(thresholds)[None, :]
        loss = np.sum(trigger & (parent[:, None] == labels[indices, None]), axis=0)
        recorded = ledger["quality"][name]
        quality_checks.append(
            np.array_equal(loss, np.asarray(recorded["loss_counts"], dtype=np.int64))
            and bool(np.all(loss <= math.floor(0.01 * len(indices) + 1e-12)))
            and bool(recorded["coordinate_wise_nonimproving"])
        )
    checks["quality_reconstructed"] = all(quality_checks)

    reproduced: dict[str, dict[str, np.ndarray]] = {}
    for prefix, seeds in (
        ("search", common.SELECTOR_SEARCH_SEEDS),
        ("verify", common.SELECTOR_VERIFY_SEEDS),
    ):
        result = reproduce(
            predictions, labels, scores, thresholds,
            arrays[f"selector_{prefix}_indices"], seeds,
        )
        reproduced[prefix] = result
        checks[f"{prefix}_all_saved_vectors_bitwise"] = all(
            np.array_equal(value, arrays[f"{prefix}_{name}"])
            for name, value in result.items()
        )
        checks[f"{prefix}_fixed_exact_zero"] = (
            np.max(np.abs(result["fixed_terminal"])) == 0
            and np.max(np.abs(result["fixed_cumulative"])) == 0
            and np.array_equal(result["fixed_roots"], result["clean_roots"])
        )

    verify_terminal = reproduced["verify"]["terminal"]
    verify_active = reproduced["verify"]["active"]
    verify_cumulative = reproduced["verify"]["cumulative"]
    inference = {
        "terminal": common.effect_summary(verify_terminal, 98800, 98801),
        "active_minus_fixed_terminal": common.effect_summary(verify_active, 98801, 98802),
        "cumulative": common.effect_summary(verify_cumulative, 98802, 98803),
    }
    recorded_inference = ledger["selector_verify"]["inference"]
    checks["verification_inference_reconstructed"] = inference == recorded_inference
    checks["core_development_outcome_gates_all_pass"] = all(
        bool(value) for name, value in ledger["gates"].items()
        if name != "clean_parent_terminal_rate_at_least_60pct"
    )
    checks["failed_auxiliary_gate_not_mathematically_required_for_harm"] = (
        ledger["selector_search"]["clean_parent_terminal_rate"] < 0.60
        and ledger["selector_search"]["mean_terminal_delta"] >= 0.005
        and ledger["selector_search"]["parent_to_challenger_rate"]
        > ledger["selector_search"]["challenger_to_parent_rate"]
    )

    checks = {name: bool(passed) for name, passed in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]
    output = {
        "validation_id": "STEP98_STAGEA_INDEPENDENT_VALIDATION_V1",
        "date": common.DATE,
        "decision": "PASS_STEP98_STAGEA_RECONSTRUCTION_RETAIN_ORIGINAL_NO_GO" if not failed else "FAIL_STEP98_STAGEA_RECONSTRUCTION",
        "original_step98_decision_unchanged": ledger["decision"],
        "checks": checks,
        "failed_checks": failed,
        "recomputed": {
            "search_terminal_pp": 100 * float(np.mean(reproduced["search"]["terminal"])),
            "verify_terminal_pp": 100 * float(np.mean(verify_terminal)),
            "verify_terminal_ci_pp": [100 * float(value) for value in inference["terminal"]["bootstrap_95"]],
            "verify_active_ci_pp": [100 * float(value) for value in inference["active_minus_fixed_terminal"]["bootstrap_95"]],
        },
        "authority_sha256": {
            ARRAYS.name: common.sha256_path(ARRAYS),
            LEDGER.name: common.sha256_path(LEDGER),
            CONFIG.name: common.sha256_path(CONFIG),
            SELECTION.name: common.sha256_path(SELECTION),
            Path(__file__).name: common.sha256_path(Path(__file__)),
        },
        "sealed_outcome_loaded": False,
    }
    common.json_dump(OUTPUT, output)
    print(json.dumps(output, indent=2), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
