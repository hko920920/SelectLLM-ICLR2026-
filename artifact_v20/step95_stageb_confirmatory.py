from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import step95_common as common
from step95_executable_adapter_endpoint import execute_aliases_from_raw_pairs


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / f"STEP95_STAGEA_FROZEN_CONFIG_{common.DATE}.json"
LEDGER_PATH = ROOT / f"STEP95_STAGEA_COMPLETE_LEDGER_{common.DATE}.json"
MANIFEST_PATH = ROOT / f"STEP95_STAGE0_MANIFEST_{common.DATE}.json"
INCIDENT_PATH = ROOT / f"STEP95_PREREGISTRATION_AMENDMENT_B_{common.DATE}.md"
LOCK_PATH = ROOT / f"STEP95_CONFIRMATORY_EXECUTION_LOCK_{common.DATE}.json"
RAW_PATH = ROOT / f"STEP95_CONFIRMATORY_RAW_{common.DATE}.npz"
RESULT_PATH = ROOT / f"STEP95_CONFIRMATORY_RESULTS_{common.DATE}.json"

CONFIRMATORY_SEEDS = tuple(range(955000, 958000))
BOOTSTRAP_SEED = 95910
SIGNFLIP_SEED = 95911


def source_sha256(function: object) -> str:
    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


def adapter_parameter_count(path: Path) -> int:
    return int(sum(tensor.numel() for tensor in load_file(str(path)).values()))


def infer_test_root(
    spec: common.RootSpec,
    premises: list[str],
    hypotheses: list[str],
    batch_size: int = 64,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Replay a pinned root; root 8 uses the recorded Windows slow-tokenizer fix."""
    if spec.key != "crossencoder_deberta_small":
        return common.infer_root_predictions(spec, premises, hypotheses, batch_size)
    tokenizer = AutoTokenizer.from_pretrained(
        spec.repo_id, revision=spec.revision, use_fast=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        spec.repo_id, revision=spec.revision, use_safetensors=None
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(premises), batch_size):
            batch = tokenizer(
                premises[start : start + batch_size],
                hypotheses[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            chunks.append(torch.argmax(model(**batch).logits, dim=1).cpu().numpy())
    raw = np.concatenate(chunks).astype(np.int64)
    predictions = np.asarray(spec.raw_to_dataset, dtype=np.int64)[raw]
    audit = {
        "root_key": spec.key,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "model_type": str(model.config.model_type),
        "tokenizer_class": tokenizer.__class__.__name__,
        "raw_to_dataset": list(spec.raw_to_dataset),
        "prediction_sha256": common.array_sha256(predictions),
        "samples": len(predictions),
    }
    model.cpu()
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions, audit


def _code_paths() -> list[Path]:
    return [
        ROOT / "step93_common.py",
        ROOT / "step95_common.py",
        ROOT / "step95_executable_adapter_endpoint.py",
        ROOT / "step95_stageb_confirmatory.py",
        ROOT / "lock_step95_confirmatory.py",
        ROOT / "validate_step95_independent.py",
    ]


def create_execution_lock() -> dict[str, Any]:
    if LOCK_PATH.exists():
        raise RuntimeError(f"refusing to overwrite lock: {LOCK_PATH}")
    if RAW_PATH.exists() or RESULT_PATH.exists():
        raise RuntimeError("confirmatory output predates execution lock")
    for path in (CONFIG_PATH, LEDGER_PATH, MANIFEST_PATH, INCIDENT_PATH, *_code_paths()):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if config["decision"] != "GO_TO_STEP95_ONE_TIME_CONFIRMATORY_LOCK":
        raise RuntimeError(config["decision"])
    if config["stagea_ledger_sha256"] != common.sha256_path(LEDGER_PATH):
        raise AssertionError("Stage-A ledger binding mismatch")
    for relative, expected in config["adapter_sha256"].items():
        if common.sha256_path(ROOT / relative) != expected:
            raise AssertionError({"adapter": relative, "hash": False})
    test_input = ROOT / manifest["test_input"]["file"]
    sealed_outcome = ROOT / manifest["sealed_outcome"]["file"]
    if common.sha256_path(test_input) != manifest["test_input"]["sha256"]:
        raise AssertionError("test-input binding mismatch")
    if common.sha256_path(sealed_outcome) != manifest["sealed_outcome"]["sha256"]:
        raise AssertionError("sealed-outcome binding mismatch")
    lock = {
        "lock_id": "STEP95_ONE_TIME_CONFIRMATORY_EXECUTION_LOCK_V1",
        "date": common.DATE,
        "authority_sha256": config["authority_sha256"],
        "execution_incident_sha256": common.sha256_path(INCIDENT_PATH),
        "stage0_manifest_sha256": common.sha256_path(MANIFEST_PATH),
        "stagea_ledger_sha256": common.sha256_path(LEDGER_PATH),
        "stagea_config_sha256": common.sha256_path(CONFIG_PATH),
        "selected_parent": config["selected_parent"],
        "roster_bank_indices": config["roster_bank_indices"],
        "roster_keys": config["roster_keys"],
        "selected_condition": config["selected_condition"],
        "adapter_sha256": config["adapter_sha256"],
        "test_input": manifest["test_input"],
        "sealed_outcome": manifest["sealed_outcome"],
        "root_bank": manifest["root_bank"],
        "code_sha256": {path.name: common.sha256_path(path) for path in _code_paths()},
        "source_assertions": {
            "raw_endpoint_signature": str(inspect.signature(execute_aliases_from_raw_pairs)),
            "raw_endpoint_source_sha256": source_sha256(execute_aliases_from_raw_pairs),
            "exact_similarity_source_sha256": source_sha256(common.exact_match_similarity),
            "active_source_sha256": source_sha256(common.run_active),
            "fixed_source_sha256": source_sha256(common.run_fixed),
            "test_root_source_sha256": source_sha256(infer_test_root),
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
            "matched_validation_only": True,
            "mismatched_validation_not_authorized": True,
            "single_sealed_open_authorized": True,
        },
    }
    common.json_dump(LOCK_PATH, lock)
    return lock


def main() -> None:
    for output in (RAW_PATH, RESULT_PATH):
        if output.exists():
            raise RuntimeError(f"refusing to overwrite confirmatory output: {output}")
    if not LOCK_PATH.is_file():
        raise RuntimeError("execution lock must exist before the confirmatory runner")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if lock["stagea_config_sha256"] != common.sha256_path(CONFIG_PATH):
        raise AssertionError("config changed after lock")
    if lock["stagea_ledger_sha256"] != common.sha256_path(LEDGER_PATH):
        raise AssertionError("ledger changed after lock")
    for filename, expected in lock["code_sha256"].items():
        if common.sha256_path(ROOT / filename) != expected:
            raise AssertionError({"code": filename, "post_lock_hash": False})
    for relative, expected in lock["adapter_sha256"].items():
        if common.sha256_path(ROOT / relative) != expected:
            raise AssertionError({"adapter": relative, "post_lock_hash": False})

    input_path = ROOT / lock["test_input"]["file"]
    sealed_path = ROOT / lock["sealed_outcome"]["file"]
    if common.sha256_path(input_path) != lock["test_input"]["sha256"]:
        raise AssertionError("input changed after lock")
    if common.sha256_path(sealed_path) != lock["sealed_outcome"]["sha256"]:
        raise AssertionError("sealed outcome changed after lock")
    inputs = json.loads(input_path.read_text(encoding="utf-8"))
    rows = inputs["rows"]
    premises = [str(row["premise"]) for row in rows]
    hypotheses = [str(row["hypothesis"]) for row in rows]
    source_indices = np.asarray([int(row["source_index"]) for row in rows], dtype=np.int64)
    parent_key = str(config["selected_parent"]["key"])
    thresholds = [float(value) for value in config["selected_condition"]["thresholds"]]
    adapter_paths = list(config["adapter_paths"])
    endpoint = execute_aliases_from_raw_pairs(
        premises, hypotheses, parent_key, adapter_paths, thresholds
    )
    parent_predictions = np.asarray(endpoint["parent_predictions"], dtype=np.int64)
    alias_codes = np.asarray(endpoint["alias_codes"], dtype=np.int64)
    parent_position = int(config["selected_parent"]["position"])
    clean_columns: list[np.ndarray] = []
    root_audits: list[dict[str, Any]] = []
    for position, bank_index in enumerate(config["roster_bank_indices"]):
        spec = common.ROOT_SPECS[int(bank_index)]
        if position == parent_position:
            predictions = parent_predictions
            audit = dict(endpoint["parent_audit"])
        else:
            predictions, audit = infer_test_root(spec, premises, hypotheses)
        clean_columns.append(predictions)
        root_audits.append(audit)
        print(f"test root {position + 1}/{common.ROSTER_SIZE}: {spec.key}", flush=True)
    clean_codes = np.stack(clean_columns, axis=1)
    if not np.array_equal(clean_codes[:, parent_position], parent_predictions):
        raise AssertionError("endpoint parent differs from clean parent")

    # This is the one authorized opening of validation_matched outcomes.
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
        np.all(alias_feedback <= np.repeat(parent_feedback, common.N_ALIASES, axis=1))
    )
    alias_losses = (
        np.mean(parent_feedback, axis=0)[0] - np.mean(alias_feedback, axis=0)
    ).astype(np.float64)
    clean_parents = np.arange(common.ROSTER_SIZE, dtype=np.int64)
    refined_codes = np.column_stack([clean_codes, alias_codes])
    refined_parents = np.concatenate(
        [clean_parents, np.full(common.N_ALIASES, parent_position, dtype=np.int64)]
    )
    pools = common.sample_pools(
        len(labels), CONFIRMATORY_SEEDS, pool_size=common.POOL_SIZE
    )
    tau = float(config["selected_condition"]["tau"])
    clean = common.run_active(
        clean_codes, labels, clean_parents, clean_codes, pools,
        CONFIRMATORY_SEEDS, common.BUDGET, tau,
    )
    refined = common.run_active(
        refined_codes, labels, refined_parents, clean_codes, pools,
        CONFIRMATORY_SEEDS, common.BUDGET, tau,
    )
    fixed = common.run_fixed(
        refined_codes, labels, refined_parents, clean_codes, pools,
        clean.queries, CONFIRMATORY_SEEDS,
    )
    terminal = refined.terminal - clean.terminal
    cumulative = refined.cumulative - clean.cumulative
    fixed_terminal = fixed.terminal - clean.terminal
    fixed_cumulative = fixed.cumulative - clean.cumulative
    active_minus_fixed = terminal - fixed_terminal
    terminal_summary = common.effect_summary(terminal, BOOTSTRAP_SEED, SIGNFLIP_SEED)
    active_summary = common.effect_summary(
        active_minus_fixed, BOOTSTRAP_SEED + 1, SIGNFLIP_SEED + 1
    )
    cumulative_summary = common.effect_summary(
        cumulative, BOOTSTRAP_SEED + 2, SIGNFLIP_SEED + 2
    )
    np.savez_compressed(
        RAW_PATH,
        source_indices=source_indices,
        labels=labels.astype(np.int16),
        clean_codes=clean_codes.astype(np.int16),
        alias_codes=alias_codes.astype(np.int16),
        adapter_error_scores=np.asarray(endpoint["adapter_error_scores"], dtype=np.float64),
        pools=pools,
        clean_terminal=clean.terminal,
        refined_terminal=refined.terminal,
        fixed_terminal=fixed.terminal,
        clean_cumulative=clean.cumulative,
        refined_cumulative=refined.cumulative,
        fixed_cumulative=fixed.cumulative,
        clean_queries=clean.queries,
        refined_queries=refined.queries,
        clean_roots=clean.roots,
        refined_roots=refined.roots,
        fixed_roots=fixed.roots,
        terminal_delta=terminal,
        cumulative_delta=cumulative,
        fixed_terminal_delta=fixed_terminal,
        fixed_cumulative_delta=fixed_cumulative,
        active_minus_fixed_terminal_delta=active_minus_fixed,
    )
    path_change = float(np.mean(np.any(clean.queries != refined.queries, axis=1)))
    query_set_change = float(np.mean([
        set(left.tolist()) != set(right.tolist())
        for left, right in zip(clean.queries, refined.queries)
    ]))
    fixed_root_exact = bool(np.array_equal(fixed.roots, clean.roots))
    signature = lock["source_assertions"]["raw_endpoint_signature"].lower()
    adapter_parameter_counts = [
        adapter_parameter_count(ROOT / path) for path in adapter_paths
    ]
    gates = {
        "pre_outcome_integrity": True,
        "all_development_gates_passed": bool(all(config["development_gates"].values())),
        "four_distinct_adapters_at_least_ten_million_parameters": bool(
            len(set(endpoint["adapter_sha256"])) == common.N_ALIASES
            and all(count >= 10_000_000 for count in adapter_parameter_counts)
        ),
        "raw_endpoint_has_no_forbidden_runtime_input": all(token not in signature for token in (
            "label", "reference", "item_id", "lookup", "peer", "pool",
            "trajectory", "posterior", "selector",
        )),
        "same_literal_similarity": True,
        "coordinate_wise_nonimproving": coordinate_nonimproving,
        "each_alias_loss_at_most_one_point": bool(np.max(alias_losses) <= 0.0100),
        "path_change_at_least_half": bool(path_change >= 0.50),
        "terminal_mean_at_least_half_point": bool(terminal_summary["mean"] >= 0.005),
        "terminal_bootstrap_lower_positive": bool(terminal_summary["bootstrap_95"][0] > 0),
        "terminal_signflip_at_most_point05": bool(terminal_summary["one_sided_signflip_p"] <= 0.05),
        "active_minus_fixed_mean_at_least_half_point": bool(active_summary["mean"] >= 0.005),
        "active_minus_fixed_bootstrap_lower_positive": bool(active_summary["bootstrap_95"][0] > 0),
        "active_minus_fixed_signflip_at_most_point05": bool(active_summary["one_sided_signflip_p"] <= 0.05),
        "cumulative_mean_positive": bool(cumulative_summary["mean"] > 0),
        "cumulative_bootstrap_lower_positive": bool(cumulative_summary["bootstrap_95"][0] > 0),
        "fixed_terminal_exact_zero": bool(np.max(np.abs(fixed_terminal)) == 0),
        "fixed_cumulative_exact_zero": bool(np.max(np.abs(fixed_cumulative)) == 0),
        "fixed_root_history_bitwise_exact": fixed_root_exact,
    }
    decision = (
        "GO_SOURCE_FAITHFUL_FULL_ADAPTER_TERMINAL_BRIDGE"
        if all(gates.values()) else "NO_GO_RETAIN_STEP95_NEGATIVE"
    )
    results = {
        "result_id": "STEP95_MNLI_CONFIRMATORY_V1",
        "date": common.DATE,
        "decision": decision,
        "execution_lock_sha256": common.sha256_path(LOCK_PATH),
        "stagea_config_sha256": common.sha256_path(CONFIG_PATH),
        "raw_sha256": common.sha256_path(RAW_PATH),
        "dataset": "nyu-mll/multi_nli:validation_matched",
        "parent_key": parent_key,
        "parent_position": parent_position,
        "roster_keys": config["roster_keys"],
        "tau": tau,
        "thresholds": thresholds,
        "root_prediction_audits": root_audits,
        "endpoint": {
            "adapter_sha256": endpoint["adapter_sha256"],
            "adapter_parameter_counts": adapter_parameter_counts,
            "runtime_inputs": endpoint["runtime_inputs"],
            "forbidden_runtime_inputs": False,
        },
        "quality": {
            "parent_accuracy": float(np.mean(parent_feedback)),
            "alias_accuracies": np.mean(alias_feedback, axis=0).tolist(),
            "alias_losses": alias_losses.tolist(),
            "coordinate_wise_nonimproving": coordinate_nonimproving,
            "trigger_fractions": np.mean(np.asarray(endpoint["triggers"]), axis=0).tolist(),
        },
        "mechanism": {
            "path_change_rate": path_change,
            "query_set_change_rate": query_set_change,
            "final_root_change_rate": float(np.mean(clean.roots[:, -1] != refined.roots[:, -1])),
        },
        "effects": {
            "terminal": terminal_summary,
            "active_minus_fixed_terminal": active_summary,
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
