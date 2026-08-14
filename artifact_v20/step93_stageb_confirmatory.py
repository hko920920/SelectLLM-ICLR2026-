from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download

import step93_common as common
from step93_executable_abstention_endpoint import (
    FORBIDDEN_RUNTIME_INPUTS,
    RUNTIME_INPUT_CONTRACT,
    execute_raw_text_endpoint,
)


ROOT = Path(__file__).resolve().parent
DATE = common.DATE
CONFIG_PATH = ROOT / f"STEP93_STAGEA_FROZEN_CONFIG_{DATE}.json"
LOCK_PATH = ROOT / f"STEP93_CONFIRMATORY_EXECUTION_LOCK_{DATE}.json"
INPUT_PATH = ROOT / f"STEP93_BANKING77_HOLDOUT_INPUTS_{DATE}.json"
SEALED_PATH = ROOT / "external_data" / "step93_sealed" / f"STEP93_BANKING77_SEALED_OUTCOMES_{DATE}.npz"
MODEL_DIR = ROOT / "step93_models"
RAW_PATH = ROOT / f"STEP93_SOURCE_FAITHFUL_CONFIRMATORY_RAW_{DATE}.npz"
RESULT_PATH = ROOT / f"STEP93_SOURCE_FAITHFUL_CONFIRMATORY_RESULTS_{DATE}.json"
SEEDS = tuple(range(931000, 932000))
MIN_TERMINAL_DELTA = 0.005
MIN_PATH_CHANGE = 0.50
MAX_ALIAS_ACCURACY_LOSS = 0.010
BOOTSTRAP_SEED = 93500
SIGNFLIP_SEED = 93501


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def source_text_sha256(function: object) -> str:
    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


def pairwise_hamming(codes: np.ndarray) -> tuple[list[float], float, float]:
    values = [
        float(np.mean(codes[:, left] != codes[:, right]))
        for left in range(common.N_ALIASES)
        for right in range(left + 1, common.N_ALIASES)
    ]
    return values, float(np.mean(values)), float(np.min(values))


def main() -> None:
    if not LOCK_PATH.exists():
        raise FileNotFoundError("confirmatory execution lock missing")
    if RAW_PATH.exists() or RESULT_PATH.exists():
        raise RuntimeError("refusing to overwrite Step 93 confirmatory output")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    for relative, expected in lock["locked_sha256"].items():
        path = ROOT / Path(relative)
        observed = common.sha256_path(path)
        if observed != expected:
            raise AssertionError({"path": relative, "expected": expected, "observed": observed})

    base_path = Path(
        hf_hub_download(
            common.BASE_MODEL,
            "model.safetensors",
            revision=common.BASE_REVISION,
            local_files_only=True,
        )
    )
    if common.sha256_path(base_path) != common.BASE_MODEL_SHA256:
        raise AssertionError("base model hash mismatch")

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    selected = config["selected"]
    parent = int(selected["parent_root"])
    inputs = json.loads(INPUT_PATH.read_text(encoding="utf-8"))["rows"]
    sealed = np.load(SEALED_PATH, allow_pickle=False)
    source_indices = np.asarray([row["source_index"] for row in inputs], dtype=np.int64)
    if not np.array_equal(source_indices, sealed["source_indices"]):
        raise AssertionError("holdout source order mismatch")
    labels = sealed["labels"].astype(np.int64)
    if len(inputs) != 3_076 or len(labels) != 3_076:
        raise AssertionError((len(inputs), len(labels)))
    texts = [str(row["text"]) for row in inputs]

    tokenizer, encoder, device = common.load_encoder()
    hidden = common.encode_texts(texts, tokenizer, encoder, device)
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    root_logits: list[np.ndarray] = []
    root_predictions: list[np.ndarray] = []
    for root in config["roster_roots"]:
        path = MODEL_DIR / f"step93_clean_root_{int(root):02d}.safetensors"
        model = common.load_classifier(path, torch.device("cpu"))
        logits = common.infer_classifier(model, hidden)
        root_logits.append(logits)
        root_predictions.append(np.argmax(logits, axis=1).astype(np.int64))
    logits_tensor = np.stack(root_logits, axis=1)
    predictions = np.stack(root_predictions, axis=1)

    selected_adapter_paths = [
        MODEL_DIR / f"step93_parent_{parent:02d}_abstention_adapter_{alias}.safetensors"
        for alias in range(common.N_ALIASES)
    ]
    thresholds = [float(value) for value in selected["thresholds"]]
    internal_scores: list[np.ndarray] = []
    adapter_audits: dict[str, Any] = {}
    for path in selected_adapter_paths:
        model = common.load_abstention(path, torch.device("cpu"))
        internal_scores.append(common.infer_error_score(model, hidden))
        adapter_audits[path.name] = common.tensor_audit(path)
    error_scores = np.stack(internal_scores, axis=1)
    parent_predictions = predictions[:, parent].astype(np.int64)
    alias_codes, alias_triggers = common.make_alias_codes(
        parent_predictions, error_scores, thresholds
    )

    endpoint = execute_raw_text_endpoint(
        texts,
        MODEL_DIR / f"step93_clean_root_{parent:02d}.safetensors",
        selected_adapter_paths,
        thresholds,
    )
    endpoint_exact_replay = bool(
        np.array_equal(endpoint["parent_predictions"], parent_predictions)
        and np.array_equal(endpoint["learned_error_scores"], error_scores)
        and np.array_equal(endpoint["response_codes"], alias_codes)
        and np.array_equal(endpoint["triggers"], alias_triggers)
    )

    clean_codes = predictions.astype(np.int64)
    clean_parents = np.arange(common.N_ROOTS, dtype=np.int64)
    refined_codes = np.column_stack([clean_codes, alias_codes]).astype(np.int64)
    refined_parents = np.concatenate(
        [clean_parents, np.full(common.N_ALIASES, parent, dtype=np.int64)]
    )
    core_feedback = common.exact_match_feedback(clean_codes, labels)
    parent_feedback = core_feedback[:, parent]
    alias_feedback = common.exact_match_feedback(alias_codes, labels)
    coordinate_nonimproving = bool(np.all(alias_feedback <= parent_feedback[:, None]))
    parent_accuracy = float(np.mean(parent_feedback))
    alias_accuracies = np.mean(alias_feedback, axis=0)
    alias_accuracy_losses = parent_accuracy - alias_accuracies

    same_similarity_expected = config["single_similarity_assertion"]
    same_similarity_observed = {
        "definition": "s(a,b)=1[a=b]",
        "function": "step93_common.exact_match_similarity",
        "function_source_sha256": source_text_sha256(common.exact_match_similarity),
        "feedback_function_source_sha256": source_text_sha256(common.exact_match_feedback),
        "group_builder_source_sha256": source_text_sha256(common.build_source_faithful_groups),
        "run_active_source_sha256": source_text_sha256(common.run_active),
        "run_active_accepts_external_feedback": "feedback" in inspect.signature(
            common.run_active
        ).parameters,
    }
    same_similarity_gate = bool(
        same_similarity_observed["definition"] == same_similarity_expected["definition"]
        and same_similarity_observed["function"] == same_similarity_expected["function"]
        and same_similarity_observed["function_source_sha256"]
        == same_similarity_expected["function_source_sha256"]
        and same_similarity_observed["feedback_function_source_sha256"]
        == same_similarity_expected["feedback_function_source_sha256"]
        and same_similarity_observed["group_builder_source_sha256"]
        == same_similarity_expected["group_builder_source_sha256"]
        and same_similarity_observed["run_active_source_sha256"]
        == same_similarity_expected["run_active_source_sha256"]
        and same_similarity_expected["run_active_accepts_external_feedback"] is False
        and same_similarity_observed["run_active_accepts_external_feedback"] is False
    )
    dense_probe = common.exact_match_agreement(refined_codes[:64])
    groups_probe = common.build_source_faithful_groups(refined_codes[:64])
    probe_posterior = np.arange(1, refined_codes.shape[1] + 1, dtype=np.float64)
    probe_posterior /= np.sum(probe_posterior)
    dense_acquisition = np.einsum(
        "nij,i,j->n", dense_probe, probe_posterior, probe_posterior, optimize=True
    )
    sparse_acquisition = common.select_source_faithful_acquisition(
        groups_probe, probe_posterior
    )
    same_similarity_numeric_probe = bool(
        np.allclose(dense_acquisition, sparse_acquisition, rtol=0.0, atol=2e-16)
        and np.array_equal(
            common.exact_match_feedback(refined_codes[:64], labels[:64]),
            common.exact_match_similarity(
                refined_codes[:64], labels[:64, None]
            ).astype(np.float64),
        )
    )

    pools = common.sample_pools(len(labels), SEEDS, int(selected["pool_size"]))
    clean = common.run_active(
        clean_codes,
        labels,
        clean_parents,
        clean_codes,
        pools,
        SEEDS,
        int(selected["budget"]),
        float(selected["tau"]),
    )
    refined = common.run_active(
        refined_codes,
        labels,
        refined_parents,
        clean_codes,
        pools,
        SEEDS,
        int(selected["budget"]),
        float(selected["tau"]),
    )
    clean_fixed = common.run_fixed(
        clean_codes,
        labels,
        clean_parents,
        clean_codes,
        pools,
        clean.queries,
        SEEDS,
    )
    refined_fixed = common.run_fixed(
        refined_codes,
        labels,
        refined_parents,
        clean_codes,
        pools,
        clean.queries,
        SEEDS,
    )

    terminal_delta = refined.terminal - clean.terminal
    cumulative_delta = refined.cumulative - clean.cumulative
    fixed_terminal_delta = refined_fixed.terminal - clean_fixed.terminal
    fixed_cumulative_delta = refined_fixed.cumulative - clean_fixed.cumulative
    active_minus_fixed = terminal_delta - fixed_terminal_delta
    fixed_exact = bool(
        np.array_equal(fixed_terminal_delta, np.zeros_like(fixed_terminal_delta))
        and np.array_equal(fixed_cumulative_delta, np.zeros_like(fixed_cumulative_delta))
        and np.array_equal(clean_fixed.roots, refined_fixed.roots)
        and np.array_equal(clean_fixed.roots, clean.roots)
    )
    terminal_summary = common.effect_summary(
        terminal_delta, BOOTSTRAP_SEED, SIGNFLIP_SEED
    )
    cumulative_summary = common.effect_summary(
        cumulative_delta, BOOTSTRAP_SEED, SIGNFLIP_SEED
    )
    active_minus_fixed_summary = common.effect_summary(
        active_minus_fixed, BOOTSTRAP_SEED, SIGNFLIP_SEED
    )
    path_change = float(np.mean(np.any(clean.queries != refined.queries, axis=1)))
    pairwise_values, pairwise_mean, pairwise_min = pairwise_hamming(alias_codes)

    adapter_hashes = [row["sha256"] for row in adapter_audits.values()]
    adapter_gate = bool(
        len(set(adapter_hashes)) == common.N_ALIASES
        and all(row["parameter_count"] == 50_817 for row in adapter_audits.values())
        and all(row["variance"] > 0.0 for row in adapter_audits.values())
        and all(row["nonzero_fraction"] >= 0.95 for row in adapter_audits.values())
    )
    runtime_gate = bool(
        endpoint_exact_replay
        and tuple(endpoint["runtime_input_contract"]) == RUNTIME_INPUT_CONTRACT
        and tuple(endpoint["forbidden_runtime_inputs"]) == FORBIDDEN_RUNTIME_INPUTS
        and not set(RUNTIME_INPUT_CONTRACT).intersection(FORBIDDEN_RUNTIME_INPUTS)
    )
    gates = {
        "all_locked_hashes_match": True,
        "same_exact_similarity_for_acquisition_and_posterior": (
            same_similarity_gate and same_similarity_numeric_probe
        ),
        "four_genuine_distinct_50817_parameter_adapters": adapter_gate,
        "raw_text_endpoint_exact_replay_and_runtime_exclusions": runtime_gate,
        "coordinate_wise_nonimproving": coordinate_nonimproving,
        "each_alias_accuracy_loss_at_most_1pp": bool(
            np.all(alias_accuracy_losses <= MAX_ALIAS_ACCURACY_LOSS + 1e-15)
        ),
        "path_change_at_least_half": path_change >= MIN_PATH_CHANGE,
        "mean_terminal_delta_at_least_0_5pp": terminal_summary["mean"] >= MIN_TERMINAL_DELTA,
        "terminal_bootstrap_lower_above_zero": terminal_summary["bootstrap_95"][0] > 0.0,
        "terminal_one_sided_signflip_p_at_most_0_05": (
            terminal_summary["one_sided_signflip_p"] <= 0.05
        ),
        "active_minus_fixed_at_least_0_5pp_and_lower_above_zero": bool(
            active_minus_fixed_summary["mean"] >= MIN_TERMINAL_DELTA
            and active_minus_fixed_summary["bootstrap_95"][0] > 0.0
        ),
        "fixed_query_terminal_and_cumulative_exact_zero": fixed_exact,
    }
    validity_keys = (
        "all_locked_hashes_match",
        "same_exact_similarity_for_acquisition_and_posterior",
        "four_genuine_distinct_50817_parameter_adapters",
        "raw_text_endpoint_exact_replay_and_runtime_exclusions",
        "coordinate_wise_nonimproving",
        "fixed_query_terminal_and_cumulative_exact_zero",
    )
    if not all(gates[key] for key in validity_keys):
        decision = "INVALID_STEP93_IMPLEMENTATION"
    elif all(gates.values()):
        decision = "GO_SOURCE_FAITHFUL_LEARNED_ABSTENTION_BRIDGE"
    else:
        decision = "NO_GO_RETAIN_SOURCE_FAITHFUL_NEGATIVE"

    arrays = {
        "source_indices": source_indices,
        "labels": labels.astype(np.int16),
        "clean_root_logits": logits_tensor,
        "clean_root_predictions": predictions.astype(np.int16),
        "parent_predictions": parent_predictions.astype(np.int16),
        "learned_error_scores": error_scores,
        "adapter_triggers": alias_triggers,
        "adapter_response_codes": alias_codes.astype(np.int16),
        "alias_feedback": alias_feedback.astype(np.int8),
        "pools": pools,
        "clean_queries": clean.queries,
        "refined_queries": refined.queries,
        "clean_roots": clean.roots,
        "refined_roots": refined.roots,
        "clean_fixed_roots": clean_fixed.roots,
        "refined_fixed_roots": refined_fixed.roots,
        "clean_terminal": clean.terminal,
        "refined_terminal": refined.terminal,
        "terminal_delta": terminal_delta,
        "cumulative_delta": cumulative_delta,
        "fixed_terminal_delta": fixed_terminal_delta,
        "fixed_cumulative_delta": fixed_cumulative_delta,
        "active_minus_fixed_terminal": active_minus_fixed,
    }
    np.savez_compressed(RAW_PATH, **arrays)
    results = {
        "result_id": "STEP93_SOURCE_FAITHFUL_LEARNED_ABSTENTION_CONFIRMATORY_V1",
        "decision": decision,
        "config_sha256": common.sha256_path(CONFIG_PATH),
        "execution_lock_sha256": common.sha256_path(LOCK_PATH),
        "raw_sha256": common.sha256_path(RAW_PATH),
        "task": {
            "name": "Banking77",
            "dataset": common.DATASET_ID,
            "revision": common.DATASET_REVISION,
            "n_holdout": len(labels),
            "paired_runs": len(SEEDS),
            "selected": selected,
            "clean_root_holdout_accuracies": np.mean(core_feedback, axis=0).tolist(),
            "parent_holdout_accuracy": parent_accuracy,
            "alias_holdout_accuracies": alias_accuracies.tolist(),
            "alias_parent_accuracy_losses": alias_accuracy_losses.tolist(),
            "realized_trigger_fractions": np.mean(alias_triggers, axis=0).tolist(),
            "pairwise_alias_response_hamming": pairwise_values,
            "mean_pairwise_alias_response_hamming": pairwise_mean,
            "minimum_pairwise_alias_response_hamming": pairwise_min,
            "path_change_rate": path_change,
            "query_set_change_rate": float(
                np.mean(
                    [
                        set(left.tolist()) != set(right.tolist())
                        for left, right in zip(clean.queries, refined.queries)
                    ]
                )
            ),
            "final_root_change_rate": float(
                np.mean(clean.roots[:, -1] != refined.roots[:, -1])
            ),
            "terminal_regret_delta": terminal_summary,
            "cumulative_regret_delta": cumulative_summary,
            "active_minus_fixed_terminal": active_minus_fixed_summary,
            "max_abs_fixed_terminal_delta": float(np.max(np.abs(fixed_terminal_delta))),
            "max_abs_fixed_cumulative_delta": float(np.max(np.abs(fixed_cumulative_delta))),
            "same_similarity_observed": same_similarity_observed,
            "same_similarity_numeric_probe": same_similarity_numeric_probe,
            "endpoint_exact_replay": endpoint_exact_replay,
            "runtime_input_contract": list(RUNTIME_INPUT_CONTRACT),
            "forbidden_runtime_inputs": list(FORBIDDEN_RUNTIME_INPUTS),
            "learned_adapter_audits": adapter_audits,
            "gates": gates,
        },
        "claim_boundary": (
            "One fresh task with learned no-lookup same-root abstention refinements and one "
            "source-faithful exact-match similarity; not live registry admission, an independently "
            "fine-tuned full language model, secret-task compromise, universal harm, or prevalence."
        ),
    }
    write_json(RESULT_PATH, results)
    print(
        json.dumps(
            {
                "decision": decision,
                "mean_terminal_delta_pp": 100.0 * terminal_summary["mean"],
                "terminal_bootstrap_95_pp": [
                    100.0 * value for value in terminal_summary["bootstrap_95"]
                ],
                "one_sided_signflip_p": terminal_summary["one_sided_signflip_p"],
                "active_minus_fixed_pp": 100.0 * active_minus_fixed_summary["mean"],
                "path_change_rate": path_change,
                "alias_accuracy_losses_pp": (100.0 * alias_accuracy_losses).tolist(),
                "fixed_query_exact_zero": fixed_exact,
                "same_similarity_gate": same_similarity_gate and same_similarity_numeric_probe,
                "gates": gates,
                "raw_sha256": results["raw_sha256"],
                "result_sha256": common.sha256_path(RESULT_PATH),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
