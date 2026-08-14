from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download

import step92_common as common


ROOT = Path(__file__).resolve().parent
DATE = common.DATE
CONFIG_PATH = ROOT / f"STEP92_STAGEA_FROZEN_CONFIG_{DATE}.json"
LOCK_PATH = ROOT / f"STEP92_CONFIRMATORY_EXECUTION_LOCK_{DATE}.json"
INPUT_PATH = ROOT / f"STEP92_AGNEWS_HOLDOUT_INPUTS_{DATE}.json"
SEALED_PATH = ROOT / "external_data" / "step92_sealed" / f"STEP92_AGNEWS_SEALED_OUTCOMES_{DATE}.npz"
MODEL_DIR = ROOT / "step92_models"
RAW_PATH = ROOT / f"STEP92_LEARNED_ADAPTER_CONFIRMATORY_RAW_{DATE}.npz"
RESULT_PATH = ROOT / f"STEP92_LEARNED_ADAPTER_CONFIRMATORY_RESULTS_{DATE}.json"
SEEDS = tuple(range(1000))
MIN_TERMINAL_DELTA = 0.005
MIN_PATH_CHANGE = 0.50


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def pairwise_hamming(codes: np.ndarray) -> tuple[list[float], float, float]:
    values = [
        float(np.mean(codes[:, left] != codes[:, right]))
        for left in range(4)
        for right in range(left + 1, 4)
    ]
    return values, float(np.mean(values)), float(np.min(values))


def main() -> None:
    if not LOCK_PATH.exists():
        raise FileNotFoundError("confirmatory execution lock missing")
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
    labels = sealed["labels"].astype(np.int8)
    if len(inputs) != 7600 or len(labels) != 7600:
        raise AssertionError((len(inputs), len(labels)))

    tokenizer, encoder, device = common.load_encoder()
    hidden = common.encode_texts([str(row["text"]) for row in inputs], tokenizer, encoder, device)
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    root_logits: list[np.ndarray] = []
    root_predictions: list[np.ndarray] = []
    for root in config["roster_roots"]:
        path = MODEL_DIR / f"step92_clean_root_{int(root):02d}.safetensors"
        model = common.load_classifier(path, torch.device("cpu"))
        logits = common.infer_classifier(model, hidden)
        root_logits.append(logits)
        root_predictions.append(np.argmax(logits, axis=1).astype(np.int8))
    logits_tensor = np.stack(root_logits, axis=1)
    predictions = np.stack(root_predictions, axis=1)

    adapter_logs: list[np.ndarray] = []
    adapter_audits: dict[str, Any] = {}
    for alias in range(4):
        path = MODEL_DIR / f"step92_parent_{parent:02d}_learned_adapter_{alias}.safetensors"
        model = common.load_temperature(path, torch.device("cpu"))
        adapter_logs.append(common.infer_log_temperature(model, hidden))
        adapter_audits[path.name] = common.adapter_tensor_audit(path)
    log_temperatures = np.stack(adapter_logs, axis=1)
    parent_logits = logits_tensor[:, parent, :]
    alias_codes, alias_confidences, alias_triggers = common.learned_response_codes(
        parent_logits,
        log_temperatures,
        [float(value) for value in selected["thresholds"]],
        int(selected["bins"]),
    )
    parent_hard = predictions[:, parent].astype(np.int64)
    alias_hard = np.repeat(parent_hard[:, None], 4, axis=1)
    hard_equal = bool(np.array_equal(alias_hard, np.repeat(parent_hard[:, None], 4, axis=1)))

    clean_codes = predictions.astype(np.int64)
    core_feedback = (predictions == labels[:, None]).astype(np.float64)
    clean_feedback = core_feedback.copy()
    clean_parents = np.arange(predictions.shape[1], dtype=np.int64)
    refined_codes = np.column_stack([clean_codes, alias_codes]).astype(np.int64)
    refined_feedback = np.column_stack(
        [clean_feedback, np.repeat(core_feedback[:, [parent]], 4, axis=1)]
    )
    refined_parents = np.concatenate(
        [clean_parents, np.full(4, parent, dtype=np.int64)]
    )
    if not np.array_equal(
        refined_feedback[:, -4:], np.repeat(core_feedback[:, [parent]], 4, axis=1)
    ):
        raise AssertionError("coordinate-wise utility mismatch")

    pools = common.sample_pools(len(labels), SEEDS, int(selected["pool_size"]))
    clean = common.run_active(
        clean_codes,
        clean_feedback,
        clean_parents,
        core_feedback,
        pools,
        SEEDS,
        int(selected["budget"]),
        float(selected["tau"]),
    )
    refined = common.run_active(
        refined_codes,
        refined_feedback,
        refined_parents,
        core_feedback,
        pools,
        SEEDS,
        int(selected["budget"]),
        float(selected["tau"]),
    )
    clean_fixed = common.run_fixed(
        clean_feedback, clean_parents, core_feedback, pools, clean.queries, SEEDS
    )
    refined_fixed = common.run_fixed(
        refined_feedback, refined_parents, core_feedback, pools, clean.queries, SEEDS
    )

    terminal_delta = refined.terminal - clean.terminal
    cumulative_delta = refined.cumulative - clean.cumulative
    fixed_terminal = refined_fixed.terminal - clean_fixed.terminal
    fixed_cumulative = refined_fixed.cumulative - clean_fixed.cumulative
    active_minus_fixed = terminal_delta - fixed_terminal
    fixed_exact = bool(
        np.array_equal(fixed_terminal, np.zeros_like(fixed_terminal))
        and np.array_equal(fixed_cumulative, np.zeros_like(fixed_cumulative))
        and np.array_equal(clean_fixed.roots, refined_fixed.roots)
    )
    terminal_summary = common.effect_summary(terminal_delta, 920_100)
    cumulative_summary = common.effect_summary(cumulative_delta, 920_200)
    active_minus_fixed_summary = common.effect_summary(active_minus_fixed, 920_300)
    path_change = float(np.mean(np.any(clean.queries != refined.queries, axis=1)))
    pairwise_values, pairwise_mean, pairwise_min = pairwise_hamming(alias_codes)

    adapter_hashes = [row["sha256"] for row in adapter_audits.values()]
    adapter_gate = bool(
        len(set(adapter_hashes)) == 4
        and all(row["parameter_count"] >= 49_000 for row in adapter_audits.values())
        and all(row["variance"] > 0.0 for row in adapter_audits.values())
        and all(row["nonzero_count"] > 0 for row in adapter_audits.values())
    )
    gates = {
        "all_locked_hashes_match": True,
        "genuine_learned_adapter_files": adapter_gate,
        "coordinate_wise_hard_utility_equal": hard_equal,
        "fixed_query_exact_zero": fixed_exact,
        "path_change_at_least_half": path_change >= MIN_PATH_CHANGE,
        "mean_terminal_delta_at_least_0_5pp": terminal_summary["mean"] >= MIN_TERMINAL_DELTA,
        "terminal_bootstrap_lower_above_zero": terminal_summary["bootstrap_95"][0] > 0.0,
        "one_sided_signflip_p_at_most_0_05": terminal_summary["one_sided_signflip_p"] <= 0.05,
        "active_minus_fixed_at_least_0_5pp": (
            active_minus_fixed_summary["mean"] >= MIN_TERMINAL_DELTA
            and active_minus_fixed_summary["bootstrap_95"][0] > 0.0
        ),
    }
    if not all(gates[key] for key in (
        "all_locked_hashes_match",
        "genuine_learned_adapter_files",
        "coordinate_wise_hard_utility_equal",
        "fixed_query_exact_zero",
    )):
        decision = "INVALID_STEP92_IMPLEMENTATION"
    elif all(gates.values()):
        decision = "GO_SCORE_CHANGING_LEARNED_ADAPTER_BRIDGE"
    else:
        decision = "NO_SCORE_CHANGING_LEARNED_ADAPTER_BRIDGE"

    arrays = {
        "source_indices": source_indices,
        "labels": labels,
        "clean_root_logits": logits_tensor,
        "clean_root_predictions": predictions,
        "parent_hard": parent_hard,
        "adapter_log_temperatures": log_temperatures,
        "adapter_confidences": alias_confidences,
        "adapter_triggers": alias_triggers,
        "adapter_response_codes": alias_codes,
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
        "fixed_terminal_delta": fixed_terminal,
        "fixed_cumulative_delta": fixed_cumulative,
        "active_minus_fixed_terminal": active_minus_fixed,
    }
    np.savez_compressed(RAW_PATH, **arrays)
    results = {
        "result_id": "STEP92_LEARNED_CONTEXTUAL_ADAPTER_CONFIRMATORY_V1",
        "decision": decision,
        "config_sha256": common.sha256_path(CONFIG_PATH),
        "execution_lock_sha256": common.sha256_path(LOCK_PATH),
        "raw_sha256": common.sha256_path(RAW_PATH),
        "task": {
            "name": "AG News",
            "dataset": common.DATASET_ID,
            "revision": common.DATASET_REVISION,
            "n_holdout": len(labels),
            "paired_runs": len(SEEDS),
            "selected": selected,
            "clean_root_holdout_accuracies": np.mean(core_feedback, axis=0).tolist(),
            "parent_holdout_accuracy": float(np.mean(parent_hard == labels)),
            "adapter_parent_accuracy_gaps": [0.0, 0.0, 0.0, 0.0],
            "realized_trigger_fractions": np.mean(alias_triggers, axis=0).tolist(),
            "pairwise_alias_response_hamming": pairwise_values,
            "mean_pairwise_alias_response_hamming": pairwise_mean,
            "minimum_pairwise_alias_response_hamming": pairwise_min,
            "path_change_rate": path_change,
            "query_set_change_rate": float(
                np.mean([set(a.tolist()) != set(b.tolist()) for a, b in zip(clean.queries, refined.queries)])
            ),
            "final_root_change_rate": float(np.mean(clean.roots[:, -1] != refined.roots[:, -1])),
            "terminal_regret_delta": terminal_summary,
            "cumulative_regret_delta": cumulative_summary,
            "active_minus_fixed_terminal": active_minus_fixed_summary,
            "max_abs_fixed_terminal_delta": float(np.max(np.abs(fixed_terminal))),
            "max_abs_fixed_cumulative_delta": float(np.max(np.abs(fixed_cumulative))),
            "learned_adapter_audits": adapter_audits,
            "gates": gates,
        },
        "claim_boundary": (
            "New raw-input task with nontrivially learned contextual calibration adapters; "
            "not live registry admission, secret-task compromise, universal harm, or deployment prevalence."
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
                "fixed_query_exact_zero": fixed_exact,
                "gates": gates,
                "raw_sha256": results["raw_sha256"],
                "result_sha256": common.sha256_path(RESULT_PATH),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
