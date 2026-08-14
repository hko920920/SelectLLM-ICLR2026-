from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset

import step92_common as common


ROOT = Path(__file__).resolve().parent
DATE = common.DATE
MANIFEST_PATH = ROOT / f"STEP92_STAGE0_MANIFEST_{DATE}.json"
CALIBRATION_PATH = ROOT / f"STEP92_AGNEWS_CALIBRATION_PACKAGE_{DATE}.json"
MODEL_DIR = ROOT / "step92_models"
LEDGER_PATH = ROOT / f"STEP92_STAGEA_COMPLETE_SEARCH_LEDGER_{DATE}.json"
CALIBRATION_OUTPUTS = ROOT / f"STEP92_STAGEA_CALIBRATION_OUTPUTS_{DATE}.npz"
CONFIG_PATH = ROOT / f"STEP92_STAGEA_FROZEN_CONFIG_{DATE}.json"

PILOT_SEEDS = tuple(range(20))
VERIFY_SEEDS = tuple(range(300))
TAUS = (0.25, 1.0, 4.0)
BINS = (8, 16, 32, 64)
TRIGGER_FRACTIONS = (0.05, 0.10, 0.20, 0.30, 0.50)
TOP_TO_VERIFY = 12


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def root_subset_indices(source_indices: np.ndarray, root: int, fraction: float) -> np.ndarray:
    keys = np.asarray(
        [common.stable_u64("step92-root-order", root, int(index)) for index in source_indices],
        dtype=np.uint64,
    )
    order = np.argsort(keys, kind="stable")
    count = max(1, int(math.ceil(fraction * len(order))))
    return order[:count].astype(np.int64)


def prepare_registry(
    predictions: np.ndarray,
    labels: np.ndarray,
    parent: int,
    alias_codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    clean_codes = predictions.astype(np.int64)
    core_feedback = (predictions == labels[:, None]).astype(np.float64)
    clean_parents = np.arange(predictions.shape[1], dtype=np.int64)
    refined_codes = np.column_stack([clean_codes, alias_codes]).astype(np.int64)
    refined_feedback = np.column_stack(
        [core_feedback, np.repeat(core_feedback[:, [parent]], 4, axis=1)]
    )
    refined_parents = np.concatenate(
        [clean_parents, np.full(4, parent, dtype=np.int64)]
    )
    return core_feedback, refined_codes, refined_feedback, refined_parents


def evaluate(
    predictions: np.ndarray,
    labels: np.ndarray,
    parent_logits: np.ndarray,
    adapter_log_temperatures: np.ndarray,
    config: dict[str, Any],
    seeds: tuple[int, ...],
    clean_cache: dict[tuple[tuple[int, ...], float], common.RunSummary],
    pool_cache: dict[tuple[int, ...], np.ndarray],
) -> dict[str, Any]:
    parent = int(config["parent_root"])
    trigger_fraction = float(config["trigger_fraction"])
    bins = int(config["bins"])
    tau = float(config["tau"])
    scaled_confidences = []
    for alias in range(4):
        scaled = parent_logits * np.exp(adapter_log_temperatures[:, [alias]])
        scaled_confidences.append(common.softmax_confidence(scaled))
    confidence_matrix = np.stack(scaled_confidences, axis=1)
    thresholds = common.quantile_thresholds(confidence_matrix, trigger_fraction)
    alias_codes, _, triggers = common.learned_response_codes(
        parent_logits, adapter_log_temperatures, thresholds, bins
    )
    core_feedback, refined_codes, refined_feedback, refined_parents = prepare_registry(
        predictions, labels, parent, alias_codes
    )
    clean_codes = predictions.astype(np.int64)
    clean_feedback = core_feedback.copy()
    clean_parents = np.arange(predictions.shape[1], dtype=np.int64)
    if seeds not in pool_cache:
        pool_cache[seeds] = common.sample_pools(len(labels), seeds)
    pools = pool_cache[seeds]
    cache_key = (seeds, tau)
    if cache_key not in clean_cache:
        clean_cache[cache_key] = common.run_active(
            clean_codes,
            clean_feedback,
            clean_parents,
            core_feedback,
            pools,
            seeds,
            common.BUDGET,
            tau,
        )
    clean = clean_cache[cache_key]
    refined = common.run_active(
        refined_codes,
        refined_feedback,
        refined_parents,
        core_feedback,
        pools,
        seeds,
        common.BUDGET,
        tau,
    )
    terminal_delta = refined.terminal - clean.terminal
    cumulative_delta = refined.cumulative - clean.cumulative
    pairwise = [
        float(np.mean(alias_codes[:, left] != alias_codes[:, right]))
        for left in range(4)
        for right in range(left + 1, 4)
    ]
    return {
        **config,
        "thresholds": thresholds,
        "seeds": len(seeds),
        "mean_terminal_delta": float(np.mean(terminal_delta)),
        "mean_cumulative_delta": float(np.mean(cumulative_delta)),
        "path_change_rate": float(np.mean(np.any(clean.queries != refined.queries, axis=1))),
        "final_root_change_rate": float(np.mean(clean.roots[:, -1] != refined.roots[:, -1])),
        "positive_terminal_fraction": float(np.mean(terminal_delta > 0)),
        "negative_terminal_fraction": float(np.mean(terminal_delta < 0)),
        "realized_trigger_fractions": np.mean(triggers, axis=0).tolist(),
        "mean_pairwise_alias_response_hamming": float(np.mean(pairwise)),
        "minimum_pairwise_alias_response_hamming": float(np.min(pairwise)),
        "terminal_delta_values": terminal_delta.tolist() if len(seeds) >= len(VERIFY_SEEDS) else None,
        "clean_query_sha256": hashlib.sha256(clean.queries.tobytes()).hexdigest(),
        "refined_query_sha256": hashlib.sha256(refined.queries.tobytes()).hexdigest(),
    }


def rank_key(row: dict[str, Any]) -> tuple[float, float, float, str]:
    serial = json.dumps(
        {
            key: row[key]
            for key in ("parent_root", "tau", "bins", "trigger_fraction")
        },
        sort_keys=True,
    )
    return (
        float(row["mean_terminal_delta"]),
        float(row["path_change_rate"]),
        float(row["mean_cumulative_delta"]),
        serial,
    )


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["protocol_sha256"] != common.sha256_path(
        ROOT / f"STEP92_LEARNED_CONTEXTUAL_ADAPTER_PREREGISTRATION_{DATE}.md"
    ):
        raise AssertionError("protocol hash mismatch")
    if manifest["outputs"][CALIBRATION_PATH.name] != common.sha256_path(CALIBRATION_PATH):
        raise AssertionError("calibration package hash mismatch")

    train = load_dataset(
        common.DATASET_ID,
        revision=common.DATASET_REVISION,
        split="train",
    )
    partitions = np.asarray([common.train_partition(index) for index in range(len(train))])
    model_source = np.flatnonzero(partitions == "model_train").astype(np.int64)
    adapter_source = np.flatnonzero(partitions == "adapter_train").astype(np.int64)
    calibration_source = np.flatnonzero(partitions == "stagea_calibration").astype(np.int64)
    expected_counts = manifest["dataset"]["counts"]
    if [len(model_source), len(adapter_source), len(calibration_source)] != [
        expected_counts["model_train"],
        expected_counts["adapter_train"],
        expected_counts["stagea_calibration"],
    ]:
        raise AssertionError("partition count mismatch")
    texts = [str(value) for value in train["text"]]
    labels = np.asarray(train["label"], dtype=np.int64)

    tokenizer, encoder, device = common.load_encoder()
    print(f"encoding {len(texts)} AG News train inputs on {device}", flush=True)
    hidden = common.encode_texts(texts, tokenizer, encoder, device)
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model_hidden = hidden[model_source]
    model_labels = labels[model_source]
    adapter_hidden = hidden[adapter_source]
    adapter_labels = labels[adapter_source]
    calibration_hidden = hidden[calibration_source]
    calibration_labels = labels[calibration_source]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    root_rows: list[dict[str, Any]] = []
    calibration_root_logits: list[np.ndarray] = []
    adapter_root_logits: dict[int, np.ndarray] = {}
    for root, fraction in enumerate(common.ROOT_FRACTIONS):
        seed = common.ROOT_SEED_BASE + root
        subset = root_subset_indices(model_source, root, fraction)
        model = common.train_classifier(model_hidden, model_labels, subset, seed)
        path = MODEL_DIR / f"step92_clean_root_{root:02d}.safetensors"
        common.save_module(
            model,
            path,
            {
                "kind": "learned_bottleneck_classification_adapter",
                "base_model": common.BASE_MODEL,
                "base_revision": common.BASE_REVISION,
                "root": root,
                "seed": seed,
                "train_fraction": fraction,
                "train_examples": len(subset),
                "runtime_inputs": "raw_text_only",
            },
        )
        calibration_logits = common.infer_classifier(model, calibration_hidden)
        calibration_root_logits.append(calibration_logits)
        if root in common.ELIGIBLE_PARENTS:
            adapter_root_logits[root] = common.infer_classifier(model, adapter_hidden)
        audit = common.adapter_tensor_audit(path)
        root_rows.append(
            {
                "root": root,
                "fraction": fraction,
                "seed": seed,
                "train_examples": len(subset),
                "training_source_index_sha256": common.canonical_json_sha256(
                    model_source[subset].tolist()
                ),
                "checkpoint": path.name,
                "checkpoint_audit": audit,
                "calibration_accuracy": float(
                    np.mean(np.argmax(calibration_logits, axis=1) == calibration_labels)
                ),
            }
        )
        print(f"trained clean root {root + 1}/12", flush=True)

    root_logits = np.stack(calibration_root_logits, axis=1)
    predictions = np.argmax(root_logits, axis=2).astype(np.int8)
    learned_rows: dict[str, Any] = {}
    calibration_log_temperatures: dict[int, np.ndarray] = {}
    for parent in common.ELIGIBLE_PARENTS:
        logs: list[np.ndarray] = []
        parent_rows: list[dict[str, Any]] = []
        for alias in range(4):
            seed = common.CALIBRATION_SEED_BASE + alias
            model = common.train_temperature_adapter(
                adapter_hidden,
                adapter_root_logits[parent],
                adapter_labels,
                seed,
            )
            path = MODEL_DIR / f"step92_parent_{parent:02d}_learned_adapter_{alias}.safetensors"
            common.save_module(
                model,
                path,
                {
                    "kind": "learned_contextual_temperature_adapter",
                    "base_model": common.BASE_MODEL,
                    "base_revision": common.BASE_REVISION,
                    "parent_root": parent,
                    "alias": alias,
                    "seed": seed,
                    "train_examples": len(adapter_source),
                    "hard_label_invariant": True,
                    "runtime_inputs": "raw_text_parent_hidden_parent_logits",
                    "item_lookup_table": False,
                },
            )
            logs.append(common.infer_log_temperature(model, calibration_hidden))
            parent_rows.append(
                {
                    "alias": alias,
                    "seed": seed,
                    "checkpoint": path.name,
                    "checkpoint_audit": common.adapter_tensor_audit(path),
                }
            )
        calibration_log_temperatures[parent] = np.stack(logs, axis=1)
        learned_rows[str(parent)] = parent_rows
        print(f"trained four learned adapters for parent {parent}", flush=True)

    arrays: dict[str, np.ndarray] = {
        "calibration_source_indices": calibration_source,
        "calibration_labels": calibration_labels.astype(np.int8),
        "clean_root_logits": root_logits,
        "clean_root_predictions": predictions,
    }
    for parent, values in calibration_log_temperatures.items():
        arrays[f"parent_{parent:02d}_adapter_log_temperatures"] = values
    np.savez_compressed(CALIBRATION_OUTPUTS, **arrays)

    pilot: list[dict[str, Any]] = []
    clean_cache: dict[tuple[tuple[int, ...], float], common.RunSummary] = {}
    pool_cache: dict[tuple[int, ...], np.ndarray] = {}
    grid = list(
        itertools.product(common.ELIGIBLE_PARENTS, TAUS, BINS, TRIGGER_FRACTIONS)
    )
    for index, (parent, tau, bins, trigger_fraction) in enumerate(grid, start=1):
        config = {
            "parent_root": parent,
            "tau": tau,
            "bins": bins,
            "trigger_fraction": trigger_fraction,
        }
        pilot.append(
            evaluate(
                predictions,
                calibration_labels,
                root_logits[:, parent, :],
                calibration_log_temperatures[parent],
                config,
                PILOT_SEEDS,
                clean_cache,
                pool_cache,
            )
        )
        if index % 25 == 0 or index == len(grid):
            print(f"pilot {index}/{len(grid)}", flush=True)

    ranked = sorted(pilot, key=rank_key, reverse=True)
    verified: list[dict[str, Any]] = []
    for index, row in enumerate(ranked[:TOP_TO_VERIFY], start=1):
        config = {
            key: row[key] for key in ("parent_root", "tau", "bins", "trigger_fraction")
        }
        parent = int(config["parent_root"])
        verified.append(
            evaluate(
                predictions,
                calibration_labels,
                root_logits[:, parent, :],
                calibration_log_temperatures[parent],
                config,
                VERIFY_SEEDS,
                clean_cache,
                pool_cache,
            )
        )
        print(f"verify {index}/{TOP_TO_VERIFY}", flush=True)
    selected = max(verified, key=rank_key)

    ledger = {
        "ledger_id": "STEP92_LEARNED_CONTEXTUAL_ADAPTER_STAGEA_V1",
        "stage0_manifest_sha256": common.sha256_path(MANIFEST_PATH),
        "calibration_package_sha256": common.sha256_path(CALIBRATION_PATH),
        "base_model": common.BASE_MODEL,
        "base_revision": common.BASE_REVISION,
        "training_partitions": {
            "model_train": len(model_source),
            "adapter_train": len(adapter_source),
            "stagea_calibration": len(calibration_source),
        },
        "clean_roots": root_rows,
        "learned_adapters": learned_rows,
        "search_space": {
            "pilot_seeds": [PILOT_SEEDS[0], PILOT_SEEDS[-1]],
            "verification_seeds": [VERIFY_SEEDS[0], VERIFY_SEEDS[-1]],
            "parent_roots": list(common.ELIGIBLE_PARENTS),
            "tau": list(TAUS),
            "bins": list(BINS),
            "trigger_fractions": list(TRIGGER_FRACTIONS),
            "roster_roots": list(range(12)),
            "pool_size": common.POOL_SIZE,
            "budget": common.BUDGET,
            "pilot_configurations": len(pilot),
            "verified_configurations": len(verified),
        },
        "selection_rule": (
            "maximum verified terminal delta, then path change, cumulative delta, serialized config"
        ),
        "pilot_results": pilot,
        "verification_results": verified,
        "selected": selected,
        "holdout_access": False,
    }
    write_json(LEDGER_PATH, ledger)

    selected_parent = int(selected["parent_root"])
    selected_adapter_files = [
        MODEL_DIR / f"step92_parent_{selected_parent:02d}_learned_adapter_{alias}.safetensors"
        for alias in range(4)
    ]
    config = {
        "config_id": "STEP92_LEARNED_CONTEXTUAL_ADAPTER_CONFIRMATORY_V1",
        "stage0_manifest_sha256": common.sha256_path(MANIFEST_PATH),
        "stagea_ledger_sha256": common.sha256_path(LEDGER_PATH),
        "calibration_outputs_sha256": common.sha256_path(CALIBRATION_OUTPUTS),
        "dataset": {"id": common.DATASET_ID, "revision": common.DATASET_REVISION},
        "base_model": {
            "id": common.BASE_MODEL,
            "revision": common.BASE_REVISION,
            "model_safetensors_sha256": common.BASE_MODEL_SHA256,
        },
        "roster_roots": list(range(12)),
        "clean_root_sha256": {
            f"step92_clean_root_{root:02d}.safetensors": common.sha256_path(
                MODEL_DIR / f"step92_clean_root_{root:02d}.safetensors"
            )
            for root in range(12)
        },
        "selected": {
            "parent_root": selected_parent,
            "tau": float(selected["tau"]),
            "bins": int(selected["bins"]),
            "trigger_fraction": float(selected["trigger_fraction"]),
            "thresholds": [float(value) for value in selected["thresholds"]],
            "pool_size": common.POOL_SIZE,
            "budget": common.BUDGET,
        },
        "learned_adapter_sha256": {
            path.name: common.sha256_path(path) for path in selected_adapter_files
        },
        "learned_adapter_audits": {
            path.name: common.adapter_tensor_audit(path) for path in selected_adapter_files
        },
        "construction_access": {
            "public_train_labels": True,
            "stagea_calibration_labels": True,
            "test_inputs_or_labels": False,
            "item_or_prompt_lookup": False,
            "peer_outputs_at_runtime": False,
        },
    }
    write_json(CONFIG_PATH, config)
    print(
        json.dumps(
            {
                "config": CONFIG_PATH.name,
                "config_sha256": common.sha256_path(CONFIG_PATH),
                "selected": config["selected"],
                "calibration_terminal_delta_pp": 100.0 * selected["mean_terminal_delta"],
                "calibration_path_change_rate": selected["path_change_rate"],
                "adapter_parameter_counts": [
                    config["learned_adapter_audits"][path.name]["parameter_count"]
                    for path in selected_adapter_files
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
