from __future__ import annotations

import hashlib
import inspect
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset

import step93_common as common


ROOT = Path(__file__).resolve().parent
DATE = common.DATE
PROTOCOL_PATH = ROOT / f"STEP93_SOURCE_FAITHFUL_LEARNED_ABSTENTION_PREREGISTRATION_{DATE}.md"
AMENDMENT_A_PATH = ROOT / f"STEP93_PREREGISTRATION_AMENDMENT_A_{DATE}.md"
AMENDMENT_B_PATH = ROOT / f"STEP93_PREREGISTRATION_AMENDMENT_B_{DATE}.md"
MANIFEST_PATH = ROOT / f"STEP93_STAGE0_MANIFEST_{DATE}.json"
CALIBRATION_PATH = ROOT / f"STEP93_BANKING77_CALIBRATION_PACKAGE_{DATE}.json"
MODEL_DIR = ROOT / "step93_models"
LEDGER_PATH = ROOT / f"STEP93_STAGEA_COMPLETE_SEARCH_LEDGER_{DATE}.json"
CALIBRATION_OUTPUTS = ROOT / f"STEP93_STAGEA_CALIBRATION_OUTPUTS_{DATE}.npz"
CONFIG_PATH = ROOT / f"STEP93_STAGEA_FROZEN_CONFIG_{DATE}.json"

PILOT_SEEDS = tuple(range(930000, 930020))
VERIFY_SEEDS = tuple(range(930100, 930400))
TAUS = (0.10, 0.25, 0.50, 1.0, 2.0, 4.0)
LOSS_CAPS = (0.0025, 0.0050, 0.0075)
TRIGGER_CAPS = (0.05, 0.10, 0.20, 0.30)
TOP_TO_VERIFY = 16


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def source_text_sha256(function: object) -> str:
    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


def root_subset_indices(source_indices: np.ndarray, root: int, fraction: float) -> np.ndarray:
    keys = np.asarray(
        [common.stable_u64("step93-root-order", root, int(index)) for index in source_indices],
        dtype=np.uint64,
    )
    order = np.argsort(keys, kind="stable")
    count = max(1, int(math.ceil(float(fraction) * len(order))))
    return order[:count].astype(np.int64)


def prepare_construction(
    predictions: np.ndarray,
    labels: np.ndarray,
    parent: int,
    error_scores: np.ndarray,
    loss_cap: float,
    trigger_cap: float,
) -> dict[str, Any]:
    parent_predictions = predictions[:, parent].astype(np.int64)
    parent_correct = parent_predictions == labels
    thresholds: list[float] = []
    threshold_audits: list[dict[str, Any]] = []
    expected_triggers: list[np.ndarray] = []
    for alias in range(common.N_ALIASES):
        threshold, trigger, audit = common.threshold_from_caps(
            error_scores[:, alias], parent_correct, loss_cap, trigger_cap
        )
        thresholds.append(float(threshold))
        expected_triggers.append(trigger)
        threshold_audits.append({"alias": alias, **audit})
    alias_codes, triggers = common.make_alias_codes(parent_predictions, error_scores, thresholds)
    if not np.array_equal(triggers, np.stack(expected_triggers, axis=1)):
        raise AssertionError("threshold replay mismatch")
    parent_feedback = common.exact_match_feedback(parent_predictions[:, None], labels)
    alias_feedback = common.exact_match_feedback(alias_codes, labels)
    coordinate_nonimproving = bool(
        np.all(alias_feedback <= np.repeat(parent_feedback, common.N_ALIASES, axis=1))
    )
    losses = (
        np.mean(parent_feedback, axis=0)[0] - np.mean(alias_feedback, axis=0)
    ).astype(np.float64)
    pairwise_hamming = [
        float(np.mean(alias_codes[:, left] != alias_codes[:, right]))
        for left in range(common.N_ALIASES)
        for right in range(left + 1, common.N_ALIASES)
    ]
    eligible = bool(
        coordinate_nonimproving
        and np.all(np.isfinite(np.asarray(thresholds)))
        and np.all(losses <= float(loss_cap) + 1e-15)
        and np.all(np.mean(triggers, axis=0) <= float(trigger_cap) + 1e-15)
    )
    return {
        "thresholds": thresholds,
        "threshold_audits": threshold_audits,
        "alias_codes": alias_codes,
        "triggers": triggers,
        "alias_utility_losses": losses.tolist(),
        "realized_trigger_fractions": np.mean(triggers, axis=0).tolist(),
        "coordinate_wise_nonimproving": coordinate_nonimproving,
        "pairwise_alias_response_hamming": pairwise_hamming,
        "mean_pairwise_alias_response_hamming": float(np.mean(pairwise_hamming)),
        "minimum_pairwise_alias_response_hamming": float(np.min(pairwise_hamming)),
        "eligible": eligible,
    }


def evaluate(
    predictions: np.ndarray,
    labels: np.ndarray,
    parent: int,
    construction: dict[str, Any],
    tau: float,
    loss_cap: float,
    trigger_cap: float,
    seeds: tuple[int, ...],
    clean_cache: dict[tuple[tuple[int, ...], float], common.RunSummary],
    pool_cache: dict[tuple[int, ...], np.ndarray],
    retain_vectors: bool,
) -> dict[str, Any]:
    clean_codes = predictions.astype(np.int64)
    clean_parents = np.arange(predictions.shape[1], dtype=np.int64)
    alias_codes = np.asarray(construction["alias_codes"], dtype=np.int64)
    refined_codes = np.column_stack([clean_codes, alias_codes]).astype(np.int64)
    refined_parents = np.concatenate(
        [clean_parents, np.full(common.N_ALIASES, parent, dtype=np.int64)]
    )
    if seeds not in pool_cache:
        pool_cache[seeds] = common.sample_pools(len(labels), seeds)
    pools = pool_cache[seeds]
    cache_key = (seeds, float(tau))
    if cache_key not in clean_cache:
        clean_cache[cache_key] = common.run_active(
            clean_codes,
            labels,
            clean_parents,
            clean_codes,
            pools,
            seeds,
            common.BUDGET,
            tau,
        )
    clean = clean_cache[cache_key]
    refined = common.run_active(
        refined_codes,
        labels,
        refined_parents,
        clean_codes,
        pools,
        seeds,
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
        seeds,
    )
    terminal_delta = refined.terminal - clean.terminal
    cumulative_delta = refined.cumulative - clean.cumulative
    fixed_terminal_delta = refined_fixed.terminal - clean.terminal
    fixed_cumulative_delta = refined_fixed.cumulative - clean.cumulative
    active_minus_fixed = terminal_delta - fixed_terminal_delta
    row: dict[str, Any] = {
        "parent_root": int(parent),
        "tau": float(tau),
        "max_calibration_utility_loss": float(loss_cap),
        "max_trigger_fraction": float(trigger_cap),
        "thresholds": [float(value) for value in construction["thresholds"]],
        "threshold_audits": construction["threshold_audits"],
        "alias_utility_losses": construction["alias_utility_losses"],
        "realized_trigger_fractions": construction["realized_trigger_fractions"],
        "coordinate_wise_nonimproving": construction["coordinate_wise_nonimproving"],
        "mean_pairwise_alias_response_hamming": construction[
            "mean_pairwise_alias_response_hamming"
        ],
        "minimum_pairwise_alias_response_hamming": construction[
            "minimum_pairwise_alias_response_hamming"
        ],
        "eligible": bool(construction["eligible"]),
        "seeds": len(seeds),
        "mean_active_minus_fixed_terminal_delta": float(np.mean(active_minus_fixed)),
        "mean_terminal_delta": float(np.mean(terminal_delta)),
        "mean_cumulative_delta": float(np.mean(cumulative_delta)),
        "mean_fixed_terminal_delta": float(np.mean(fixed_terminal_delta)),
        "mean_fixed_cumulative_delta": float(np.mean(fixed_cumulative_delta)),
        "max_abs_fixed_terminal_delta": float(np.max(np.abs(fixed_terminal_delta))),
        "max_abs_fixed_cumulative_delta": float(np.max(np.abs(fixed_cumulative_delta))),
        "fixed_root_history_exact": bool(np.array_equal(refined_fixed.roots, clean.roots)),
        "path_change_rate": float(np.mean(np.any(clean.queries != refined.queries, axis=1))),
        "query_set_change_rate": float(
            np.mean(
                [
                    set(left.tolist()) != set(right.tolist())
                    for left, right in zip(clean.queries, refined.queries)
                ]
            )
        ),
        "final_root_change_rate": float(np.mean(clean.roots[:, -1] != refined.roots[:, -1])),
        "positive_terminal_fraction": float(np.mean(terminal_delta > 0)),
        "negative_terminal_fraction": float(np.mean(terminal_delta < 0)),
        "clean_query_sha256": hashlib.sha256(clean.queries.tobytes()).hexdigest(),
        "refined_query_sha256": hashlib.sha256(refined.queries.tobytes()).hexdigest(),
    }
    if retain_vectors:
        row.update(
            {
                "terminal_delta_values": terminal_delta.tolist(),
                "cumulative_delta_values": cumulative_delta.tolist(),
                "active_minus_fixed_terminal_values": active_minus_fixed.tolist(),
            }
        )
    return row


def rank_key(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(row["mean_active_minus_fixed_terminal_delta"]),
        float(row["mean_terminal_delta"]),
        float(row["mean_cumulative_delta"]),
        float(row["path_change_rate"]),
        -float(row["max_calibration_utility_loss"]),
        -float(row["max_trigger_fraction"]),
        -float(row["parent_root"]),
        -float(row["tau"]),
    )


def main() -> None:
    for output in (LEDGER_PATH, CALIBRATION_OUTPUTS, CONFIG_PATH):
        if output.exists():
            raise RuntimeError(f"refusing to overwrite Step 93 Stage A output: {output}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    authority_hashes = {
        "protocol_sha256": common.sha256_path(PROTOCOL_PATH),
        "amendment_a_sha256": common.sha256_path(AMENDMENT_A_PATH),
        "amendment_b_sha256": common.sha256_path(AMENDMENT_B_PATH),
    }
    for key, observed in authority_hashes.items():
        if manifest[key] != observed:
            raise AssertionError({"authority": key, "expected": manifest[key], "observed": observed})
    if manifest["outputs"][CALIBRATION_PATH.name] != common.sha256_path(CALIBRATION_PATH):
        raise AssertionError("calibration package hash mismatch")

    train = load_dataset(common.DATASET_ID, revision=common.DATASET_REVISION, split="train")
    if len(train) != int(manifest["dataset"]["counts"]["train"]):
        raise AssertionError("training count mismatch")
    partitions = np.asarray([common.train_partition(index) for index in range(len(train))])
    root_source = np.flatnonzero(partitions == "root_train").astype(np.int64)
    adapter_source = np.flatnonzero(partitions == "adapter_train").astype(np.int64)
    calibration_source = np.flatnonzero(partitions == "stagea_calibration").astype(np.int64)
    expected_counts = manifest["dataset"]["counts"]
    if [len(root_source), len(adapter_source), len(calibration_source)] != [
        expected_counts["root_train"],
        expected_counts["adapter_train"],
        expected_counts["stagea_calibration"],
    ]:
        raise AssertionError("partition count mismatch")

    calibration_package = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))["rows"]
    package_indices = np.asarray([row["source_index"] for row in calibration_package], dtype=np.int64)
    package_labels = np.asarray([row["label"] for row in calibration_package], dtype=np.int64)
    if not np.array_equal(package_indices, calibration_source):
        raise AssertionError("calibration index order mismatch")
    texts = [str(value) for value in train["text"]]
    labels = np.asarray(train["label"], dtype=np.int64)
    if not np.array_equal(labels[calibration_source], package_labels):
        raise AssertionError("calibration labels differ from sealed Stage 0 package")

    tokenizer, encoder, device = common.load_encoder()
    print(f"encoding {len(texts)} Banking77 development inputs on {device}", flush=True)
    hidden = common.encode_texts(texts, tokenizer, encoder, device)
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    root_hidden = hidden[root_source]
    root_labels = labels[root_source]
    adapter_hidden = hidden[adapter_source]
    adapter_labels = labels[adapter_source]
    calibration_hidden = hidden[calibration_source]
    calibration_labels = labels[calibration_source]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    root_rows: list[dict[str, Any]] = []
    calibration_root_logits: list[np.ndarray] = []
    adapter_parent_predictions: dict[int, np.ndarray] = {}
    for root, fraction in enumerate(common.ROOT_FRACTIONS):
        seed = common.ROOT_SEED_BASE + root
        subset = root_subset_indices(root_source, root, fraction)
        model = common.train_classifier(root_hidden, root_labels, subset, seed)
        path = MODEL_DIR / f"step93_clean_root_{root:02d}.safetensors"
        common.save_module(
            model,
            path,
            {
                "kind": "learned_bottleneck_intent_classifier",
                "base_model": common.BASE_MODEL,
                "base_revision": common.BASE_REVISION,
                "root": root,
                "seed": seed,
                "train_fraction": fraction,
                "train_examples": len(subset),
                "runtime_inputs": "raw_text_frozen_hidden",
            },
        )
        calibration_logits = common.infer_classifier(model, calibration_hidden)
        calibration_root_logits.append(calibration_logits)
        if root in common.ELIGIBLE_PARENTS:
            adapter_logits = common.infer_classifier(model, adapter_hidden)
            adapter_parent_predictions[root] = np.argmax(adapter_logits, axis=1).astype(np.int64)
        audit = common.tensor_audit(path)
        if audit["parameter_count"] < 109_000 or audit["variance"] <= 0.0:
            raise AssertionError({"invalid_root": root, "audit": audit})
        root_rows.append(
            {
                "root": root,
                "fraction": fraction,
                "seed": seed,
                "train_examples": len(subset),
                "training_source_index_sha256": common.canonical_json_sha256(
                    root_source[subset].tolist()
                ),
                "checkpoint": path.name,
                "checkpoint_audit": audit,
                "calibration_accuracy": float(
                    np.mean(np.argmax(calibration_logits, axis=1) == calibration_labels)
                ),
            }
        )
        print(f"trained clean root {root + 1}/{common.N_ROOTS}", flush=True)

    root_logits = np.stack(calibration_root_logits, axis=1)
    predictions = np.argmax(root_logits, axis=2).astype(np.int16)
    learned_rows: dict[str, Any] = {}
    calibration_error_scores: dict[int, np.ndarray] = {}
    for parent in common.ELIGIBLE_PARENTS:
        scores: list[np.ndarray] = []
        parent_rows: list[dict[str, Any]] = []
        for alias, seed in enumerate(common.ABSTENTION_SEEDS):
            model = common.train_abstention_adapter(
                adapter_hidden,
                adapter_parent_predictions[parent],
                adapter_labels,
                seed,
            )
            path = MODEL_DIR / f"step93_parent_{parent:02d}_abstention_adapter_{alias}.safetensors"
            common.save_module(
                model,
                path,
                {
                    "kind": "learned_parent_error_abstention_adapter",
                    "base_model": common.BASE_MODEL,
                    "base_revision": common.BASE_REVISION,
                    "parent_root": parent,
                    "alias": alias,
                    "seed": seed,
                    "train_examples": len(adapter_source),
                    "runtime_inputs": "raw_text_frozen_hidden",
                    "forbidden_item_lookup": True,
                    "abstention_token": f"ABSTAIN_{alias}",
                },
            )
            audit = common.tensor_audit(path)
            if (
                audit["parameter_count"] != 50_817
                or audit["variance"] <= 0.0
                or audit["nonzero_fraction"] < 0.95
            ):
                raise AssertionError({"invalid_abstention_adapter": path.name, "audit": audit})
            scores.append(common.infer_error_score(model, calibration_hidden))
            parent_rows.append(
                {
                    "alias": alias,
                    "seed": seed,
                    "checkpoint": path.name,
                    "checkpoint_audit": audit,
                    "adapter_training_error_prevalence": float(
                        np.mean(adapter_parent_predictions[parent] != adapter_labels)
                    ),
                }
            )
        score_matrix = np.stack(scores, axis=1)
        calibration_error_scores[parent] = score_matrix
        learned_rows[str(parent)] = parent_rows
        print(f"trained four learned abstention adapters for parent {parent}", flush=True)

    arrays: dict[str, np.ndarray] = {
        "calibration_source_indices": calibration_source,
        "calibration_labels": calibration_labels.astype(np.int16),
        "clean_root_logits": root_logits,
        "clean_root_predictions": predictions,
    }
    for parent, values in calibration_error_scores.items():
        arrays[f"parent_{parent:02d}_learned_error_scores"] = values
    np.savez_compressed(CALIBRATION_OUTPUTS, **arrays)

    construction_cache: dict[tuple[int, float, float], dict[str, Any]] = {}
    for parent, loss_cap, trigger_cap in itertools.product(
        common.ELIGIBLE_PARENTS, LOSS_CAPS, TRIGGER_CAPS
    ):
        construction_cache[(parent, loss_cap, trigger_cap)] = prepare_construction(
            predictions,
            calibration_labels,
            parent,
            calibration_error_scores[parent],
            loss_cap,
            trigger_cap,
        )

    pilot: list[dict[str, Any]] = []
    clean_cache: dict[tuple[tuple[int, ...], float], common.RunSummary] = {}
    pool_cache: dict[tuple[int, ...], np.ndarray] = {}
    grid = list(itertools.product(common.ELIGIBLE_PARENTS, TAUS, LOSS_CAPS, TRIGGER_CAPS))
    if len(grid) != 504:
        raise AssertionError(len(grid))
    for index, (parent, tau, loss_cap, trigger_cap) in enumerate(grid, start=1):
        construction = construction_cache[(parent, loss_cap, trigger_cap)]
        pilot.append(
            evaluate(
                predictions,
                calibration_labels,
                parent,
                construction,
                tau,
                loss_cap,
                trigger_cap,
                PILOT_SEEDS,
                clean_cache,
                pool_cache,
                retain_vectors=False,
            )
        )
        if index % 24 == 0 or index == len(grid):
            print(f"pilot {index}/{len(grid)}", flush=True)

    eligible_pilot = [row for row in pilot if row["eligible"]]
    if len(eligible_pilot) < TOP_TO_VERIFY:
        raise AssertionError({"eligible_conditions": len(eligible_pilot)})
    ranked = sorted(eligible_pilot, key=rank_key, reverse=True)
    verified: list[dict[str, Any]] = []
    for index, pilot_row in enumerate(ranked[:TOP_TO_VERIFY], start=1):
        parent = int(pilot_row["parent_root"])
        loss_cap = float(pilot_row["max_calibration_utility_loss"])
        trigger_cap = float(pilot_row["max_trigger_fraction"])
        verified_row = evaluate(
            predictions,
            calibration_labels,
            parent,
            construction_cache[(parent, loss_cap, trigger_cap)],
            float(pilot_row["tau"]),
            loss_cap,
            trigger_cap,
            VERIFY_SEEDS,
            clean_cache,
            pool_cache,
            retain_vectors=True,
        )
        verified_row["pilot_rank"] = index
        verified.append(verified_row)
        print(f"verification {index}/{TOP_TO_VERIFY}", flush=True)
    selected = max(verified, key=rank_key)

    same_similarity_assertion = {
        "definition": "s(a,b)=1[a=b]",
        "function": "step93_common.exact_match_similarity",
        "function_source_sha256": source_text_sha256(common.exact_match_similarity),
        "feedback_function_source_sha256": source_text_sha256(common.exact_match_feedback),
        "group_builder_source_sha256": source_text_sha256(common.build_source_faithful_groups),
        "run_active_source_sha256": source_text_sha256(common.run_active),
        "run_active_accepts_external_feedback": False,
    }
    ledger = {
        "ledger_id": "STEP93_SOURCE_FAITHFUL_LEARNED_ABSTENTION_STAGEA_V1",
        "date": DATE,
        "authority_sha256": authority_hashes,
        "stage0_manifest_sha256": common.sha256_path(MANIFEST_PATH),
        "calibration_package_sha256": common.sha256_path(CALIBRATION_PATH),
        "calibration_outputs_sha256": common.sha256_path(CALIBRATION_OUTPUTS),
        "base_model": {
            "id": common.BASE_MODEL,
            "revision": common.BASE_REVISION,
            "model_safetensors_sha256": common.BASE_MODEL_SHA256,
        },
        "training_partitions": {
            "root_train": len(root_source),
            "adapter_train": len(adapter_source),
            "stagea_calibration": len(calibration_source),
        },
        "single_similarity_assertion": same_similarity_assertion,
        "clean_roots": root_rows,
        "learned_abstention_adapters": learned_rows,
        "search_space": {
            "conditions": len(grid),
            "parents": list(common.ELIGIBLE_PARENTS),
            "temperatures": list(TAUS),
            "maximum_calibration_utility_losses": list(LOSS_CAPS),
            "maximum_trigger_fractions": list(TRIGGER_CAPS),
            "pilot_seeds": [PILOT_SEEDS[0], PILOT_SEEDS[-1]],
            "verification_seeds": [VERIFY_SEEDS[0], VERIFY_SEEDS[-1]],
            "top_verified": TOP_TO_VERIFY,
            "pool_size": common.POOL_SIZE,
            "budget": common.BUDGET,
        },
        "pilot_complete_504_condition_ledger": pilot,
        "verification_top_16": verified,
        "selected": selected,
    }
    write_json(LEDGER_PATH, ledger)

    selected_parent = int(selected["parent_root"])
    all_root_files = [
        MODEL_DIR / f"step93_clean_root_{root:02d}.safetensors"
        for root in range(common.N_ROOTS)
    ]
    all_adapter_files = [
        MODEL_DIR / f"step93_parent_{parent:02d}_abstention_adapter_{alias}.safetensors"
        for parent in common.ELIGIBLE_PARENTS
        for alias in range(common.N_ALIASES)
    ]
    selected_adapter_files = [
        MODEL_DIR / f"step93_parent_{selected_parent:02d}_abstention_adapter_{alias}.safetensors"
        for alias in range(common.N_ALIASES)
    ]
    config = {
        "config_id": "STEP93_SOURCE_FAITHFUL_LEARNED_ABSTENTION_FROZEN_V1",
        "date": DATE,
        "authority_sha256": authority_hashes,
        "stage0_manifest_sha256": common.sha256_path(MANIFEST_PATH),
        "stagea_ledger_sha256": common.sha256_path(LEDGER_PATH),
        "calibration_outputs_sha256": common.sha256_path(CALIBRATION_OUTPUTS),
        "single_similarity_assertion": same_similarity_assertion,
        "dataset": {
            "id": common.DATASET_ID,
            "revision": common.DATASET_REVISION,
        },
        "base_model": {
            "id": common.BASE_MODEL,
            "revision": common.BASE_REVISION,
            "model_safetensors_sha256": common.BASE_MODEL_SHA256,
        },
        "roster_roots": list(range(common.N_ROOTS)),
        "selected": {
            "parent_root": selected_parent,
            "tau": float(selected["tau"]),
            "max_calibration_utility_loss": float(selected["max_calibration_utility_loss"]),
            "max_trigger_fraction": float(selected["max_trigger_fraction"]),
            "thresholds": [float(value) for value in selected["thresholds"]],
            "calibration_alias_utility_losses": selected["alias_utility_losses"],
            "calibration_trigger_fractions": selected["realized_trigger_fractions"],
            "calibration_active_minus_fixed_terminal_delta": selected[
                "mean_active_minus_fixed_terminal_delta"
            ],
            "calibration_terminal_delta": selected["mean_terminal_delta"],
            "calibration_cumulative_delta": selected["mean_cumulative_delta"],
            "calibration_path_change_rate": selected["path_change_rate"],
            "pool_size": common.POOL_SIZE,
            "budget": common.BUDGET,
        },
        "clean_root_sha256": {path.name: common.sha256_path(path) for path in all_root_files},
        "all_learned_adapter_sha256": {
            path.name: common.sha256_path(path) for path in all_adapter_files
        },
        "selected_learned_adapter_sha256": {
            path.name: common.sha256_path(path) for path in selected_adapter_files
        },
        "selected_learned_adapter_audits": {
            path.name: common.tensor_audit(path) for path in selected_adapter_files
        },
        "construction_access": {
            "public_training_labels": True,
            "stagea_calibration_labels": True,
            "holdout_inputs_or_labels": False,
            "item_id_or_prompt_lookup": False,
            "peer_outputs": False,
            "selector_state": False,
        },
    }
    write_json(CONFIG_PATH, config)
    print(
        json.dumps(
            {
                "config": CONFIG_PATH.name,
                "config_sha256": common.sha256_path(CONFIG_PATH),
                "ledger": LEDGER_PATH.name,
                "ledger_sha256": common.sha256_path(LEDGER_PATH),
                "selected": config["selected"],
                "selected_adapter_parameter_counts": [
                    config["selected_learned_adapter_audits"][path.name]["parameter_count"]
                    for path in selected_adapter_files
                ],
                "all_model_files": len(all_root_files) + len(all_adapter_files),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
