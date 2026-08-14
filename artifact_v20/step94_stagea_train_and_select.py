from __future__ import annotations

import hashlib
import inspect
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download

import step94_common as common


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / f"STEP94_STAGE0_MANIFEST_{common.DATE}.json"
MODEL_DIR = ROOT / "step94_models"
OUTPUT_DIR = ROOT / "step94_stagea_outputs"
LEDGER_PATH = ROOT / f"STEP94_STAGEA_COMPLETE_LEDGER_{common.DATE}.json"
CONFIG_PATH = ROOT / f"STEP94_STAGEA_FROZEN_CONFIG_{common.DATE}.json"

SEARCH_SEEDS = tuple(range(940000, 940040))
VERIFY_SEEDS = tuple(range(940100, 940600))
TAUS = (0.05, 0.10, 0.25, 0.50, 1.0, 2.0)
LOSS_CAPS = (0.0025, 0.0050, 0.0100)
TRIGGER_CAPS = (0.05, 0.10, 0.20, 0.40)
TOP_TO_VERIFY = 24


def source_sha256(function: object) -> str:
    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


def task_manifest(manifest: dict[str, Any], task_key: str) -> dict[str, Any]:
    matches = [row for row in manifest["tasks"] if row["task_key"] == task_key]
    if len(matches) != 1:
        raise AssertionError({"task_key": task_key, "manifest_matches": len(matches)})
    return matches[0]


def load_development(
    task: common.TaskSpec, manifest_row: dict[str, Any]
) -> tuple[list[str], np.ndarray]:
    paths: dict[str, str] = {}
    for split in task.development_splits:
        path = Path(
            hf_hub_download(
                repo_id=task.repo_id,
                filename=task.filenames[split],
                repo_type="dataset",
                revision=task.revision,
            )
        )
        if common.sha256_path(path) != manifest_row["raw_file_sha256"][split]:
            raise AssertionError({"task": task.key, "split": split, "raw_hash": False})
        paths[split] = str(path)
    payload = load_dataset("json", data_files=paths)
    texts: list[str] = []
    labels: list[int] = []
    for split in task.development_splits:
        texts.extend(str(value) for value in payload[split][task.text_column])
        labels.extend(int(value) for value in payload[split][task.label_column])
    labels_array = np.asarray(labels, dtype=np.int64)
    if common.canonical_json_sha256(texts) != manifest_row["development_text_sha256"]:
        raise AssertionError({"task": task.key, "development_text_hash": False})
    if common.array_sha256(labels_array.astype(np.int16)) != manifest_row[
        "development_label_sha256"
    ]:
        raise AssertionError({"task": task.key, "development_label_hash": False})
    return texts, labels_array


def replay_construction(
    roster_predictions: np.ndarray,
    labels: np.ndarray,
    parent_position: int,
    gate_scores: np.ndarray,
    thresholds: list[float],
    n_classes: int,
) -> dict[str, Any]:
    parent_predictions = roster_predictions[:, parent_position].astype(np.int64)
    parent_correct = parent_predictions == labels
    alias_codes, triggers = common.make_alias_codes(
        parent_predictions, gate_scores, thresholds, n_classes
    )
    parent_feedback = common.exact_match_feedback(parent_predictions[:, None], labels)
    alias_feedback = common.exact_match_feedback(alias_codes, labels)
    losses = np.mean(parent_feedback, axis=0)[0] - np.mean(alias_feedback, axis=0)
    pairwise = [
        float(np.mean(alias_codes[:, left] != alias_codes[:, right]))
        for left in range(common.N_ALIASES)
        for right in range(left + 1, common.N_ALIASES)
    ]
    return {
        "thresholds": [float(value) for value in thresholds],
        "threshold_audits": [],
        "alias_codes": alias_codes,
        "triggers": triggers,
        "alias_utility_losses": np.asarray(losses, dtype=np.float64).tolist(),
        "realized_trigger_fractions": np.mean(triggers, axis=0).tolist(),
        "coordinate_wise_nonimproving": bool(
            np.all(alias_feedback <= np.repeat(parent_feedback, common.N_ALIASES, axis=1))
        ),
        "mean_pairwise_alias_response_hamming": float(np.mean(pairwise)),
        "minimum_pairwise_alias_response_hamming": float(np.min(pairwise)),
        "parent_accuracy": float(np.mean(parent_correct)),
        "alias_accuracies": np.mean(alias_feedback, axis=0).tolist(),
    }


def evaluate(
    task_key: str,
    partition_name: str,
    roster: tuple[int, ...],
    roster_predictions: np.ndarray,
    labels: np.ndarray,
    parent_position: int,
    parent_bank_id: int,
    construction: dict[str, Any],
    tau: float,
    loss_cap: float,
    trigger_cap: float,
    seeds: tuple[int, ...],
    clean_cache: dict[tuple[Any, ...], common.RunSummary],
    pool_cache: dict[tuple[str, tuple[int, ...]], np.ndarray],
    retain_vectors: bool,
) -> dict[str, Any]:
    clean_codes = roster_predictions.astype(np.int64)
    clean_parents = np.arange(common.N_ROSTER_ROOTS, dtype=np.int64)
    alias_codes = np.asarray(construction["alias_codes"], dtype=np.int64)
    refined_codes = np.column_stack([clean_codes, alias_codes]).astype(np.int64)
    refined_parents = np.concatenate(
        [clean_parents, np.full(common.N_ALIASES, parent_position, dtype=np.int64)]
    )
    pool_key = (partition_name, seeds)
    if pool_key not in pool_cache:
        pool_cache[pool_key] = common.sample_pools(
            len(labels), seeds, pool_size=common.POOL_SIZE
        )
    pools = pool_cache[pool_key]
    cache_key = (partition_name, roster, seeds, float(tau))
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
    terminal = refined.terminal - clean.terminal
    cumulative = refined.cumulative - clean.cumulative
    fixed_terminal = refined_fixed.terminal - clean.terminal
    fixed_cumulative = refined_fixed.cumulative - clean.cumulative
    active_minus_fixed = terminal - fixed_terminal
    row: dict[str, Any] = {
        "task_key": task_key,
        "partition": partition_name,
        "roster_bank_root_ids": list(roster),
        "parent_position": int(parent_position),
        "parent_bank_root_id": int(parent_bank_id),
        "tau": float(tau),
        "max_utility_loss": float(loss_cap),
        "max_trigger_fraction": float(trigger_cap),
        "thresholds": [float(value) for value in construction["thresholds"]],
        "threshold_audits": construction.get("threshold_audits", []),
        "alias_utility_losses": construction["alias_utility_losses"],
        "realized_trigger_fractions": construction["realized_trigger_fractions"],
        "coordinate_wise_nonimproving": construction[
            "coordinate_wise_nonimproving"
        ],
        "mean_pairwise_alias_response_hamming": construction[
            "mean_pairwise_alias_response_hamming"
        ],
        "minimum_pairwise_alias_response_hamming": construction[
            "minimum_pairwise_alias_response_hamming"
        ],
        "seeds": len(seeds),
        "mean_active_minus_fixed_terminal_delta": float(np.mean(active_minus_fixed)),
        "mean_terminal_delta": float(np.mean(terminal)),
        "mean_cumulative_delta": float(np.mean(cumulative)),
        "mean_fixed_terminal_delta": float(np.mean(fixed_terminal)),
        "mean_fixed_cumulative_delta": float(np.mean(fixed_cumulative)),
        "max_abs_fixed_terminal_delta": float(np.max(np.abs(fixed_terminal))),
        "max_abs_fixed_cumulative_delta": float(np.max(np.abs(fixed_cumulative))),
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
        "final_root_change_rate": float(
            np.mean(clean.roots[:, -1] != refined.roots[:, -1])
        ),
        "positive_terminal_fraction": float(np.mean(terminal > 0)),
        "negative_terminal_fraction": float(np.mean(terminal < 0)),
        "clean_query_sha256": hashlib.sha256(clean.queries.tobytes()).hexdigest(),
        "refined_query_sha256": hashlib.sha256(refined.queries.tobytes()).hexdigest(),
    }
    if retain_vectors:
        row.update(
            {
                "terminal_delta_values": terminal.tolist(),
                "cumulative_delta_values": cumulative.tolist(),
                "fixed_terminal_delta_values": fixed_terminal.tolist(),
                "fixed_cumulative_delta_values": fixed_cumulative.tolist(),
                "active_minus_fixed_terminal_values": active_minus_fixed.tolist(),
            }
        )
    return row


def rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    template_order = {"dense": 0, "mixed": 1, "wide": 2}
    return (
        float(row["mean_active_minus_fixed_terminal_delta"]),
        float(row["mean_terminal_delta"]),
        float(row["mean_cumulative_delta"]),
        float(row["path_change_rate"]),
        -float(row["max_utility_loss"]),
        -float(row["max_trigger_fraction"]),
        -template_order[str(row["primary_template"])],
        -int(row["parent_bank_root_id"]),
        -float(row["tau"]),
    )


def best_roster_member(
    roster: tuple[int, ...], bank_accuracies: np.ndarray
) -> int:
    return min(
        roster,
        key=lambda root: (-float(bank_accuracies[int(root)]), int(root)),
    )


def task_model_path(task_key: str, kind: str, identifier: str) -> Path:
    return MODEL_DIR / task_key / f"step94_{task_key}_{kind}_{identifier}.safetensors"


def run_task(
    task: common.TaskSpec,
    manifest: dict[str, Any],
    tokenizer: Any,
    encoder: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    manifest_row = task_manifest(manifest, task.key)
    texts, labels = load_development(task, manifest_row)
    partitions = np.asarray(
        [common.development_partition(task.key, index) for index in range(len(texts))]
    )
    partition_indices = {
        name: np.flatnonzero(partitions == name).astype(np.int64)
        for name in (
            "root_train",
            "gate_train",
            "stagea_search",
            "stagea_verification",
        )
    }
    observed_counts = {key: len(value) for key, value in partition_indices.items()}
    if observed_counts != manifest_row["partition_counts"]:
        raise AssertionError(
            {"task": task.key, "partition_counts": observed_counts, "manifest": manifest_row["partition_counts"]}
        )
    print(f"[{task.key}] encoding {len(texts)} development inputs on {device}", flush=True)
    hidden = common.encode_texts(texts, tokenizer, encoder, device)
    root_source = partition_indices["root_train"]
    gate_source = partition_indices["gate_train"]
    search_source = partition_indices["stagea_search"]
    verify_source = partition_indices["stagea_verification"]

    root_hidden = hidden[root_source]
    root_labels = labels[root_source]
    root_models: dict[int, common.ClassificationAdapter] = {}
    root_rows: list[dict[str, Any]] = []
    search_logits: list[np.ndarray] = []
    verify_logits: list[np.ndarray] = []
    gate_logits_by_root: dict[int, np.ndarray] = {}
    for bank_root, fraction in enumerate(common.ROOT_FRACTIONS):
        seed = 94400 + 100 * task.order + bank_root
        subset = common.root_subset_indices(
            len(root_source), task.key, bank_root, float(fraction)
        )
        model = common.train_classifier(
            root_hidden,
            root_labels,
            subset,
            task.n_classes,
            seed,
        )
        path = task_model_path(task.key, "root", f"{bank_root:02d}")
        common.save_module(
            model,
            path,
            {
                "kind": "step94_clean_root_classifier",
                "task": task.key,
                "dataset": task.repo_id,
                "revision": task.revision,
                "base_model": common.BASE_MODEL,
                "base_revision": common.BASE_REVISION,
                "bank_root": bank_root,
                "seed": seed,
                "train_fraction": fraction,
                "train_examples": len(subset),
                "runtime_inputs": "raw_text_frozen_encoder",
            },
        )
        audit = common.tensor_audit(path)
        if audit["parameter_count"] < 98_000 or audit["variance"] <= 0:
            raise AssertionError({"task": task.key, "root": bank_root, "audit": audit})
        root_models[bank_root] = model
        search_value = common.infer_classifier(model, hidden[search_source])
        verify_value = common.infer_classifier(model, hidden[verify_source])
        gate_value = common.infer_classifier(model, hidden[gate_source])
        search_logits.append(search_value)
        verify_logits.append(verify_value)
        gate_logits_by_root[bank_root] = gate_value
        root_rows.append(
            {
                "bank_root": bank_root,
                "fraction": fraction,
                "seed": seed,
                "train_examples": len(subset),
                "training_position_sha256": common.canonical_json_sha256(subset.tolist()),
                "checkpoint": str(path.relative_to(ROOT)).replace("\\", "/"),
                "checkpoint_audit": audit,
                "search_accuracy": float(
                    np.mean(np.argmax(search_value, axis=1) == labels[search_source])
                ),
                "verification_accuracy": float(
                    np.mean(np.argmax(verify_value, axis=1) == labels[verify_source])
                ),
            }
        )
        print(f"[{task.key}] trained root {bank_root + 1}/{common.N_BANK_ROOTS}", flush=True)

    search_bank_logits = np.stack(search_logits, axis=1)
    verify_bank_logits = np.stack(verify_logits, axis=1)
    search_bank_predictions = np.argmax(search_bank_logits, axis=2).astype(np.int64)
    verify_bank_predictions = np.argmax(verify_bank_logits, axis=2).astype(np.int64)
    search_accuracies = np.mean(
        search_bank_predictions == labels[search_source, None], axis=0
    )
    verify_accuracies = np.mean(
        verify_bank_predictions == labels[verify_source, None], axis=0
    )

    roster_map: dict[tuple[int, ...], dict[str, Any]] = {}
    roster_audit: list[dict[str, Any]] = []
    for template_name in common.GAP_TEMPLATES:
        roster = common.select_roster(search_accuracies, template_name)
        roster_audit.append({"template": template_name, "roster": list(roster)})
        if roster not in roster_map:
            roster_map[roster] = {
                "primary_template": template_name,
                "templates": [template_name],
            }
        else:
            roster_map[roster]["templates"].append(template_name)

    roster_parents: dict[tuple[int, ...], tuple[int, ...]] = {}
    for roster in roster_map:
        ordered = sorted(
            roster,
            key=lambda root: (-float(search_accuracies[int(root)]), int(root)),
        )
        roster_parents[roster] = tuple(int(value) for value in ordered[:3])
    union_parents = sorted(
        set(parent for parents in roster_parents.values() for parent in parents)
    )

    gate_models: dict[tuple[int, int], common.ErrorGate] = {}
    gate_rows: list[dict[str, Any]] = []
    search_gate_scores: dict[int, np.ndarray] = {}
    verify_gate_scores: dict[int, np.ndarray] = {}
    for parent in union_parents:
        parent_gate_logits = gate_logits_by_root[parent]
        search_parent_logits = search_bank_logits[:, parent, :]
        verify_parent_logits = verify_bank_logits[:, parent, :]
        parent_search_scores: list[np.ndarray] = []
        parent_verify_scores: list[np.ndarray] = []
        for alias in range(common.N_ALIASES):
            seed = 94600 + 100 * task.order + 10 * parent + alias
            gate = common.train_error_gate(
                hidden[gate_source],
                parent_gate_logits,
                labels[gate_source],
                task.n_classes,
                seed,
            )
            path = task_model_path(task.key, "parent_gate", f"{parent:02d}_{alias}")
            common.save_module(
                gate,
                path,
                {
                    "kind": "step94_learned_parent_error_gate",
                    "task": task.key,
                    "parent_bank_root": parent,
                    "alias": alias,
                    "seed": seed,
                    "runtime_inputs": "raw_text_frozen_encoder_parent_logits",
                    "forbidden_lookup": True,
                },
            )
            audit = common.tensor_audit(path)
            if audit["variance"] <= 0 or audit["nonzero_fraction"] < 0.95:
                raise AssertionError(
                    {"task": task.key, "parent": parent, "alias": alias, "audit": audit}
                )
            gate_models[(parent, alias)] = gate
            parent_search_scores.append(
                common.infer_gate_scores(
                    gate, hidden[search_source], search_parent_logits
                )
            )
            parent_verify_scores.append(
                common.infer_gate_scores(
                    gate, hidden[verify_source], verify_parent_logits
                )
            )
            gate_rows.append(
                {
                    "parent_bank_root": parent,
                    "alias": alias,
                    "seed": seed,
                    "checkpoint": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "checkpoint_audit": audit,
                    "gate_training_parent_error_prevalence": float(
                        np.mean(
                            np.argmax(parent_gate_logits, axis=1)
                            != labels[gate_source]
                        )
                    ),
                }
            )
        search_gate_scores[parent] = np.stack(parent_search_scores, axis=1)
        verify_gate_scores[parent] = np.stack(parent_verify_scores, axis=1)
        print(f"[{task.key}] trained gates for parent {parent}", flush=True)

    task_output_path = OUTPUT_DIR / f"{task.key}_development_outputs.npz"
    task_output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "search_source_indices": search_source,
        "verification_source_indices": verify_source,
        "search_labels": labels[search_source].astype(np.int16),
        "verification_labels": labels[verify_source].astype(np.int16),
        "search_bank_logits": search_bank_logits,
        "verification_bank_logits": verify_bank_logits,
    }
    for parent in union_parents:
        arrays[f"parent_{parent:02d}_search_gate_scores"] = search_gate_scores[parent]
        arrays[f"parent_{parent:02d}_verification_gate_scores"] = verify_gate_scores[parent]
    np.savez_compressed(task_output_path, **arrays)

    search_rows: list[dict[str, Any]] = []
    search_clean_cache: dict[tuple[Any, ...], common.RunSummary] = {}
    search_pool_cache: dict[tuple[str, tuple[int, ...]], np.ndarray] = {}
    construction_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    grid: list[tuple[Any, ...]] = []
    for roster, parent, tau, loss_cap, trigger_cap in itertools.product(
        roster_map.keys(), union_parents, TAUS, LOSS_CAPS, TRIGGER_CAPS
    ):
        if parent not in roster_parents[roster]:
            continue
        grid.append((roster, parent, tau, loss_cap, trigger_cap))
    for index, (roster, parent, tau, loss_cap, trigger_cap) in enumerate(grid, start=1):
        parent_position = roster.index(parent)
        cache_key = (roster, parent, loss_cap, trigger_cap)
        if cache_key not in construction_cache:
            construction_cache[cache_key] = common.prepare_construction(
                search_bank_predictions[:, roster],
                labels[search_source],
                parent_position,
                search_gate_scores[parent],
                loss_cap,
                trigger_cap,
                task.n_classes,
            )
        construction = construction_cache[cache_key]
        row = evaluate(
            task.key,
            "stagea_search",
            roster,
            search_bank_predictions[:, roster],
            labels[search_source],
            parent_position,
            parent,
            construction,
            tau,
            loss_cap,
            trigger_cap,
            SEARCH_SEEDS,
            search_clean_cache,
            search_pool_cache,
            retain_vectors=False,
        )
        row.update(roster_map[roster])
        search_rows.append(row)
        if index % 72 == 0 or index == len(grid):
            print(f"[{task.key}] search {index}/{len(grid)}", flush=True)

    eligible_search = [
        row
        for row in search_rows
        if row["coordinate_wise_nonimproving"]
        and all(float(value) <= row["max_utility_loss"] + 1e-15 for value in row["alias_utility_losses"])
        and all(float(value) <= row["max_trigger_fraction"] + 1e-15 for value in row["realized_trigger_fractions"])
    ]
    if len(eligible_search) < TOP_TO_VERIFY:
        raise AssertionError({"task": task.key, "eligible_search": len(eligible_search)})
    ranked_search = sorted(eligible_search, key=rank_key, reverse=True)

    verification_rows: list[dict[str, Any]] = []
    verify_clean_cache: dict[tuple[Any, ...], common.RunSummary] = {}
    verify_pool_cache: dict[tuple[str, tuple[int, ...]], np.ndarray] = {}
    for pilot_rank, search_row in enumerate(ranked_search[:TOP_TO_VERIFY], start=1):
        roster = tuple(int(value) for value in search_row["roster_bank_root_ids"])
        parent = int(search_row["parent_bank_root_id"])
        parent_position = int(search_row["parent_position"])
        construction = replay_construction(
            verify_bank_predictions[:, roster],
            labels[verify_source],
            parent_position,
            verify_gate_scores[parent],
            [float(value) for value in search_row["thresholds"]],
            task.n_classes,
        )
        row = evaluate(
            task.key,
            "stagea_verification",
            roster,
            verify_bank_predictions[:, roster],
            labels[verify_source],
            parent_position,
            parent,
            construction,
            float(search_row["tau"]),
            float(search_row["max_utility_loss"]),
            float(search_row["max_trigger_fraction"]),
            VERIFY_SEEDS,
            verify_clean_cache,
            verify_pool_cache,
            retain_vectors=True,
        )
        row.update(roster_map[roster])
        row["search_rank"] = pilot_rank
        row["search_mean_active_minus_fixed_terminal_delta"] = search_row[
            "mean_active_minus_fixed_terminal_delta"
        ]
        row["search_mean_terminal_delta"] = search_row["mean_terminal_delta"]
        row["search_mean_cumulative_delta"] = search_row["mean_cumulative_delta"]
        verification_rows.append(row)
        print(f"[{task.key}] verification {pilot_rank}/{TOP_TO_VERIFY}", flush=True)

    selected = max(verification_rows, key=rank_key)
    selected_roster = tuple(int(value) for value in selected["roster_bank_root_ids"])
    selected_parent = int(selected["parent_bank_root_id"])
    search_best = best_roster_member(selected_roster, search_accuracies)
    verification_best = best_roster_member(selected_roster, verify_accuracies)
    descriptive_parent_rank_checks = {
        "selected_parent_best_on_search": bool(selected_parent == search_best),
        "selected_parent_best_on_verification": bool(selected_parent == verification_best),
    }
    eligibility_checks = {
        "search_active_minus_fixed_positive": bool(
            selected["search_mean_active_minus_fixed_terminal_delta"] > 0
        ),
        "verification_active_minus_fixed_positive": bool(
            selected["mean_active_minus_fixed_terminal_delta"] > 0
        ),
        "verification_path_change_at_least_half": bool(
            selected["path_change_rate"] >= 0.50
        ),
        "verification_alias_losses_at_most_one_point": bool(
            max(float(value) for value in selected["alias_utility_losses"]) <= 0.0100
        ),
        "fixed_terminal_exact_zero": bool(selected["max_abs_fixed_terminal_delta"] == 0),
        "fixed_cumulative_exact_zero": bool(selected["max_abs_fixed_cumulative_delta"] == 0),
        "fixed_root_history_exact": bool(selected["fixed_root_history_exact"]),
    }
    task_eligible = bool(all(eligibility_checks.values()))

    integrated_rows: list[dict[str, Any]] = []
    for alias in range(common.N_ALIASES):
        integrated = common.make_integrated_variant(
            root_models[selected_parent], gate_models[(selected_parent, alias)], task.n_classes
        )
        path = task_model_path(task.key, "integrated_variant", f"{selected_parent:02d}_{alias}")
        common.save_module(
            integrated,
            path,
            {
                "kind": "step94_integrated_learned_raw_input_variant",
                "task": task.key,
                "parent_bank_root": selected_parent,
                "alias": alias,
                "threshold": selected["thresholds"][alias],
                "abstention_token": f"ABSTAIN_{alias}",
                "runtime_inputs": "raw_text_only",
                "forbidden_lookup": True,
            },
        )
        integrated_rows.append(
            {
                "alias": alias,
                "checkpoint": str(path.relative_to(ROOT)).replace("\\", "/"),
                "checkpoint_audit": common.tensor_audit(path),
            }
        )

    return {
        "task_key": task.key,
        "task_order": task.order,
        "dataset": {
            "id": task.repo_id,
            "revision": task.revision,
            "n_classes": task.n_classes,
        },
        "development_output_file": str(task_output_path.relative_to(ROOT)).replace("\\", "/"),
        "development_output_sha256": common.sha256_path(task_output_path),
        "partition_counts": {key: len(value) for key, value in partition_indices.items()},
        "root_bank": root_rows,
        "roster_audit": roster_audit,
        "unique_rosters": [
            {
                "roster": list(roster),
                **metadata,
                "eligible_parents": list(roster_parents[roster]),
            }
            for roster, metadata in roster_map.items()
        ],
        "gate_bank": gate_rows,
        "search_grid_count": len(grid),
        "search_complete_ledger": search_rows,
        "verification_top_24": verification_rows,
        "selected": selected,
        "selected_integrated_variants": integrated_rows,
        "search_best_roster_member": search_best,
        "verification_best_roster_member": verification_best,
        "calibration_task_eligibility_checks": eligibility_checks,
        "descriptive_parent_rank_checks": descriptive_parent_rank_checks,
        "calibration_task_eligible": task_eligible,
    }


def main() -> None:
    for authority in (
        common.PROTOCOL,
        common.AMENDMENT_A,
        ROOT / f"STEP94_PREREGISTRATION_AMENDMENT_B_{common.DATE}.md",
        ROOT / f"STEP94_PREREGISTRATION_AMENDMENT_C_{common.DATE}.md",
        ROOT / f"STEP94_PREREGISTRATION_AMENDMENT_D_{common.DATE}.md",
        MANIFEST_PATH,
    ):
        if not authority.exists():
            raise FileNotFoundError(authority)
    for output in (MODEL_DIR, OUTPUT_DIR, LEDGER_PATH, CONFIG_PATH):
        if output.exists():
            raise RuntimeError(f"refusing to overwrite Step 94 Stage-A output: {output}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    stage0_authorities = {
        common.PROTOCOL.name: common.sha256_path(common.PROTOCOL),
        common.AMENDMENT_A.name: common.sha256_path(common.AMENDMENT_A),
        f"STEP94_PREREGISTRATION_AMENDMENT_B_{common.DATE}.md": common.sha256_path(
            ROOT / f"STEP94_PREREGISTRATION_AMENDMENT_B_{common.DATE}.md"
        ),
        f"STEP94_PREREGISTRATION_AMENDMENT_C_{common.DATE}.md": common.sha256_path(
            ROOT / f"STEP94_PREREGISTRATION_AMENDMENT_C_{common.DATE}.md"
        ),
    }
    if manifest["authority_sha256"] != stage0_authorities:
        raise AssertionError("Stage-0 authority binding mismatch")
    expected_authorities = {
        **stage0_authorities,
        f"STEP94_PREREGISTRATION_AMENDMENT_D_{common.DATE}.md": common.sha256_path(
            ROOT / f"STEP94_PREREGISTRATION_AMENDMENT_D_{common.DATE}.md"
        ),
    }

    tokenizer, encoder, device = common.load_encoder()
    task_results: list[dict[str, Any]] = []
    for task in common.TASKS:
        task_results.append(run_task(task, manifest, tokenizer, encoder, device))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    eligible_tasks = [row for row in task_results if row["calibration_task_eligible"]]
    if eligible_tasks:
        selected_task = max(
            eligible_tasks,
            key=lambda row: (
                float(row["selected"]["mean_active_minus_fixed_terminal_delta"]),
                -int(row["task_order"]),
            ),
        )
        decision = "GO_TO_ONE_TIME_CONFIRMATORY_LOCK"
    else:
        selected_task = None
        decision = "NO_GO_NO_CALIBRATION_ELIGIBLE_TASK"

    similarity_assertion = {
        "definition": "s(a,b)=1[a=b]",
        "function": "step93_common.exact_match_similarity",
        "function_source_sha256": source_sha256(common.exact_match_similarity),
        "feedback_source_sha256": source_sha256(common.exact_match_feedback),
        "group_builder_source_sha256": source_sha256(common.build_source_faithful_groups),
        "run_active_source_sha256": source_sha256(common.run_active),
        "run_fixed_source_sha256": source_sha256(common.run_fixed),
        "run_active_accepts_external_feedback": False,
    }
    ledger = {
        "ledger_id": "STEP94_FRESH_TASK_STAGEA_COMPLETE_V1",
        "date": common.DATE,
        "authority_sha256": expected_authorities,
        "stage0_manifest_sha256": common.sha256_path(MANIFEST_PATH),
        "single_similarity_assertion": similarity_assertion,
        "search_protocol": {
            "search_seeds": [SEARCH_SEEDS[0], SEARCH_SEEDS[-1]],
            "verification_seeds": [VERIFY_SEEDS[0], VERIFY_SEEDS[-1]],
            "temperatures": list(TAUS),
            "loss_caps": list(LOSS_CAPS),
            "trigger_caps": list(TRIGGER_CAPS),
            "top_to_verify": TOP_TO_VERIFY,
            "pool_size": common.POOL_SIZE,
            "budget": common.BUDGET,
        },
        "tasks": task_results,
        "stagea_decision": decision,
        "selected_task_key": selected_task["task_key"] if selected_task else None,
    }
    common.json_dump(LEDGER_PATH, ledger)

    all_model_files = sorted(MODEL_DIR.rglob("*.safetensors"))
    config = {
        "config_id": "STEP94_FRESH_TASK_FROZEN_CONFIG_V1",
        "date": common.DATE,
        "decision": decision,
        "authority_sha256": expected_authorities,
        "stage0_manifest_sha256": common.sha256_path(MANIFEST_PATH),
        "stagea_ledger_sha256": common.sha256_path(LEDGER_PATH),
        "single_similarity_assertion": similarity_assertion,
        "all_model_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): common.sha256_path(path)
            for path in all_model_files
        },
        "selected_task": (
            {
                "task_key": selected_task["task_key"],
                "task_order": selected_task["task_order"],
                "dataset": selected_task["dataset"],
                "selected_condition": selected_task["selected"],
                "integrated_variants": selected_task["selected_integrated_variants"],
                "calibration_eligibility_checks": selected_task[
                    "calibration_task_eligibility_checks"
                ],
                "test_input_file": task_manifest(
                    manifest, selected_task["task_key"]
                )["test_input_file"],
                "test_input_sha256": task_manifest(
                    manifest, selected_task["task_key"]
                )["test_input_sha256"],
                "sealed_outcome_file": task_manifest(
                    manifest, selected_task["task_key"]
                )["sealed_outcome_file"],
                "sealed_outcome_sha256": task_manifest(
                    manifest, selected_task["task_key"]
                )["sealed_outcome_sha256"],
            }
            if selected_task
            else None
        ),
        "construction_access": {
            "public_development_labels": True,
            "official_test_inputs": False,
            "official_test_labels": False,
            "item_id_or_prompt_lookup": False,
            "peer_outputs": False,
            "selector_state": False,
        },
    }
    common.json_dump(CONFIG_PATH, config)
    print(
        json.dumps(
            {
                "config": CONFIG_PATH.name,
                "config_sha256": common.sha256_path(CONFIG_PATH),
                "ledger": LEDGER_PATH.name,
                "ledger_sha256": common.sha256_path(LEDGER_PATH),
                "stagea_decision": decision,
                "task_summaries": [
                    {
                        "task_key": row["task_key"],
                        "eligible": row["calibration_task_eligible"],
                        "selected_active_minus_fixed_terminal": row["selected"][
                            "mean_active_minus_fixed_terminal_delta"
                        ],
                        "selected_terminal": row["selected"]["mean_terminal_delta"],
                        "selected_path_change": row["selected"]["path_change_rate"],
                        "selected_parent": row["selected"]["parent_bank_root_id"],
                        "search_best": row["search_best_roster_member"],
                        "verification_best": row["verification_best_roster_member"],
                    }
                    for row in task_results
                ],
                "selected_task_key": selected_task["task_key"] if selected_task else None,
                "model_files": len(all_model_files),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
