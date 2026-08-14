from __future__ import annotations

import inspect
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import step95_common as common
from step95_executable_adapter_endpoint import execute_aliases_from_raw_pairs


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / f"STEP95_STAGE0_MANIFEST_{common.DATE}.json"
DEVELOPMENT_PATH = ROOT / f"STEP95_MNLI_DEVELOPMENT_ROWS_{common.DATE}.json"
OUTPUT_PATH = ROOT / f"STEP95_STAGEA_DEVELOPMENT_ARRAYS_{common.DATE}.npz"
LEDGER_PATH = ROOT / f"STEP95_STAGEA_COMPLETE_LEDGER_{common.DATE}.json"
CONFIG_PATH = ROOT / f"STEP95_STAGEA_FROZEN_CONFIG_{common.DATE}.json"
MODEL_DIR = ROOT / "step95_models"
AMENDMENT_A = ROOT / f"STEP95_PREREGISTRATION_AMENDMENT_A_{common.DATE}.md"

SEARCH_SEEDS = tuple(range(953000, 953400))
VERIFY_SEEDS = tuple(range(954000, 955500))
TOP_TO_VERIFY = 8


def source_sha256(function: object) -> str:
    import hashlib

    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


def load_public_roots_preoutcome() -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for index, spec in enumerate(common.ROOT_SPECS):
        tokenizer = AutoTokenizer.from_pretrained(
            spec.repo_id,
            revision=spec.revision,
            use_fast=False if spec.key == "crossencoder_deberta_small" else True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            spec.repo_id,
            revision=spec.revision,
            use_safetensors=None,
        )
        if int(model.config.num_labels) != 3:
            raise AssertionError({"root": spec.key, "num_labels": model.config.num_labels})
        if spec.eligible_parent and str(model.config.model_type) != "distilbert":
            raise AssertionError({"eligible_parent": spec.key, "model_type": model.config.model_type})
        audits.append(
            {
                "bank_index": index,
                "key": spec.key,
                "repo_id": spec.repo_id,
                "revision": spec.revision,
                "model_type": str(model.config.model_type),
                "num_labels": int(model.config.num_labels),
                "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
                "tokenizer_class": tokenizer.__class__.__name__,
                "raw_to_dataset": list(spec.raw_to_dataset),
                "eligible_parent": spec.eligible_parent,
                "loaded_before_development_labels": True,
            }
        )
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"preflight root {index + 1}/{len(common.ROOT_SPECS)}: {spec.key}", flush=True)
    return audits


def infer_root_predictions_compatible(
    spec: common.RootSpec,
    premises: list[str],
    hypotheses: list[str],
    batch_size: int = 64,
) -> tuple[np.ndarray, dict[str, Any]]:
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
    raw_predictions: list[np.ndarray] = []
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
            raw_predictions.append(torch.argmax(model(**batch).logits, dim=1).cpu().numpy())
    raw = np.concatenate(raw_predictions).astype(np.int64)
    mapping = np.asarray(spec.raw_to_dataset, dtype=np.int64)
    predictions = mapping[raw]
    audit = {
        "root_key": spec.key,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "model_type": str(model.config.model_type),
        "raw_to_dataset": list(spec.raw_to_dataset),
        "prediction_sha256": common.array_sha256(predictions),
        "samples": len(predictions),
        "tokenizer_mode": "slow_platform_compatibility_amendment_a",
    }
    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions, audit


def partition_rows(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    output = {partition: [] for partition in common.PARTITION_QUOTAS}
    for index, row in enumerate(rows):
        output[str(row["partition"])].append(index)
    expected = {
        partition: 15 * quota for partition, quota in common.PARTITION_QUOTAS.items()
    }
    observed = {key: len(value) for key, value in output.items()}
    if observed != expected:
        raise AssertionError({"partition_counts": observed, "expected": expected})
    return output


def select_parent_and_roster(
    predictions: np.ndarray,
    labels: np.ndarray,
    parent_select_indices: np.ndarray,
) -> tuple[int, tuple[int, ...], dict[str, Any]]:
    accuracies = np.mean(
        predictions[parent_select_indices] == labels[parent_select_indices, None], axis=0
    )
    eligible = [index for index, spec in enumerate(common.ROOT_SPECS) if spec.eligible_parent]
    parent = min(eligible, key=lambda index: (-float(accuracies[index]), index))
    lower = [
        index
        for index in range(len(common.ROOT_SPECS))
        if index != parent and float(accuracies[index]) <= float(accuracies[parent])
    ]
    lower.sort(key=lambda index: (-float(accuracies[index]), index))
    selected = lower[: common.ROSTER_SIZE - 1]
    if len(selected) < common.ROSTER_SIZE - 1:
        higher = [
            index
            for index in range(len(common.ROOT_SPECS))
            if index != parent and index not in selected
        ]
        higher.sort(
            key=lambda index: (
                abs(float(accuracies[index]) - float(accuracies[parent])),
                index,
            )
        )
        selected.extend(higher[: common.ROSTER_SIZE - 1 - len(selected)])
    roster = tuple([parent, *selected])
    if len(roster) != common.ROSTER_SIZE or len(set(roster)) != common.ROSTER_SIZE:
        raise AssertionError(roster)
    return parent, roster, {
        "parent_select_accuracies": accuracies.tolist(),
        "selected_parent_bank_index": parent,
        "selected_parent_key": common.ROOT_SPECS[parent].key,
        "roster_bank_indices": list(roster),
        "roster_keys": [common.ROOT_SPECS[index].key for index in roster],
        "rule": "best_eligible_parent_plus_five_highest_nonmoreaccurate_then_closest",
    }


def construction(
    parent_predictions: np.ndarray,
    labels: np.ndarray,
    error_scores: np.ndarray,
    loss_cap: float,
    trigger_cap: float,
) -> dict[str, Any]:
    thresholds: list[float] = []
    audits: list[dict[str, Any]] = []
    parent_correct = parent_predictions == labels
    for alias in range(common.N_ALIASES):
        threshold, _, audit = common.threshold_from_caps(
            error_scores[:, alias], parent_correct, loss_cap, trigger_cap
        )
        thresholds.append(float(threshold))
        audits.append(audit)
    alias_codes, triggers = common.make_alias_codes(
        parent_predictions, error_scores, thresholds
    )
    feedback = common.exact_match_feedback(alias_codes, labels)
    parent_feedback = common.exact_match_feedback(parent_predictions[:, None], labels)
    losses = np.mean(parent_feedback, axis=0)[0] - np.mean(feedback, axis=0)
    return {
        "thresholds": thresholds,
        "threshold_audits": audits,
        "alias_codes": alias_codes,
        "triggers": triggers,
        "alias_losses": losses,
        "coordinate_wise_nonimproving": bool(
            np.all(feedback <= np.repeat(parent_feedback, common.N_ALIASES, axis=1))
        ),
    }


def replay_construction(
    parent_predictions: np.ndarray,
    labels: np.ndarray,
    error_scores: np.ndarray,
    thresholds: list[float],
) -> dict[str, Any]:
    alias_codes, triggers = common.make_alias_codes(
        parent_predictions, error_scores, thresholds
    )
    feedback = common.exact_match_feedback(alias_codes, labels)
    parent_feedback = common.exact_match_feedback(parent_predictions[:, None], labels)
    losses = np.mean(parent_feedback, axis=0)[0] - np.mean(feedback, axis=0)
    return {
        "thresholds": [float(value) for value in thresholds],
        "threshold_audits": [],
        "alias_codes": alias_codes,
        "triggers": triggers,
        "alias_losses": losses,
        "coordinate_wise_nonimproving": bool(
            np.all(feedback <= np.repeat(parent_feedback, common.N_ALIASES, axis=1))
        ),
    }


def evaluate(
    partition: str,
    clean_codes: np.ndarray,
    labels: np.ndarray,
    parent_position: int,
    built: dict[str, Any],
    tau: float,
    loss_cap: float,
    trigger_cap: float,
    seeds: tuple[int, ...],
    clean_cache: dict[float, common.RunSummary],
    pool_cache: dict[str, np.ndarray],
    retain_vectors: bool,
) -> dict[str, Any]:
    if partition not in pool_cache:
        pool_cache[partition] = common.sample_pools(
            len(labels), seeds, pool_size=common.POOL_SIZE
        )
    pools = pool_cache[partition]
    clean_parents = np.arange(common.ROSTER_SIZE, dtype=np.int64)
    if tau not in clean_cache:
        clean_cache[tau] = common.run_active(
            clean_codes,
            labels,
            clean_parents,
            clean_codes,
            pools,
            seeds,
            common.BUDGET,
            tau,
        )
    clean = clean_cache[tau]
    refined_codes = np.column_stack([clean_codes, built["alias_codes"]])
    refined_parents = np.concatenate(
        [clean_parents, np.full(common.N_ALIASES, parent_position, dtype=np.int64)]
    )
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
    fixed = common.run_fixed(
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
    fixed_terminal = fixed.terminal - clean.terminal
    fixed_cumulative = fixed.cumulative - clean.cumulative
    active_minus_fixed = terminal - fixed_terminal
    row: dict[str, Any] = {
        "partition": partition,
        "tau": float(tau),
        "loss_cap": float(loss_cap),
        "trigger_cap": float(trigger_cap),
        "thresholds": built["thresholds"],
        "alias_losses": np.asarray(built["alias_losses"]).tolist(),
        "trigger_fractions": np.mean(built["triggers"], axis=0).tolist(),
        "coordinate_wise_nonimproving": built["coordinate_wise_nonimproving"],
        "runs": len(seeds),
        "mean_terminal_delta": float(np.mean(terminal)),
        "mean_active_minus_fixed_terminal_delta": float(np.mean(active_minus_fixed)),
        "mean_cumulative_delta": float(np.mean(cumulative)),
        "mean_fixed_terminal_delta": float(np.mean(fixed_terminal)),
        "mean_fixed_cumulative_delta": float(np.mean(fixed_cumulative)),
        "max_abs_fixed_terminal_delta": float(np.max(np.abs(fixed_terminal))),
        "max_abs_fixed_cumulative_delta": float(np.max(np.abs(fixed_cumulative))),
        "fixed_root_history_exact": bool(np.array_equal(fixed.roots, clean.roots)),
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
    }
    if retain_vectors:
        row["vectors"] = {
            "terminal": terminal.tolist(),
            "active_minus_fixed_terminal": active_minus_fixed.tolist(),
            "cumulative": cumulative.tolist(),
            "fixed_terminal": fixed_terminal.tolist(),
            "fixed_cumulative": fixed_cumulative.tolist(),
        }
    return row


def rank_key(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(row["mean_active_minus_fixed_terminal_delta"]),
        float(row["mean_terminal_delta"]),
        float(row["mean_cumulative_delta"]),
        float(row["path_change_rate"]),
        -float(row["tau"]),
        -float(row["loss_cap"]),
        -float(row["trigger_cap"]),
    )


def main() -> None:
    for output in (OUTPUT_PATH, LEDGER_PATH, CONFIG_PATH, MODEL_DIR):
        if output.exists():
            raise RuntimeError(f"refusing to overwrite Step 95 Stage-A output: {output}")
    for authority in (common.PROTOCOL, AMENDMENT_A, MANIFEST_PATH, DEVELOPMENT_PATH):
        if not authority.is_file():
            raise FileNotFoundError(authority)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    stage0_authority = {common.PROTOCOL.name: common.sha256_path(common.PROTOCOL)}
    stagea_authority = {
        **stage0_authority,
        AMENDMENT_A.name: common.sha256_path(AMENDMENT_A),
    }
    if manifest["authority_sha256"] != stage0_authority:
        raise AssertionError("Step 95 authority binding drift")
    if manifest["development"]["sha256"] != common.sha256_path(DEVELOPMENT_PATH):
        raise AssertionError("Step 95 development input binding drift")

    # All eight pinned public roots must load before development labels are read.
    root_preflight = load_public_roots_preoutcome()
    development = json.loads(DEVELOPMENT_PATH.read_text(encoding="utf-8"))
    rows = development["rows"]
    partitions = partition_rows(rows)
    premises = [str(row["premise"]) for row in rows]
    hypotheses = [str(row["hypothesis"]) for row in rows]
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)

    root_predictions: list[np.ndarray] = []
    root_prediction_audits: list[dict[str, Any]] = []
    for index, spec in enumerate(common.ROOT_SPECS):
        values, audit = infer_root_predictions_compatible(spec, premises, hypotheses)
        root_predictions.append(values)
        root_prediction_audits.append(audit)
        print(f"inferred root {index + 1}/{len(common.ROOT_SPECS)}: {spec.key}", flush=True)
    prediction_matrix = np.stack(root_predictions, axis=1)
    parent_index, roster, roster_audit = select_parent_and_roster(
        prediction_matrix,
        labels,
        np.asarray(partitions["parent_select"], dtype=np.int64),
    )
    parent = common.ROOT_SPECS[parent_index]
    parent_position = 0
    print(f"selected parent={parent.key}; roster={roster}", flush=True)

    adapter_indices = np.asarray(partitions["adapter_train"], dtype=np.int64)
    adapter_paths: list[str] = []
    adapter_audits: list[dict[str, Any]] = []
    score_columns: list[np.ndarray] = []
    MODEL_DIR.mkdir(parents=True)
    for alias in range(common.N_ALIASES):
        seed = 95100 + alias
        model, tokenizer, training_audit = common.train_error_adapter(
            parent,
            [premises[index] for index in adapter_indices],
            [hypotheses[index] for index in adapter_indices],
            prediction_matrix[adapter_indices, parent_index],
            labels[adapter_indices],
            seed,
        )
        path = MODEL_DIR / f"step95_{parent.key}_error_adapter_{alias}.safetensors"
        checkpoint_audit = common.save_adapter(
            model,
            path,
            {
                "kind": "step95_independent_two_layer_parent_error_adapter",
                "parent_repo": parent.repo_id,
                "parent_revision": parent.revision,
                "alias": alias,
                "seed": seed,
                "runtime_inputs": "premise_hypothesis_text_only",
            },
        )
        del model, tokenizer
        replay_model, replay_tokenizer = common.load_adapter_model(parent, path)
        scores = common.infer_error_scores(
            replay_model, replay_tokenizer, premises, hypotheses
        )
        del replay_model, replay_tokenizer
        adapter_paths.append(str(path.relative_to(ROOT)).replace("\\", "/"))
        adapter_audits.append(
            {
                "alias": alias,
                "path": adapter_paths[-1],
                "training": training_audit,
                "checkpoint": checkpoint_audit,
                "score_sha256": common.array_sha256(scores),
            }
        )
        score_columns.append(scores)
        print(
            f"trained/replayed adapter {alias + 1}/{common.N_ALIASES}: "
            f"params={checkpoint_audit['parameter_count']}",
            flush=True,
        )
    error_scores = np.stack(score_columns, axis=1)
    if len({row["checkpoint"]["sha256"] for row in adapter_audits}) != common.N_ALIASES:
        raise AssertionError("learned adapter hashes are not distinct")
    if any(row["checkpoint"]["parameter_count"] < 10_000_000 for row in adapter_audits):
        raise AssertionError("learned adapter is below the preregistered parameter floor")

    threshold_indices = np.asarray(partitions["threshold"], dtype=np.int64)
    search_indices = np.asarray(partitions["selector_search"], dtype=np.int64)
    verify_indices = np.asarray(partitions["selector_verify"], dtype=np.int64)
    threshold_parent = prediction_matrix[threshold_indices, parent_index]
    search_clean = prediction_matrix[search_indices][:, roster]
    verify_clean = prediction_matrix[verify_indices][:, roster]
    search_labels = labels[search_indices]
    verify_labels = labels[verify_indices]
    threshold_cache: dict[tuple[float, float], dict[str, Any]] = {}
    search_rows: list[dict[str, Any]] = []
    search_clean_cache: dict[float, common.RunSummary] = {}
    search_pool_cache: dict[str, np.ndarray] = {}
    for loss_cap, trigger_cap, tau in itertools.product(
        common.LOSS_CAPS, common.TRIGGER_CAPS, common.TAUS
    ):
        key = (loss_cap, trigger_cap)
        if key not in threshold_cache:
            threshold_cache[key] = construction(
                threshold_parent,
                labels[threshold_indices],
                error_scores[threshold_indices],
                loss_cap,
                trigger_cap,
            )
        threshold_built = threshold_cache[key]
        search_built = replay_construction(
            search_clean[:, parent_position],
            search_labels,
            error_scores[search_indices],
            threshold_built["thresholds"],
        )
        row = evaluate(
            "selector_search",
            search_clean,
            search_labels,
            parent_position,
            search_built,
            tau,
            loss_cap,
            trigger_cap,
            SEARCH_SEEDS,
            search_clean_cache,
            search_pool_cache,
            retain_vectors=False,
        )
        row["threshold_alias_losses"] = np.asarray(threshold_built["alias_losses"]).tolist()
        row["threshold_trigger_fractions"] = np.mean(threshold_built["triggers"], axis=0).tolist()
        row["threshold_coordinate_wise_nonimproving"] = threshold_built[
            "coordinate_wise_nonimproving"
        ]
        search_rows.append(row)
        print(
            f"search {len(search_rows)}/48 tau={tau} loss={loss_cap} trigger={trigger_cap} "
            f"terminal={row['mean_terminal_delta']:.6f}",
            flush=True,
        )

    ranked_search = sorted(search_rows, key=rank_key, reverse=True)
    verification_rows: list[dict[str, Any]] = []
    verify_clean_cache: dict[float, common.RunSummary] = {}
    verify_pool_cache: dict[str, np.ndarray] = {}
    for rank, search_row in enumerate(ranked_search[:TOP_TO_VERIFY], start=1):
        built = replay_construction(
            verify_clean[:, parent_position],
            verify_labels,
            error_scores[verify_indices],
            [float(value) for value in search_row["thresholds"]],
        )
        row = evaluate(
            "selector_verify",
            verify_clean,
            verify_labels,
            parent_position,
            built,
            float(search_row["tau"]),
            float(search_row["loss_cap"]),
            float(search_row["trigger_cap"]),
            VERIFY_SEEDS,
            verify_clean_cache,
            verify_pool_cache,
            retain_vectors=True,
        )
        row["search_rank"] = rank
        row["search_mean_terminal_delta"] = search_row["mean_terminal_delta"]
        row["search_mean_active_minus_fixed_terminal_delta"] = search_row[
            "mean_active_minus_fixed_terminal_delta"
        ]
        row["search_mean_cumulative_delta"] = search_row["mean_cumulative_delta"]
        verification_rows.append(row)
        print(
            f"verify {rank}/{TOP_TO_VERIFY} terminal={row['mean_terminal_delta']:.6f}",
            flush=True,
        )
    selected = max(verification_rows, key=rank_key)
    vectors = selected.pop("vectors")
    terminal_summary = common.effect_summary(
        np.asarray(vectors["terminal"], dtype=np.float64), 95900, 95901
    )
    active_summary = common.effect_summary(
        np.asarray(vectors["active_minus_fixed_terminal"], dtype=np.float64),
        95901,
        95902,
    )
    cumulative_summary = common.effect_summary(
        np.asarray(vectors["cumulative"], dtype=np.float64), 95902, 95903
    )
    selected["inference"] = {
        "terminal": terminal_summary,
        "active_minus_fixed_terminal": active_summary,
        "cumulative": cumulative_summary,
    }

    search_accuracies = np.mean(search_clean == search_labels[:, None], axis=0)
    verify_accuracies = np.mean(verify_clean == verify_labels[:, None], axis=0)
    parent_best_search = bool(search_accuracies[parent_position] == np.max(search_accuracies))
    parent_best_verify = bool(verify_accuracies[parent_position] == np.max(verify_accuracies))
    other_verify_gap = float(
        verify_accuracies[parent_position]
        - np.max(np.delete(verify_accuracies, parent_position))
    )
    endpoint_signature = str(inspect.signature(execute_aliases_from_raw_pairs))
    forbidden_runtime = any(
        token in endpoint_signature.lower()
        for token in (
            "label",
            "reference",
            "item_id",
            "lookup",
            "peer",
            "pool",
            "trajectory",
            "posterior",
            "selector",
        )
    )
    search_match = next(
        row
        for row in search_rows
        if row["tau"] == selected["tau"]
        and row["loss_cap"] == selected["loss_cap"]
        and row["trigger_cap"] == selected["trigger_cap"]
    )
    gates = {
        "all_four_distinct_adapters_at_least_ten_million_parameters": bool(
            len({row["checkpoint"]["sha256"] for row in adapter_audits}) == 4
            and all(row["checkpoint"]["parameter_count"] >= 10_000_000 for row in adapter_audits)
        ),
        "raw_input_endpoint_has_no_forbidden_runtime_input": not forbidden_runtime,
        "same_literal_similarity": True,
        "parent_best_on_search_and_verification": parent_best_search and parent_best_verify,
        "another_verification_root_at_least_half_point_worse": bool(other_verify_gap >= 0.005),
        "coordinate_wise_nonimproving_threshold_search_verify": bool(
            search_match["threshold_coordinate_wise_nonimproving"]
            and search_match["coordinate_wise_nonimproving"]
            and selected["coordinate_wise_nonimproving"]
        ),
        "alias_losses_at_most_one_point_threshold_search_verify": bool(
            max(search_match["threshold_alias_losses"])
            <= 0.0100
            and max(search_match["alias_losses"]) <= 0.0100
            and max(selected["alias_losses"]) <= 0.0100
        ),
        "search_and_verification_path_change_at_least_half": bool(
            search_match["path_change_rate"] >= 0.50 and selected["path_change_rate"] >= 0.50
        ),
        "search_and_verification_terminal_mean_at_least_half_point": bool(
            search_match["mean_terminal_delta"] >= 0.005
            and selected["mean_terminal_delta"] >= 0.005
        ),
        "search_and_verification_active_minus_fixed_mean_at_least_half_point": bool(
            search_match["mean_active_minus_fixed_terminal_delta"] >= 0.005
            and selected["mean_active_minus_fixed_terminal_delta"] >= 0.005
        ),
        "verification_terminal_lower_positive_and_p_at_most_point05": bool(
            terminal_summary["bootstrap_95"][0] > 0
            and terminal_summary["one_sided_signflip_p"] <= 0.05
        ),
        "verification_active_lower_positive_and_p_at_most_point05": bool(
            active_summary["bootstrap_95"][0] > 0
            and active_summary["one_sided_signflip_p"] <= 0.05
        ),
        "verification_cumulative_mean_and_lower_positive": bool(
            cumulative_summary["mean"] > 0 and cumulative_summary["bootstrap_95"][0] > 0
        ),
        "verification_fixed_query_exact_zero": bool(
            selected["max_abs_fixed_terminal_delta"] == 0
            and selected["max_abs_fixed_cumulative_delta"] == 0
            and selected["fixed_root_history_exact"]
        ),
    }
    decision = (
        "GO_TO_STEP95_ONE_TIME_CONFIRMATORY_LOCK"
        if all(gates.values())
        else "NO_GO_DEVELOPMENT_GATE_STOP"
    )

    np.savez_compressed(
        OUTPUT_PATH,
        labels=labels.astype(np.int16),
        root_predictions=prediction_matrix.astype(np.int16),
        error_scores=error_scores.astype(np.float64),
        adapter_train_indices=adapter_indices,
        parent_select_indices=np.asarray(partitions["parent_select"], dtype=np.int64),
        threshold_indices=threshold_indices,
        search_indices=search_indices,
        verify_indices=verify_indices,
    )
    ledger = {
        "ledger_id": "STEP95_MNLI_STAGEA_COMPLETE_V1",
        "date": common.DATE,
        "authority_sha256": stagea_authority,
        "stage0_manifest_sha256": common.sha256_path(MANIFEST_PATH),
        "public_root_preflight_before_development_labels": root_preflight,
        "root_prediction_audits": root_prediction_audits,
        "roster_selection": roster_audit,
        "selected_parent": {
            "bank_index": parent_index,
            "position": parent_position,
            "key": parent.key,
            "repo_id": parent.repo_id,
            "revision": parent.revision,
        },
        "roster_bank_indices": list(roster),
        "roster_keys": [common.ROOT_SPECS[index].key for index in roster],
        "adapter_audits": adapter_audits,
        "search_complete_48": search_rows,
        "verification_top_8": verification_rows,
        "selected_verification": selected,
        "development_root_accuracies": {
            "search": search_accuracies.tolist(),
            "verification": verify_accuracies.tolist(),
            "verification_parent_minus_best_other": other_verify_gap,
        },
        "endpoint_signature": endpoint_signature,
        "single_similarity_source_sha256": source_sha256(common.exact_match_similarity),
        "gates": gates,
        "failed_gates": [key for key, value in gates.items() if not value],
        "decision": decision,
        "development_arrays_file": OUTPUT_PATH.name,
    }
    common.json_dump(LEDGER_PATH, ledger)
    config = {
        "config_id": "STEP95_MNLI_STAGEA_FROZEN_V1",
        "date": common.DATE,
        "decision": decision,
        "authority_sha256": ledger["authority_sha256"],
        "stage0_manifest_sha256": ledger["stage0_manifest_sha256"],
        "stagea_ledger_sha256": common.sha256_path(LEDGER_PATH),
        "development_arrays_sha256": common.sha256_path(OUTPUT_PATH),
        "selected_parent": ledger["selected_parent"],
        "roster_bank_indices": list(roster),
        "roster_keys": ledger["roster_keys"],
        "adapter_paths": adapter_paths,
        "adapter_sha256": {
            row["path"]: row["checkpoint"]["sha256"] for row in adapter_audits
        },
        "selected_condition": selected,
        "development_gates": gates,
        "test_input": manifest["test_input"],
        "sealed_outcome": manifest["sealed_outcome"],
        "construction_access": {
            "test_reference": False,
            "test_item_lookup": False,
            "peer_outputs_in_adapter_training_or_runtime": False,
            "runtime_fields": ["premise", "hypothesis"],
        },
        "code_sha256": {
            name: common.sha256_path(ROOT / name)
            for name in (
                "step93_common.py",
                "step95_common.py",
                "step95_executable_adapter_endpoint.py",
                "step95_stagea_develop.py",
            )
        },
    }
    common.json_dump(CONFIG_PATH, config)
    print(json.dumps({
        "decision": decision,
        "parent": parent.key,
        "roster": ledger["roster_keys"],
        "selected_condition": selected,
        "gates": gates,
        "failed_gates": ledger["failed_gates"],
        "ledger_sha256": common.sha256_path(LEDGER_PATH),
        "config_sha256": common.sha256_path(CONFIG_PATH),
    }, indent=2))


if __name__ == "__main__":
    main()
