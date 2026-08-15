#!/usr/bin/env python3
"""SRSA Stage 4: single authorized outcome opening and paired inference.

The job verifies the Stage-3 pre-outcome lock before reading the physically
separate label seal. It then executes every frozen root attack on CIFAR-100
(primary) and CIFAR-10 (non-substitutable replication), materializes active,
fixed-query, exact-wrapper, and root-aware controls, applies four-root
familywise inference, and writes a literal GO/NO_GO decision. A scientific
NO_GO exits successfully; exceptions are reserved for technical INVALID runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from srsa_core import (
    canonical_sha256,
    clean_registry,
    complete_root_regret,
    digest_array,
    exact_refinement,
    holm_adjust,
    semantic_alias_equality,
    structured_refinement,
)
from srsa_selector_frozen import run_active_frozen, run_fixed_frozen

SCHEMA = "srsa.cifar.stage4.outcome.v1"


class Stage4Error(RuntimeError):
    pass


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_strings(values: np.ndarray) -> str:
    payload = "\n".join(str(value) for value in values.tolist()) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def seed_for(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def bootstrap_means_discrete(
    values: np.ndarray,
    *,
    seed: int,
    repetitions: int,
    batch: int = 2000,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise Stage4Error(values.shape)
    support, counts = np.unique(values, return_counts=True)
    probabilities = counts.astype(np.float64) / len(values)
    rng = np.random.default_rng(seed)
    output: list[np.ndarray] = []
    remaining = int(repetitions)
    while remaining > 0:
        count = min(batch, remaining)
        draws = rng.multinomial(len(values), probabilities, size=count)
        output.append((draws @ support) / len(values))
        remaining -= count
    return np.concatenate(output)


def bonferroni_lower_bound_discrete(
    values: np.ndarray,
    *,
    seed: int,
    repetitions: int,
    family_size: int,
    alpha: float = 0.05,
) -> float:
    means = bootstrap_means_discrete(
        values,
        seed=seed,
        repetitions=repetitions,
    )
    return float(np.quantile(means, alpha / family_size))


def bootstrap_interval_discrete(
    values: np.ndarray,
    *,
    seed: int,
    repetitions: int,
    alpha: float = 0.05,
) -> list[float]:
    means = bootstrap_means_discrete(
        values,
        seed=seed,
        repetitions=repetitions,
    )
    return [
        float(np.quantile(means, alpha / 2.0)),
        float(np.quantile(means, 1.0 - alpha / 2.0)),
    ]


def signflip_pvalue_discrete(
    values: np.ndarray,
    *,
    seed: int,
    repetitions: int,
    batch: int = 10000,
) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise Stage4Error(values.shape)
    observed = float(np.mean(values))
    magnitudes, counts = np.unique(
        np.abs(values[values != 0.0]),
        return_counts=True,
    )
    if not len(magnitudes):
        return 1.0
    rng = np.random.default_rng(seed)
    exceed = 0
    remaining = int(repetitions)
    counts_float = counts.astype(np.float64)
    while remaining > 0:
        count = min(batch, remaining)
        positive = rng.binomial(counts, 0.5, size=(count, len(counts)))
        signed_sum = (2.0 * positive - counts_float[None, :]) @ magnitudes
        randomized_means = signed_sum / len(values)
        exceed += int(np.sum(randomized_means >= observed))
        remaining -= count
    return float((exceed + 1) / (repetitions + 1))


def load_preoutcome_task(
    root: Path,
    task: str,
    stage3: dict[str, Any],
) -> dict[str, np.ndarray]:
    path = root / task / "preoutcome.npz"
    if not path.is_file():
        raise Stage4Error(f"missing preoutcome arrays: {task}")
    binding = stage3["tasks"][task]["arrays"]
    if file_digest(path) != binding["file_sha256"]:
        raise Stage4Error(f"preoutcome file drift: {task}")
    with np.load(path, allow_pickle=False) as arrays:
        expected = {
            "uids",
            "image_sha256",
            "partition_codes",
            "clean_predictions",
            "alias_tags",
            "primary_indices",
            "seeds",
            "pools",
            "root_digests",
        }
        if set(arrays.files) != expected:
            raise Stage4Error((task, arrays.files))
        value = {name: arrays[name].copy() for name in arrays.files}
    checks = {
        "uids_sha256": digest_strings(value["uids"].astype(str)),
        "image_sha256_array_sha256": digest_strings(
            value["image_sha256"].astype(str)
        ),
        "partition_codes_sha256": digest_array(value["partition_codes"]),
        "clean_predictions_sha256": digest_array(value["clean_predictions"]),
        "alias_tags_sha256": digest_array(value["alias_tags"]),
        "primary_indices_sha256": digest_array(value["primary_indices"]),
        "seeds_sha256": digest_array(value["seeds"]),
        "pools_sha256": digest_array(value["pools"]),
        "root_digests_sha256": digest_strings(
            value["root_digests"].astype(str)
        ),
    }
    for key, observed in checks.items():
        if observed != binding[key]:
            raise Stage4Error(
                {
                    "task": task,
                    "binding": key,
                    "observed": observed,
                    "expected": binding[key],
                }
            )
    return value


def load_sealed_task(
    root: Path,
    task: str,
    stage1: dict[str, Any],
) -> dict[str, np.ndarray]:
    directory = root / task
    manifest_path = directory / "manifest.json"
    arrays_path = directory / "labels.npz"
    if not manifest_path.is_file() or not arrays_path.is_file():
        raise Stage4Error(f"missing sealed task artifact: {task}")
    manifest = read_json(manifest_path)
    binding = stage1["tasks"][task]
    if manifest.get("manifest_sha256") != binding["sealed_manifest_sha256"]:
        raise Stage4Error(f"sealed manifest drift: {task}")
    if file_digest(arrays_path) != binding["sealed_npz_sha256"]:
        raise Stage4Error(f"sealed arrays drift: {task}")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"uids", "partition_codes", "labels"}:
            raise Stage4Error((task, arrays.files))
        return {name: arrays[name].copy() for name in arrays.files}


def load_endpoint(
    directory: Path,
    task: str,
    model_name: str,
    stage2: dict[str, Any],
) -> dict[str, np.ndarray]:
    receipt_path = directory / "receipt.json"
    arrays_path = directory / "predictions.npz"
    if not receipt_path.is_file() or not arrays_path.is_file():
        raise Stage4Error(f"missing endpoint artifact: {model_name}")
    receipt = read_json(receipt_path)
    binding = stage2["tasks"][task]["endpoints"][model_name]
    if receipt.get("decision") != "PASS_SRSA_STAGE2_CLEAN_ENDPOINT":
        raise Stage4Error(f"endpoint not PASS: {model_name}")
    if receipt.get("receipt_sha256") != binding["receipt_sha256"]:
        raise Stage4Error(f"endpoint receipt drift: {model_name}")
    if file_digest(receipt_path) != binding["receipt_file_sha256"]:
        raise Stage4Error(f"endpoint receipt file drift: {model_name}")
    if file_digest(arrays_path) != binding["arrays_sha256"]:
        raise Stage4Error(f"endpoint arrays drift: {model_name}")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        expected = {"uids", "partition_codes", "predictions", "logits"}
        if set(arrays.files) != expected:
            raise Stage4Error((model_name, arrays.files))
        value = {name: arrays[name].copy() for name in arrays.files}
    if digest_array(value["predictions"]) != binding["predictions_sha256"]:
        raise Stage4Error(f"endpoint prediction digest drift: {model_name}")
    if digest_array(value["logits"]) != binding["logits_sha256"]:
        raise Stage4Error(f"endpoint logit digest drift: {model_name}")
    return value


def nll_per_root(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 3 or logits.shape[0] != len(labels):
        raise Stage4Error((logits.shape, labels.shape))
    maximum = np.max(logits, axis=2, keepdims=True)
    log_normalizer = maximum[:, :, 0] + np.log(
        np.sum(np.exp(logits - maximum), axis=2)
    )
    selected = np.take_along_axis(
        logits,
        labels[:, None, None],
        axis=2,
    )[:, :, 0]
    return np.mean(log_normalizer - selected, axis=0).astype(np.float64)


def equal_run(left: Any, right: Any, *, include_posterior: bool) -> bool:
    fields = (
        "queries",
        "roots",
        "terminal",
        "cumulative",
        "attacker_selected",
    )
    if any(
        not np.array_equal(getattr(left, field), getattr(right, field))
        for field in fields
    ):
        return False
    if include_posterior and not np.array_equal(
        left.posterior_roots,
        right.posterior_roots,
    ):
        return False
    return True


def add_run_arrays(
    output: dict[str, np.ndarray],
    prefix: str,
    summary: Any,
) -> None:
    output[f"{prefix}_queries"] = summary.queries
    output[f"{prefix}_roots"] = summary.roots
    output[f"{prefix}_terminal"] = summary.terminal
    output[f"{prefix}_cumulative"] = summary.cumulative
    output[f"{prefix}_attacker_selected"] = summary.attacker_selected.astype(
        np.uint8
    )
    output[f"{prefix}_posterior_roots"] = summary.posterior_roots


def outcome_task(
    task: str,
    config: dict[str, Any],
    stage1: dict[str, Any],
    stage2: dict[str, Any],
    stage3: dict[str, Any],
    sealed_root: Path,
    endpoints_root: Path,
    preoutcome_root: Path,
    run_arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    pre = load_preoutcome_task(preoutcome_root, task, stage3)
    sealed = load_sealed_task(sealed_root, task, stage1)
    if not np.array_equal(pre["uids"], sealed["uids"]):
        raise Stage4Error(f"sealed/preoutcome UID mismatch: {task}")
    if not np.array_equal(
        pre["partition_codes"],
        sealed["partition_codes"],
    ):
        raise Stage4Error(f"sealed/preoutcome partition mismatch: {task}")

    task_lock = stage3["tasks"][task]
    root_ids = list(task_lock["root_order"])
    root_digests = pre["root_digests"].astype(str).tolist()
    endpoints = [
        load_endpoint(endpoints_root / model_name, task, model_name, stage2)
        for model_name in root_ids
    ]
    for endpoint in endpoints:
        if not np.array_equal(endpoint["uids"], pre["uids"]):
            raise Stage4Error(f"endpoint/preoutcome UID mismatch: {task}")
        if not np.array_equal(
            endpoint["partition_codes"],
            pre["partition_codes"],
        ):
            raise Stage4Error(f"endpoint/preoutcome partition mismatch: {task}")
    endpoint_predictions = np.column_stack(
        [endpoint["predictions"] for endpoint in endpoints]
    ).astype(np.int16)
    if not np.array_equal(endpoint_predictions, pre["clean_predictions"]):
        raise Stage4Error(f"clean prediction rebinding failed: {task}")

    primary_indices = pre["primary_indices"].astype(np.int32)
    expected_primary = np.flatnonzero(pre["partition_codes"] == 1)
    if not np.array_equal(primary_indices, expected_primary):
        raise Stage4Error(f"primary index drift: {task}")
    primary_labels = sealed["labels"][primary_indices].astype(np.int64)
    primary_predictions = pre["clean_predictions"][primary_indices].astype(
        np.int64
    )
    primary_uids = pre["uids"][primary_indices].astype(str).tolist()
    primary_image_keys = pre["image_sha256"][primary_indices].astype(str).tolist()
    expected_alias_tags = pre["alias_tags"][primary_indices]
    seeds = pre["seeds"].astype(np.int64).tolist()
    pools = pre["pools"].astype(np.int32)

    clean = clean_registry(primary_predictions, root_ids)
    accuracies, regrets = complete_root_regret(
        primary_predictions,
        primary_labels,
    )
    clean_run = run_active_frozen(
        clean,
        clean,
        primary_labels,
        primary_uids,
        pools,
        seeds,
        root_digests,
        budget=int(config["selector"]["budget"]),
        temperature=float(config["selector"]["temperature"]),
    )
    fixed_clean = run_fixed_frozen(
        clean,
        clean,
        primary_labels,
        clean_run.queries,
        root_digests,
    )
    root_aware_run = run_active_frozen(
        clean,
        clean,
        primary_labels,
        primary_uids,
        pools,
        seeds,
        root_digests,
        budget=int(config["selector"]["budget"]),
        temperature=float(config["selector"]["temperature"]),
    )
    root_aware_exact = equal_run(
        clean_run,
        root_aware_run,
        include_posterior=True,
    )
    fixed_clean_exact = equal_run(
        clean_run,
        fixed_clean,
        include_posterior=False,
    )
    if not root_aware_exact or not fixed_clean_exact:
        raise Stage4Error(f"clean control determinism failed: {task}")
    add_run_arrays(run_arrays, f"{task}_clean", clean_run)
    add_run_arrays(run_arrays, f"{task}_root_aware", root_aware_run)

    response = config["response"]
    bootstrap_repetitions = int(config["selector"]["bootstrap_replicates"])
    signflip_repetitions = int(config["selector"]["signflip_replicates"])
    family_size = int(config["attack_family"]["familywise_roots"])
    root_summaries: dict[str, Any] = {}
    raw_pvalues: dict[str, float] = {}

    deployment_indices = np.flatnonzero(pre["partition_codes"] == 2)
    deployment_labels = sealed["labels"][deployment_indices].astype(np.int64)
    deployment_predictions = endpoint_predictions[deployment_indices]
    deployment_logits = np.stack(
        [endpoint["logits"][deployment_indices] for endpoint in endpoints],
        axis=1,
    )
    deployment_accuracy = np.mean(
        deployment_predictions == deployment_labels[:, None],
        axis=0,
    ).astype(np.float64)
    deployment_loss = 1.0 - deployment_accuracy
    deployment_nll = nll_per_root(deployment_logits, deployment_labels)

    for attack_root, root_id in enumerate(root_ids):
        structured = structured_refinement(
            clean,
            primary_image_keys,
            attack_root,
            root_ids,
            alias_count=int(response["alias_count"]),
            numerator=int(response["canary_numerator"]),
            denominator=int(response["canary_denominator"]),
            salt=str(response["canary_salt"]),
        )
        exact = exact_refinement(
            clean,
            attack_root,
            root_ids,
            alias_count=int(response["alias_count"]),
        )
        if not semantic_alias_equality(clean, structured):
            raise Stage4Error(
                f"structured semantic equality failed: {task}/{root_id}"
            )
        if not semantic_alias_equality(clean, exact):
            raise Stage4Error(f"exact semantic equality failed: {task}/{root_id}")
        observed_alias_tags = structured.tags[:, clean.labels.shape[1] :]
        if not np.array_equal(observed_alias_tags, expected_alias_tags):
            raise Stage4Error(
                f"Stage-3 canary rebinding failed: {task}/{root_id}"
            )

        structured_run = run_active_frozen(
            structured,
            clean,
            primary_labels,
            primary_uids,
            pools,
            seeds,
            root_digests,
            budget=int(config["selector"]["budget"]),
            temperature=float(config["selector"]["temperature"]),
            attack_root=attack_root,
        )
        exact_run = run_active_frozen(
            exact,
            clean,
            primary_labels,
            primary_uids,
            pools,
            seeds,
            root_digests,
            budget=int(config["selector"]["budget"]),
            temperature=float(config["selector"]["temperature"]),
            attack_root=attack_root,
        )
        fixed_structured = run_fixed_frozen(
            structured,
            clean,
            primary_labels,
            clean_run.queries,
            root_digests,
            attack_root=attack_root,
        )

        terminal_difference = structured_run.terminal - clean_run.terminal
        cumulative_difference = (
            structured_run.cumulative - clean_run.cumulative
        )
        fixed_terminal_difference = (
            fixed_structured.terminal - fixed_clean.terminal
        )
        fixed_cumulative_difference = (
            fixed_structured.cumulative - fixed_clean.cumulative
        )
        active_minus_fixed = (
            structured_run.terminal - fixed_structured.terminal
        )
        exact_terminal_difference = exact_run.terminal - clean_run.terminal
        exact_cumulative_difference = (
            exact_run.cumulative - clean_run.cumulative
        )
        path_changed = np.any(
            structured_run.queries != clean_run.queries,
            axis=1,
        )
        final_root_changed = (
            structured_run.roots[:, -1] != clean_run.roots[:, -1]
        )
        zeros_terminal = np.zeros_like(fixed_terminal_difference)
        zeros_cumulative = np.zeros_like(fixed_cumulative_difference)
        fixed_zero = (
            np.array_equal(fixed_structured.queries, fixed_clean.queries)
            and np.array_equal(fixed_structured.roots, fixed_clean.roots)
            and np.array_equal(fixed_terminal_difference, zeros_terminal)
            and np.array_equal(fixed_cumulative_difference, zeros_cumulative)
        )
        active_fixed_identity = np.array_equal(
            terminal_difference,
            active_minus_fixed,
        )

        terminal_mean = float(np.mean(terminal_difference))
        cumulative_mean = float(np.mean(cumulative_difference))
        active_fixed_mean = float(np.mean(active_minus_fixed))
        terminal_lower = bonferroni_lower_bound_discrete(
            terminal_difference,
            seed=seed_for(task, root_id, "terminal-bootstrap"),
            repetitions=bootstrap_repetitions,
            family_size=family_size,
        )
        cumulative_lower = bonferroni_lower_bound_discrete(
            cumulative_difference,
            seed=seed_for(task, root_id, "cumulative-bootstrap"),
            repetitions=bootstrap_repetitions,
            family_size=family_size,
        )
        raw_pvalue = signflip_pvalue_discrete(
            terminal_difference,
            seed=seed_for(task, root_id, "terminal-signflip"),
            repetitions=signflip_repetitions,
        )
        raw_pvalues[root_id] = raw_pvalue

        clean_selected = clean_run.roots[:, -1]
        attacked_selected = structured_run.roots[:, -1]
        deployment_loss_difference = (
            deployment_loss[attacked_selected] - deployment_loss[clean_selected]
        )
        deployment_nll_difference = (
            deployment_nll[attacked_selected] - deployment_nll[clean_selected]
        )
        deployment_accuracy_difference = (
            deployment_accuracy[attacked_selected]
            - deployment_accuracy[clean_selected]
        )

        add_run_arrays(
            run_arrays,
            f"{task}_structured_{attack_root}",
            structured_run,
        )
        add_run_arrays(
            run_arrays,
            f"{task}_exact_{attack_root}",
            exact_run,
        )
        add_run_arrays(
            run_arrays,
            f"{task}_fixed_structured_{attack_root}",
            fixed_structured,
        )
        run_arrays[f"{task}_terminal_difference_{attack_root}"] = (
            terminal_difference
        )
        run_arrays[f"{task}_cumulative_difference_{attack_root}"] = (
            cumulative_difference
        )
        run_arrays[f"{task}_deployment_loss_difference_{attack_root}"] = (
            deployment_loss_difference
        )
        run_arrays[f"{task}_deployment_nll_difference_{attack_root}"] = (
            deployment_nll_difference
        )

        root_summaries[root_id] = {
            "attack_root_index": attack_root,
            "path_change_rate": float(np.mean(path_changed)),
            "final_root_change_rate": float(np.mean(final_root_changed)),
            "attacked_root_selection_probability_clean": float(
                np.mean(clean_selected == attack_root)
            ),
            "attacked_root_selection_probability_structured": float(
                np.mean(attacked_selected == attack_root)
            ),
            "attacked_root_selection_probability_change": float(
                np.mean(attacked_selected == attack_root)
                - np.mean(clean_selected == attack_root)
            ),
            "terminal": {
                "mean": terminal_mean,
                "mean_percentage_points": 100.0 * terminal_mean,
                "bonferroni_simultaneous_lower": terminal_lower,
                "raw_one_sided_signflip_p": raw_pvalue,
            },
            "cumulative": {
                "mean": cumulative_mean,
                "mean_percentage_points": 100.0 * cumulative_mean,
                "bonferroni_simultaneous_lower": cumulative_lower,
            },
            "fixed_query": {
                "root_history_bitwise_identical": bool(
                    np.array_equal(fixed_structured.roots, fixed_clean.roots)
                ),
                "terminal_effect_exactly_zero": bool(
                    np.array_equal(fixed_terminal_difference, zeros_terminal)
                ),
                "cumulative_effect_exactly_zero": bool(
                    np.array_equal(
                        fixed_cumulative_difference,
                        zeros_cumulative,
                    )
                ),
                "all_required_effects_zero": fixed_zero,
            },
            "active_minus_fixed": {
                "mean": active_fixed_mean,
                "mean_percentage_points": 100.0 * active_fixed_mean,
                "numerically_identical_to_active_terminal_difference": (
                    active_fixed_identity
                ),
            },
            "exact_wrapper_anchor": {
                "path_change_rate": float(
                    np.mean(
                        np.any(
                            exact_run.queries != clean_run.queries,
                            axis=1,
                        )
                    )
                ),
                "terminal_mean": float(np.mean(exact_terminal_difference)),
                "cumulative_mean": float(np.mean(exact_cumulative_difference)),
            },
            "root_aware": {
                "canonical_input_equals_clean": bool(
                    task_lock["attacks"][root_id][
                        "root_aware_canonical_equals_clean"
                    ]
                ),
                "queries_bitwise_identical": bool(
                    np.array_equal(root_aware_run.queries, clean_run.queries)
                ),
                "posterior_root_history_bitwise_identical": bool(
                    np.array_equal(
                        root_aware_run.posterior_roots,
                        clean_run.posterior_roots,
                    )
                ),
                "root_history_bitwise_identical": bool(
                    np.array_equal(root_aware_run.roots, clean_run.roots)
                ),
                "terminal_bitwise_identical": bool(
                    np.array_equal(
                        root_aware_run.terminal,
                        clean_run.terminal,
                    )
                ),
                "cumulative_bitwise_identical": bool(
                    np.array_equal(
                        root_aware_run.cumulative,
                        clean_run.cumulative,
                    )
                ),
                "all_required_histories_identical": root_aware_exact,
            },
            "deployment_secondary": {
                "loss_mean_difference": float(
                    np.mean(deployment_loss_difference)
                ),
                "loss_mean_difference_percentage_points": (
                    100.0 * float(np.mean(deployment_loss_difference))
                ),
                "loss_95_percent_bootstrap_interval": (
                    bootstrap_interval_discrete(
                        deployment_loss_difference,
                        seed=seed_for(
                            task,
                            root_id,
                            "deployment-loss-bootstrap",
                        ),
                        repetitions=bootstrap_repetitions,
                    )
                ),
                "nll_mean_difference": float(
                    np.mean(deployment_nll_difference)
                ),
                "nll_95_percent_bootstrap_interval": (
                    bootstrap_interval_discrete(
                        deployment_nll_difference,
                        seed=seed_for(
                            task,
                            root_id,
                            "deployment-nll-bootstrap",
                        ),
                        repetitions=bootstrap_repetitions,
                    )
                ),
                "accuracy_mean_difference": float(
                    np.mean(deployment_accuracy_difference)
                ),
            },
        }

    adjusted = holm_adjust(raw_pvalues)
    gates_config = config["primary_gates"]
    for root_id, summary in root_summaries.items():
        summary["terminal"]["holm_adjusted_one_sided_p"] = float(
            adjusted[root_id]
        )
        gates = {
            "bindings_pass": True,
            "semantic_alias_equality_all_rows": bool(
                task_lock["attacks"][root_id][
                    "semantic_prediction_identical_all_rows"
                ]
            ),
            "canary_fraction_4_to_6_percent": bool(
                task_lock["canary"]["gate_pass"]
            ),
            "ordered_path_change_at_least_50_percent": (
                summary["path_change_rate"]
                >= float(gates_config["path_change_min"])
            ),
            "terminal_harm_at_least_half_point": (
                summary["terminal"]["mean"]
                >= float(gates_config["terminal_harm_min"])
            ),
            "active_minus_fixed_at_least_half_point": (
                summary["active_minus_fixed"]["mean"]
                >= float(
                    gates_config[
                        "active_minus_fixed_terminal_harm_min"
                    ]
                )
            ),
            "terminal_simultaneous_lower_above_zero": (
                summary["terminal"]["bonferroni_simultaneous_lower"]
                > float(
                    gates_config[
                        "bonferroni_simultaneous_lower_bound_gt"
                    ]
                )
            ),
            "holm_adjusted_signflip_at_most_005": (
                summary["terminal"]["holm_adjusted_one_sided_p"]
                <= float(gates_config["holm_adjusted_one_sided_p_max"])
            ),
            "cumulative_mean_above_zero": (
                summary["cumulative"]["mean"]
                > float(gates_config["cumulative_mean_gt"])
            ),
            "cumulative_simultaneous_lower_above_zero": (
                summary["cumulative"]["bonferroni_simultaneous_lower"]
                > float(
                    gates_config[
                        "cumulative_simultaneous_lower_bound_gt"
                    ]
                )
            ),
            "fixed_query_effect_exactly_zero": bool(
                summary["fixed_query"]["all_required_effects_zero"]
            ),
            "root_aware_histories_bitwise_identical": bool(
                summary["root_aware"][
                    "all_required_histories_identical"
                ]
            ),
            "active_fixed_identity": bool(
                summary["active_minus_fixed"][
                    "numerically_identical_to_active_terminal_difference"
                ]
            ),
        }
        summary["gates"] = gates
        summary["passes_all_gates"] = bool(all(gates.values()))

    terminal_means = {
        root_id: float(summary["terminal"]["mean"])
        for root_id, summary in root_summaries.items()
    }
    adversarial_root = max(
        root_ids,
        key=lambda root_id: (terminal_means[root_id], root_id),
    )
    passing_roots = [
        root_id
        for root_id in root_ids
        if root_summaries[root_id]["passes_all_gates"]
    ]
    if task == "cifar100":
        task_decision = (
            "GO_SRSA_PRIMARY" if passing_roots else "NO_GO_SRSA_PRIMARY"
        )
    else:
        task_decision = (
            "POSITIVE_SRSA_REPLICATION"
            if passing_roots
            else "NEGATIVE_SRSA_REPLICATION"
        )

    return {
        "role": config["tasks"][task]["role"],
        "decision": task_decision,
        "passing_roots": passing_roots,
        "adversarial_estimand": {
            "definition": (
                "maximum mean terminal harm over all four frozen root attacks"
            ),
            "root": adversarial_root,
            "mean": terminal_means[adversarial_root],
            "mean_percentage_points": 100.0
            * terminal_means[adversarial_root],
        },
        "clean_primary": {
            "root_order": root_ids,
            "accuracy": [float(value) for value in accuracies],
            "loss": [float(1.0 - value) for value in accuracies],
            "regret": [float(value) for value in regrets],
            "oracle_best_roots": [
                root_ids[index]
                for index in np.flatnonzero(regrets == 0.0)
            ],
        },
        "clean_deployment_secondary": {
            "accuracy": [float(value) for value in deployment_accuracy],
            "loss": [float(value) for value in deployment_loss],
            "nll": [float(value) for value in deployment_nll],
        },
        "canary": task_lock["canary"],
        "roots": root_summaries,
        "familywise": {
            "roots": family_size,
            "bootstrap_replicates": bootstrap_repetitions,
            "signflip_replicates": signflip_repetitions,
            "holm_adjusted_pvalues": {
                root_id: float(adjusted[root_id]) for root_id in root_ids
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage0-lock", type=Path, required=True)
    parser.add_argument("--stage1-lock", type=Path, required=True)
    parser.add_argument("--stage2-lock", type=Path, required=True)
    parser.add_argument("--stage3-lock", type=Path, required=True)
    parser.add_argument("--stage1-sealed-root", type=Path, required=True)
    parser.add_argument("--endpoints-root", type=Path, required=True)
    parser.add_argument("--preoutcome-root", type=Path, required=True)
    parser.add_argument("--stage3-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = read_json(args.config)
    stage0 = read_json(args.stage0_lock)
    stage1 = read_json(args.stage1_lock)
    stage2 = read_json(args.stage2_lock)
    stage3 = read_json(args.stage3_lock)
    if stage0.get("decision") != "PASS_SRSA_STAGE0_PUBLIC_METADATA":
        raise Stage4Error("Stage 0 lock is not PASS")
    if stage1.get("decision") != "PASS_SRSA_STAGE1_DATA_SEAL":
        raise Stage4Error("Stage 1 lock is not PASS")
    if stage2.get("decision") != "PASS_SRSA_STAGE2_CLEAN_ENDPOINTS":
        raise Stage4Error("Stage 2 lock is not PASS")
    if stage3.get("decision") != "PASS_SRSA_STAGE3_PREOUTCOME":
        raise Stage4Error("Stage 3 lock is not PASS; labels must remain sealed")
    if file_digest(args.config) != stage3["config_sha256"]:
        raise Stage4Error("config changed after Stage 3")
    if file_digest(args.stage0_lock) != stage3["stage0_lock_sha256"]:
        raise Stage4Error("Stage 0 lock changed after Stage 3")
    if file_digest(args.stage1_lock) != stage3["stage1_lock_sha256"]:
        raise Stage4Error("Stage 1 lock changed after Stage 3")
    if file_digest(args.stage2_lock) != stage3["stage2_lock_sha256"]:
        raise Stage4Error("Stage 2 lock changed after Stage 3")
    here = Path(__file__).resolve()
    if file_digest(here) != stage3["code"]["stage4_sha256"]:
        raise Stage4Error("Stage 4 executable changed after pre-outcome lock")
    if file_digest(here.with_name("srsa_core.py")) != stage3["code"][
        "core_sha256"
    ]:
        raise Stage4Error("SRSA core changed after pre-outcome lock")
    if file_digest(here.with_name("srsa_selector_frozen.py")) != stage3["code"][
        "frozen_selector_sha256"
    ]:
        raise Stage4Error("frozen selector changed after pre-outcome lock")

    args.output.mkdir(parents=True, exist_ok=True)
    run_arrays: dict[str, np.ndarray] = {}
    tasks = {
        task: outcome_task(
            task,
            config,
            stage1,
            stage2,
            stage3,
            args.stage1_sealed_root,
            args.endpoints_root,
            args.preoutcome_root,
            run_arrays,
        )
        for task in ("cifar100", "cifar10")
    }
    decision = tasks["cifar100"]["decision"]
    if decision not in {"GO_SRSA_PRIMARY", "NO_GO_SRSA_PRIMARY"}:
        raise Stage4Error(decision)

    runs_path = args.output / "SRSA_STAGE4_PAIRED_RUNS.npz"
    np.savez_compressed(runs_path, **run_arrays)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "decision": decision,
        "primary_task": "cifar100",
        "replication_task": "cifar10",
        "replication_cannot_rescue_primary": True,
        "no_third_task": True,
        "stage3_preoutcome_commit": args.stage3_commit,
        "bindings": {
            "config_sha256": file_digest(args.config),
            "stage0_lock_sha256": file_digest(args.stage0_lock),
            "stage1_lock_sha256": file_digest(args.stage1_lock),
            "stage2_lock_sha256": file_digest(args.stage2_lock),
            "stage3_lock_sha256": file_digest(args.stage3_lock),
            "stage3_declared_lock_sha256": stage3["lock_sha256"],
            "stage4_code_sha256": file_digest(here),
            "paired_runs_file_sha256": file_digest(runs_path),
            "paired_runs_file_bytes": runs_path.stat().st_size,
        },
        "tasks": tasks,
        "information_boundary": {
            "stage3_lock_verified_before_label_read": True,
            "sealed_label_artifact_downloaded_only_by_stage4": True,
            "sealed_labels_opened": True,
            "label_opening_jobs": 1,
            "primary_outcome_opened": True,
            "deployment_outcome_opened": True,
            "all_four_roots_executed_and_reported": True,
            "third_task_introduced": False,
        },
        "closure": {
            "scientific_no_go_is_successful_workflow_completion": True,
            "technical_invalid_raises_exception": True,
        },
    }
    result["result_sha256"] = canonical_sha256(result)
    result_path = args.output / "SRSA_STAGE4_RESULT.json"
    result_path.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "result_sha256": result["result_sha256"],
                "cifar100_passing_roots": tasks["cifar100"][
                    "passing_roots"
                ],
                "cifar10_passing_roots": tasks["cifar10"][
                    "passing_roots"
                ],
                "cifar100_adversarial": tasks["cifar100"][
                    "adversarial_estimand"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
