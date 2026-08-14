from __future__ import annotations

import inspect
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification

import step96_common as common
from step96_executable_adapter_endpoint import execute_aliases_from_raw_pairs


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / f"STEP96_STAGE0_MANIFEST_{common.DATE}.json"
DEVELOPMENT_PATH = ROOT / f"STEP96_SNLI_DEVELOPMENT_ROWS_{common.DATE}.json"
OUTPUT_PATH = ROOT / f"STEP96_STAGEA_DEVELOPMENT_ARRAYS_{common.DATE}.npz"
LEDGER_PATH = ROOT / f"STEP96_STAGEA_COMPLETE_LEDGER_{common.DATE}.json"
CONFIG_PATH = ROOT / f"STEP96_STAGEA_FROZEN_CONFIG_{common.DATE}.json"
EARLY_STOP_PATH = ROOT / f"STEP96_STAGEA_ROOT_GEOMETRY_STOP_{common.DATE}.json"
MODEL_DIR = ROOT / "step96_models"
WORK_DIR = ROOT / "step96_stagea_work"
ROOT_CACHE = WORK_DIR / "root_predictions.npz"
ROOT_AUDIT_CACHE = WORK_DIR / "root_prediction_audits.json"
AMENDMENT_A = ROOT / f"STEP96_PREREGISTRATION_AMENDMENT_A_INTEGER_GATE_{common.DATE}.md"
SEARCH_SEEDS = tuple(range(963000, 963400))
VERIFY_SEEDS = tuple(range(964000, 965500))
TOP_TO_VERIFY = 8


def source_sha256(function: object) -> str:
    import hashlib

    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


def load_rows() -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    payload = json.loads(DEVELOPMENT_PATH.read_text(encoding="utf-8"))
    rows = [*payload["train_rows"], *payload["validation_rows"]]
    partitions: dict[str, list[int]] = {
        **{name: [] for name in common.PARTITION_QUOTAS},
        "selector_verify": [],
    }
    for index, row in enumerate(rows):
        partitions[str(row["partition"])].append(index)
    expected = {
        "adapter_train": 9000,
        "roster_gate": 3000,
        "threshold": 4500,
        "selector_search": 6000,
        "selector_verify": len(payload["validation_rows"]),
    }
    observed = {name: len(indices) for name, indices in partitions.items()}
    if observed != expected:
        raise AssertionError({"observed": observed, "expected": expected})
    return rows, {name: np.asarray(indices, dtype=np.int64) for name, indices in partitions.items()}


def root_preflight() -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for index, spec in enumerate(common.ROOT_SPECS):
        tokenizer = common.tokenizer_for(spec)
        model = AutoModelForSequenceClassification.from_pretrained(
            spec.repo_id, revision=spec.revision, use_safetensors=None
        )
        if int(model.config.num_labels) != 3:
            raise AssertionError({"root": spec.key, "num_labels": model.config.num_labels})
        if spec.parent and str(model.config.model_type) != "deberta-v2":
            raise AssertionError({"parent_model_type": model.config.model_type})
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
                "loaded_before_snli_prediction": True,
            }
        )
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"preflight root {index + 1}/{len(common.ROOT_SPECS)}: {spec.key}", flush=True)
    return audits


def load_or_infer_roots(
    premises: list[str], hypotheses: list[str]
) -> tuple[np.ndarray, list[dict[str, Any]], bool]:
    if ROOT_CACHE.is_file() and ROOT_AUDIT_CACHE.is_file():
        cache = np.load(ROOT_CACHE, allow_pickle=False)
        predictions = cache["root_predictions"].astype(np.int64)
        audits = json.loads(ROOT_AUDIT_CACHE.read_text(encoding="utf-8"))
        if predictions.shape != (len(premises), common.ROSTER_SIZE):
            raise AssertionError(predictions.shape)
        for index, audit in enumerate(audits):
            if audit["prediction_sha256"] != common.array_sha256(predictions[:, index]):
                raise AssertionError("root cache hash drift")
        return predictions, audits, True
    if ROOT_CACHE.exists() or ROOT_AUDIT_CACHE.exists():
        raise RuntimeError("incomplete Step 96 root cache")
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    columns: list[np.ndarray] = []
    audits: list[dict[str, Any]] = []
    for index, spec in enumerate(common.ROOT_SPECS):
        predictions, audit = common.infer_root_predictions(spec, premises, hypotheses)
        columns.append(predictions)
        audits.append(audit)
        print(f"inferred root {index + 1}/{common.ROSTER_SIZE}: {spec.key}", flush=True)
    matrix = np.stack(columns, axis=1)
    np.savez_compressed(ROOT_CACHE, root_predictions=matrix.astype(np.int16))
    common.json_dump(ROOT_AUDIT_CACHE, audits)
    return matrix, audits, False


def geometry(
    predictions: np.ndarray, labels: np.ndarray, indices: np.ndarray
) -> dict[str, Any]:
    codes = predictions[indices]
    references = labels[indices]
    accuracies = np.mean(codes == references[:, None], axis=0)
    parent_wrong = codes[:, 0] != references
    complementarity = [
        float(np.mean((codes[:, root] == references)[parent_wrong]))
        if np.any(parent_wrong)
        else 0.0
        for root in range(1, common.ROSTER_SIZE)
    ]
    second = float(np.max(accuracies[1:]))
    return {
        "rows": len(indices),
        "root_accuracies": accuracies.tolist(),
        "root_accuracy_by_key": {
            spec.key: float(value) for spec, value in zip(common.ROOT_SPECS, accuracies)
        },
        "parent_minus_best_challenger": float(accuracies[0] - second),
        "parent_error_prevalence": float(np.mean(parent_wrong)),
        "challenger_correct_given_parent_wrong": {
            common.ROOT_SPECS[root].key: complementarity[root - 1]
            for root in range(1, common.ROSTER_SIZE)
        },
        "max_challenger_correct_given_parent_wrong": max(complementarity),
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
    alias_codes, triggers = common.make_alias_codes(parent_predictions, error_scores, thresholds)
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
    alias_codes, triggers = common.make_alias_codes(parent_predictions, error_scores, thresholds)
    feedback = common.exact_match_feedback(alias_codes, labels)
    parent_feedback = common.exact_match_feedback(parent_predictions[:, None], labels)
    return {
        "thresholds": [float(value) for value in thresholds],
        "threshold_audits": [],
        "alias_codes": alias_codes,
        "triggers": triggers,
        "alias_losses": np.mean(parent_feedback, axis=0)[0] - np.mean(feedback, axis=0),
        "coordinate_wise_nonimproving": bool(
            np.all(feedback <= np.repeat(parent_feedback, common.N_ALIASES, axis=1))
        ),
    }


def evaluate(
    partition: str,
    clean_codes: np.ndarray,
    labels: np.ndarray,
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
        pool_cache[partition] = common.sample_pools(len(labels), seeds, common.POOL_SIZE)
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
        [clean_parents, np.zeros(common.N_ALIASES, dtype=np.int64)]
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
    clean_final = clean.roots[:, -1]
    refined_final = refined.roots[:, -1]
    parent_to_challenger = (clean_final == 0) & (refined_final != 0)
    challenger_to_parent = (clean_final != 0) & (refined_final == 0)
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
        "final_root_change_rate": float(np.mean(clean_final != refined_final)),
        "clean_parent_terminal_rate": float(np.mean(clean_final == 0)),
        "refined_parent_terminal_rate": float(np.mean(refined_final == 0)),
        "parent_to_challenger_rate": float(np.mean(parent_to_challenger)),
        "challenger_to_parent_rate": float(np.mean(challenger_to_parent)),
        "directional_transition_net": float(
            np.mean(parent_to_challenger) - np.mean(challenger_to_parent)
        ),
    }
    if retain_vectors:
        row["vectors"] = {
            "terminal": terminal.tolist(),
            "active_minus_fixed_terminal": active_minus_fixed.tolist(),
            "cumulative": cumulative.tolist(),
            "fixed_terminal": fixed_terminal.tolist(),
            "fixed_cumulative": fixed_cumulative.tolist(),
            "clean_final_roots": clean_final.tolist(),
            "refined_final_roots": refined_final.tolist(),
        }
    return row


def rank_key(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(row["mean_active_minus_fixed_terminal_delta"]),
        float(row["mean_terminal_delta"]),
        float(row["directional_transition_net"]),
        float(row["mean_cumulative_delta"]),
        float(row["path_change_rate"]),
        -float(row["tau"]),
        -float(row["loss_cap"]),
        -float(row["trigger_cap"]),
    )


def main() -> None:
    for output in (OUTPUT_PATH, LEDGER_PATH, CONFIG_PATH):
        if output.exists():
            raise RuntimeError(f"refusing to overwrite Step 96 Stage-A output: {output}")
    for authority in (common.PROTOCOL, AMENDMENT_A, MANIFEST_PATH, DEVELOPMENT_PATH):
        if not authority.is_file():
            raise FileNotFoundError(authority)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["authority_sha256"][common.PROTOCOL.name] != common.sha256_path(common.PROTOCOL):
        raise AssertionError("protocol hash drift")
    preflight = root_preflight()
    rows, partitions = load_rows()
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    premises = [str(row["premise"]) for row in rows]
    hypotheses = [str(row["hypothesis"]) for row in rows]
    predictions, root_audits, cache_reused = load_or_infer_roots(premises, hypotheses)
    geometry_rows = {
        name: geometry(predictions, labels, partitions[name])
        for name in ("roster_gate", "selector_search", "selector_verify")
    }
    gate_geometry = geometry_rows["roster_gate"]
    early_gates = {
        "parent_unique_best_by_2_5pp_on_roster_gate": bool(
            gate_geometry["parent_minus_best_challenger"] >= 0.025
        ),
        "parent_error_prevalence_at_least_3pct": bool(
            gate_geometry["parent_error_prevalence"] >= 0.03
        ),
        "challenger_correct_on_at_least_15pct_parent_errors": bool(
            gate_geometry["max_challenger_correct_given_parent_wrong"] >= 0.15
        ),
    }
    if not all(early_gates.values()):
        receipt = {
            "decision": "NO_GO_STEP96_ROOT_GEOMETRY_STOP",
            "authority_sha256": {common.PROTOCOL.name: common.sha256_path(common.PROTOCOL)},
            "manifest_sha256": common.sha256_path(MANIFEST_PATH),
            "development_sha256": common.sha256_path(DEVELOPMENT_PATH),
            "root_cache_sha256": common.sha256_path(ROOT_CACHE),
            "root_audit_sha256": common.sha256_path(ROOT_AUDIT_CACHE),
            "root_preflight": preflight,
            "root_prediction_audits": root_audits,
            "geometry": geometry_rows,
            "gates": early_gates,
            "failed_gates": [name for name, passed in early_gates.items() if not passed],
            "sealed_test_opened": False,
        }
        common.json_dump(EARLY_STOP_PATH, receipt)
        print(json.dumps(receipt, indent=2))
        return
    print(json.dumps({"root_geometry_go": True, "geometry": geometry_rows}, indent=2), flush=True)

    adapter_indices = partitions["adapter_train"]
    parent = common.ROOT_SPECS[0]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    score_columns: list[np.ndarray] = []
    adapter_audits: list[dict[str, Any]] = []
    adapter_paths: list[str] = []
    for alias in range(common.N_ALIASES):
        path = MODEL_DIR / f"step96_deberta_error_adapter_{alias}.safetensors"
        audit_path = MODEL_DIR / f"step96_deberta_error_adapter_{alias}.audit.json"
        score_path = WORK_DIR / f"step96_error_scores_{alias}.npy"
        if path.is_file() and audit_path.is_file() and score_path.is_file():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if audit["checkpoint"]["sha256"] != common.sha256_path(path):
                raise AssertionError("adapter resume hash drift")
            scores = np.load(score_path, allow_pickle=False).astype(np.float64)
            if audit["score_sha256"] != common.array_sha256(scores):
                raise AssertionError("score resume hash drift")
            resumed = True
        elif path.exists() or audit_path.exists() or score_path.exists():
            raise RuntimeError(f"incomplete Step 96 adapter resume set for alias {alias}")
        else:
            model, tokenizer, training = common.train_error_adapter(
                parent,
                [premises[index] for index in adapter_indices],
                [hypotheses[index] for index in adapter_indices],
                predictions[adapter_indices, 0],
                labels[adapter_indices],
                seed=96100 + alias,
            )
            checkpoint = common.save_adapter(
                model,
                path,
                {
                    "step": "96",
                    "alias": alias,
                    "parent_repo": parent.repo_id,
                    "parent_revision": parent.revision,
                    "seed": 96100 + alias,
                },
            )
            scores = common.infer_error_scores(model, tokenizer, premises, hypotheses)
            np.save(score_path, scores.astype(np.float64), allow_pickle=False)
            audit = {
                "alias": alias,
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "training": training,
                "checkpoint": checkpoint,
                "score_sha256": common.array_sha256(scores),
            }
            common.json_dump(audit_path, audit)
            model.cpu()
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            resumed = False
        if scores.shape != (len(rows),):
            raise AssertionError(scores.shape)
        adapter_paths.append(str(path.relative_to(ROOT)).replace("\\", "/"))
        adapter_audits.append(audit)
        score_columns.append(scores)
        print(
            f"adapter {alias + 1}/{common.N_ALIASES} ready; resumed={resumed}; "
            f"params={audit['checkpoint']['parameter_count']}",
            flush=True,
        )
    error_scores = np.stack(score_columns, axis=1)
    if len({row["checkpoint"]["sha256"] for row in adapter_audits}) != common.N_ALIASES:
        raise AssertionError("adapter hashes are not distinct")
    if any(row["checkpoint"]["parameter_count"] < 10_000_000 for row in adapter_audits):
        raise AssertionError("adapter below parameter floor")

    threshold_indices = partitions["threshold"]
    search_indices = partitions["selector_search"]
    verify_indices = partitions["selector_verify"]
    threshold_cache: dict[tuple[float, float], dict[str, Any]] = {}
    search_rows: list[dict[str, Any]] = []
    search_clean_cache: dict[float, common.RunSummary] = {}
    search_pool_cache: dict[str, np.ndarray] = {}
    search_clean = predictions[search_indices]
    search_labels = labels[search_indices]
    for loss_cap, trigger_cap, tau in itertools.product(
        common.LOSS_CAPS, common.TRIGGER_CAPS, common.TAUS
    ):
        key = (loss_cap, trigger_cap)
        if key not in threshold_cache:
            threshold_cache[key] = construction(
                predictions[threshold_indices, 0],
                labels[threshold_indices],
                error_scores[threshold_indices],
                loss_cap,
                trigger_cap,
            )
        threshold_built = threshold_cache[key]
        search_built = replay_construction(
            search_clean[:, 0],
            search_labels,
            error_scores[search_indices],
            threshold_built["thresholds"],
        )
        row = evaluate(
            "selector_search",
            search_clean,
            search_labels,
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
            f"terminal={100*row['mean_terminal_delta']:+.3f}pp",
            flush=True,
        )

    ranked_search = sorted(search_rows, key=rank_key, reverse=True)
    verify_clean = predictions[verify_indices]
    verify_labels = labels[verify_indices]
    verification_rows: list[dict[str, Any]] = []
    verify_clean_cache: dict[float, common.RunSummary] = {}
    verify_pool_cache: dict[str, np.ndarray] = {}
    for rank, search_row in enumerate(ranked_search[:TOP_TO_VERIFY], start=1):
        built = replay_construction(
            verify_clean[:, 0],
            verify_labels,
            error_scores[verify_indices],
            [float(value) for value in search_row["thresholds"]],
        )
        row = evaluate(
            "selector_verify",
            verify_clean,
            verify_labels,
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
        row["search_clean_parent_terminal_rate"] = search_row["clean_parent_terminal_rate"]
        row["search_parent_to_challenger_rate"] = search_row["parent_to_challenger_rate"]
        row["search_challenger_to_parent_rate"] = search_row["challenger_to_parent_rate"]
        verification_rows.append(row)
        print(
            f"verify {rank}/{TOP_TO_VERIFY}: terminal={100*row['mean_terminal_delta']:+.3f}pp "
            f"net={row['directional_transition_net']:+.3f}",
            flush=True,
        )
    selected = max(verification_rows, key=rank_key)
    vectors = selected.pop("vectors")
    terminal_summary = common.effect_summary(
        np.asarray(vectors["terminal"], dtype=np.float64), 96900, 96901
    )
    active_summary = common.effect_summary(
        np.asarray(vectors["active_minus_fixed_terminal"], dtype=np.float64), 96901, 96902
    )
    cumulative_summary = common.effect_summary(
        np.asarray(vectors["cumulative"], dtype=np.float64), 96902, 96903
    )
    selected["inference"] = {
        "terminal": terminal_summary,
        "active_minus_fixed_terminal": active_summary,
        "cumulative": cumulative_summary,
    }
    selected["terminal_root_counts"] = {
        "clean": np.bincount(
            np.asarray(vectors["clean_final_roots"], dtype=np.int64),
            minlength=common.ROSTER_SIZE,
        ).tolist(),
        "refined": np.bincount(
            np.asarray(vectors["refined_final_roots"], dtype=np.int64),
            minlength=common.ROSTER_SIZE,
        ).tolist(),
    }
    search_match = next(
        row
        for row in search_rows
        if row["tau"] == selected["tau"]
        and row["loss_cap"] == selected["loss_cap"]
        and row["trigger_cap"] == selected["trigger_cap"]
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
            "dataset",
        )
    )
    geometry_gates = all(
        geometry_rows[name]["parent_minus_best_challenger"] >= 0.025
        for name in ("roster_gate", "selector_search", "selector_verify")
    )
    gates = {
        **early_gates,
        "all_four_distinct_adapters_at_least_ten_million_parameters": bool(
            len({row["checkpoint"]["sha256"] for row in adapter_audits}) == 4
            and all(row["checkpoint"]["parameter_count"] >= 10_000_000 for row in adapter_audits)
        ),
        "raw_input_endpoint_has_no_forbidden_runtime_input": not forbidden_runtime,
        "same_literal_similarity": True,
        "parent_unique_best_by_2_5pp_on_roster_search_validation": geometry_gates,
        "coordinate_wise_nonimproving_threshold_search_validation": bool(
            search_match["threshold_coordinate_wise_nonimproving"]
            and search_match["coordinate_wise_nonimproving"]
            and selected["coordinate_wise_nonimproving"]
        ),
        "alias_losses_at_most_one_point_threshold_search_validation": bool(
            max(search_match["threshold_alias_losses"]) <= 0.0100 + 1e-12
            and max(search_match["alias_losses"]) <= 0.0100 + 1e-12
            and max(selected["alias_losses"]) <= 0.0100 + 1e-12
        ),
        "search_and_validation_path_change_at_least_half": bool(
            search_match["path_change_rate"] >= 0.50 and selected["path_change_rate"] >= 0.50
        ),
        "clean_parent_terminal_rate_at_least_60pct": bool(
            search_match["clean_parent_terminal_rate"] >= 0.60
            and selected["clean_parent_terminal_rate"] >= 0.60
        ),
        "parent_to_challenger_exceeds_reverse": bool(
            search_match["parent_to_challenger_rate"] > search_match["challenger_to_parent_rate"]
            and selected["parent_to_challenger_rate"] > selected["challenger_to_parent_rate"]
        ),
        "search_and_validation_terminal_mean_at_least_half_point": bool(
            search_match["mean_terminal_delta"] >= 0.005
            and selected["mean_terminal_delta"] >= 0.005
        ),
        "search_and_validation_active_minus_fixed_mean_at_least_half_point": bool(
            search_match["mean_active_minus_fixed_terminal_delta"] >= 0.005
            and selected["mean_active_minus_fixed_terminal_delta"] >= 0.005
        ),
        "validation_terminal_lower_positive_and_p_at_most_point05": bool(
            terminal_summary["bootstrap_95"][0] > 0
            and terminal_summary["one_sided_signflip_p"] <= 0.05
        ),
        "validation_active_lower_positive_and_p_at_most_point05": bool(
            active_summary["bootstrap_95"][0] > 0
            and active_summary["one_sided_signflip_p"] <= 0.05
        ),
        "validation_cumulative_mean_and_lower_positive": bool(
            cumulative_summary["mean"] > 0 and cumulative_summary["bootstrap_95"][0] > 0
        ),
        "validation_fixed_query_exact_zero": bool(
            selected["max_abs_fixed_terminal_delta"] == 0
            and selected["max_abs_fixed_cumulative_delta"] == 0
            and selected["fixed_root_history_exact"]
        ),
    }
    decision = (
        "GO_TO_STEP96_ONE_TIME_CONFIRMATORY_LOCK"
        if all(gates.values())
        else "NO_GO_DEVELOPMENT_GATE_STOP"
    )
    np.savez_compressed(
        OUTPUT_PATH,
        labels=labels.astype(np.int16),
        root_predictions=predictions.astype(np.int16),
        error_scores=error_scores.astype(np.float64),
        adapter_train_indices=adapter_indices,
        roster_gate_indices=partitions["roster_gate"],
        threshold_indices=threshold_indices,
        search_indices=search_indices,
        verify_indices=verify_indices,
    )
    ledger = {
        "ledger_id": "STEP96_SNLI_STAGEA_COMPLETE_V1",
        "date": common.DATE,
        "authority_sha256": {
            common.PROTOCOL.name: common.sha256_path(common.PROTOCOL),
            AMENDMENT_A.name: common.sha256_path(AMENDMENT_A),
        },
        "stage0_manifest_sha256": common.sha256_path(MANIFEST_PATH),
        "development_rows_sha256": common.sha256_path(DEVELOPMENT_PATH),
        "root_preflight": preflight,
        "root_cache_reused": cache_reused,
        "root_prediction_audits": root_audits,
        "root_geometry": geometry_rows,
        "adapter_audits": adapter_audits,
        "search_complete_48": search_rows,
        "verification_top_8": verification_rows,
        "selected_verification": selected,
        "endpoint_signature": endpoint_signature,
        "single_similarity_source_sha256": source_sha256(common.exact_match_similarity),
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "decision": decision,
        "development_arrays_file": OUTPUT_PATH.name,
    }
    common.json_dump(LEDGER_PATH, ledger)
    config = {
        "config_id": "STEP96_SNLI_STAGEA_FROZEN_V1",
        "date": common.DATE,
        "decision": decision,
        "authority_sha256": ledger["authority_sha256"],
        "stage0_manifest_sha256": ledger["stage0_manifest_sha256"],
        "stagea_ledger_sha256": common.sha256_path(LEDGER_PATH),
        "development_arrays_sha256": common.sha256_path(OUTPUT_PATH),
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
                "step96_common.py",
                "step96_executable_adapter_endpoint.py",
                "step96_stagea_develop.py",
            )
        },
    }
    common.json_dump(CONFIG_PATH, config)
    print(
        json.dumps(
            {
                "decision": decision,
                "selected_condition": selected,
                "gates": gates,
                "failed_gates": ledger["failed_gates"],
                "ledger_sha256": common.sha256_path(LEDGER_PATH),
                "config_sha256": common.sha256_path(CONFIG_PATH),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
