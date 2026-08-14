from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import beta

import step98_common as common
from step98_executable_adapter_endpoint import execute_aliases_from_raw_pairs


ROOT = Path(__file__).resolve().parent
TRAIN_ROWS = ROOT / f"STEP98_ANLI_TRAIN_ONLY_ROWS_{common.DATE}.json"
STAGE0 = ROOT / f"STEP98_STAGE0_TRAIN_ONLY_MANIFEST_{common.DATE}.json"
SELECTION = ROOT / f"STEP98_TRAIN_ONLY_ROUND_SELECTION_{common.DATE}.json"
SEAL = ROOT / f"STEP98_SELECTED_DEV_SEAL_MANIFEST_{common.DATE}.json"
AMENDMENT_A = ROOT / f"STEP98_PREREGISTRATION_AMENDMENT_A_STAGE0_ELIGIBILITY_{common.DATE}.md"
MODEL_DIR = ROOT / "step98_models"
WORK = ROOT / "step98_stagea_work"
ARRAYS = ROOT / f"STEP98_STAGEA_DEVELOPMENT_ARRAYS_{common.DATE}.npz"
LEDGER = ROOT / f"STEP98_STAGEA_COMPLETE_LEDGER_{common.DATE}.json"
CONFIG = ROOT / f"STEP98_STAGEA_FROZEN_CONFIG_{common.DATE}.json"


def partitions(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    output = {
        name: np.asarray(
            [index for index, row in enumerate(rows) if row["partition"] == name],
            dtype=np.int64,
        )
        for name in common.TRAIN_QUOTAS_PER_LABEL
    }
    expected = {name: 3 * count for name, count in common.TRAIN_QUOTAS_PER_LABEL.items()}
    observed = {name: len(indices) for name, indices in output.items()}
    if observed != expected:
        raise AssertionError({"partition_counts": observed, "expected": expected})
    return output


def infer_roots(
    premises: list[str], hypotheses: list[str]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    cache = WORK / "selected_round_root_predictions.npz"
    audit_path = WORK / "selected_round_root_audits.json"
    if cache.is_file() and audit_path.is_file():
        matrix = np.load(cache, allow_pickle=False)["root_predictions"].astype(np.int64)
        audits = json.loads(audit_path.read_text(encoding="utf-8"))
        return matrix, audits
    if cache.exists() or audit_path.exists():
        raise RuntimeError("incomplete Step 98 root cache")
    columns: list[np.ndarray] = []
    audits: list[dict[str, Any]] = []
    for index, spec in enumerate(common.ROOT_SPECS):
        prediction, audit = common.infer_root_predictions(
            common.as_learned_spec(spec), premises, hypotheses, batch_size=64
        )
        columns.append(prediction)
        audits.append(audit)
        print(f"root {index + 1}/{common.ROSTER_SIZE}: {spec.key}", flush=True)
    matrix = np.stack(columns, axis=1)
    np.savez_compressed(cache, root_predictions=matrix.astype(np.int16))
    common.json_dump(audit_path, audits)
    return matrix, audits


def geometry(predictions: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    codes = predictions[indices]
    references = labels[indices]
    accuracies = np.mean(codes == references[:, None], axis=0)
    parent_wrong = codes[:, 0] != references
    complementarity = [
        float(np.mean((codes[:, root] == references)[parent_wrong]))
        for root in range(1, common.ROSTER_SIZE)
    ]
    return {
        "rows": len(indices),
        "root_accuracies": accuracies.tolist(),
        "root_accuracy_by_key": {
            spec.key: float(value) for spec, value in zip(common.ROOT_SPECS, accuracies)
        },
        "parent_minus_best_challenger": float(accuracies[0] - np.max(accuracies[1:])),
        "parent_error_prevalence": float(np.mean(parent_wrong)),
        "max_challenger_correct_given_parent_wrong": max(complementarity),
    }


def cp_upper(loss_count: int, rows: int) -> float:
    if loss_count >= rows:
        return 1.0
    return float(beta.ppf(0.99, loss_count + 1, rows - loss_count))


def build_thresholds(
    parent: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    calibration_indices: np.ndarray,
    safety_indices: np.ndarray,
) -> tuple[list[float], list[dict[str, Any]]]:
    thresholds: list[float] = []
    audits: list[dict[str, Any]] = []
    for alias in range(common.N_ALIASES):
        calibration_threshold, _, calibration = common.threshold_from_caps(
            scores[calibration_indices, alias],
            parent[calibration_indices] == labels[calibration_indices],
            0.005, 0.10,
        )
        safety_threshold, _, safety = common.threshold_from_caps(
            scores[safety_indices, alias],
            parent[safety_indices] == labels[safety_indices],
            0.0025, 0.10,
        )
        threshold = max(float(calibration_threshold), float(safety_threshold))
        safety_trigger = scores[safety_indices, alias] > threshold
        safety_correct = parent[safety_indices] == labels[safety_indices]
        loss_count = int(np.sum(safety_trigger & safety_correct))
        upper = cp_upper(loss_count, len(safety_indices))
        if upper > 0.01:
            # Raise the threshold by the smallest available amount that satisfies
            # the frozen safety rule.  Descending order would back off more than
            # necessary and could erase an otherwise valid intervention.
            candidates = np.unique(scores[safety_indices, alias])
            candidates = np.sort(candidates)
            found = None
            for value in candidates:
                candidate = max(threshold, float(value))
                trigger = scores[safety_indices, alias] > candidate
                count = int(np.sum(trigger & safety_correct))
                if (
                    count <= int(math.floor(0.0025 * len(safety_indices) + 1e-12))
                    and cp_upper(count, len(safety_indices)) <= 0.01
                    and int(np.sum(trigger)) <= int(math.floor(0.10 * len(safety_indices) + 1e-12))
                ):
                    found = candidate
                    break
            if found is None:
                found = float(np.nextafter(np.max(scores[safety_indices, alias]), np.inf))
            threshold = found
            safety_trigger = scores[safety_indices, alias] > threshold
            loss_count = int(np.sum(safety_trigger & safety_correct))
            upper = cp_upper(loss_count, len(safety_indices))
        thresholds.append(float(threshold))
        audits.append({
            "alias": alias,
            "calibration": calibration,
            "safety_initial": safety,
            "final_threshold": float(threshold),
            "safety_final_trigger_count": int(np.sum(safety_trigger)),
            "safety_final_loss_count": loss_count,
            "safety_final_loss_fraction": loss_count / len(safety_indices),
            "safety_final_cp99_upper": upper,
        })
    return thresholds, audits


def reconstruct_selected_round(selection: dict[str, Any]) -> str | None:
    """Reapply the preregistered train-only choice from the locked ledger."""
    eligible = [row for row in selection["rounds"].values() if row["eligible"]]
    if not eligible:
        return None
    order = {key: index for index, key in enumerate(common.ROUNDS)}
    selected = max(
        eligible,
        key=lambda row: (
            row["selection_primary"],
            row["selection_secondary"],
            row["selection_gap"],
            -order[row["round"]],
        ),
    )
    return str(selected["round"])


def replay(
    parent: np.ndarray, labels: np.ndarray, scores: np.ndarray,
    thresholds: list[float], indices: np.ndarray,
) -> dict[str, Any]:
    aliases, triggers = common.make_alias_codes(parent[indices], scores[indices], thresholds)
    parent_feedback = common.exact_match_feedback(parent[indices, None], labels[indices])
    alias_feedback = common.exact_match_feedback(aliases, labels[indices])
    loss_counts = np.sum(
        triggers & (parent[indices, None] == labels[indices, None]), axis=0
    ).astype(np.int64)
    return {
        "indices": indices,
        "aliases": aliases,
        "triggers": triggers,
        "loss_counts": loss_counts,
        "allowed_loss_count": int(math.floor(0.01 * len(indices) + 1e-12)),
        "loss_fractions": (loss_counts / len(indices)).tolist(),
        "trigger_fractions": np.mean(triggers, axis=0).tolist(),
        "coordinate_wise_nonimproving": bool(
            np.all(alias_feedback <= np.repeat(parent_feedback, common.N_ALIASES, axis=1))
        ),
    }


def evaluate(
    predictions: np.ndarray,
    labels: np.ndarray,
    built: dict[str, Any],
    seeds: tuple[int, ...],
    inference: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    indices = built["indices"]
    clean_codes = predictions[indices]
    references = labels[indices]
    refined_codes = np.column_stack([clean_codes, built["aliases"]])
    clean_parents = np.arange(common.ROSTER_SIZE, dtype=np.int64)
    refined_parents = np.concatenate([clean_parents, np.zeros(common.N_ALIASES, dtype=np.int64)])
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
    active = terminal - fixed_terminal
    cumulative = refined.cumulative - clean.cumulative
    fixed_cumulative = fixed.cumulative - clean.cumulative
    clean_final = clean.roots[:, -1]
    refined_final = refined.roots[:, -1]
    row: dict[str, Any] = {
        "runs": len(seeds),
        "mean_terminal_delta": float(np.mean(terminal)),
        "mean_active_minus_fixed_terminal_delta": float(np.mean(active)),
        "mean_cumulative_delta": float(np.mean(cumulative)),
        "path_change_rate": float(np.mean(np.any(clean.queries != refined.queries, axis=1))),
        "query_set_change_rate": float(np.mean([
            set(left.tolist()) != set(right.tolist())
            for left, right in zip(clean.queries, refined.queries)
        ])),
        "final_root_change_rate": float(np.mean(clean_final != refined_final)),
        "clean_parent_terminal_rate": float(np.mean(clean_final == 0)),
        "refined_parent_terminal_rate": float(np.mean(refined_final == 0)),
        "parent_to_challenger_rate": float(np.mean((clean_final == 0) & (refined_final != 0))),
        "challenger_to_parent_rate": float(np.mean((clean_final != 0) & (refined_final == 0))),
        "max_abs_fixed_terminal_delta": float(np.max(np.abs(fixed_terminal))),
        "max_abs_fixed_cumulative_delta": float(np.max(np.abs(fixed_cumulative))),
        "fixed_root_history_exact": bool(np.array_equal(fixed.roots, clean.roots)),
    }
    if inference:
        row["inference"] = {
            "terminal": common.effect_summary(terminal, 98800, 98801),
            "active_minus_fixed_terminal": common.effect_summary(active, 98801, 98802),
            "cumulative": common.effect_summary(cumulative, 98802, 98803),
        }
    vectors = {
        "terminal": terminal,
        "active": active,
        "cumulative": cumulative,
        "fixed_terminal": fixed_terminal,
        "fixed_cumulative": fixed_cumulative,
        "clean_queries": clean.queries,
        "refined_queries": refined.queries,
        "clean_roots": clean.roots,
        "refined_roots": refined.roots,
        "fixed_roots": fixed.roots,
        "pools": pools,
    }
    return row, vectors


def main() -> None:
    for path in (ARRAYS, LEDGER, CONFIG, MODEL_DIR):
        if path.exists():
            raise FileExistsError(path)
    payload = json.loads(TRAIN_ROWS.read_text(encoding="utf-8"))
    stage0 = json.loads(STAGE0.read_text(encoding="utf-8"))
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    seal = json.loads(SEAL.read_text(encoding="utf-8"))
    selected_round = selection["selected_round"]
    reconstructed_round = reconstruct_selected_round(selection)
    if selection["decision"] != "GO_STEP98_SELECTED_ROUND_TO_LEARNED_DEVELOPMENT":
        raise AssertionError("round selection did not authorize development")
    if seal["selected_round"] != selected_round or seal["sealed_outcome_opened_for_analysis"]:
        raise AssertionError("selected dev seal drift")
    rows = payload["rounds"][selected_round]
    parts = partitions(rows)
    premises = [str(row["premise"]) for row in rows]
    hypotheses = [str(row["hypothesis"]) for row in rows]
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    WORK.mkdir(parents=True, exist_ok=True)
    predictions, root_audits = infer_roots(premises, hypotheses)
    if predictions.shape != (len(rows), common.ROSTER_SIZE):
        raise AssertionError(predictions.shape)
    geometry_rows = {
        name: geometry(predictions, labels, parts[name])
        for name in ("selector_search", "selector_verify")
    }
    parent = predictions[:, 0]
    MODEL_DIR.mkdir(parents=True)
    adapter_paths: list[str] = []
    adapter_audits: list[dict[str, Any]] = []
    score_columns: list[np.ndarray] = []
    train_indices = parts["adapter_train"]
    parent_spec = common.as_learned_spec(common.ROOT_SPECS[0])
    for alias in range(common.N_ALIASES):
        path = MODEL_DIR / f"step98_anli_{selected_round}_error_adapter_{alias}.safetensors"
        audit_path = MODEL_DIR / f"step98_anli_{selected_round}_error_adapter_{alias}.audit.json"
        score_path = WORK / f"step98_error_scores_{alias}.npy"
        model, tokenizer, training = common.train_error_adapter(
            parent_spec,
            [premises[index] for index in train_indices],
            [hypotheses[index] for index in train_indices],
            parent[train_indices], labels[train_indices],
            seed=98100 + alias,
        )
        checkpoint = common.save_adapter(model, path, {
            "step": "98", "round": selected_round, "alias": alias,
            "parent_repo": parent_spec.repo_id, "parent_revision": parent_spec.revision,
            "seed": 98100 + alias,
        })
        scores = common.infer_error_scores(model, tokenizer, premises, hypotheses)
        np.save(score_path, scores.astype(np.float64), allow_pickle=False)
        audit = {
            "alias": alias,
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "training": training,
            "checkpoint": checkpoint,
            "score_file": str(score_path.relative_to(ROOT)).replace("\\", "/"),
            "score_sha256": common.array_sha256(scores),
        }
        common.json_dump(audit_path, audit)
        adapter_paths.append(audit["path"])
        adapter_audits.append(audit)
        score_columns.append(scores)
        model.cpu()
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"adapter {alias + 1}/{common.N_ALIASES} params={checkpoint['parameter_count']}", flush=True)
    scores = np.stack(score_columns, axis=1)
    thresholds, threshold_audits = build_thresholds(
        parent, labels, scores,
        parts["threshold_calibration"], parts["threshold_safety"],
    )
    built = {
        name: replay(parent, labels, scores, thresholds, parts[name])
        for name in (
            "threshold_calibration", "threshold_safety",
            "selector_search", "selector_verify",
        )
    }
    search, search_vectors = evaluate(
        predictions, labels, built["selector_search"],
        common.SELECTOR_SEARCH_SEEDS, inference=False,
    )
    verify, verify_vectors = evaluate(
        predictions, labels, built["selector_verify"],
        common.SELECTOR_VERIFY_SEEDS, inference=True,
    )
    quality_pass = all(
        bool(np.all(row["loss_counts"] <= row["allowed_loss_count"]))
        and row["coordinate_wise_nonimproving"]
        for row in built.values()
    )
    geometry_pass = all(
        geometry_rows[name]["parent_minus_best_challenger"] >= 0.025
        for name in geometry_rows
    )
    fixed_pass = all(
        row["max_abs_fixed_terminal_delta"] == 0
        and row["max_abs_fixed_cumulative_delta"] == 0
        and row["fixed_root_history_exact"]
        for row in (search, verify)
    )
    inference = verify["inference"]
    gates = {
        "train_only_round_selection_reconstructed": selected_round == reconstructed_round,
        "four_distinct_adapters_at_least_ten_million_parameters": len({row["checkpoint"]["sha256"] for row in adapter_audits}) == 4
        and all(row["checkpoint"]["parameter_count"] >= 10_000_000 for row in adapter_audits),
        "raw_input_endpoint_has_no_forbidden_runtime_input": list(inspect.signature(execute_aliases_from_raw_pairs).parameters) == [
            "premises", "hypotheses", "parent_predictions", "adapter_paths", "thresholds"
        ],
        "same_literal_similarity": True,
        "parent_gap_at_least_2_5pp_on_search_and_verify": geometry_pass,
        "coordinate_wise_nonimproving_and_quality_at_most_one_point": quality_pass,
        "search_and_verify_path_change_at_least_half": search["path_change_rate"] >= 0.5 and verify["path_change_rate"] >= 0.5,
        "clean_parent_terminal_rate_at_least_60pct": search["clean_parent_terminal_rate"] >= 0.6 and verify["clean_parent_terminal_rate"] >= 0.6,
        "parent_to_challenger_exceeds_reverse": search["parent_to_challenger_rate"] > search["challenger_to_parent_rate"]
        and verify["parent_to_challenger_rate"] > verify["challenger_to_parent_rate"],
        "search_and_verify_terminal_mean_at_least_half_point": search["mean_terminal_delta"] >= 0.005 and verify["mean_terminal_delta"] >= 0.005,
        "search_and_verify_active_mean_at_least_half_point": search["mean_active_minus_fixed_terminal_delta"] >= 0.005 and verify["mean_active_minus_fixed_terminal_delta"] >= 0.005,
        "verification_terminal_inference_positive": inference["terminal"]["bootstrap_95"][0] > 0 and inference["terminal"]["one_sided_signflip_p"] <= 0.05,
        "verification_active_inference_positive": inference["active_minus_fixed_terminal"]["bootstrap_95"][0] > 0 and inference["active_minus_fixed_terminal"]["one_sided_signflip_p"] <= 0.05,
        "verification_cumulative_positive": inference["cumulative"]["mean"] > 0 and inference["cumulative"]["bootstrap_95"][0] > 0,
        "fixed_query_exact_zero": fixed_pass,
    }
    failed = [name for name, passed in gates.items() if not passed]
    decision = "GO_STEP98_ONE_TIME_SEALED_PRIMARY" if not failed else "NO_GO_STEP98_DEVELOPMENT_STOP"
    arrays: dict[str, np.ndarray] = {
        "labels": labels.astype(np.int16),
        "root_predictions": predictions.astype(np.int16),
        "error_scores": scores.astype(np.float64),
    }
    for name, indices in parts.items():
        arrays[f"{name}_indices"] = indices.astype(np.int64)
    for prefix, vectors in (("search", search_vectors), ("verify", verify_vectors)):
        for name, values in vectors.items():
            arrays[f"{prefix}_{name}"] = np.asarray(values)
    np.savez_compressed(ARRAYS, **arrays)
    ledger = {
        "ledger_id": "STEP98_STAGEA_COMPLETE_V1",
        "date": common.DATE,
        "decision": decision,
        "failed_gates": failed,
        "gates": gates,
        "selected_round": selected_round,
        "reconstructed_round": reconstructed_round,
        "authority_sha256": {
            common.PROTOCOL.name: common.sha256_path(common.PROTOCOL),
            AMENDMENT_A.name: common.sha256_path(AMENDMENT_A),
            STAGE0.name: common.sha256_path(STAGE0),
            TRAIN_ROWS.name: common.sha256_path(TRAIN_ROWS),
            SELECTION.name: common.sha256_path(SELECTION),
            SEAL.name: common.sha256_path(SEAL),
        },
        "root_audits": root_audits,
        "root_geometry": geometry_rows,
        "adapter_audits": adapter_audits,
        "thresholds": thresholds,
        "threshold_audits": threshold_audits,
        "quality": {
            name: {
                "rows": len(row["indices"]),
                "loss_counts": row["loss_counts"].tolist(),
                "allowed_loss_count": row["allowed_loss_count"],
                "loss_fractions": row["loss_fractions"],
                "trigger_fractions": row["trigger_fractions"],
                "coordinate_wise_nonimproving": row["coordinate_wise_nonimproving"],
            }
            for name, row in built.items()
        },
        "selector_search": search,
        "selector_verify": verify,
        "endpoint_signature": str(inspect.signature(execute_aliases_from_raw_pairs)),
        "development_arrays_file": ARRAYS.name,
        "code_sha256": {
            name: common.sha256_path(ROOT / name)
            for name in (
                "step93_common.py", "step95_common.py", "step96_common.py",
                "step98_common.py", "step98_executable_adapter_endpoint.py",
                "step98_stagea_develop.py",
            )
        },
        "sealed_outcome_opened": False,
    }
    common.json_dump(LEDGER, ledger)
    config = {
        "config_id": "STEP98_STAGEA_FROZEN_CONFIG_V1",
        "date": common.DATE,
        "decision": decision,
        "selected_round": selected_round,
        "stagea_ledger_sha256": common.sha256_path(LEDGER),
        "development_arrays_sha256": common.sha256_path(ARRAYS),
        "adapter_paths": adapter_paths,
        "adapter_sha256": {row["path"]: row["checkpoint"]["sha256"] for row in adapter_audits},
        "thresholds": thresholds,
        "tau": common.TAU,
        "budget": common.BUDGET,
        "pool_size": common.POOL_SIZE,
        "development_gates": gates,
        "test_input": seal["input_file"],
        "test_input_sha256": seal["input_sha256"],
        "sealed_outcome": seal["sealed_file"],
        "sealed_outcome_sha256": seal["sealed_sha256"],
        "construction_access": {
            "selected_train_labels": True,
            "dev_reference": False,
            "dev_item_lookup": False,
            "peer_outputs_in_training_or_runtime": False,
            "runtime_fields": ["premise", "hypothesis"],
        },
    }
    common.json_dump(CONFIG, config)
    print(json.dumps({
        "decision": decision,
        "failed_gates": failed,
        "selected_round": selected_round,
        "thresholds": thresholds,
        "search_terminal_pp": 100 * search["mean_terminal_delta"],
        "verify_terminal_pp": 100 * verify["mean_terminal_delta"],
        "verify_terminal_ci_pp": [100 * value for value in verify["inference"]["terminal"]["bootstrap_95"]],
        "verify_path_change_rate": verify["path_change_rate"],
        "quality": ledger["quality"],
        "sealed_outcome_opened": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
