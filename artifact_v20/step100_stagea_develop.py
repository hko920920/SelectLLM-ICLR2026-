from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import beta

import step100_common as common
from step100_executable_adapter_endpoint import execute_aliases_from_raw_text


ROOT = Path(__file__).resolve().parent
DEV = ROOT / f"STEP100_IMDB_DEVELOPMENT_ROWS_{common.DATE}.json"
STAGE0 = ROOT / f"STEP100_STAGE0_DATA_AND_SEAL_MANIFEST_{common.DATE}.json"
AMENDMENT_A = ROOT / f"STEP100_PREREGISTRATION_AMENDMENT_A_GEOMETRY_AND_QUALITY_{common.DATE}.md"
MODEL_DIR = ROOT / "step100_models"
WORK = ROOT / "step100_stagea_work"
ARRAYS = ROOT / f"STEP100_STAGEA_DEVELOPMENT_ARRAYS_{common.DATE}.npz"
GRID = ROOT / f"STEP100_STAGEA_COMPLETE_GRID_{common.DATE}.json"
LEDGER = ROOT / f"STEP100_STAGEA_COMPLETE_LEDGER_{common.DATE}.json"
CONFIG = ROOT / f"STEP100_STAGEA_FROZEN_CONFIG_{common.DATE}.json"


def partition_indices(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    output = {
        name: np.asarray(
            [index for index, row in enumerate(rows) if row["partition"] == name],
            dtype=np.int64,
        )
        for name in common.DEV_QUOTAS_PER_LABEL
    }
    expected = {name: 2 * count for name, count in common.DEV_QUOTAS_PER_LABEL.items()}
    if {name: len(value) for name, value in output.items()} != expected:
        raise AssertionError("development partition drift")
    return output


def infer_roots(texts: list[str]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    cache = WORK / "root_predictions.npz"
    audit_path = WORK / "root_audits.json"
    if cache.is_file() and audit_path.is_file():
        return (
            np.load(cache, allow_pickle=False)["root_predictions"].astype(np.int64),
            json.loads(audit_path.read_text(encoding="utf-8")),
        )
    if cache.exists() or audit_path.exists():
        raise RuntimeError("incomplete Step 100 root cache")
    columns: list[np.ndarray] = []
    audits: list[dict[str, Any]] = []
    for index, spec in enumerate(common.ROOT_SPECS):
        prediction, audit = common.infer_root_predictions(spec, texts)
        columns.append(prediction)
        audits.append(audit)
        print(f"root {index + 1}/{common.ROSTER_SIZE}: {spec.key}", flush=True)
    matrix = np.stack(columns, axis=1)
    np.savez_compressed(cache, root_predictions=matrix.astype(np.int16))
    common.json_dump(audit_path, audits)
    return matrix, audits


def run_condition(
    predictions: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    aliases: np.ndarray,
    seeds: tuple[int, ...],
    budget: int,
    tau: float,
    inference_seeds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    clean_codes = predictions[indices]
    references = labels[indices]
    refined_codes = np.column_stack([clean_codes, aliases[indices]])
    clean_parents = np.arange(common.ROSTER_SIZE, dtype=np.int64)
    refined_parents = np.concatenate(
        [clean_parents, np.zeros(common.N_ALIASES, dtype=np.int64)]
    )
    pools = common.sample_pools(len(indices), seeds, common.POOL_SIZE)
    clean = common.run_active(
        clean_codes, references, clean_parents, clean_codes, pools, seeds, budget, tau
    )
    refined = common.run_active(
        refined_codes, references, refined_parents, clean_codes, pools, seeds, budget, tau
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
        "budget": budget,
        "tau": tau,
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
    if inference_seeds is not None:
        row["inference"] = {
            "terminal": common.effect_summary(terminal, *inference_seeds[0]),
            "active_minus_fixed_terminal": common.effect_summary(active, *inference_seeds[1]),
            "cumulative": common.effect_summary(cumulative, *inference_seeds[2]),
        }
    vectors = {
        "terminal": terminal,
        "active": active,
        "cumulative": cumulative,
        "fixed_terminal": fixed_terminal,
        "fixed_cumulative": fixed_cumulative,
        "pools": pools,
        "clean_queries": clean.queries,
        "refined_queries": refined.queries,
        "clean_roots": clean.roots,
        "refined_roots": refined.roots,
        "fixed_roots": fixed.roots,
    }
    return row, vectors


def oracle_geometry(
    predictions: np.ndarray, labels: np.ndarray, indices: np.ndarray,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    codes = predictions[indices]
    references = labels[indices]
    accuracies = np.mean(codes == references[:, None], axis=0)
    parent_wrong = codes[:, 0] != references
    aliases_local = np.repeat(codes[:, 0, None], common.N_ALIASES, axis=1)
    for alias in range(common.N_ALIASES):
        aliases_local[parent_wrong, alias] = common.N_CLASSES + alias
    aliases = np.zeros((len(labels), common.N_ALIASES), dtype=np.int64)
    aliases[indices] = aliases_local
    effect, _ = run_condition(
        predictions, labels, indices, aliases, seeds, budget=10, tau=0.025
    )
    complementarity = [
        float(np.mean((codes[:, root] == references)[parent_wrong]))
        for root in range(1, common.ROSTER_SIZE)
    ]
    row = {
        "rows": len(indices),
        "root_accuracies": accuracies.tolist(),
        "root_accuracy_by_key": {
            spec.key: float(value) for spec, value in zip(common.ROOT_SPECS, accuracies)
        },
        "parent_minus_best_challenger": float(accuracies[0] - np.max(accuracies[1:])),
        "parent_error_prevalence": float(np.mean(parent_wrong)),
        "max_challenger_correct_given_parent_wrong": max(complementarity),
        "oracle": effect,
    }
    row["gates"] = {
        "parent_gap_at_least_2_5pp": row["parent_minus_best_challenger"] >= 0.025,
        "parent_error_at_least_5pct": row["parent_error_prevalence"] >= 0.05,
        "challenger_complementarity_at_least_15pct": row["max_challenger_correct_given_parent_wrong"] >= 0.15,
        "oracle_terminal_at_least_1pp": effect["mean_terminal_delta"] >= 0.01,
        "oracle_parent_to_challenger_exceeds_reverse": effect["parent_to_challenger_rate"] > effect["challenger_to_parent_rate"],
        "oracle_fixed_exact_zero": effect["max_abs_fixed_terminal_delta"] == 0
        and effect["max_abs_fixed_cumulative_delta"] == 0
        and effect["fixed_root_history_exact"],
    }
    row["eligible"] = all(row["gates"].values())
    return row


def cp_upper(loss_count: int, rows: int) -> float:
    return 1.0 if loss_count >= rows else float(beta.ppf(0.99, loss_count + 1, rows - loss_count))


def thresholds_for_pair(
    parent: np.ndarray, labels: np.ndarray, scores: np.ndarray,
    calibration: np.ndarray, safety: np.ndarray,
    loss_cap: float, trigger_cap: float,
) -> tuple[list[float], list[dict[str, Any]]]:
    thresholds: list[float] = []
    audits: list[dict[str, Any]] = []
    for alias in range(common.N_ALIASES):
        cal_threshold, _, cal_audit = common.threshold_from_caps(
            scores[calibration, alias], parent[calibration] == labels[calibration],
            loss_cap, trigger_cap,
        )
        safe_threshold, _, safe_audit = common.threshold_from_caps(
            scores[safety, alias], parent[safety] == labels[safety],
            0.005, trigger_cap,
        )
        threshold = max(cal_threshold, safe_threshold)
        safe_correct = parent[safety] == labels[safety]
        candidates = np.sort(np.unique(scores[safety, alias]))
        while True:
            trigger = scores[safety, alias] > threshold
            loss_count = int(np.sum(trigger & safe_correct))
            if (
                loss_count <= math.floor(0.005 * len(safety) + 1e-12)
                and cp_upper(loss_count, len(safety)) <= 0.01
                and int(np.sum(trigger)) <= math.floor(trigger_cap * len(safety) + 1e-12)
            ):
                break
            larger = candidates[candidates > threshold]
            threshold = (
                float(larger[0])
                if len(larger)
                else float(np.nextafter(np.max(scores[safety, alias]), np.inf))
            )
        thresholds.append(float(threshold))
        audits.append({
            "alias": alias,
            "calibration": cal_audit,
            "safety_initial": safe_audit,
            "final_threshold": float(threshold),
            "safety_final_trigger_count": int(np.sum(trigger)),
            "safety_final_loss_count": loss_count,
            "safety_final_cp99_upper": cp_upper(loss_count, len(safety)),
        })
    return thresholds, audits


def quality_for_thresholds(
    predictions: np.ndarray, labels: np.ndarray, scores: np.ndarray,
    thresholds: list[float], parts: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any], bool]:
    aliases, triggers = common.make_alias_codes(predictions[:, 0], scores, thresholds)
    quality: dict[str, Any] = {}
    passed = True
    for name, indices in parts.items():
        loss = np.sum(
            triggers[indices] & (predictions[indices, 0, None] == labels[indices, None]),
            axis=0,
        ).astype(np.int64)
        allowed = math.floor(0.01 * len(indices) + 1e-12)
        coordinate_nonimproving = bool(np.all(
            common.exact_match_feedback(aliases[indices], labels[indices])
            <= np.repeat(
                common.exact_match_feedback(predictions[indices, 0, None], labels[indices]),
                common.N_ALIASES, axis=1,
            )
        ))
        row = {
            "rows": len(indices),
            "loss_counts": loss.tolist(),
            "allowed_loss_count": allowed,
            "loss_fractions": (loss / len(indices)).tolist(),
            "trigger_fractions": np.mean(triggers[indices], axis=0).tolist(),
            "coordinate_wise_nonimproving": coordinate_nonimproving,
        }
        quality[name] = row
        passed = passed and bool(np.all(loss <= allowed)) and coordinate_nonimproving
    return aliases, quality, bool(passed)


def main() -> None:
    for path in (ARRAYS, GRID, LEDGER, CONFIG, MODEL_DIR):
        if path.exists():
            raise FileExistsError(path)
    stage0 = json.loads(STAGE0.read_text(encoding="utf-8"))
    payload = json.loads(DEV.read_text(encoding="utf-8"))
    if stage0["protocol_sha256"] != common.sha256_path(common.PROTOCOL):
        raise AssertionError("protocol drift")
    if stage0["development_sha256"] != common.sha256_path(DEV):
        raise AssertionError("development binding drift")
    rows = payload["rows"]
    texts = [str(row["text"]) for row in rows]
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    parts = partition_indices(rows)
    WORK.mkdir(parents=True, exist_ok=True)
    predictions, root_audits = infer_roots(texts)
    geometry = {
        "selector_search": oracle_geometry(
            predictions, labels, parts["selector_search"], common.SEARCH_SEEDS
        ),
        "selector_verify": oracle_geometry(
            predictions, labels, parts["selector_verify"], common.VERIFY_SEEDS
        ),
    }
    geometry_pass = all(row["eligible"] for row in geometry.values())
    if not geometry_pass:
        ledger = {
            "ledger_id": "STEP100_STAGEA_COMPLETE_V1",
            "date": common.DATE,
            "decision": "NO_GO_STEP100_GEOMETRY_STOP",
            "geometry": geometry,
            "sealed_outcome_opened": False,
        }
        common.json_dump(LEDGER, ledger)
        print(json.dumps(ledger, indent=2), flush=True)
        return

    MODEL_DIR.mkdir(parents=True)
    parent = predictions[:, 0]
    train_indices = parts["adapter_train"]
    adapter_audits: list[dict[str, Any]] = []
    adapter_paths: list[str] = []
    score_columns: list[np.ndarray] = []
    for alias in range(common.N_ALIASES):
        model, tokenizer, training = common.train_error_adapter(
            common.ROOT_SPECS[0], [texts[index] for index in train_indices],
            parent[train_indices], labels[train_indices], seed=100100 + alias,
        )
        path = MODEL_DIR / f"step100_imdb_error_adapter_{alias}.safetensors"
        checkpoint = common.save_adapter(model, path, {
            "step": 100, "alias": alias, "seed": 100100 + alias,
            "parent_repo": common.ROOT_SPECS[0].repo_id,
            "parent_revision": common.ROOT_SPECS[0].revision,
        })
        scores = common.infer_error_scores(model, tokenizer, texts)
        audit = {
            "alias": alias,
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "training": training,
            "checkpoint": checkpoint,
            "error_score_sha256": common.array_sha256(scores),
        }
        audit_path = MODEL_DIR / f"step100_imdb_error_adapter_{alias}.audit.json"
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

    threshold_rows: dict[str, Any] = {}
    grid_rows: list[dict[str, Any]] = []
    grid_terminal: list[np.ndarray] = []
    grid_cumulative: list[np.ndarray] = []
    aliases_by_pair: dict[str, np.ndarray] = {}
    quality_by_pair: dict[str, dict[str, Any]] = {}
    for loss_cap in common.LOSS_CAPS:
        for trigger_cap in common.TRIGGER_CAPS:
            pair_id = f"l{loss_cap:.4f}_t{trigger_cap:.3f}"
            thresholds, threshold_audits = thresholds_for_pair(
                parent, labels, scores, parts["threshold_calibration"],
                parts["threshold_safety"], loss_cap, trigger_cap,
            )
            aliases, quality, quality_pass = quality_for_thresholds(
                predictions, labels, scores, thresholds, parts
            )
            aliases_by_pair[pair_id] = aliases
            quality_by_pair[pair_id] = quality
            threshold_rows[pair_id] = {
                "loss_cap": loss_cap,
                "trigger_cap": trigger_cap,
                "thresholds": thresholds,
                "audits": threshold_audits,
                "quality": quality,
                "quality_pass": quality_pass,
            }
            for budget in common.BUDGETS:
                for tau in common.TAUS:
                    effect, vectors = run_condition(
                        predictions, labels, parts["selector_search"], aliases,
                        common.SEARCH_SEEDS, budget, tau,
                    )
                    gates = {
                        "quality": quality_pass,
                        "path_change_at_least_half": effect["path_change_rate"] >= 0.50,
                        "directionality": effect["parent_to_challenger_rate"] > effect["challenger_to_parent_rate"],
                        "terminal_at_least_half_point": effect["mean_terminal_delta"] >= 0.005,
                        "active_at_least_half_point": effect["mean_active_minus_fixed_terminal_delta"] >= 0.005,
                        "cumulative_positive": effect["mean_cumulative_delta"] > 0,
                        "fixed_exact_zero": effect["max_abs_fixed_terminal_delta"] == 0
                        and effect["max_abs_fixed_cumulative_delta"] == 0
                        and effect["fixed_root_history_exact"],
                    }
                    cell_id = f"{pair_id}_b{budget}_tau{tau:.3f}"
                    grid_rows.append({
                        "cell_id": cell_id,
                        "pair_id": pair_id,
                        "loss_cap": loss_cap,
                        "trigger_cap": trigger_cap,
                        "budget": budget,
                        "tau": tau,
                        "effect": effect,
                        "gates": gates,
                        "eligible": all(gates.values()),
                    })
                    grid_terminal.append(vectors["terminal"])
                    grid_cumulative.append(vectors["cumulative"])
            print(f"grid pair {pair_id} complete", flush=True)
    eligible = [row for row in grid_rows if row["eligible"]]
    if eligible:
        selected = max(
            eligible,
            key=lambda row: (
                row["effect"]["mean_terminal_delta"],
                row["effect"]["mean_cumulative_delta"],
                row["effect"]["path_change_rate"],
                -row["loss_cap"], -row["trigger_cap"],
                -row["budget"], -row["tau"],
            ),
        )
    else:
        selected = None

    verify = None
    verify_vectors = None
    verify_gates: dict[str, bool] = {}
    if selected is not None:
        verify, verify_vectors = run_condition(
            predictions, labels, parts["selector_verify"],
            aliases_by_pair[selected["pair_id"]], common.VERIFY_SEEDS,
            int(selected["budget"]), float(selected["tau"]),
            inference_seeds=((100700, 100701), (100701, 100702), (100702, 100703)),
        )
        inference = verify["inference"]
        verify_gates = {
            "quality": threshold_rows[selected["pair_id"]]["quality_pass"],
            "path_change_at_least_half": verify["path_change_rate"] >= 0.50,
            "directionality": verify["parent_to_challenger_rate"] > verify["challenger_to_parent_rate"],
            "terminal_at_least_half_point": verify["mean_terminal_delta"] >= 0.005,
            "active_at_least_half_point": verify["mean_active_minus_fixed_terminal_delta"] >= 0.005,
            "terminal_inference_positive": inference["terminal"]["bootstrap_95"][0] > 0
            and inference["terminal"]["one_sided_signflip_p"] <= 0.05,
            "active_inference_positive": inference["active_minus_fixed_terminal"]["bootstrap_95"][0] > 0
            and inference["active_minus_fixed_terminal"]["one_sided_signflip_p"] <= 0.05,
            "cumulative_inference_positive": inference["cumulative"]["mean"] > 0
            and inference["cumulative"]["bootstrap_95"][0] > 0,
            "fixed_exact_zero": verify["max_abs_fixed_terminal_delta"] == 0
            and verify["max_abs_fixed_cumulative_delta"] == 0
            and verify["fixed_root_history_exact"],
        }
    development_go = selected is not None and all(verify_gates.values())
    decision = "GO_STEP100_TO_ONE_TIME_SEALED_PRIMARY" if development_go else "NO_GO_STEP100_DEVELOPMENT_STOP"

    arrays: dict[str, np.ndarray] = {
        "labels": labels.astype(np.int16),
        "root_predictions": predictions.astype(np.int16),
        "error_scores": scores.astype(np.float64),
        "grid_terminal": np.stack(grid_terminal),
        "grid_cumulative": np.stack(grid_cumulative),
    }
    for name, indices in parts.items():
        arrays[f"{name}_indices"] = indices
    if verify_vectors is not None:
        for name, value in verify_vectors.items():
            arrays[f"verify_{name}"] = value
    np.savez_compressed(ARRAYS, **arrays)
    common.json_dump(GRID, {
        "grid_id": "STEP100_COMPLETE_81_CELL_GRID_V1",
        "rows": grid_rows,
        "selected_cell_id": selected["cell_id"] if selected else None,
        "selection_rule": "max terminal, cumulative, path; then smaller loss, trigger, budget, tau",
    })
    all_code = (
        "step93_common.py", "step100_common.py",
        "step100_executable_adapter_endpoint.py", "step100_stagea_develop.py",
    )
    ledger = {
        "ledger_id": "STEP100_STAGEA_COMPLETE_V1",
        "date": common.DATE,
        "decision": decision,
        "protocol_sha256": common.sha256_path(common.PROTOCOL),
        "amendment_a_sha256": common.sha256_path(AMENDMENT_A),
        "stage0_sha256": common.sha256_path(STAGE0),
        "development_sha256": common.sha256_path(DEV),
        "geometry": geometry,
        "root_audits": root_audits,
        "adapter_audits": adapter_audits,
        "threshold_pairs": threshold_rows,
        "grid_file": GRID.name,
        "grid_sha256": common.sha256_path(GRID),
        "eligible_cell_count": len(eligible),
        "selected": selected,
        "verification": verify,
        "verification_gates": verify_gates,
        "failed_verification_gates": [name for name, passed in verify_gates.items() if not passed],
        "endpoint_signature": str(inspect.signature(execute_aliases_from_raw_text)),
        "arrays_file": ARRAYS.name,
        "arrays_sha256": common.sha256_path(ARRAYS),
        "code_sha256": {name: common.sha256_path(ROOT / name) for name in all_code},
        "sealed_outcome_opened": False,
    }
    common.json_dump(LEDGER, ledger)
    selected_pair = threshold_rows[selected["pair_id"]] if selected else None
    config = {
        "config_id": "STEP100_STAGEA_FROZEN_CONFIG_V1",
        "date": common.DATE,
        "decision": decision,
        "stagea_ledger_sha256": common.sha256_path(LEDGER),
        "arrays_sha256": common.sha256_path(ARRAYS),
        "grid_sha256": common.sha256_path(GRID),
        "selected": selected,
        "thresholds": selected_pair["thresholds"] if selected_pair else None,
        "adapter_paths": adapter_paths,
        "adapter_sha256": {row["path"]: row["checkpoint"]["sha256"] for row in adapter_audits},
        "pool_size": common.POOL_SIZE,
        "test_seed_start": common.TEST_SEEDS[0],
        "test_seed_stop_exclusive": common.TEST_SEEDS[-1] + 1,
        "test_seed_count": len(common.TEST_SEEDS),
        "test_input": stage0["input_file"],
        "test_input_sha256": stage0["input_sha256"],
        "sealed_outcome": stage0["sealed_file"],
        "sealed_outcome_sha256": stage0["sealed_sha256"],
        "construction_access": {
            "development_labels": True,
            "heldout_reference": False,
            "heldout_item_lookup": False,
            "peer_outputs_in_training_or_runtime": False,
            "runtime_fields": ["text"],
        },
    }
    common.json_dump(CONFIG, config)
    print(json.dumps({
        "decision": decision,
        "geometry": {
            name: {
                "parent_gap_pp": 100 * row["parent_minus_best_challenger"],
                "oracle_terminal_pp": 100 * row["oracle"]["mean_terminal_delta"],
                "eligible": row["eligible"],
            }
            for name, row in geometry.items()
        },
        "eligible_cell_count": len(eligible),
        "selected_cell": selected["cell_id"] if selected else None,
        "search_terminal_pp": 100 * selected["effect"]["mean_terminal_delta"] if selected else None,
        "verify_terminal_pp": 100 * verify["mean_terminal_delta"] if verify else None,
        "verify_terminal_ci_pp": [100 * value for value in verify["inference"]["terminal"]["bootstrap_95"]] if verify else None,
        "failed_verification_gates": ledger["failed_verification_gates"],
        "sealed_outcome_opened": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
