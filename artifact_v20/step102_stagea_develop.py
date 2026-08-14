from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import step102_common as common
from step100_stagea_develop import (
    partition_indices,
    quality_for_thresholds,
    run_condition,
    thresholds_for_pair,
)
from step102_executable_adapter_endpoint import execute_aliases_from_raw_text


ROOT = Path(__file__).resolve().parent
DEV = ROOT / f"STEP100_IMDB_DEVELOPMENT_ROWS_{common.DATE}.json"
STAGE0 = ROOT / f"STEP100_STAGE0_DATA_AND_SEAL_MANIFEST_{common.DATE}.json"
STEP100_LEDGER = ROOT / f"STEP100_STAGEA_COMPLETE_LEDGER_{common.DATE}.json"
STEP101 = ROOT / f"STEP101_IMDB_DEVELOPMENT_PARENT_ROSTER_SELECTION_{common.DATE}.json"
LEGACY = ROOT / "step100_stagea_work" / "root_predictions.npz"
PARENT = ROOT / "step101_candidate_predictions" / "wrmurray_roberta.npz"
MODEL_DIR = ROOT / "step102_models"
ARRAYS = ROOT / f"STEP102_STAGEA_DEVELOPMENT_ARRAYS_{common.DATE}.npz"
GRID = ROOT / f"STEP102_STAGEA_COMPLETE_GRID_{common.DATE}.json"
LEDGER = ROOT / f"STEP102_STAGEA_COMPLETE_LEDGER_{common.DATE}.json"
CONFIG = ROOT / f"STEP102_STAGEA_FROZEN_CONFIG_{common.DATE}.json"


def oracle_geometry(
    predictions: np.ndarray, labels: np.ndarray, indices: np.ndarray,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    local = predictions[indices]
    references = labels[indices]
    accuracies = np.mean(local == references[:, None], axis=0)
    parent_wrong = local[:, 0] != references
    local_aliases = np.repeat(local[:, 0, None], common.N_ALIASES, axis=1)
    for alias in range(common.N_ALIASES):
        local_aliases[parent_wrong, alias] = common.N_CLASSES + alias
    aliases = np.zeros((len(labels), common.N_ALIASES), dtype=np.int64)
    aliases[indices] = local_aliases
    effect, _ = run_condition(
        predictions, labels, indices, aliases, seeds, budget=10, tau=0.025
    )
    complementarity = [
        float(np.mean((local[:, root] == references)[parent_wrong]))
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
        "parent_error_at_least_3pct": row["parent_error_prevalence"] >= 0.03,
        "challenger_complementarity_at_least_15pct": row["max_challenger_correct_given_parent_wrong"] >= 0.15,
        "oracle_terminal_at_least_1pp": effect["mean_terminal_delta"] >= 0.01,
        "path_change_at_least_half": effect["path_change_rate"] >= 0.50,
        "directionality": effect["parent_to_challenger_rate"] > effect["challenger_to_parent_rate"],
        "fixed_exact_zero": effect["max_abs_fixed_terminal_delta"] == 0
        and effect["max_abs_fixed_cumulative_delta"] == 0
        and effect["fixed_root_history_exact"],
    }
    row["eligible"] = all(row["gates"].values())
    return row


def main() -> None:
    for path in (MODEL_DIR, ARRAYS, GRID, LEDGER, CONFIG):
        if path.exists():
            raise FileExistsError(path)
    step100 = json.loads(STEP100_LEDGER.read_text(encoding="utf-8"))
    step101 = json.loads(STEP101.read_text(encoding="utf-8"))
    if step100["decision"] != "NO_GO_STEP100_GEOMETRY_STOP":
        raise AssertionError("Step 100 non-relabeling drift")
    if step101["decision"] != "NO_GO_STEP101_PARENT_GEOMETRY_STOP":
        raise AssertionError("Step 101 non-relabeling drift")
    payload = json.loads(DEV.read_text(encoding="utf-8"))
    rows = payload["rows"]
    texts = [str(row["text"]) for row in rows]
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    parts = partition_indices(rows)
    legacy = np.load(LEGACY, allow_pickle=False)["root_predictions"].astype(np.int64)
    parent = np.load(PARENT, allow_pickle=False)["predictions"].astype(np.int64)
    predictions = np.column_stack([parent, legacy[:, 0], legacy[:, 1], legacy[:, 3]])
    geometry = {
        "selector_search": oracle_geometry(
            predictions, labels, parts["selector_search"], common.SEARCH_SEEDS
        ),
        "selector_verify": oracle_geometry(
            predictions, labels, parts["selector_verify"], common.VERIFY_SEEDS
        ),
    }
    print(json.dumps({
        "geometry_confirmation": {
            name: {
                "gap_pp": 100 * row["parent_minus_best_challenger"],
                "oracle_terminal_pp": 100 * row["oracle"]["mean_terminal_delta"],
                "path_change_rate": row["oracle"]["path_change_rate"],
                "eligible": row["eligible"],
            }
            for name, row in geometry.items()
        },
        "sealed_outcome_opened": False,
    }, indent=2), flush=True)
    if not all(row["eligible"] for row in geometry.values()):
        output = {
            "ledger_id": "STEP102_STAGEA_COMPLETE_V1",
            "date": common.DATE,
            "decision": "NO_GO_STEP102_GEOMETRY_CONFIRMATION_STOP",
            "geometry": geometry,
            "sealed_outcome_opened": False,
        }
        common.json_dump(LEDGER, output)
        print(json.dumps(output, indent=2), flush=True)
        return

    MODEL_DIR.mkdir(parents=True)
    train_indices = parts["adapter_train"]
    adapter_audits: list[dict[str, Any]] = []
    adapter_paths: list[str] = []
    score_columns: list[np.ndarray] = []
    for alias in range(common.N_ALIASES):
        model, tokenizer, training = common.train_error_adapter(
            [texts[index] for index in train_indices],
            parent[train_indices], labels[train_indices], seed=102100 + alias,
        )
        path = MODEL_DIR / f"step102_imdb_wrmurray_error_adapter_{alias}.safetensors"
        checkpoint = common.save_adapter(model, path, {
            "step": 102, "alias": alias, "seed": 102100 + alias,
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
        common.json_dump(
            MODEL_DIR / f"step102_imdb_wrmurray_error_adapter_{alias}.audit.json", audit
        )
        adapter_audits.append(audit)
        adapter_paths.append(audit["path"])
        score_columns.append(scores)
        model.cpu()
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"adapter {alias + 1}/{common.N_ALIASES} params={checkpoint['parameter_count']}", flush=True)
    scores = np.stack(score_columns, axis=1)

    threshold_rows: dict[str, Any] = {}
    aliases_by_pair: dict[str, np.ndarray] = {}
    grid_rows: list[dict[str, Any]] = []
    grid_terminal: list[np.ndarray] = []
    grid_cumulative: list[np.ndarray] = []
    for loss_cap in common.LOSS_CAPS:
        for trigger_cap in common.TRIGGER_CAPS:
            pair_id = f"l{loss_cap:.4f}_t{trigger_cap:.3f}"
            thresholds, audits = thresholds_for_pair(
                parent, labels, scores, parts["threshold_calibration"],
                parts["threshold_safety"], loss_cap, trigger_cap,
            )
            aliases, quality, quality_pass = quality_for_thresholds(
                predictions, labels, scores, thresholds, parts
            )
            aliases_by_pair[pair_id] = aliases
            threshold_rows[pair_id] = {
                "loss_cap": loss_cap,
                "trigger_cap": trigger_cap,
                "thresholds": thresholds,
                "audits": audits,
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
                    grid_rows.append({
                        "cell_id": f"{pair_id}_b{budget}_tau{tau:.3f}",
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
    selected = max(
        eligible,
        key=lambda row: (
            row["effect"]["mean_terminal_delta"],
            row["effect"]["mean_cumulative_delta"],
            row["effect"]["path_change_rate"],
            -row["loss_cap"], -row["trigger_cap"],
            -row["budget"], -row["tau"],
        ),
    ) if eligible else None
    verify = None
    verify_vectors = None
    verify_gates: dict[str, bool] = {}
    if selected:
        verify, verify_vectors = run_condition(
            predictions, labels, parts["selector_verify"],
            aliases_by_pair[selected["pair_id"]], common.VERIFY_SEEDS,
            selected["budget"], selected["tau"],
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
    decision = "GO_STEP102_TO_HELDOUT_LOCK" if selected and all(verify_gates.values()) else "NO_GO_STEP102_LEARNED_DEVELOPMENT_STOP"

    arrays: dict[str, np.ndarray] = {
        "labels": labels.astype(np.int16),
        "root_predictions": predictions.astype(np.int16),
        "error_scores": scores,
        "grid_terminal": np.stack(grid_terminal),
        "grid_cumulative": np.stack(grid_cumulative),
    }
    for name, indices in parts.items():
        arrays[f"{name}_indices"] = indices
    if verify_vectors:
        for name, value in verify_vectors.items():
            arrays[f"verify_{name}"] = value
    np.savez_compressed(ARRAYS, **arrays)
    common.json_dump(GRID, {
        "grid_id": "STEP102_COMPLETE_81_CELL_GRID_V1",
        "rows": grid_rows,
        "selected_cell_id": selected["cell_id"] if selected else None,
    })
    ledger = {
        "ledger_id": "STEP102_STAGEA_COMPLETE_V1",
        "date": common.DATE,
        "decision": decision,
        "protocol_sha256": common.sha256_path(common.PROTOCOL),
        "authority_sha256": {
            STEP100_LEDGER.name: common.sha256_path(STEP100_LEDGER),
            STEP101.name: common.sha256_path(STEP101),
            DEV.name: common.sha256_path(DEV),
            LEGACY.name: common.sha256_path(LEGACY),
            PARENT.name: common.sha256_path(PARENT),
        },
        "geometry": geometry,
        "roster": [spec.__dict__ for spec in common.ROOT_SPECS],
        "adapter_audits": adapter_audits,
        "threshold_pairs": threshold_rows,
        "eligible_cell_count": len(eligible),
        "selected": selected,
        "verification": verify,
        "verification_gates": verify_gates,
        "failed_verification_gates": [name for name, passed in verify_gates.items() if not passed],
        "endpoint_signature": str(inspect.signature(execute_aliases_from_raw_text)),
        "grid_file": GRID.name,
        "grid_sha256": common.sha256_path(GRID),
        "arrays_file": ARRAYS.name,
        "arrays_sha256": common.sha256_path(ARRAYS),
        "sealed_outcome_opened": False,
        "code_sha256": {
            name: common.sha256_path(ROOT / name)
            for name in (
                "step93_common.py", "step100_common.py", "step100_stagea_develop.py",
                "step102_common.py", "step102_executable_adapter_endpoint.py",
                "step102_stagea_develop.py",
            )
        },
    }
    common.json_dump(LEDGER, ledger)
    selected_pair = threshold_rows[selected["pair_id"]] if selected else None
    stage0 = json.loads(STAGE0.read_text(encoding="utf-8"))
    config = {
        "config_id": "STEP102_STAGEA_FROZEN_CONFIG_V1",
        "date": common.DATE,
        "decision": decision,
        "stagea_ledger_sha256": common.sha256_path(LEDGER),
        "arrays_sha256": common.sha256_path(ARRAYS),
        "grid_sha256": common.sha256_path(GRID),
        "selected": selected,
        "thresholds": selected_pair["thresholds"] if selected_pair else None,
        "adapter_paths": adapter_paths,
        "adapter_sha256": {row["path"]: row["checkpoint"]["sha256"] for row in adapter_audits},
        "test_input": stage0["input_file"],
        "test_input_sha256": stage0["input_sha256"],
        "sealed_outcome": stage0["sealed_file"],
        "sealed_outcome_sha256": stage0["sealed_sha256"],
        "test_seed_start": common.TEST_SEEDS[0],
        "test_seed_stop_exclusive": common.TEST_SEEDS[-1] + 1,
        "test_seed_count": len(common.TEST_SEEDS),
    }
    common.json_dump(CONFIG, config)
    print(json.dumps({
        "decision": decision,
        "geometry": {
            name: {
                "gap_pp": 100 * row["parent_minus_best_challenger"],
                "oracle_terminal_pp": 100 * row["oracle"]["mean_terminal_delta"],
                "eligible": row["eligible"],
            }
            for name, row in geometry.items()
        },
        "eligible_cell_count": len(eligible),
        "selected_cell": selected["cell_id"] if selected else None,
        "search_terminal_pp": 100 * selected["effect"]["mean_terminal_delta"] if selected else None,
        "verify_terminal_pp": 100 * verify["mean_terminal_delta"] if verify else None,
        "verify_terminal_ci_pp": [100 * x for x in verify["inference"]["terminal"]["bootstrap_95"]] if verify else None,
        "failed_verification_gates": ledger["failed_verification_gates"],
        "sealed_outcome_opened": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
