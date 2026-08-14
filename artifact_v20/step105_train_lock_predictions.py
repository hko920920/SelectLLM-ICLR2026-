from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import step105_common as common


ROOT = Path(__file__).resolve().parent
DEV = ROOT / f"STEP105_YELP_DEVELOPMENT_ROWS_{common.DATE}.json"
STAGE0 = ROOT / f"STEP105_STAGE0_DATA_AND_SEAL_MANIFEST_{common.DATE}.json"
TEST_INPUT = ROOT / "step105_test_inputs" / "yelp_polarity_test_inputs.json"
MODEL_DIR = ROOT / "step105_models"
WORK_DIR = ROOT / "step105_preoutcome_work"
PREDICTIONS = ROOT / f"STEP105_PREOUTCOME_PREDICTIONS_{common.DATE}.npz"
LEDGER = ROOT / f"STEP105_TRAIN_AND_PREOUTCOME_PREDICTION_LEDGER_{common.DATE}.json"


def main() -> None:
    if not common.PREDATA_LOCK.is_file():
        raise FileNotFoundError(common.PREDATA_LOCK)
    if any(path.exists() for path in (MODEL_DIR, WORK_DIR, PREDICTIONS, LEDGER)):
        raise FileExistsError("Step 105 train/prediction output already exists")
    predata = json.loads(common.PREDATA_LOCK.read_text(encoding="utf-8"))
    stage0 = json.loads(STAGE0.read_text(encoding="utf-8"))
    if stage0["predata_lock_sha256"] != common.sha256_path(common.PREDATA_LOCK):
        raise AssertionError("pre-data lock drift")
    dev_payload = json.loads(DEV.read_text(encoding="utf-8"))
    dev_rows = dev_payload["rows"]
    dev_texts = [str(row["text"]) for row in dev_rows]
    dev_labels = np.asarray([int(row["label"]) for row in dev_rows], dtype=np.int64)
    parts = {
        name: np.asarray([i for i, row in enumerate(dev_rows) if row["partition"] == name], dtype=np.int64)
        for name in common.DEV_QUOTAS_PER_LABEL
    }
    if {key: len(value) for key, value in parts.items()} != {
        "adapter_train": 6000, "threshold_calibration": 1000, "threshold_safety": 2000,
    }:
        raise AssertionError({key: len(value) for key, value in parts.items()})
    test_payload = json.loads(TEST_INPUT.read_text(encoding="utf-8"))
    if test_payload.get("labels_present") is not False:
        raise AssertionError("test input must remain label-free")
    test_texts = [str(row["text"]) for row in test_payload["rows"]]
    if len(test_texts) != 38_000:
        raise AssertionError(len(test_texts))

    MODEL_DIR.mkdir()
    WORK_DIR.mkdir()
    dev_root_columns: list[np.ndarray] = []
    test_root_columns: list[np.ndarray] = []
    root_audits: dict[str, Any] = {}
    for spec in common.ROOT_SPECS:
        dev_predictions, dev_audit = common.infer_root_predictions(spec, dev_texts)
        test_predictions, test_audit = common.infer_root_predictions(spec, test_texts)
        dev_root_columns.append(dev_predictions)
        test_root_columns.append(test_predictions)
        root_audits[spec.key] = {"development": dev_audit, "test_label_free": test_audit}
        print(json.dumps({
            "root": spec.key,
            "development_prediction_sha256": dev_audit["prediction_sha256"],
            "test_prediction_sha256": test_audit["prediction_sha256"],
        }), flush=True)
    dev_roots = np.stack(dev_root_columns, axis=1).astype(np.int64)
    test_roots = np.stack(test_root_columns, axis=1).astype(np.int64)

    adapter_rows: list[dict[str, Any]] = []
    dev_scores: list[np.ndarray] = []
    test_scores: list[np.ndarray] = []
    train_indices = parts["adapter_train"]
    for alias, seed in enumerate(common.ADAPTER_SEEDS):
        model, tokenizer, train_audit = common.train_error_adapter(
            [dev_texts[i] for i in train_indices],
            dev_roots[train_indices, 0], dev_labels[train_indices], seed,
        )
        path = MODEL_DIR / f"step105_yelp_error_adapter_{alias}.safetensors"
        weight_audit = common.save_adapter(model, path, {
            "step": 105, "alias": alias, "seed": seed,
            "parent_repo": common.ROOT_SPECS[0].repo_id,
            "parent_revision": common.ROOT_SPECS[0].revision,
        })
        if int(weight_audit["parameter_count"]) < 10_000_000:
            raise AssertionError({"alias": alias, "parameters": weight_audit["parameter_count"]})
        dev_score = common.infer_error_scores(model, tokenizer, dev_texts)
        test_score = common.infer_error_scores(model, tokenizer, test_texts)
        dev_scores.append(dev_score)
        test_scores.append(test_score)
        row = {
            "alias": alias,
            "seed": seed,
            "weight_file": str(path.relative_to(ROOT)).replace("\\", "/"),
            "weight_audit": weight_audit,
            "training": train_audit,
            "development_score_sha256": common.array_sha256(dev_score),
            "test_score_sha256": common.array_sha256(test_score),
        }
        audit_path = path.with_suffix(".audit.json")
        common.json_dump(audit_path, row)
        row["audit_file"] = str(audit_path.relative_to(ROOT)).replace("\\", "/")
        row["audit_sha256"] = common.sha256_path(audit_path)
        adapter_rows.append(row)
        model.cpu()
        print(json.dumps({
            "alias": alias,
            "adapter_sha256": weight_audit["sha256"],
            "parameters": weight_audit["parameter_count"],
            "train_parent_error_prevalence": train_audit["parent_error_prevalence"],
        }), flush=True)
    if len({row["weight_audit"]["sha256"] for row in adapter_rows}) != common.N_ALIASES:
        raise AssertionError("adapter hashes are not distinct")
    dev_score_matrix = np.stack(dev_scores, axis=1)
    test_score_matrix = np.stack(test_scores, axis=1)
    thresholds, threshold_audits = common.fit_frozen_thresholds(
        dev_roots[:, 0], dev_labels, dev_score_matrix,
        parts["threshold_calibration"], parts["threshold_safety"],
    )
    dev_aliases, dev_triggers = common.make_alias_codes(
        dev_roots[:, 0], dev_score_matrix, thresholds,
    )
    test_aliases, test_triggers = common.make_alias_codes(
        test_roots[:, 0], test_score_matrix, thresholds,
    )
    development_quality: dict[str, Any] = {}
    for name, indices in parts.items():
        parent_correct = dev_roots[indices, 0] == dev_labels[indices]
        loss_counts = np.sum(dev_triggers[indices] & parent_correct[:, None], axis=0).astype(np.int64)
        coordinate_nonimproving = bool(np.all(
            common.exact_match_feedback(dev_aliases[indices], dev_labels[indices])
            <= np.repeat(
                common.exact_match_feedback(dev_roots[indices, 0, None], dev_labels[indices]),
                common.N_ALIASES, axis=1,
            )
        ))
        development_quality[name] = {
            "rows": len(indices),
            "parent_accuracy": float(np.mean(parent_correct)),
            "loss_counts": loss_counts.tolist(),
            "loss_fractions": (loss_counts / len(indices)).tolist(),
            "trigger_counts": np.sum(dev_triggers[indices], axis=0).astype(int).tolist(),
            "trigger_fractions": np.mean(dev_triggers[indices], axis=0).tolist(),
            "coordinate_wise_nonimproving": coordinate_nonimproving,
        }

    np.savez_compressed(
        PREDICTIONS,
        test_root_predictions=test_roots,
        test_error_scores=test_score_matrix,
        test_aliases=test_aliases,
        test_triggers=test_triggers.astype(np.uint8),
        thresholds=np.asarray(thresholds, dtype=np.float64),
    )
    np.savez_compressed(
        WORK_DIR / "development_predictions_and_scores.npz",
        root_predictions=dev_roots,
        error_scores=dev_score_matrix,
        aliases=dev_aliases,
        triggers=dev_triggers.astype(np.uint8),
        thresholds=np.asarray(thresholds, dtype=np.float64),
    )
    common.json_dump(WORK_DIR / "root_audits.json", root_audits)
    ledger = {
        "schema": "step105.train_and_preoutcome_prediction.v1",
        "date": common.DATE,
        "status": "PREOUTCOME_PREDICTIONS_LOCKABLE_NO_TEST_LABEL_OPEN",
        "predata_lock_sha256": common.sha256_path(common.PREDATA_LOCK),
        "stage0_manifest_sha256": common.sha256_path(STAGE0),
        "development_file_sha256": common.sha256_path(DEV),
        "test_input_sha256": common.sha256_path(TEST_INPUT),
        "target_selector_search_rows": 0,
        "target_selector_verify_rows": 0,
        "target_selector_grid_cells": 0,
        "root_audits_file": str((WORK_DIR / "root_audits.json").relative_to(ROOT)).replace("\\", "/"),
        "root_audits_sha256": common.sha256_path(WORK_DIR / "root_audits.json"),
        "development_arrays_file": str((WORK_DIR / "development_predictions_and_scores.npz").relative_to(ROOT)).replace("\\", "/"),
        "development_arrays_sha256": common.sha256_path(WORK_DIR / "development_predictions_and_scores.npz"),
        "adapters": adapter_rows,
        "thresholds": thresholds,
        "threshold_audits": threshold_audits,
        "development_quality": development_quality,
        "test_rows": len(test_texts),
        "test_predictions_file": PREDICTIONS.name,
        "test_predictions_sha256": common.sha256_path(PREDICTIONS),
        "test_root_prediction_sha256": common.array_sha256(test_roots),
        "test_error_score_sha256": common.array_sha256(test_score_matrix),
        "test_alias_sha256": common.array_sha256(test_aliases),
        "test_trigger_sha256": common.array_sha256(test_triggers.astype(np.uint8)),
        "test_outcome_opened": False,
        "test_class_specific_quality_computed": False,
        "active_selector_run_on_target": False,
        "code_sha256": {
            "step105_common.py": common.sha256_path(ROOT / "step105_common.py"),
            "step105_train_lock_predictions.py": common.sha256_path(Path(__file__)),
        },
    }
    common.json_dump(LEDGER, ledger)
    print(json.dumps({
        "status": ledger["status"],
        "thresholds": thresholds,
        "test_predictions_sha256": ledger["test_predictions_sha256"],
        "test_outcome_opened": False,
        "active_selector_run_on_target": False,
        "output": LEDGER.name,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
