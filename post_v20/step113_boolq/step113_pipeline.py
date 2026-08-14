from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

import step113_common as common


PUBLIC = common.ROOT / "external_data" / "step113_public"
SEALED = common.ROOT / "external_data" / "step113_sealed"
TRAIN_PARQUET = PUBLIC / "boolq_train.parquet"
VALIDATION_PARQUET = SEALED / "boolq_validation_source.parquet"
VALIDATION_INPUTS = SEALED / "boolq_validation_inputs.json"
VALIDATION_OUTCOMES = SEALED / "boolq_validation_sealed_outcomes.npz"
CODE = (
    "step113_common.py",
    "step113_endpoint.py",
    "step113_pipeline.py",
    "step113_validate_independent.py",
    "step93_common.py",
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path) -> str:
    return str(path.relative_to(common.ROOT)).replace("\\", "/")


def verify_hashes(bindings: dict[str, str]) -> None:
    failures = {}
    for name, expected in bindings.items():
        path = common.ROOT / name
        actual = common.sha256_path(path) if path.exists() else None
        if actual != expected:
            failures[name] = {"expected": expected, "actual": actual}
    if failures:
        raise AssertionError({"hash_failures": failures})


def create_predata_lock() -> None:
    if common.PREDATA_LOCK.exists():
        raise AssertionError("Step 113 predata lock already exists")
    if any(
        path.exists()
        for path in (
            common.STAGE0_MANIFEST,
            common.DEVELOPMENT_LEDGER,
            common.PREOUTCOME_LEDGER,
            common.PRIMARY_LEDGER,
        )
    ):
        raise AssertionError("Step 113 data/result file predates predata lock")
    metadata = load_json(common.METADATA)
    if metadata["prior_project_token_audit"] != "NO_PRIOR_BOOLQ_STUDY_TOKEN":
        raise AssertionError(metadata["prior_project_token_audit"])
    lock = {
        "schema": "step113.predata_lock.v1",
        "date": common.DATE,
        "status": "LOCKED_BEFORE_ANY_BOOLQ_ROW_ACCESS_MODEL_INFERENCE_OR_SELECTOR_EFFECT",
        "claim_status": "POSTREVIEW_PROSPECTIVE_CROSS_TASK_ONE_SHOT",
        "protocol_sha256": common.sha256_path(common.PROTOCOL),
        "metadata_sha256": common.sha256_path(common.METADATA),
        "dataset_id": common.DATASET_ID,
        "dataset_revision": common.DATASET_REVISION,
        "registry": [spec.__dict__ for spec in common.CANDIDATES],
        "input_contract": {
            "fields": ["question", "passage"],
            "max_length": common.MAX_LENGTH,
            "labels": {"0": "False", "1": "True"},
        },
        "partitions": {
            "error_head_train": common.TRAIN_ROWS_HEAD,
            "threshold_calibration": common.TRAIN_ROWS_THRESHOLD,
            "safety": common.TRAIN_ROWS_SAFETY,
            "salt": common.TRAIN_SPLIT_SALT,
        },
        "error_head_recipe": {
            "seeds": list(common.HEAD_SEEDS),
            "architecture": "standardized_frozen_parent_final_cls_plus_logits_to_linear_binary_head",
            "balanced_bootstrap": True,
            "epochs": common.HEAD_EPOCHS,
            "batch_size": common.HEAD_BATCH_SIZE,
            "learning_rate": common.HEAD_LEARNING_RATE,
            "weight_decay": common.HEAD_WEIGHT_DECAY,
        },
        "threshold_rule": {
            "quantile": common.THRESHOLD_QUANTILE,
            "method": "higher",
            "label_free": True,
            "selector_effect_used": False,
        },
        "preoutcome_gates": {
            "unique_best_parent_on_safety": True,
            "each_alias_nonempty_trigger": True,
            "maximum_safety_trigger_fraction": common.SAFETY_TRIGGER_MAX,
            "coordinate_wise_nonimproving": True,
            "maximum_per_alias_quality_loss": common.QUALITY_LOSS_MAX,
            "minimum_error_precision": None,
            "selector_replay_on_train_partitions": False,
        },
        "selector": {
            "pool_size": common.POOL_SIZE,
            "budget": common.BUDGET,
            "temperature": common.TAU,
            "paired_runs": len(common.TEST_SEEDS),
            "seed_range": [common.TEST_SEEDS[0], common.TEST_SEEDS[-1] + 1],
            "same_literal_exact_match": True,
            "aliases": common.N_ALIASES,
        },
        "primary_materiality": 0.005,
        "replacement_task_or_configuration_allowed": False,
        "prior_stops_retained": list(range(105, 113)),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "numpy": np.__version__,
            "pyarrow": importlib.metadata.version("pyarrow"),
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "code_sha256": {
            name: common.sha256_path(common.ROOT / name) for name in CODE
        },
        "data_access_completed": False,
        "outcome_opened": False,
    }
    common.json_dump(common.PREDATA_LOCK, lock)
    print(json.dumps(lock, indent=2))


def download(url: str, path: Path, expected_bytes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.stat().st_size != expected_bytes:
            raise AssertionError({"path": str(path), "unexpected_bytes": path.stat().st_size})
        return
    temporary = path.with_suffix(path.suffix + ".partial")
    with urllib.request.urlopen(url) as source, temporary.open("wb") as target:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            target.write(block)
    if temporary.stat().st_size != expected_bytes:
        raise AssertionError(
            {"path": str(path), "expected": expected_bytes, "actual": temporary.stat().st_size}
        )
    os.replace(temporary, path)


def _extract(table: object) -> tuple[list[str], list[str], np.ndarray]:
    questions = [str(value) for value in table["question"].to_pylist()]
    passages = [str(value) for value in table["passage"].to_pylist()]
    labels = np.asarray(table["answer"].to_pylist(), dtype=np.int64)
    return questions, passages, labels


def prepare_and_seal_data() -> None:
    if common.STAGE0_MANIFEST.exists():
        raise AssertionError("Step 113 stage-0 manifest already exists")
    lock = load_json(common.PREDATA_LOCK)
    verify_hashes(lock["code_sha256"])
    if lock["data_access_completed"] or lock["outcome_opened"]:
        raise AssertionError("invalid predata state")
    base = f"https://huggingface.co/datasets/{common.DATASET_ID}/resolve/{common.DATASET_REVISION}"
    download(f"{base}/{common.TRAIN_BLOB}", TRAIN_PARQUET, common.TRAIN_BYTES)
    download(
        f"{base}/{common.VALIDATION_BLOB}", VALIDATION_PARQUET, common.VALIDATION_BYTES
    )
    train = pq.read_table(TRAIN_PARQUET)
    validation = pq.read_table(VALIDATION_PARQUET)
    if train.num_rows != common.TRAIN_ROWS or validation.num_rows != common.OUTCOME_ROWS:
        raise AssertionError({"train": train.num_rows, "validation": validation.num_rows})
    train_questions, train_passages, _ = _extract(train)
    validation_questions, validation_passages, validation_labels = _extract(validation)
    train_uids = [
        common.uid("train", index, question, passage)
        for index, (question, passage) in enumerate(zip(train_questions, train_passages))
    ]
    head, threshold, safety = common.split_train(train_uids)
    input_rows = [
        {
            "source_index": index,
            "uid": common.uid("validation", index, question, passage),
            "question": question,
            "passage": passage,
        }
        for index, (question, passage) in enumerate(
            zip(validation_questions, validation_passages)
        )
    ]
    common.json_dump(
        VALIDATION_INPUTS,
        {
            "schema": "step113.boolq_validation_inputs.v1",
            "dataset_revision": common.DATASET_REVISION,
            "rows": input_rows,
        },
    )
    VALIDATION_OUTCOMES.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        VALIDATION_OUTCOMES,
        labels=validation_labels,
        uids=np.asarray([row["uid"] for row in input_rows]),
    )
    manifest = {
        "schema": "step113.stage0_data_and_seal.v1",
        "date": common.DATE,
        "status": "TRAIN_AVAILABLE_VALIDATION_INPUTS_SEPARATED_OUTCOMES_SEALED_UNOPENED",
        "predata_lock_sha256": common.sha256_path(common.PREDATA_LOCK),
        "train_file": relative(TRAIN_PARQUET),
        "train_sha256": common.sha256_path(TRAIN_PARQUET),
        "train_bytes": TRAIN_PARQUET.stat().st_size,
        "train_rows": common.TRAIN_ROWS,
        "validation_source_file": relative(VALIDATION_PARQUET),
        "validation_source_sha256": common.sha256_path(VALIDATION_PARQUET),
        "validation_source_bytes": VALIDATION_PARQUET.stat().st_size,
        "validation_input_file": relative(VALIDATION_INPUTS),
        "validation_input_sha256": common.sha256_path(VALIDATION_INPUTS),
        "sealed_outcome_file": relative(VALIDATION_OUTCOMES),
        "sealed_outcome_sha256": common.sha256_path(VALIDATION_OUTCOMES),
        "validation_rows": common.OUTCOME_ROWS,
        "train_uid_sha256": common.array_sha256(np.asarray(train_uids)),
        "partition_indices_sha256": {
            "error_head_train": common.array_sha256(head),
            "threshold_calibration": common.array_sha256(threshold),
            "safety": common.array_sha256(safety),
        },
        "validation_labels_or_statistics_inspected": False,
        "validation_model_inference_completed": False,
        "sealed_outcome_opened": False,
    }
    common.json_dump(common.STAGE0_MANIFEST, manifest)
    print(json.dumps(manifest, indent=2))


def _load_train() -> tuple[list[str], list[str], np.ndarray, list[str]]:
    table = pq.read_table(TRAIN_PARQUET)
    questions, passages, labels = _extract(table)
    uids = [
        common.uid("train", index, question, passage)
        for index, (question, passage) in enumerate(zip(questions, passages))
    ]
    return questions, passages, labels, uids


def develop_and_certify() -> None:
    if common.DEVELOPMENT_LEDGER.exists():
        raise AssertionError("Step 113 development ledger already exists")
    lock = load_json(common.PREDATA_LOCK)
    stage = load_json(common.STAGE0_MANIFEST)
    verify_hashes(lock["code_sha256"])
    verify_hashes(
        {
            stage["train_file"]: stage["train_sha256"],
            stage["validation_input_file"]: stage["validation_input_sha256"],
            stage["sealed_outcome_file"]: stage["sealed_outcome_sha256"],
        }
    )
    questions, passages, labels, uids = _load_train()
    if common.array_sha256(np.asarray(uids)) != stage["train_uid_sha256"]:
        raise AssertionError("train UID drift")
    head_indices, threshold_indices, safety_indices = common.split_train(uids)
    for name, values in (
        ("error_head_train", head_indices),
        ("threshold_calibration", threshold_indices),
        ("safety", safety_indices),
    ):
        if common.array_sha256(values) != stage["partition_indices_sha256"][name]:
            raise AssertionError({"partition_drift": name})

    parent_predictions, parent_logits, parent_features, parent_audit = common.infer_candidate_pairs(
        common.PARENT, questions, passages, return_features=True, local_files_only=False
    )
    if parent_features is None:
        raise AssertionError("missing parent features")
    safety_roots = np.empty((len(safety_indices), common.N_ROOTS), dtype=np.int64)
    safety_roots[:, 0] = parent_predictions[safety_indices]
    root_audits = [parent_audit]
    safety_questions = [questions[index] for index in safety_indices]
    safety_passages = [passages[index] for index in safety_indices]
    for column, spec in enumerate(common.CANDIDATES[1:], start=1):
        predictions, _, _, audit = common.infer_candidate_pairs(
            spec,
            safety_questions,
            safety_passages,
            return_features=False,
            local_files_only=False,
        )
        safety_roots[:, column] = predictions
        root_audits.append(audit)
        print(f"safety root inference {column + 1}/{common.N_ROOTS} complete", flush=True)

    common.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    targets = (parent_predictions[head_indices] != labels[head_indices]).astype(np.int64)
    head_paths: list[str] = []
    head_checkpoints: list[dict] = []
    head_training: list[dict] = []
    for alias, seed in enumerate(common.HEAD_SEEDS):
        state, audit = common.train_error_head(parent_features[head_indices], targets, seed)
        path = common.MODEL_DIR / f"step113_boolq_error_head_{alias}.safetensors"
        checkpoint = common.save_error_head(state, path, audit)
        head_paths.append(checkpoint["path"])
        head_checkpoints.append(checkpoint)
        head_training.append(audit)
        print(f"trained error head {alias + 1}/{common.N_ALIASES}", flush=True)

    threshold_scores = np.column_stack(
        [
            common.error_head_scores(parent_features[threshold_indices], common.ROOT / path)
            for path in head_paths
        ]
    )
    safety_scores = np.column_stack(
        [
            common.error_head_scores(parent_features[safety_indices], common.ROOT / path)
            for path in head_paths
        ]
    )
    thresholds = np.asarray(
        [common.threshold_p99(threshold_scores[:, alias]) for alias in range(common.N_ALIASES)],
        dtype=np.float64,
    )
    safety_aliases, safety_triggers = common.make_aliases(
        parent_predictions[safety_indices], safety_scores, thresholds
    )
    safety_labels = labels[safety_indices]
    quality = common.alias_quality_audit(
        parent_predictions[safety_indices], safety_aliases, safety_labels
    )
    clean_accuracies = np.mean(safety_roots == safety_labels[:, None], axis=0)
    best = np.flatnonzero(clean_accuracies == np.max(clean_accuracies))
    root_vectors_distinct = len(
        {common.array_sha256(safety_roots[:, column]) for column in range(common.N_ROOTS)}
    ) == common.N_ROOTS
    alias_vectors_distinct = len(
        {common.array_sha256(safety_aliases[:, column]) for column in range(common.N_ALIASES)}
    ) == common.N_ALIASES
    checkpoint_hashes = [row["sha256"] for row in head_checkpoints]
    reconstructed = np.column_stack(
        [
            common.error_head_scores(parent_features[safety_indices], common.ROOT / path)
            for path in head_paths
        ]
    )
    safety_error = (parent_predictions[safety_indices] != safety_labels).astype(np.int64)
    diagnostics = {
        "clean_root_accuracies": clean_accuracies.tolist(),
        "parent_runner_up_gap": float(
            clean_accuracies[0] - np.max(clean_accuracies[1:])
        ),
        "parent_error_rate": float(np.mean(safety_error)),
        "error_auc": [
            common.binary_auc(safety_error, safety_scores[:, alias])
            for alias in range(common.N_ALIASES)
        ],
        "trigger_fraction": np.mean(safety_triggers, axis=0).tolist(),
        "trigger_error_precision": [
            float(np.mean(safety_error[safety_triggers[:, alias]]))
            if np.any(safety_triggers[:, alias])
            else None
            for alias in range(common.N_ALIASES)
        ],
        "pairwise_trigger_overlap": [
            float(np.mean(safety_triggers[:, left] & safety_triggers[:, right]))
            for left in range(common.N_ALIASES)
            for right in range(left + 1, common.N_ALIASES)
        ],
    }
    gates = {
        "all_root_revisions_and_labels_verified": all(
            row["id2label"] == {"0": "False", "1": "True"} for row in root_audits
        ),
        "clean_response_vectors_distinct": root_vectors_distinct,
        "designated_parent_unique_best": len(best) == 1 and int(best[0]) == 0,
        "each_alias_triggers": bool(np.all(np.sum(safety_triggers, axis=0) >= 1)),
        "trigger_fraction_at_most_two_percent": bool(
            np.all(np.mean(safety_triggers, axis=0) <= common.SAFETY_TRIGGER_MAX)
        ),
        "aliases_response_distinct": alias_vectors_distinct,
        "coordinate_wise_nonimproving": quality["coordinate_wise_nonimproving"],
        "quality_loss_within_one_point": quality["loss_within_one_point"],
        "four_nonempty_hash_distinct_checkpoints": all(row["bytes"] > 0 for row in head_checkpoints)
        and len(set(checkpoint_hashes)) == common.N_ALIASES,
        "checkpoint_score_reconstruction_exact": bool(np.array_equal(reconstructed, safety_scores)),
        "no_selector_effect_used": True,
        "sealed_outcome_still_unopened": common.sha256_path(
            common.ROOT / stage["sealed_outcome_file"]
        )
        == stage["sealed_outcome_sha256"],
    }
    decision = "GO_STEP113_DEVELOPMENT_CERTIFIED" if all(gates.values()) else "STOP_STEP113_DEVELOPMENT_NOT_CERTIFIED"
    np.savez_compressed(
        common.DEVELOPMENT_ARRAYS,
        head_indices=head_indices,
        threshold_indices=threshold_indices,
        safety_indices=safety_indices,
        parent_predictions=parent_predictions,
        parent_logits=parent_logits,
        parent_features=parent_features,
        safety_roots=safety_roots,
        threshold_scores=threshold_scores,
        safety_scores=safety_scores,
        thresholds=thresholds,
        safety_aliases=safety_aliases,
        safety_triggers=safety_triggers,
        safety_labels=safety_labels,
    )
    ledger = {
        "schema": "step113.development_ledger.v1",
        "date": common.DATE,
        "decision": decision,
        "claim_status": "PREOUTCOME_DEVELOPMENT_ONLY",
        "predata_lock_sha256": common.sha256_path(common.PREDATA_LOCK),
        "stage0_manifest_sha256": common.sha256_path(common.STAGE0_MANIFEST),
        "root_audits": root_audits,
        "head_paths": head_paths,
        "head_checkpoints": head_checkpoints,
        "head_training": head_training,
        "thresholds": thresholds.tolist(),
        "threshold_rule": "label_free_empirical_p99_higher",
        "quality": quality,
        "diagnostics_not_gates": diagnostics,
        "gates": gates,
        "arrays_file": relative(common.DEVELOPMENT_ARRAYS),
        "arrays_sha256": common.sha256_path(common.DEVELOPMENT_ARRAYS),
        "validation_model_inference_completed": False,
        "sealed_outcome_opened": False,
    }
    common.json_dump(common.DEVELOPMENT_LEDGER, ledger)
    if decision.startswith("GO"):
        go_lock = {
            "schema": "step113.development_go_lock.v1",
            "date": common.DATE,
            "status": "LOCKED_AFTER_DEVELOPMENT_GO_BEFORE_VALIDATION_MODEL_INFERENCE",
            "predata_lock_sha256": common.sha256_path(common.PREDATA_LOCK),
            "stage0_manifest_sha256": common.sha256_path(common.STAGE0_MANIFEST),
            "development_ledger_sha256": common.sha256_path(common.DEVELOPMENT_LEDGER),
            "development_arrays_sha256": common.sha256_path(common.DEVELOPMENT_ARRAYS),
            "head_paths": head_paths,
            "head_sha256": {row["path"]: row["sha256"] for row in head_checkpoints},
            "thresholds": thresholds.tolist(),
            "validation_input_file": stage["validation_input_file"],
            "validation_input_sha256": stage["validation_input_sha256"],
            "sealed_outcome_file": stage["sealed_outcome_file"],
            "sealed_outcome_sha256": stage["sealed_outcome_sha256"],
            "code_sha256": lock["code_sha256"],
            "sealed_outcome_opened": False,
        }
        common.json_dump(common.DEVELOPMENT_GO_LOCK, go_lock)
    print(json.dumps(ledger, indent=2))


def _load_validation_inputs() -> tuple[list[str], list[str], list[str]]:
    package = load_json(VALIDATION_INPUTS)
    rows = package["rows"]
    if len(rows) != common.OUTCOME_ROWS:
        raise AssertionError("validation input row count drift")
    questions = [row["question"] for row in rows]
    passages = [row["passage"] for row in rows]
    uids = [row["uid"] for row in rows]
    expected = [
        common.uid("validation", index, question, passage)
        for index, (question, passage) in enumerate(zip(questions, passages))
    ]
    if uids != expected:
        raise AssertionError("validation UID drift")
    return questions, passages, uids


def predict_validation_without_outcomes() -> None:
    if common.PREOUTCOME_LEDGER.exists():
        raise AssertionError("Step 113 pre-outcome ledger already exists")
    go = load_json(common.DEVELOPMENT_GO_LOCK)
    verify_hashes(go["code_sha256"])
    verify_hashes(go["head_sha256"])
    verify_hashes(
        {
            go["validation_input_file"]: go["validation_input_sha256"],
            go["sealed_outcome_file"]: go["sealed_outcome_sha256"],
        }
    )
    questions, passages, uids = _load_validation_inputs()
    roots = np.empty((common.OUTCOME_ROWS, common.N_ROOTS), dtype=np.int64)
    parent, parent_logits, parent_features, parent_audit = common.infer_candidate_pairs(
        common.PARENT,
        questions,
        passages,
        return_features=True,
        local_files_only=True,
    )
    if parent_features is None:
        raise AssertionError("missing validation parent features")
    roots[:, 0] = parent
    root_audits = [parent_audit]
    for column, spec in enumerate(common.CANDIDATES[1:], start=1):
        predictions, _, _, audit = common.infer_candidate_pairs(
            spec,
            questions,
            passages,
            return_features=False,
            local_files_only=True,
        )
        roots[:, column] = predictions
        root_audits.append(audit)
        print(f"validation root inference {column + 1}/{common.N_ROOTS} complete", flush=True)
    scores = np.column_stack(
        [common.error_head_scores(parent_features, common.ROOT / path) for path in go["head_paths"]]
    )
    thresholds = np.asarray(go["thresholds"], dtype=np.float64)
    aliases, triggers = common.make_aliases(parent, scores, thresholds)
    clean_distinct = len(
        {common.array_sha256(roots[:, column]) for column in range(common.N_ROOTS)}
    ) == common.N_ROOTS
    alias_distinct = len(
        {common.array_sha256(aliases[:, column]) for column in range(common.N_ALIASES)}
    ) == common.N_ALIASES
    np.savez_compressed(
        common.PREOUTCOME_PREDICTIONS,
        uids=np.asarray(uids),
        roots=roots,
        parent_logits=parent_logits,
        parent_features=parent_features,
        scores=scores,
        thresholds=thresholds,
        aliases=aliases,
        triggers=triggers,
    )
    ledger = {
        "schema": "step113.preoutcome_prediction_ledger.v1",
        "date": common.DATE,
        "status": "ALL_PREDICTIONS_COMPLETE_OUTCOMES_UNOPENED",
        "development_go_lock_sha256": common.sha256_path(common.DEVELOPMENT_GO_LOCK),
        "root_audits": root_audits,
        "head_paths": go["head_paths"],
        "head_sha256": go["head_sha256"],
        "thresholds": go["thresholds"],
        "prediction_hashes": {
            "uids": common.array_sha256(np.asarray(uids)),
            "roots": common.array_sha256(roots),
            "parent_logits": common.array_sha256(parent_logits),
            "parent_features": common.array_sha256(parent_features),
            "scores": common.array_sha256(scores),
            "aliases": common.array_sha256(aliases),
            "triggers": common.array_sha256(triggers),
        },
        "clean_response_vectors_distinct": clean_distinct,
        "alias_response_vectors_distinct": alias_distinct,
        "trigger_fraction_label_free": np.mean(triggers, axis=0).tolist(),
        "predictions_file": relative(common.PREOUTCOME_PREDICTIONS),
        "predictions_sha256": common.sha256_path(common.PREOUTCOME_PREDICTIONS),
        "sealed_outcome_sha256_unchanged": common.sha256_path(
            common.ROOT / go["sealed_outcome_file"]
        )
        == go["sealed_outcome_sha256"],
        "sealed_outcome_opened": False,
    }
    common.json_dump(common.PREOUTCOME_LEDGER, ledger)
    print(json.dumps(ledger, indent=2))


def create_preoutcome_lock() -> None:
    if common.PREOUTCOME_LOCK.exists():
        raise AssertionError("Step 113 pre-outcome lock already exists")
    go = load_json(common.DEVELOPMENT_GO_LOCK)
    ledger = load_json(common.PREOUTCOME_LEDGER)
    if ledger["sealed_outcome_opened"]:
        raise AssertionError("outcome state already opened")
    verify_hashes(go["code_sha256"])
    verify_hashes(go["head_sha256"])
    verify_hashes(
        {
            ledger["predictions_file"]: ledger["predictions_sha256"],
            go["sealed_outcome_file"]: go["sealed_outcome_sha256"],
        }
    )
    lock = {
        "schema": "step113.preoutcome_lock.v1",
        "date": common.DATE,
        "status": "LOCKED_BEFORE_FIRST_OUTCOME_OPEN",
        "protocol_sha256": common.sha256_path(common.PROTOCOL),
        "metadata_sha256": common.sha256_path(common.METADATA),
        "predata_lock_sha256": common.sha256_path(common.PREDATA_LOCK),
        "stage0_manifest_sha256": common.sha256_path(common.STAGE0_MANIFEST),
        "development_go_lock_sha256": common.sha256_path(common.DEVELOPMENT_GO_LOCK),
        "preoutcome_ledger_sha256": common.sha256_path(common.PREOUTCOME_LEDGER),
        "predictions_file": ledger["predictions_file"],
        "predictions_sha256": ledger["predictions_sha256"],
        "prediction_hashes": ledger["prediction_hashes"],
        "head_paths": go["head_paths"],
        "head_sha256": go["head_sha256"],
        "thresholds": go["thresholds"],
        "sealed_outcome_file": go["sealed_outcome_file"],
        "sealed_outcome_sha256": go["sealed_outcome_sha256"],
        "selector": {
            "pool_size": common.POOL_SIZE,
            "budget": common.BUDGET,
            "temperature": common.TAU,
            "seed_range": [common.TEST_SEEDS[0], common.TEST_SEEDS[-1] + 1],
            "same_literal_exact_match": True,
        },
        "primary_gates": "verbatim_STEP113_protocol",
        "replacement_allowed": False,
        "code_sha256": go["code_sha256"],
        "outcome_opened": False,
    }
    common.json_dump(common.PREOUTCOME_LOCK, lock)
    print(json.dumps(lock, indent=2))


def _subset_summary(
    roots: np.ndarray, aliases: np.ndarray, labels: np.ndarray, indices: np.ndarray
) -> dict:
    raw, arrays = common.compute_raw_effects(roots[indices], aliases[indices], labels[indices])
    return common.add_inference(raw, arrays)


def confirm_once() -> None:
    if common.PRIMARY_LEDGER.exists():
        raise AssertionError("Step 113 primary outcome already exists")
    lock = load_json(common.PREOUTCOME_LOCK)
    verify_hashes(lock["code_sha256"])
    verify_hashes(lock["head_sha256"])
    verify_hashes(
        {
            lock["predictions_file"]: lock["predictions_sha256"],
            lock["sealed_outcome_file"]: lock["sealed_outcome_sha256"],
        }
    )
    with np.load(common.ROOT / lock["predictions_file"], allow_pickle=False) as data:
        predictions = {key: data[key] for key in data.files}
    for name, expected in lock["prediction_hashes"].items():
        if common.array_sha256(predictions[name]) != expected:
            raise AssertionError({"prediction_hash_drift": name})

    # This is the first and only confirmatory opening of the sealed labels.
    with np.load(common.ROOT / lock["sealed_outcome_file"], allow_pickle=False) as sealed:
        labels = sealed["labels"].astype(np.int64)
        sealed_uids = sealed["uids"]
    if not np.array_equal(sealed_uids, predictions["uids"]):
        raise AssertionError("sealed outcome UID mismatch")
    roots = predictions["roots"].astype(np.int64)
    aliases = predictions["aliases"].astype(np.int64)
    quality = common.alias_quality_audit(roots[:, 0], aliases, labels)
    aliases_distinct = len(
        {common.array_sha256(aliases[:, column]) for column in range(common.N_ALIASES)}
    ) == common.N_ALIASES
    raw, arrays = common.compute_raw_effects(roots, aliases, labels)
    summary = common.add_inference(raw, arrays)
    gates = common.primary_gates(summary, quality, aliases_distinct)
    uids = [str(value) for value in predictions["uids"].tolist()]
    half_a, half_b = common.diagnostic_halves(uids)
    diagnostics = {
        "root_accuracies": np.mean(roots == labels[:, None], axis=0).tolist(),
        "trigger_fraction": np.mean(predictions["triggers"], axis=0).tolist(),
        "half_a": _subset_summary(roots, aliases, labels, half_a),
        "half_b": _subset_summary(roots, aliases, labels, half_b),
    }
    decision = "GO_STEP113_PRIMARY_CONFIRMATION" if all(gates.values()) else "NO_GO_STEP113_PRIMARY_CONFIRMATION"
    np.savez_compressed(
        common.PRIMARY_ARRAYS,
        labels=labels,
        half_a=half_a,
        half_b=half_b,
        **arrays,
    )
    ledger = {
        "schema": "step113.primary_confirmatory_ledger.v1",
        "date": common.DATE,
        "decision": decision,
        "claim_status": "ONE_TIME_FROZEN_OUTCOME",
        "preoutcome_lock_sha256": common.sha256_path(common.PREOUTCOME_LOCK),
        "sealed_outcome_sha256": lock["sealed_outcome_sha256"],
        "predictions_sha256": lock["predictions_sha256"],
        "quality": quality,
        "summary": summary,
        "diagnostics_not_gates": diagnostics,
        "gates": gates,
        "arrays_file": relative(common.PRIMARY_ARRAYS),
        "arrays_sha256": common.sha256_path(common.PRIMARY_ARRAYS),
        "sealed_outcome_opened": True,
        "replacement_allowed": False,
    }
    common.json_dump(common.PRIMARY_LEDGER, ledger)
    print(json.dumps(ledger, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("lock", "stage0", "develop", "predict", "preoutcome-lock", "confirm"),
    )
    args = parser.parse_args()
    actions = {
        "lock": create_predata_lock,
        "stage0": prepare_and_seal_data,
        "develop": develop_and_certify,
        "predict": predict_validation_without_outcomes,
        "preoutcome-lock": create_preoutcome_lock,
        "confirm": confirm_once,
    }
    actions[args.command]()


if __name__ == "__main__":
    main()
