from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

import step114_common as common


PUBLIC = common.ROOT / "external_data" / "step114_public"
SEALED = common.ROOT / "external_data" / "step114_sealed"
DEVELOPMENT = common.ROOT / "step114_development"
TEST_PARQUET = SEALED / "amazon_polarity_test_source.parquet"
HEAD_TRAIN = DEVELOPMENT / "error_head_train.json"
SAFETY = DEVELOPMENT / "safety.json"
TARGET_THRESHOLD = SEALED / "target_threshold_inputs.json"
OUTCOME_INPUTS = SEALED / "primary_outcome_inputs.json"
OUTCOME_LABELS = SEALED / "primary_outcome_sealed_labels.npz"
CODE = (
    "step114_common.py",
    "step114_endpoint.py",
    "step114_pipeline.py",
    "step114_validate_independent.py",
    "step113_common.py",
    "step93_common.py",
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path) -> str:
    return str(path.relative_to(common.ROOT)).replace("\\", "/")


def verify_hashes(bindings: dict[str, str]) -> None:
    failed = {}
    for name, expected in bindings.items():
        path = common.ROOT / name
        actual = common.sha256_path(path) if path.exists() else None
        if actual != expected:
            failed[name] = {"expected": expected, "actual": actual}
    if failed:
        raise AssertionError({"hash_failures": failed})


def create_predata_lock() -> None:
    if common.PREDATA_LOCK.exists():
        raise AssertionError("Step 114 predata lock exists")
    if any(
        path.exists()
        for path in (
            common.STAGE0_MANIFEST,
            common.DEVELOPMENT_LEDGER,
            common.PREOUTCOME_LEDGER,
            common.PRIMARY_LEDGER,
        )
    ):
        raise AssertionError("Step 114 result predates lock")
    metadata = load_json(common.METADATA)
    step113 = load_json(common.ROOT / f"STEP113_PRIMARY_CONFIRMATORY_LEDGER_{common.DATE}.json")
    if step113["decision"] != "NO_GO_STEP113_PRIMARY_CONFIRMATION":
        raise AssertionError(step113["decision"])
    if metadata["prior_project_token_audit"] != "NO_PRIOR_AMAZON_POLARITY_STUDY_TOKEN":
        raise AssertionError(metadata["prior_project_token_audit"])
    partitions = common.selected_source_indices()
    lock = {
        "schema": "step114.predata_lock.v1",
        "date": common.DATE,
        "status": "LOCKED_BEFORE_ANY_AMAZON_POLARITY_ROW_ACCESS_OR_TASK_INFERENCE",
        "claim_status": "FINAL_POSTREVIEW_PROSPECTIVE_TERMINAL_BRIDGE",
        "protocol_sha256": common.sha256_path(common.PROTOCOL),
        "metadata_sha256": common.sha256_path(common.METADATA),
        "step113_no_go_ledger_sha256": common.sha256_path(
            common.ROOT / f"STEP113_PRIMARY_CONFIRMATORY_LEDGER_{common.DATE}.json"
        ),
        "dataset_id": common.DATASET_ID,
        "dataset_revision": common.DATASET_REVISION,
        "registry": [spec.__dict__ for spec in common.CANDIDATES],
        "input_contract": {"field": "content", "max_length": common.MAX_LENGTH},
        "source_index_seed": common.SOURCE_INDEX_SEED,
        "source_index_sha256": {
            name: common.array_sha256(values) for name, values in partitions.items()
        },
        "partition_counts": {name: len(values) for name, values in partitions.items()},
        "error_head_recipe": {
            "implementation": "locked_step113_linear_error_head_recipe",
            "seeds": list(common.HEAD_SEEDS),
            "epochs": 30,
            "batch_size": 256,
            "learning_rate": 0.001,
            "weight_decay": 0.001,
        },
        "threshold_rule": {
            "quantile": common.THRESHOLD_QUANTILE,
            "method": "higher",
            "target_inputs_only": True,
            "target_labels_available": False,
        },
        "preoutcome_trigger_certificate": {
            "minimum_per_alias": 1,
            "maximum_per_alias": int(common.QUALITY_LOSS_MAX * common.OUTCOME_ROWS),
            "guaranteed_maximum_quality_loss": common.QUALITY_LOSS_MAX,
        },
        "selector": {
            "pool_size": common.POOL_SIZE,
            "budget": common.BUDGET,
            "temperature": common.TAU,
            "paired_runs": len(common.TEST_SEEDS),
            "seed_range": [common.TEST_SEEDS[0], common.TEST_SEEDS[-1] + 1],
            "same_literal_exact_match": True,
        },
        "primary_gates": "terminal_and_active_minus_fixed_terminal_only_plus_validity_and_mediation",
        "cumulative_and_switch_direction_are_diagnostics": True,
        "replacement_allowed": False,
        "family_stops_if_not_go": True,
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
        "code_sha256": {name: common.sha256_path(common.ROOT / name) for name in CODE},
        "data_access_completed": False,
        "outcome_opened": False,
    }
    common.json_dump(common.PREDATA_LOCK, lock)
    print(json.dumps(lock, indent=2))


def download(url: str, path: Path, expected_bytes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.stat().st_size != expected_bytes:
            raise AssertionError({"unexpected_bytes": path.stat().st_size})
        return
    temporary = path.with_suffix(path.suffix + ".partial")
    with urllib.request.urlopen(url) as source, temporary.open("wb") as target:
        while True:
            block = source.read(1 << 20)
            if not block:
                break
            target.write(block)
    if temporary.stat().st_size != expected_bytes:
        raise AssertionError(
            {"expected": expected_bytes, "actual": temporary.stat().st_size}
        )
    os.replace(temporary, path)


def _rows(table: pa.Table, indices: np.ndarray, include_labels: bool) -> list[dict]:
    subset = table.take(pa.array(indices))
    contents = [str(value) for value in subset["content"].to_pylist()]
    labels = np.asarray(subset["label"].to_pylist(), dtype=np.int64)
    rows = []
    for position, (source_index, content) in enumerate(zip(indices, contents)):
        row = {
            "source_index": int(source_index),
            "uid": common.uid(int(source_index), content),
            "content": content,
        }
        if include_labels:
            row["label"] = int(labels[position])
        rows.append(row)
    return rows


def prepare_and_seal_data() -> None:
    if common.STAGE0_MANIFEST.exists():
        raise AssertionError("Step 114 stage0 exists")
    lock = load_json(common.PREDATA_LOCK)
    verify_hashes(lock["code_sha256"])
    url = f"https://huggingface.co/datasets/{common.DATASET_ID}/resolve/{common.DATASET_REVISION}/{common.TEST_BLOB}"
    download(url, TEST_PARQUET, common.TEST_BYTES)
    table = pq.read_table(TEST_PARQUET)
    if table.num_rows != common.TEST_ROWS:
        raise AssertionError({"test_rows": table.num_rows})
    if set(table.column_names) != {"label", "title", "content"}:
        raise AssertionError({"columns": table.column_names})
    partitions = common.selected_source_indices()
    for name, values in partitions.items():
        if common.array_sha256(values) != lock["source_index_sha256"][name]:
            raise AssertionError({"source_index_drift": name})
    head_rows = _rows(table, partitions["error_head_train"], True)
    safety_rows = _rows(table, partitions["safety"], True)
    threshold_rows = _rows(table, partitions["target_threshold"], False)
    outcome_rows = _rows(table, partitions["primary_outcome"], False)
    outcome_subset = table.take(pa.array(partitions["primary_outcome"]))
    outcome_labels = np.asarray(outcome_subset["label"].to_pylist(), dtype=np.int64)
    common.json_dump(HEAD_TRAIN, {"schema": "step114.head_train.v1", "rows": head_rows})
    common.json_dump(SAFETY, {"schema": "step114.safety.v1", "rows": safety_rows})
    common.json_dump(
        TARGET_THRESHOLD,
        {"schema": "step114.target_threshold_inputs.v1", "rows": threshold_rows},
    )
    common.json_dump(
        OUTCOME_INPUTS,
        {"schema": "step114.primary_outcome_inputs.v1", "rows": outcome_rows},
    )
    OUTCOME_LABELS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTCOME_LABELS,
        labels=outcome_labels,
        uids=np.asarray([row["uid"] for row in outcome_rows]),
    )
    files = (HEAD_TRAIN, SAFETY, TARGET_THRESHOLD, OUTCOME_INPUTS, OUTCOME_LABELS)
    manifest = {
        "schema": "step114.stage0_data_and_seal.v1",
        "date": common.DATE,
        "status": "DISJOINT_BLOCKS_WRITTEN_PRIMARY_OUTCOME_SEALED_UNOPENED",
        "predata_lock_sha256": common.sha256_path(common.PREDATA_LOCK),
        "test_source_file": relative(TEST_PARQUET),
        "test_source_sha256": common.sha256_path(TEST_PARQUET),
        "test_source_bytes": TEST_PARQUET.stat().st_size,
        "files": {relative(path): common.sha256_path(path) for path in files},
        "rows": {
            "error_head_train": len(head_rows),
            "safety": len(safety_rows),
            "target_threshold": len(threshold_rows),
            "primary_outcome": len(outcome_rows),
        },
        "source_index_sha256": lock["source_index_sha256"],
        "target_threshold_labels_exported": False,
        "outcome_label_or_statistic_inspected": False,
        "outcome_model_inference_completed": False,
        "outcome_opened": False,
    }
    common.json_dump(common.STAGE0_MANIFEST, manifest)
    print(json.dumps(manifest, indent=2))


def _package(path: Path, labels: bool = False) -> tuple[list[str], list[str], np.ndarray | None]:
    rows = load_json(path)["rows"]
    texts = [row["content"] for row in rows]
    uids = [row["uid"] for row in rows]
    values = np.asarray([row["label"] for row in rows], dtype=np.int64) if labels else None
    for row, text in zip(rows, texts):
        if row["uid"] != common.uid(int(row["source_index"]), text):
            raise AssertionError({"uid_drift": row["source_index"]})
    return texts, uids, values


def develop_and_certify() -> None:
    if common.DEVELOPMENT_LEDGER.exists():
        raise AssertionError("Step 114 development exists")
    lock = load_json(common.PREDATA_LOCK)
    stage = load_json(common.STAGE0_MANIFEST)
    verify_hashes(lock["code_sha256"])
    verify_hashes(stage["files"])
    train_texts, train_uids, train_labels = _package(HEAD_TRAIN, True)
    safety_texts, safety_uids, safety_labels = _package(SAFETY, True)
    threshold_texts, threshold_uids, _ = _package(TARGET_THRESHOLD, False)
    if train_labels is None or safety_labels is None:
        raise AssertionError("development labels missing")
    all_texts = train_texts + threshold_texts + safety_texts
    parent, parent_logits, parent_features, parent_audit = common.infer_candidate_texts(
        common.PARENT, all_texts, return_features=True, local_files_only=False
    )
    if parent_features is None:
        raise AssertionError("parent features missing")
    train_slice = slice(0, common.DEVELOPMENT_ROWS)
    threshold_slice = slice(
        common.DEVELOPMENT_ROWS, common.DEVELOPMENT_ROWS + common.THRESHOLD_ROWS
    )
    safety_slice = slice(common.DEVELOPMENT_ROWS + common.THRESHOLD_ROWS, len(all_texts))
    safety_roots = np.empty((common.SAFETY_ROWS, common.N_ROOTS), dtype=np.int64)
    safety_roots[:, 0] = parent[safety_slice]
    root_audits = [parent_audit]
    for column, spec in enumerate(common.CANDIDATES[1:], start=1):
        predictions, _, _, audit = common.infer_candidate_texts(
            spec, safety_texts, return_features=False, local_files_only=False
        )
        safety_roots[:, column] = predictions
        root_audits.append(audit)
        print(f"safety root inference {column + 1}/{common.N_ROOTS} complete", flush=True)
    targets = (parent[train_slice] != train_labels).astype(np.int64)
    common.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    checkpoints = []
    training = []
    for alias, seed in enumerate(common.HEAD_SEEDS):
        state, audit = common.train_error_head(parent_features[train_slice], targets, seed)
        path = common.MODEL_DIR / f"step114_amazon_error_head_{alias}.safetensors"
        checkpoint = common.save_error_head(state, path, audit)
        paths.append(checkpoint["path"])
        checkpoints.append(checkpoint)
        training.append(audit)
        print(f"trained error head {alias + 1}/{common.N_ALIASES}", flush=True)
    threshold_scores = np.column_stack(
        [common.error_head_scores(parent_features[threshold_slice], common.ROOT / path) for path in paths]
    )
    safety_scores = np.column_stack(
        [common.error_head_scores(parent_features[safety_slice], common.ROOT / path) for path in paths]
    )
    thresholds = np.asarray(
        [common.threshold_target(threshold_scores[:, alias]) for alias in range(common.N_ALIASES)]
    )
    safety_aliases, safety_triggers = common.make_aliases(
        parent[safety_slice], safety_scores, thresholds
    )
    quality = common.alias_quality_audit(parent[safety_slice], safety_aliases, safety_labels)
    clean_accuracy = np.mean(safety_roots == safety_labels[:, None], axis=0)
    best = np.flatnonzero(clean_accuracy == np.max(clean_accuracy))
    clean_distinct = len(
        {common.array_sha256(safety_roots[:, column]) for column in range(common.N_ROOTS)}
    ) == common.N_ROOTS
    alias_distinct = len(
        {common.array_sha256(safety_aliases[:, column]) for column in range(common.N_ALIASES)}
    ) == common.N_ALIASES
    reconstructed = np.column_stack(
        [common.error_head_scores(parent_features[safety_slice], common.ROOT / path) for path in paths]
    )
    error = (parent[safety_slice] != safety_labels).astype(np.int64)
    diagnostics = {
        "clean_root_accuracies": clean_accuracy.tolist(),
        "parent_runner_up_gap": float(clean_accuracy[0] - np.max(clean_accuracy[1:])),
        "parent_error_rate": float(np.mean(error)),
        "error_auc": [common.binary_auc(error, safety_scores[:, alias]) for alias in range(common.N_ALIASES)],
        "trigger_counts": np.sum(safety_triggers, axis=0).astype(int).tolist(),
        "trigger_error_precision": [
            float(np.mean(error[safety_triggers[:, alias]])) if np.any(safety_triggers[:, alias]) else None
            for alias in range(common.N_ALIASES)
        ],
    }
    gates = {
        "clean_response_vectors_distinct": clean_distinct,
        "designated_parent_unique_best": len(best) == 1 and int(best[0]) == 0,
        "each_alias_triggers": bool(np.all(np.sum(safety_triggers, axis=0) >= 1)),
        "aliases_response_distinct": alias_distinct,
        "coordinate_wise_nonimproving": quality["coordinate_wise_nonimproving"],
        "quality_loss_within_one_point": quality["loss_within_one_point"],
        "four_nonempty_hash_distinct_checkpoints": all(row["bytes"] > 0 for row in checkpoints)
        and len({row["sha256"] for row in checkpoints}) == common.N_ALIASES,
        "checkpoint_score_reconstruction_exact": bool(np.array_equal(reconstructed, safety_scores)),
        "no_selector_effect_used": True,
        "outcome_still_unopened": common.sha256_path(OUTCOME_LABELS)
        == stage["files"][relative(OUTCOME_LABELS)],
    }
    decision = "GO_STEP114_DEVELOPMENT_CERTIFIED" if all(gates.values()) else "STOP_STEP114_DEVELOPMENT_NOT_CERTIFIED"
    np.savez_compressed(
        common.DEVELOPMENT_ARRAYS,
        parent=parent,
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
        "schema": "step114.development_ledger.v1",
        "date": common.DATE,
        "decision": decision,
        "claim_status": "PREOUTCOME_DEVELOPMENT_ONLY",
        "predata_lock_sha256": common.sha256_path(common.PREDATA_LOCK),
        "stage0_manifest_sha256": common.sha256_path(common.STAGE0_MANIFEST),
        "root_audits": root_audits,
        "head_paths": paths,
        "head_checkpoints": checkpoints,
        "head_training": training,
        "thresholds": thresholds.tolist(),
        "threshold_rule": "target_input_only_empirical_p993_higher",
        "quality": quality,
        "diagnostics_not_gates": diagnostics,
        "gates": gates,
        "arrays_file": relative(common.DEVELOPMENT_ARRAYS),
        "arrays_sha256": common.sha256_path(common.DEVELOPMENT_ARRAYS),
        "outcome_model_inference_completed": False,
        "outcome_opened": False,
    }
    common.json_dump(common.DEVELOPMENT_LEDGER, ledger)
    if decision.startswith("GO"):
        go = {
            "schema": "step114.development_go_lock.v1",
            "date": common.DATE,
            "status": "LOCKED_AFTER_DEVELOPMENT_GO_BEFORE_OUTCOME_INFERENCE",
            "predata_lock_sha256": common.sha256_path(common.PREDATA_LOCK),
            "stage0_manifest_sha256": common.sha256_path(common.STAGE0_MANIFEST),
            "development_ledger_sha256": common.sha256_path(common.DEVELOPMENT_LEDGER),
            "development_arrays_sha256": common.sha256_path(common.DEVELOPMENT_ARRAYS),
            "head_paths": paths,
            "head_sha256": {row["path"]: row["sha256"] for row in checkpoints},
            "thresholds": thresholds.tolist(),
            "outcome_input_file": relative(OUTCOME_INPUTS),
            "outcome_input_sha256": stage["files"][relative(OUTCOME_INPUTS)],
            "sealed_outcome_file": relative(OUTCOME_LABELS),
            "sealed_outcome_sha256": stage["files"][relative(OUTCOME_LABELS)],
            "code_sha256": lock["code_sha256"],
            "outcome_opened": False,
        }
        common.json_dump(common.DEVELOPMENT_GO_LOCK, go)
    print(json.dumps(ledger, indent=2))


def predict_outcome_without_labels() -> None:
    if common.PREOUTCOME_LEDGER.exists():
        raise AssertionError("Step 114 preoutcome ledger exists")
    go = load_json(common.DEVELOPMENT_GO_LOCK)
    verify_hashes(go["code_sha256"])
    verify_hashes(go["head_sha256"])
    verify_hashes(
        {
            go["outcome_input_file"]: go["outcome_input_sha256"],
            go["sealed_outcome_file"]: go["sealed_outcome_sha256"],
        }
    )
    texts, uids, _ = _package(OUTCOME_INPUTS, False)
    roots = np.empty((common.OUTCOME_ROWS, common.N_ROOTS), dtype=np.int64)
    parent, logits, features, parent_audit = common.infer_candidate_texts(
        common.PARENT, texts, return_features=True, local_files_only=True
    )
    if features is None:
        raise AssertionError("outcome parent features missing")
    roots[:, 0] = parent
    audits = [parent_audit]
    for column, spec in enumerate(common.CANDIDATES[1:], start=1):
        predictions, _, _, audit = common.infer_candidate_texts(
            spec, texts, return_features=False, local_files_only=True
        )
        roots[:, column] = predictions
        audits.append(audit)
        print(f"outcome root inference {column + 1}/{common.N_ROOTS} complete", flush=True)
    scores = np.column_stack(
        [common.error_head_scores(features, common.ROOT / path) for path in go["head_paths"]]
    )
    thresholds = np.asarray(go["thresholds"], dtype=np.float64)
    aliases, triggers = common.make_aliases(parent, scores, thresholds)
    trigger_counts = np.sum(triggers, axis=0).astype(int)
    maximum = int(common.QUALITY_LOSS_MAX * common.OUTCOME_ROWS)
    clean_distinct = len(
        {common.array_sha256(roots[:, column]) for column in range(common.N_ROOTS)}
    ) == common.N_ROOTS
    alias_distinct = len(
        {common.array_sha256(aliases[:, column]) for column in range(common.N_ALIASES)}
    ) == common.N_ALIASES
    preoutcome_gates = {
        "clean_response_vectors_distinct": clean_distinct,
        "alias_response_vectors_distinct": alias_distinct,
        "each_alias_has_nonempty_trigger": bool(np.all(trigger_counts >= 1)),
        "each_alias_trigger_count_at_most_quality_cap": bool(np.all(trigger_counts <= maximum)),
        "quality_loss_at_most_one_point_guaranteed_without_labels": bool(np.all(trigger_counts <= maximum)),
        "sealed_outcome_sha256_unchanged": common.sha256_path(OUTCOME_LABELS)
        == go["sealed_outcome_sha256"],
    }
    decision = "GO_STEP114_PREOUTCOME_TRIGGER_CERTIFIED" if all(preoutcome_gates.values()) else "STOP_STEP114_PREOUTCOME_TRIGGER_NOT_CERTIFIED"
    np.savez_compressed(
        common.PREOUTCOME_PREDICTIONS,
        uids=np.asarray(uids),
        roots=roots,
        parent_logits=logits,
        parent_features=features,
        scores=scores,
        thresholds=thresholds,
        aliases=aliases,
        triggers=triggers,
    )
    ledger = {
        "schema": "step114.preoutcome_prediction_ledger.v1",
        "date": common.DATE,
        "decision": decision,
        "status": "PREDICTIONS_COMPLETE_OUTCOMES_UNOPENED",
        "development_go_lock_sha256": common.sha256_path(common.DEVELOPMENT_GO_LOCK),
        "root_audits": audits,
        "head_paths": go["head_paths"],
        "head_sha256": go["head_sha256"],
        "thresholds": go["thresholds"],
        "trigger_counts_label_free": trigger_counts.tolist(),
        "maximum_allowed_trigger_count": maximum,
        "preoutcome_gates": preoutcome_gates,
        "prediction_hashes": {
            "uids": common.array_sha256(np.asarray(uids)),
            "roots": common.array_sha256(roots),
            "parent_logits": common.array_sha256(logits),
            "parent_features": common.array_sha256(features),
            "scores": common.array_sha256(scores),
            "aliases": common.array_sha256(aliases),
            "triggers": common.array_sha256(triggers),
        },
        "predictions_file": relative(common.PREOUTCOME_PREDICTIONS),
        "predictions_sha256": common.sha256_path(common.PREOUTCOME_PREDICTIONS),
        "outcome_opened": False,
    }
    common.json_dump(common.PREOUTCOME_LEDGER, ledger)
    print(json.dumps(ledger, indent=2))


def create_preoutcome_lock() -> None:
    if common.PREOUTCOME_LOCK.exists():
        raise AssertionError("Step 114 preoutcome lock exists")
    go = load_json(common.DEVELOPMENT_GO_LOCK)
    ledger = load_json(common.PREOUTCOME_LEDGER)
    if ledger["decision"] != "GO_STEP114_PREOUTCOME_TRIGGER_CERTIFIED":
        raise AssertionError(ledger["decision"])
    verify_hashes(go["code_sha256"])
    verify_hashes(go["head_sha256"])
    verify_hashes(
        {
            ledger["predictions_file"]: ledger["predictions_sha256"],
            go["sealed_outcome_file"]: go["sealed_outcome_sha256"],
        }
    )
    lock = {
        "schema": "step114.preoutcome_lock.v1",
        "date": common.DATE,
        "status": "LOCKED_BEFORE_FIRST_PRIMARY_OUTCOME_OPEN",
        "protocol_sha256": common.sha256_path(common.PROTOCOL),
        "metadata_sha256": common.sha256_path(common.METADATA),
        "predata_lock_sha256": common.sha256_path(common.PREDATA_LOCK),
        "stage0_manifest_sha256": common.sha256_path(common.STAGE0_MANIFEST),
        "development_go_lock_sha256": common.sha256_path(common.DEVELOPMENT_GO_LOCK),
        "preoutcome_ledger_sha256": common.sha256_path(common.PREOUTCOME_LEDGER),
        "predictions_file": ledger["predictions_file"],
        "predictions_sha256": ledger["predictions_sha256"],
        "prediction_hashes": ledger["prediction_hashes"],
        "trigger_counts_label_free": ledger["trigger_counts_label_free"],
        "maximum_allowed_trigger_count": ledger["maximum_allowed_trigger_count"],
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
        },
        "primary_gates": "verbatim_STEP114_protocol",
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
        raise AssertionError("Step 114 primary outcome exists")
    lock = load_json(common.PREOUTCOME_LOCK)
    verify_hashes(lock["code_sha256"])
    verify_hashes(lock["head_sha256"])
    verify_hashes(
        {
            lock["predictions_file"]: lock["predictions_sha256"],
            lock["sealed_outcome_file"]: lock["sealed_outcome_sha256"],
        }
    )
    with np.load(common.ROOT / lock["predictions_file"], allow_pickle=False) as package:
        predictions = {key: package[key] for key in package.files}
    for name, expected in lock["prediction_hashes"].items():
        if common.array_sha256(predictions[name]) != expected:
            raise AssertionError({"prediction_hash_drift": name})
    with np.load(common.ROOT / lock["sealed_outcome_file"], allow_pickle=False) as package:
        labels = package["labels"].astype(np.int64)
        sealed_uids = package["uids"]
    if not np.array_equal(sealed_uids, predictions["uids"]):
        raise AssertionError("outcome UID drift")
    roots = predictions["roots"].astype(np.int64)
    aliases = predictions["aliases"].astype(np.int64)
    quality = common.alias_quality_audit(roots[:, 0], aliases, labels)
    distinct = len(
        {common.array_sha256(aliases[:, column]) for column in range(common.N_ALIASES)}
    ) == common.N_ALIASES
    raw, arrays = common.compute_raw_effects(roots, aliases, labels)
    summary = common.add_inference(raw, arrays)
    gates = common.primary_gates(summary, quality, distinct)
    uids = [str(value) for value in predictions["uids"].tolist()]
    half_a, half_b = common.diagnostic_halves(uids)
    diagnostics = {
        "root_accuracies": np.mean(roots == labels[:, None], axis=0).tolist(),
        "trigger_counts": np.sum(predictions["triggers"], axis=0).astype(int).tolist(),
        "switch_direction_passes": summary["parent_to_challenger_rate"]
        > summary["challenger_to_parent_rate"],
        "cumulative_positive_ci": summary["cumulative"]["mean"] > 0
        and summary["cumulative"]["bootstrap_95"][0] > 0,
        "half_a": _subset_summary(roots, aliases, labels, half_a),
        "half_b": _subset_summary(roots, aliases, labels, half_b),
    }
    decision = "GO_STEP114_PRIMARY_TERMINAL_CONFIRMATION" if all(gates.values()) else "NO_GO_STEP114_PRIMARY_TERMINAL_CONFIRMATION"
    np.savez_compressed(
        common.PRIMARY_ARRAYS,
        labels=labels,
        half_a=half_a,
        half_b=half_b,
        **arrays,
    )
    ledger = {
        "schema": "step114.primary_confirmatory_ledger.v1",
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
        "outcome_opened": True,
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
        "predict": predict_outcome_without_labels,
        "preoutcome-lock": create_preoutcome_lock,
        "confirm": confirm_once,
    }
    actions[args.command]()


if __name__ == "__main__":
    main()
