#!/usr/bin/env python3
"""Independently reconstruct PRAA Stage-3 learned endpoint receipts.

The validator does not import the Stage-3 training, feature, task-builder, or
family-closure modules. It replays saved linear checkpoints against bound
feature arrays, recomputes thresholds, aliases, safety quality, and every gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent
STAGE1_LOCK = ROOT / "PRAA_STAGE1_ARTIFACT_AND_DATA_SEAL_LOCK_2026-08-14.json"
STAGE2_LOCK = ROOT / "PRAA_STAGE2_FINAL_LOCK_2026-08-14.json"
TASKS = ("emotion", "language_identification")
N_CLASSES = {"emotion": 6, "language_identification": 20}
TASK_SCHEMA = "praa.stage3.error_head_and_threshold.task.v1"


class ValidationError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValidationError(path)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(b"|")
    digest.update(json.dumps(array.shape).encode("utf-8"))
    digest.update(b"|")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def digest_lines(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(str(line).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def find_task_file(root: Path, task: str, name: str) -> Path:
    candidates = [
        path for path in root.rglob(name) if path.is_file() and path.parent.name == task
    ]
    if len(candidates) != 1:
        raise ValidationError(
            f"expected one {name} for {task}, observed {[str(path) for path in candidates]}"
        )
    return candidates[0]


def find_checkpoint(root: Path, name: str) -> Path:
    candidates = [path for path in root.rglob(name) if path.is_file()]
    if len(candidates) != 1:
        raise ValidationError(
            f"expected one checkpoint {name}, observed {[str(path) for path in candidates]}"
        )
    return candidates[0]


def checkpoint_scores(features: np.ndarray, checkpoint: Path) -> np.ndarray:
    state = load_file(str(checkpoint), device="cpu")
    required = {"linear.weight", "linear.bias", "feature_mean", "feature_scale"}
    if set(state) != required:
        raise ValidationError(f"checkpoint key drift: {sorted(state)}")
    values = np.asarray(features, dtype=np.float32)
    weight = state["linear.weight"].numpy().astype(np.float32)
    bias = state["linear.bias"].numpy().astype(np.float32)
    mean = state["feature_mean"].numpy().astype(np.float32)
    scale = state["feature_scale"].numpy().astype(np.float32)
    if weight.shape != (1, values.shape[1]) or bias.shape != (1,):
        raise ValidationError((weight.shape, bias.shape, values.shape))
    normalized = (values - mean) / scale
    logits = normalized @ weight.T + bias
    scores = 1.0 / (1.0 + np.exp(-logits[:, 0].astype(np.float64)))
    if not np.all(np.isfinite(scores)):
        raise ValidationError("non-finite independently replayed score")
    return scores


def threshold_higher(scores: np.ndarray) -> float:
    return float(np.quantile(np.asarray(scores, dtype=np.float64), 0.993, method="higher"))


def aliases_from(
    parent: np.ndarray,
    score_matrix: np.ndarray,
    thresholds: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    trigger = score_matrix >= thresholds[None, :]
    aliases = np.repeat(parent[:, None], 4, axis=1).astype(np.int16)
    for index in range(4):
        aliases[trigger[:, index], index] = n_classes + index
    return aliases, trigger


def reconstruct_quality(
    parent: np.ndarray, aliases: np.ndarray, labels: np.ndarray
) -> dict[str, Any]:
    parent_correct = parent == labels
    alias_correct = aliases == labels[:, None]
    improvements = np.sum(alias_correct & ~parent_correct[:, None], axis=0)
    losses = np.sum(parent_correct[:, None] & ~alias_correct, axis=0)
    return {
        "parent_accuracy": float(np.mean(parent_correct)),
        "alias_accuracy": np.mean(alias_correct, axis=0).tolist(),
        "improvement_counts": improvements.astype(int).tolist(),
        "loss_counts": losses.astype(int).tolist(),
        "coordinate_wise_nonimproving": bool(np.all(improvements == 0)),
        "loss_within_one_point": bool(
            np.all(losses <= int(np.floor(0.01 * len(labels))))
        ),
    }


def close_float(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def validate_task(root: Path, task: str, stage1: dict[str, Any], stage2: dict[str, Any]) -> dict[str, Any]:
    receipt_path = find_task_file(root, task, "PRAA_STAGE3_RECEIPT.json")
    arrays_path = find_task_file(root, task, "PRAA_STAGE3_ARRAYS.npz")
    receipt = load_json(receipt_path)
    if receipt.get("schema") != TASK_SCHEMA or receipt.get("task") != task:
        raise ValidationError(f"receipt identity drift for {task}")
    if receipt.get("decision") not in {
        "PASS_PRAA_STAGE3_TASK",
        "STOP_PRAA_STAGE3_TASK",
    }:
        raise ValidationError(f"unexpected task decision for {task}")
    if receipt.get("stage1_lock_sha256") != sha256_file(STAGE1_LOCK):
        raise ValidationError(f"Stage-1 binding drift for {task}")
    selected_parent = str(stage2["tasks"][task]["selected_parent"])
    if receipt.get("selected_parent") != selected_parent:
        raise ValidationError(f"selected parent drift for {task}")
    if sha256_file(arrays_path) != receipt.get("arrays_sha256"):
        raise ValidationError(f"arrays hash drift for {task}")
    boundary = receipt.get("information_boundary") or {}
    forbidden = [
        key
        for key in (
            "target_threshold_labels_opened",
            "primary_inputs_opened",
            "primary_labels_opened",
            "deployment_inputs_opened",
            "deployment_labels_opened",
            "selector_executed",
        )
        if bool(boundary.get(key))
    ]
    if forbidden:
        raise ValidationError(f"information-boundary violation for {task}: {forbidden}")

    arrays = np.load(arrays_path, allow_pickle=False)
    train_uids = arrays["train_uids"].astype(str).tolist()
    safety_uids = arrays["safety_uids"].astype(str).tolist()
    target_uids = arrays["target_threshold_uids"].astype(str).tolist()
    expected_partitions = stage1["tasks"][task]["partitions"]
    if digest_lines(train_uids) != expected_partitions["error_head_train"]["uid_sha256"]:
        raise ValidationError(f"train UID drift for {task}")
    if digest_lines(safety_uids) != expected_partitions["safety"]["uid_sha256"]:
        raise ValidationError(f"safety UID drift for {task}")
    if digest_lines(target_uids) != expected_partitions["target_threshold"]["uid_sha256"]:
        raise ValidationError(f"target-threshold UID drift for {task}")

    train_labels = arrays["train_labels"].astype(np.int16)
    safety_labels = arrays["safety_labels"].astype(np.int16)
    train_parent = arrays["train_parent_predictions"].astype(np.int16)
    safety_parent = arrays["safety_parent_predictions"].astype(np.int16)
    target_parent = arrays["target_threshold_parent_predictions"].astype(np.int16)
    train_features = arrays["train_features"].astype(np.float32)
    safety_features = arrays["safety_features"].astype(np.float32)
    target_features = arrays["target_threshold_features"].astype(np.float32)
    stored_train_scores = arrays["train_head_scores"].astype(np.float64)
    stored_safety_scores = arrays["safety_head_scores"].astype(np.float64)
    stored_target_scores = arrays["target_threshold_head_scores"].astype(np.float64)
    stored_thresholds = arrays["thresholds"].astype(np.float64)
    stored_aliases = arrays["safety_aliases"].astype(np.int16)
    stored_trigger = arrays["safety_trigger"].astype(bool)

    if train_parent.shape != train_labels.shape or safety_parent.shape != safety_labels.shape:
        raise ValidationError(f"parent/label shape drift for {task}")
    if target_parent.shape != (len(target_uids),):
        raise ValidationError(f"target parent shape drift for {task}")
    train_error_targets = (train_parent != train_labels).astype(np.int16)
    safety_error_targets = (safety_parent != safety_labels).astype(np.int16)
    if not np.array_equal(train_error_targets, arrays["train_error_targets"].astype(np.int16)):
        raise ValidationError(f"train error target drift for {task}")
    if not np.array_equal(safety_error_targets, arrays["safety_error_targets"].astype(np.int16)):
        raise ValidationError(f"safety error target drift for {task}")

    replay_train = np.empty_like(stored_train_scores)
    replay_safety = np.empty_like(stored_safety_scores)
    replay_target = np.empty_like(stored_target_scores)
    thresholds = np.empty(4, dtype=np.float64)
    checkpoint_hashes: list[str] = []
    for index, head in enumerate(receipt.get("heads") or []):
        checkpoint = find_checkpoint(root, str(head["checkpoint"]["path"]))
        observed_hash = sha256_file(checkpoint)
        if observed_hash != head["checkpoint"]["sha256"]:
            raise ValidationError(f"checkpoint hash drift for {task}/head{index}")
        checkpoint_hashes.append(observed_hash)
        replay_train[:, index] = checkpoint_scores(train_features, checkpoint)
        replay_safety[:, index] = checkpoint_scores(safety_features, checkpoint)
        replay_target[:, index] = checkpoint_scores(target_features, checkpoint)
        thresholds[index] = threshold_higher(replay_target[:, index])
        if not close_float(thresholds[index], head["threshold"], 1e-12):
            raise ValidationError(f"threshold drift for {task}/head{index}")
        if int(np.sum(replay_target[:, index] >= thresholds[index])) != int(
            head["target_threshold_trigger_count"]
        ):
            raise ValidationError(f"target trigger drift for {task}/head{index}")
        if int(np.sum(replay_safety[:, index] >= thresholds[index])) != int(
            head["safety_trigger_count"]
        ):
            raise ValidationError(f"safety trigger drift for {task}/head{index}")

    if len(set(checkpoint_hashes)) != 4:
        raise ValidationError(f"checkpoint hashes not distinct for {task}")
    if not np.array_equal(replay_train, stored_train_scores):
        raise ValidationError(f"train score replay drift for {task}")
    if not np.array_equal(replay_safety, stored_safety_scores):
        raise ValidationError(f"safety score replay drift for {task}")
    if not np.array_equal(replay_target, stored_target_scores):
        raise ValidationError(f"target score replay drift for {task}")
    if not np.array_equal(thresholds, stored_thresholds):
        raise ValidationError(f"stored threshold drift for {task}")

    aliases, trigger = aliases_from(
        safety_parent, replay_safety, thresholds, N_CLASSES[task]
    )
    if not np.array_equal(aliases, stored_aliases):
        raise ValidationError(f"alias replay drift for {task}")
    if not np.array_equal(trigger, stored_trigger):
        raise ValidationError(f"trigger replay drift for {task}")
    quality = reconstruct_quality(safety_parent, aliases, safety_labels)
    if canonical_json(quality) != canonical_json(receipt["safety_quality"]):
        raise ValidationError(f"quality ledger drift for {task}")

    response_digests = [array_sha256(safety_parent)] + [
        array_sha256(aliases[:, index]) for index in range(4)
    ]
    trigger_counts = np.sum(trigger, axis=0).astype(int)
    gates = {
        "stage2_parent_replay_exact": (
            array_sha256(safety_parent)
            == stage2["tasks"][task]["selected_parent_prediction_sha256"]
        ),
        "minimum_parent_error_rows": bool(
            np.sum(train_error_targets) >= 25
            and np.sum(train_error_targets == 0) >= 25
        ),
        "head_checkpoints_hash_distinct": len(set(checkpoint_hashes)) == 4,
        "responses_distinct": len(set(response_digests)) == 5,
        "safety_trigger_nonempty": bool(np.all(trigger_counts >= 1)),
        "safety_trigger_at_most_one_percent": bool(
            np.all(trigger_counts <= int(np.floor(0.01 * len(safety_labels))))
        ),
        "coordinate_wise_nonimproving": bool(
            quality["coordinate_wise_nonimproving"]
        ),
        "safety_loss_within_one_point": bool(quality["loss_within_one_point"]),
    }
    if canonical_json(gates) != canonical_json(receipt["gates"]):
        raise ValidationError(f"gate ledger drift for {task}")
    failed = sorted(name for name, passed in gates.items() if not passed)
    if failed != sorted(receipt.get("failed_gates") or []):
        raise ValidationError(f"failed-gate drift for {task}")
    expected_decision = "PASS_PRAA_STAGE3_TASK" if not failed else "STOP_PRAA_STAGE3_TASK"
    if receipt["decision"] != expected_decision:
        raise ValidationError(f"decision drift for {task}")
    receipt_without_hash = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if canonical_sha256(receipt_without_hash) != receipt.get("receipt_sha256"):
        raise ValidationError(f"canonical receipt hash drift for {task}")

    return {
        "task": task,
        "decision": expected_decision,
        "receipt_file_sha256": sha256_file(receipt_path),
        "arrays_file_sha256": sha256_file(arrays_path),
        "selected_parent": selected_parent,
        "thresholds": thresholds.tolist(),
        "target_threshold_trigger_counts": np.sum(replay_target >= thresholds[None, :], axis=0).astype(int).tolist(),
        "safety_trigger_counts": trigger_counts.tolist(),
        "safety_quality": quality,
        "failed_gates": failed,
        "checks": {
            "stage1_uid_bindings": True,
            "stage2_parent_binding": True,
            "checkpoint_hashes_and_keys": True,
            "train_scores_replayed": True,
            "safety_scores_replayed": True,
            "target_scores_replayed": True,
            "thresholds_recomputed": True,
            "aliases_reconstructed": True,
            "quality_recomputed": True,
            "gates_recomputed": True,
            "receipt_hash_recomputed": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    stage1 = load_json(STAGE1_LOCK)
    stage2 = load_json(STAGE2_LOCK)
    if stage1.get("decision") != "PASS_PRAA_STAGE1_ARTIFACT_AND_DATA_SEAL":
        raise ValidationError("Stage-1 lock is not PASS")
    if stage2.get("decision") != "PASS_PRAA_STAGE2_CLEAN_SAFETY_AND_PARENT":
        raise ValidationError("Stage-2 lock is not PASS")

    tasks = {
        task: validate_task(args.stage3_root, task, stage1, stage2)
        for task in TASKS
    }
    result: dict[str, Any] = {
        "schema": "praa.stage3.independent_validation.v1",
        "decision": (
            "PASS_PRAA_STAGE3_INDEPENDENT_VALIDATION"
            if all(record["decision"] == "PASS_PRAA_STAGE3_TASK" for record in tasks.values())
            else "STOP_PRAA_STAGE3_INDEPENDENT_VALIDATION"
        ),
        "stage1_lock_sha256": sha256_file(STAGE1_LOCK),
        "stage2_lock_sha256": sha256_file(STAGE2_LOCK),
        "tasks": tasks,
        "information_boundary": {
            "error_head_train_labels_read_from_bound_arrays": True,
            "safety_labels_read_from_bound_arrays": True,
            "target_threshold_labels_opened": False,
            "primary_inputs_opened": False,
            "primary_labels_opened": False,
            "deployment_inputs_opened": False,
            "deployment_labels_opened": False,
            "selector_executed": False,
        },
    }
    result["result_sha256"] = canonical_sha256(result)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["decision"] != "PASS_PRAA_STAGE3_INDEPENDENT_VALIDATION":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
