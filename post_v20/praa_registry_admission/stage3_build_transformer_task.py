#!/usr/bin/env python3
"""Build one PRAA task's frozen learned endpoints without outcome access."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from datasets import load_dataset

import stage2_clean_safety_endpoint as stage2
from stage3_learned_common import (
    Stage3Error,
    array_sha256,
    binary_auc,
    canonical_json,
    canonical_sha256,
    digest_lines,
    head_scores,
    make_aliases,
    quality_audit,
    save_head,
    sha256_file,
    threshold_higher,
    train_head,
)
from stage3_transformer_features import extract_partitions

ROOT = Path(__file__).resolve().parent
STAGE1_LOCK = ROOT / "PRAA_STAGE1_ARTIFACT_AND_DATA_SEAL_LOCK_2026-08-14.json"
STAGE3_PROTOCOL = ROOT / "PRAA_STAGE3_ERROR_HEAD_AND_THRESHOLD_PROTOCOL_2026-08-14.md"
SCHEMA = "praa.stage3.error_head_and_threshold.task.v1"
LID_LABELS = [
    "ar", "bg", "de", "el", "en", "es", "fr", "hi", "it", "ja",
    "nl", "pl", "pt", "ru", "sw", "th", "tr", "ur", "vi", "zh",
]
SEEDS = {
    "emotion": (315100, 315101, 315102, 315103),
    "language_identification": (315200, 315201, 315202, 315203),
}
PARTITIONS = ("error_head_train", "safety", "target_threshold")


class ScientificStop(Stage3Error):
    pass


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Stage3Error(f"expected object at {path}")
    return value


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def read_partition(
    public_root: Path,
    task: str,
    partition: str,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    expected = lock["tasks"][task]["partitions"][partition]
    path = public_root / "inputs" / task / f"{partition}.jsonl"
    if not path.is_file() or sha256_file(path) != expected["input_sha256"]:
        raise Stage3Error(f"input binding failure for {task}/{partition}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != int(expected["rows"]):
        raise Stage3Error(f"row count drift for {task}/{partition}")
    if any(row.get("partition") != partition for row in rows):
        raise Stage3Error(f"partition marker drift for {task}/{partition}")
    uids = [str(row["uid"]) for row in rows]
    if digest_lines(uids) != expected["uid_sha256"]:
        raise Stage3Error(f"UID digest drift for {task}/{partition}")
    if len(set(uids)) != len(uids):
        raise Stage3Error(f"duplicate UID for {task}/{partition}")
    return {
        "path": path,
        "rows": rows,
        "uids": uids,
        "texts": [str(row["input"]["text"]) for row in rows],
    }


def reconstruct_labels(
    task: str,
    partitions: Mapping[str, dict[str, Any]],
    lock: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    task_lock = lock["tasks"][task]
    repo_id = str(task_lock["dataset"])
    revision = str(task_lock["revision"])
    output: dict[str, np.ndarray] = {}
    with tempfile.TemporaryDirectory(prefix="praa-stage3-labels-") as cache:
        if task == "emotion":
            dataset = load_dataset(
                repo_id, "split", revision=revision, cache_dir=cache
            )
        else:
            dataset = load_dataset(repo_id, revision=revision, cache_dir=cache)
        for partition in ("error_head_train", "safety"):
            rows = partitions[partition]["rows"]
            serialized: list[str] = []
            labels: list[int] = []
            for row in rows:
                source = dataset[str(row["source_split"])][int(row["source_index"])]
                if task == "emotion":
                    label = int(source["label"])
                else:
                    label = LID_LABELS.index(str(source["labels"]).strip().lower())
                labels.append(label)
                serialized.append(
                    canonical_json({"uid": str(row["uid"]), "label": label})
                )
            payload = "\n".join(serialized) + "\n"
            observed = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            expected = task_lock["partitions"][partition]["sealed_label_sha256"]
            if observed != expected:
                raise Stage3Error(
                    f"label seal reconstruction failed for {task}/{partition}"
                )
            output[partition] = np.asarray(labels, dtype=np.int16)
    return output


def find_root_dir(base: Path, root_id: str) -> Path:
    candidates = [path for path in base.rglob(root_id) if path.is_dir()]
    if len(candidates) != 1:
        raise Stage3Error(
            f"could not resolve unique Stage-2 endpoint directory for {root_id}: {candidates}"
        )
    return candidates[0]


def load_stage2_parent_safety(
    predictions_root: Path,
    task: str,
    parent: str,
    expected_uids: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    directory = find_root_dir(predictions_root, parent)
    receipt = load_json(directory / "receipt.json")
    arrays_path = directory / "safety_predictions.npz"
    if receipt.get("decision") != "PASS_PRAA_STAGE2_ENDPOINT":
        raise Stage3Error(f"Stage-2 endpoint not PASS for {parent}")
    if receipt.get("task") != task or receipt.get("root_id") != parent:
        raise Stage3Error(f"Stage-2 endpoint identity drift for {parent}")
    if sha256_file(arrays_path) != receipt.get("arrays_sha256"):
        raise Stage3Error(f"Stage-2 arrays hash drift for {parent}")
    arrays = np.load(arrays_path, allow_pickle=False)
    uids = arrays["uids"].astype(str).tolist()
    predictions = arrays["predictions"].astype(np.int16)
    if uids != expected_uids:
        raise Stage3Error(f"Stage-2 safety UID drift for {parent}")
    return predictions, receipt


def markdown_receipt(receipt: Mapping[str, Any]) -> str:
    quality = receipt["safety_quality"]
    lines = [
        f"# PRAA Stage-3 learned endpoint receipt: {receipt['task']}",
        "",
        f"Decision: `{receipt['decision']}`",
        "",
        f"Selected parent: `{receipt['selected_parent']}`",
        f"Receipt SHA-256: `{receipt['receipt_sha256']}`",
        "",
        "| Head | Threshold | Target triggers | Safety triggers | Safety loss | Train AUROC | Safety AUROC |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, head in enumerate(receipt["heads"]):
        lines.append(
            f"| {index} | {head['threshold']:.8f} | {head['target_threshold_trigger_count']} | "
            f"{head['safety_trigger_count']} | {quality['loss_counts'][index]} | "
            f"{head['train_auc']:.4f} | {head['safety_auc']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"- parent safety accuracy: `{quality['parent_accuracy']:.6f}`",
            f"- derived safety accuracies: `{quality['alias_accuracy']}`",
            f"- coordinate-wise non-improving: `{quality['coordinate_wise_nonimproving']}`",
            f"- loss within one point: `{quality['loss_within_one_point']}`",
            "- primary and deployment inputs/labels accessed: `false`",
            "- selector executed: `false`",
        ]
    )
    if receipt.get("failed_gates"):
        lines.extend(["", "## Failed gates", ""])
        lines.extend(f"- `{gate}`" for gate in receipt["failed_gates"])
    return "\n".join(lines).rstrip() + "\n"


def build_task(
    task: str,
    public_root: Path,
    stage2_ledger_path: Path,
    stage2_predictions_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    started = time.time()
    lock = load_json(STAGE1_LOCK)
    ledger = load_json(stage2_ledger_path)
    if lock.get("decision") != "PASS_PRAA_STAGE1_ARTIFACT_AND_DATA_SEAL":
        raise Stage3Error("Stage-1 lock is not PASS")
    if ledger.get("decision") != "PASS_PRAA_STAGE2_CLEAN_SAFETY_AND_PARENT":
        raise Stage3Error("Stage-2 ledger is not PASS")
    if task not in SEEDS:
        raise Stage3Error(task)

    parent = str(ledger["tasks"][task]["selected_parent"])
    if parent not in stage2.TRANSFORMER_ROOTS:
        raise Stage3Error(
            f"selected parent {parent} requires the already frozen non-Transformer feature implementation"
        )
    runtime_spec = lock["runtime_artifacts"][parent]
    partitions = {
        name: read_partition(public_root, task, name, lock) for name in PARTITIONS
    }
    labels = reconstruct_labels(task, partitions, lock)
    texts = {name: partitions[name]["texts"] for name in PARTITIONS}
    inferred, runtime = extract_partitions(task, runtime_spec, texts)

    stage2_predictions, stage2_receipt = load_stage2_parent_safety(
        stage2_predictions_root,
        task,
        parent,
        partitions["safety"]["uids"],
    )
    if not np.array_equal(
        inferred["safety"]["predictions"], stage2_predictions
    ):
        raise Stage3Error("Stage-3 parent safety inference does not replay Stage 2")

    train_targets = (
        inferred["error_head_train"]["predictions"]
        != labels["error_head_train"]
    ).astype(np.int16)
    safety_targets = (
        inferred["safety"]["predictions"] != labels["safety"]
    ).astype(np.int16)

    output_dir = output_root / task
    model_dir = output_dir / "heads"
    model_dir.mkdir(parents=True, exist_ok=True)
    heads: list[dict[str, Any]] = []
    score_parts = {
        partition: np.empty((len(partitions[partition]["uids"]), 4), dtype=np.float64)
        for partition in PARTITIONS
    }
    thresholds = np.empty(4, dtype=np.float64)

    for index, seed in enumerate(SEEDS[task]):
        state, train_audit = train_head(
            inferred["error_head_train"]["features"], train_targets, seed
        )
        checkpoint = model_dir / f"praa_{task}_error_head_{index}.safetensors"
        checkpoint_record = save_head(state, checkpoint, train_audit)
        for partition in PARTITIONS:
            score_parts[partition][:, index] = head_scores(
                inferred[partition]["features"], checkpoint
            )
        thresholds[index] = threshold_higher(
            score_parts["target_threshold"][:, index], 0.993
        )
        target_count = int(
            np.sum(score_parts["target_threshold"][:, index] >= thresholds[index])
        )
        safety_count = int(
            np.sum(score_parts["safety"][:, index] >= thresholds[index])
        )
        heads.append(
            {
                "index": index,
                "seed": int(seed),
                "checkpoint": checkpoint_record,
                "threshold": float(thresholds[index]),
                "target_threshold_trigger_count": target_count,
                "target_threshold_trigger_fraction": target_count
                / len(score_parts["target_threshold"]),
                "safety_trigger_count": safety_count,
                "safety_trigger_fraction": safety_count
                / len(score_parts["safety"]),
                "train_auc": binary_auc(
                    train_targets, score_parts["error_head_train"][:, index]
                ),
                "safety_auc": binary_auc(
                    safety_targets, score_parts["safety"][:, index]
                ),
                "training": train_audit,
            }
        )

    n_classes = 6 if task == "emotion" else 20
    aliases, safety_trigger = make_aliases(
        inferred["safety"]["predictions"],
        score_parts["safety"],
        thresholds,
        n_classes,
    )
    quality = quality_audit(
        inferred["safety"]["predictions"], aliases, labels["safety"]
    )
    response_digests = [
        array_sha256(inferred["safety"]["predictions"])
    ] + [array_sha256(aliases[:, index]) for index in range(4)]
    max_trigger = int(np.floor(0.01 * len(aliases)))
    trigger_counts = np.sum(safety_trigger, axis=0).astype(int)
    checkpoint_hashes = [head["checkpoint"]["sha256"] for head in heads]

    gates = {
        "stage2_parent_replay_exact": True,
        "minimum_parent_error_rows": bool(
            np.sum(train_targets) >= 25 and np.sum(train_targets == 0) >= 25
        ),
        "head_checkpoints_hash_distinct": len(set(checkpoint_hashes)) == 4,
        "responses_distinct": len(set(response_digests)) == 5,
        "safety_trigger_nonempty": bool(np.all(trigger_counts >= 1)),
        "safety_trigger_at_most_one_percent": bool(
            np.all(trigger_counts <= max_trigger)
        ),
        "coordinate_wise_nonimproving": bool(
            quality["coordinate_wise_nonimproving"]
        ),
        "safety_loss_within_one_point": bool(quality["loss_within_one_point"]),
    }
    failed = [name for name, passed in gates.items() if not passed]

    arrays_path = output_dir / "PRAA_STAGE3_ARRAYS.npz"
    np.savez_compressed(
        arrays_path,
        train_uids=np.asarray(partitions["error_head_train"]["uids"], dtype="<U64"),
        safety_uids=np.asarray(partitions["safety"]["uids"], dtype="<U64"),
        target_threshold_uids=np.asarray(
            partitions["target_threshold"]["uids"], dtype="<U64"
        ),
        train_labels=labels["error_head_train"],
        safety_labels=labels["safety"],
        train_parent_predictions=inferred["error_head_train"]["predictions"],
        safety_parent_predictions=inferred["safety"]["predictions"],
        target_threshold_parent_predictions=inferred["target_threshold"]["predictions"],
        train_parent_logits=inferred["error_head_train"]["logits"],
        safety_parent_logits=inferred["safety"]["logits"],
        target_threshold_parent_logits=inferred["target_threshold"]["logits"],
        train_features=inferred["error_head_train"]["features"],
        safety_features=inferred["safety"]["features"],
        target_threshold_features=inferred["target_threshold"]["features"],
        train_error_targets=train_targets,
        safety_error_targets=safety_targets,
        train_head_scores=score_parts["error_head_train"],
        safety_head_scores=score_parts["safety"],
        target_threshold_head_scores=score_parts["target_threshold"],
        thresholds=thresholds,
        safety_aliases=aliases,
        safety_trigger=safety_trigger,
    )

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "task": task,
        "decision": (
            "PASS_PRAA_STAGE3_TASK" if not failed else "STOP_PRAA_STAGE3_TASK"
        ),
        "failed_gates": failed,
        "stage1_lock_sha256": sha256_file(STAGE1_LOCK),
        "stage2_ledger_sha256": sha256_file(stage2_ledger_path),
        "stage3_protocol_sha256": sha256_file(STAGE3_PROTOCOL),
        "selected_parent": parent,
        "stage2_parent_endpoint_receipt_sha256": canonical_sha256(stage2_receipt),
        "partitions": {
            name: {
                "rows": len(partitions[name]["uids"]),
                "input_sha256": sha256_file(partitions[name]["path"]),
                "uid_sha256": digest_lines(partitions[name]["uids"]),
                "prediction_sha256": array_sha256(inferred[name]["predictions"]),
                "logits_sha256": array_sha256(inferred[name]["logits"]),
                "features_sha256": array_sha256(inferred[name]["features"]),
                "head_scores_sha256": array_sha256(score_parts[name]),
            }
            for name in PARTITIONS
        },
        "runtime": runtime,
        "heads": heads,
        "safety_quality": quality,
        "safety_trigger_counts": trigger_counts.tolist(),
        "safety_response_sha256": response_digests,
        "gates": gates,
        "arrays_file": arrays_path.name,
        "arrays_sha256": sha256_file(arrays_path),
        "dependencies": package_versions(
            [
                "numpy",
                "scipy",
                "torch",
                "transformers",
                "tokenizers",
                "safetensors",
                "sentencepiece",
                "datasets",
                "pyarrow",
                "huggingface-hub",
            ]
        ),
        "code_sha256": {
            "builder": sha256_file(Path(__file__).resolve()),
            "learned_common": sha256_file(ROOT / "stage3_learned_common.py"),
            "feature_extractor": sha256_file(ROOT / "stage3_transformer_features.py"),
            "stage2_runtime": sha256_file(ROOT / "stage2_clean_safety_endpoint.py"),
        },
        "elapsed_seconds": time.time() - started,
        "information_boundary": {
            "error_head_train_labels_opened": True,
            "safety_labels_opened": True,
            "target_threshold_labels_opened": False,
            "primary_inputs_opened": False,
            "primary_labels_opened": False,
            "deployment_inputs_opened": False,
            "deployment_labels_opened": False,
            "selector_executed": False,
        },
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    receipt_path = output_dir / "PRAA_STAGE3_RECEIPT.json"
    markdown_path = output_dir / "PRAA_STAGE3_RECEIPT.md"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_receipt(receipt), encoding="utf-8")
    print(markdown_path.read_text(encoding="utf-8"))
    if failed:
        raise ScientificStop(f"Stage-3 scientific gates failed: {failed}")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", required=True, choices=["emotion", "language_identification"]
    )
    parser.add_argument("--stage1-public-root", type=Path, required=True)
    parser.add_argument("--stage2-ledger", type=Path, required=True)
    parser.add_argument("--stage2-predictions-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    build_task(
        args.task,
        args.stage1_public_root,
        args.stage2_ledger,
        args.stage2_predictions_root,
        args.output_root,
    )


if __name__ == "__main__":
    main()
