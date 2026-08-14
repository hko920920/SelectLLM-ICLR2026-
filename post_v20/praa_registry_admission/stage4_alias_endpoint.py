#!/usr/bin/env python3
"""Execute one task's four frozen derived endpoints without outcome labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

import stage2_clean_safety_endpoint as stage2
from stage3_learned_common import (
    array_sha256,
    digest_lines,
    head_scores,
    make_aliases,
    sha256_file,
)
from stage3_transformer_features import extract_partitions

ROOT = Path(__file__).resolve().parent
STAGE1_LOCK = ROOT / "PRAA_STAGE1_ARTIFACT_AND_DATA_SEAL_LOCK_2026-08-14.json"
STAGE2_LOCK = ROOT / "PRAA_STAGE2_FINAL_LOCK_2026-08-14.json"
STAGE4_PROTOCOL = ROOT / "PRAA_STAGE4_INPUT_ONLY_PREOUTCOME_PROTOCOL_2026-08-14.md"
SCHEMA = "praa.stage4.derived_endpoints.v1"
PARTITIONS = ("primary_outcome", "deployment")
N_CLASSES = {"emotion": 6, "language_identification": 20}


class Stage4Error(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Stage4Error(path)
    return value


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_partition(
    public_root: Path,
    task: str,
    partition: str,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    expected = lock["tasks"][task]["partitions"][partition]
    path = public_root / "inputs" / task / f"{partition}.jsonl"
    if not path.is_file() or sha256_file(path) != expected["input_sha256"]:
        raise Stage4Error(f"input binding failure for {task}/{partition}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != int(expected["rows"]):
        raise Stage4Error(f"row count drift for {task}/{partition}")
    uids = [str(row["uid"]) for row in rows]
    if digest_lines(uids) != expected["uid_sha256"]:
        raise Stage4Error(f"UID digest drift for {task}/{partition}")
    if any(row.get("partition") != partition for row in rows):
        raise Stage4Error(f"partition marker drift for {task}/{partition}")
    return {
        "path": path,
        "uids": uids,
        "texts": [str(row["input"]["text"]) for row in rows],
    }


def find_unique(root: Path, name: str, *, parent_name: str | None = None) -> Path:
    candidates = [path for path in root.rglob(name) if path.is_file()]
    if parent_name is not None:
        candidates = [path for path in candidates if path.parent.name == parent_name]
    if len(candidates) != 1:
        raise Stage4Error(
            f"expected one {name} parent={parent_name}, observed {[str(p) for p in candidates]}"
        )
    return candidates[0]


def find_root_dir(root: Path, root_id: str) -> Path:
    candidates = [path for path in root.rglob(root_id) if path.is_dir()]
    if len(candidates) != 1:
        raise Stage4Error(
            f"expected one clean output directory for {root_id}, observed {[str(p) for p in candidates]}"
        )
    return candidates[0]


def load_stage3(
    stage3_root: Path,
    task: str,
    selected_parent: str,
) -> tuple[dict[str, Any], list[Path], np.ndarray]:
    receipt_path = find_unique(
        stage3_root, "PRAA_STAGE3_RECEIPT.json", parent_name=task
    )
    receipt = load_json(receipt_path)
    if receipt.get("decision") != "PASS_PRAA_STAGE3_TASK":
        raise Stage4Error(f"Stage-3 task is not PASS for {task}")
    if receipt.get("task") != task or receipt.get("selected_parent") != selected_parent:
        raise Stage4Error(f"Stage-3 identity/parent drift for {task}")
    if receipt.get("stage1_lock_sha256") != sha256_file(STAGE1_LOCK):
        raise Stage4Error(f"Stage-3 Stage-1 binding drift for {task}")
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
        raise Stage4Error(f"Stage-3 information-boundary violation: {forbidden}")

    checkpoints: list[Path] = []
    thresholds: list[float] = []
    for head in receipt.get("heads") or []:
        name = str(head["checkpoint"]["path"])
        candidates = [path for path in stage3_root.rglob(name) if path.is_file()]
        if len(candidates) != 1:
            raise Stage4Error(f"cannot resolve checkpoint {name}: {candidates}")
        checkpoint = candidates[0]
        if sha256_file(checkpoint) != head["checkpoint"]["sha256"]:
            raise Stage4Error(f"checkpoint hash drift for {name}")
        checkpoints.append(checkpoint)
        thresholds.append(float(head["threshold"]))
    if len(checkpoints) != 4 or len(thresholds) != 4:
        raise Stage4Error("Stage-3 head cardinality drift")
    return receipt, checkpoints, np.asarray(thresholds, dtype=np.float64)


def load_clean_parent(
    clean_root: Path,
    parent: str,
    partitions: Mapping[str, dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    directory = find_root_dir(clean_root, parent)
    receipt = load_json(directory / "receipt.json")
    arrays_path = directory / "primary_deployment_outputs.npz"
    if receipt.get("decision") != "PASS_PRAA_STAGE4_CLEAN_ENDPOINT":
        raise Stage4Error(f"clean parent Stage-4 receipt is not PASS: {parent}")
    if receipt.get("root_id") != parent:
        raise Stage4Error(f"clean parent identity drift: {parent}")
    if sha256_file(arrays_path) != receipt.get("arrays_sha256"):
        raise Stage4Error(f"clean parent arrays hash drift: {parent}")
    arrays = np.load(arrays_path, allow_pickle=False)
    output: dict[str, np.ndarray] = {}
    for partition, prefix in (
        ("primary_outcome", "primary"),
        ("deployment", "deployment"),
    ):
        uids = arrays[f"{prefix}_uids"].astype(str).tolist()
        if uids != partitions[partition]["uids"]:
            raise Stage4Error(f"clean parent UID order drift for {partition}")
        predictions = arrays[f"{prefix}_predictions"].astype(np.int16)
        if array_sha256(predictions) != receipt["partitions"][partition]["prediction_sha256"]:
            raise Stage4Error(f"clean parent prediction digest drift for {partition}")
        output[partition] = predictions
    return output, receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", required=True, choices=["emotion", "language_identification"]
    )
    parser.add_argument("--stage1-public-root", type=Path, required=True)
    parser.add_argument("--stage3-root", type=Path, required=True)
    parser.add_argument("--clean-outputs-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    started = time.time()
    stage1 = load_json(STAGE1_LOCK)
    stage2_lock = load_json(STAGE2_LOCK)
    if stage1.get("decision") != "PASS_PRAA_STAGE1_ARTIFACT_AND_DATA_SEAL":
        raise Stage4Error("Stage-1 lock is not PASS")
    if stage2_lock.get("decision") != "PASS_PRAA_STAGE2_CLEAN_SAFETY_AND_PARENT":
        raise Stage4Error("Stage-2 final lock is not PASS")
    parent = str(stage2_lock["tasks"][args.task]["selected_parent"])
    if parent not in stage2.TRANSFORMER_ROOTS:
        raise Stage4Error(
            f"selected parent {parent} does not match the frozen Transformer Stage-3 path"
        )
    partitions = {
        name: read_partition(args.stage1_public_root, args.task, name, stage1)
        for name in PARTITIONS
    }
    stage3_receipt, checkpoints, thresholds = load_stage3(
        args.stage3_root, args.task, parent
    )
    clean_parent, clean_receipt = load_clean_parent(
        args.clean_outputs_root, parent, partitions
    )

    texts = {name: value["texts"] for name, value in partitions.items()}
    runtime_spec = stage1["runtime_artifacts"][parent]
    inferred, runtime = extract_partitions(args.task, runtime_spec, texts)
    for partition in PARTITIONS:
        if not np.array_equal(
            inferred[partition]["predictions"], clean_parent[partition]
        ):
            raise Stage4Error(f"selected-parent replay mismatch for {partition}")

    score_matrices: dict[str, np.ndarray] = {}
    aliases: dict[str, np.ndarray] = {}
    triggers: dict[str, np.ndarray] = {}
    partition_receipts: dict[str, Any] = {}
    n_classes = N_CLASSES[args.task]
    for partition in PARTITIONS:
        scores = np.column_stack(
            [
                head_scores(inferred[partition]["features"], checkpoint)
                for checkpoint in checkpoints
            ]
        ).astype(np.float64)
        alias_matrix, trigger_matrix = make_aliases(
            inferred[partition]["predictions"], scores, thresholds, n_classes
        )
        counts = np.sum(trigger_matrix, axis=0).astype(int)
        maximum = int(np.floor(0.01 * len(scores)))
        response_hashes = [
            array_sha256(inferred[partition]["predictions"])
        ] + [array_sha256(alias_matrix[:, index]) for index in range(4)]
        gates = {
            "trigger_nonempty": bool(np.all(counts >= 1)),
            "trigger_at_most_one_percent": bool(np.all(counts <= maximum)),
            "responses_distinct": len(set(response_hashes)) == 5,
            "parent_replay_exact": True,
        }
        score_matrices[partition] = scores
        aliases[partition] = alias_matrix
        triggers[partition] = trigger_matrix
        partition_receipts[partition] = {
            "rows": len(scores),
            "input_sha256": sha256_file(partitions[partition]["path"]),
            "uid_sha256": digest_lines(partitions[partition]["uids"]),
            "parent_prediction_sha256": response_hashes[0],
            "feature_sha256": array_sha256(inferred[partition]["features"]),
            "logits_sha256": array_sha256(inferred[partition]["logits"]),
            "head_score_sha256": array_sha256(scores),
            "trigger_sha256": array_sha256(trigger_matrix),
            "alias_response_sha256": response_hashes[1:],
            "trigger_counts": counts.tolist(),
            "maximum_trigger_count": maximum,
            "gates": gates,
        }

    all_gates = {
        f"{partition}.{name}": passed
        for partition, record in partition_receipts.items()
        for name, passed in record["gates"].items()
    }
    failed = sorted(name for name, passed in all_gates.items() if not passed)

    output_dir = args.output_root / args.task
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "PRAA_STAGE4_DERIVED_OUTPUTS.npz"
    np.savez_compressed(
        arrays_path,
        primary_uids=np.asarray(partitions["primary_outcome"]["uids"], dtype="<U64"),
        deployment_uids=np.asarray(partitions["deployment"]["uids"], dtype="<U64"),
        primary_parent_predictions=inferred["primary_outcome"]["predictions"],
        deployment_parent_predictions=inferred["deployment"]["predictions"],
        primary_parent_logits=inferred["primary_outcome"]["logits"],
        deployment_parent_logits=inferred["deployment"]["logits"],
        primary_features=inferred["primary_outcome"]["features"],
        deployment_features=inferred["deployment"]["features"],
        primary_head_scores=score_matrices["primary_outcome"],
        deployment_head_scores=score_matrices["deployment"],
        thresholds=thresholds,
        primary_aliases=aliases["primary_outcome"],
        deployment_aliases=aliases["deployment"],
        primary_triggers=triggers["primary_outcome"],
        deployment_triggers=triggers["deployment"],
    )
    receipt = {
        "schema": SCHEMA,
        "task": args.task,
        "decision": (
            "PASS_PRAA_STAGE4_DERIVED_ENDPOINTS"
            if not failed
            else "STOP_PRAA_STAGE4_DERIVED_ENDPOINTS"
        ),
        "failed_gates": failed,
        "stage1_lock_sha256": sha256_file(STAGE1_LOCK),
        "stage2_lock_sha256": sha256_file(STAGE2_LOCK),
        "stage3_receipt_file_sha256": canonical(
            stage3_receipt
        ) and hashlib.sha256(canonical(stage3_receipt).encode("utf-8")).hexdigest(),
        "stage4_protocol_sha256": sha256_file(STAGE4_PROTOCOL),
        "selected_parent": parent,
        "stage4_clean_parent_receipt_sha256": hashlib.sha256(
            canonical(clean_receipt).encode("utf-8")
        ).hexdigest(),
        "thresholds": thresholds.tolist(),
        "head_checkpoint_sha256": [sha256_file(path) for path in checkpoints],
        "runtime": runtime,
        "partitions": partition_receipts,
        "gates": all_gates,
        "arrays_file": arrays_path.name,
        "arrays_sha256": sha256_file(arrays_path),
        "code_sha256": {
            "derived_runner": sha256_file(Path(__file__).resolve()),
            "feature_extractor": sha256_file(ROOT / "stage3_transformer_features.py"),
            "head_runtime": sha256_file(ROOT / "stage3_learned_common.py"),
            "stage2_runtime": sha256_file(ROOT / "stage2_clean_safety_endpoint.py"),
        },
        "elapsed_seconds": time.time() - started,
        "information_boundary": {
            "labels_received": False,
            "primary_labels_opened": False,
            "deployment_labels_opened": False,
            "selector_executed": False,
        },
    }
    receipt_path = output_dir / "PRAA_STAGE4_DERIVED_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
