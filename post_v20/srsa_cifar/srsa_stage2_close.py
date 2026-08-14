#!/usr/bin/env python3
"""Close the label-free SRSA Stage-2 clean endpoint family."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from srsa_core import canonical_sha256, digest_array

SCHEMA = "srsa.cifar.stage2.clean_family.v1"


class Stage2ClosureError(RuntimeError):
    pass


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_strings(values: np.ndarray) -> str:
    payload = "\n".join(str(value) for value in values.tolist()) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_endpoint(directory: Path, task: str, model_name: str) -> dict[str, Any]:
    receipt_path = directory / "receipt.json"
    arrays_path = directory / "predictions.npz"
    if not receipt_path.is_file() or not arrays_path.is_file():
        raise Stage2ClosureError(f"missing endpoint artifact: {directory}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("decision") != "PASS_SRSA_STAGE2_CLEAN_ENDPOINT":
        raise Stage2ClosureError(f"endpoint is not PASS: {model_name}")
    if receipt.get("task") != task or receipt.get("model_name") != model_name:
        raise Stage2ClosureError(f"endpoint identity drift: {model_name}")
    if file_digest(arrays_path) != receipt["outputs"]["arrays_sha256"]:
        raise Stage2ClosureError(f"endpoint arrays hash drift: {model_name}")
    boundary = receipt["information_boundary"]
    prohibited = (
        "stage1_sealed_labels_accessed",
        "labels_accessed",
        "structured_wrappers_constructed",
        "selector_executed",
        "candidate_quality_computed",
        "primary_or_deployment_outcome_opened",
    )
    if any(bool(boundary[key]) for key in prohibited):
        raise Stage2ClosureError(f"endpoint boundary violation: {model_name}")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        if set(arrays.files) != {
            "uids",
            "partition_codes",
            "predictions",
            "logits",
        }:
            raise Stage2ClosureError((model_name, arrays.files))
        value = {key: arrays[key].copy() for key in arrays.files}
    if value["predictions"].shape != (10000,):
        raise Stage2ClosureError((model_name, value["predictions"].shape))
    if value["logits"].shape[0] != 10000 or not np.all(np.isfinite(value["logits"])):
        raise Stage2ClosureError((model_name, value["logits"].shape))
    if digest_array(value["predictions"]) != receipt["outputs"]["predictions_sha256"]:
        raise Stage2ClosureError(f"prediction digest drift: {model_name}")
    if digest_array(value["logits"]) != receipt["outputs"]["logits_sha256"]:
        raise Stage2ClosureError(f"logit digest drift: {model_name}")
    return {
        "receipt": receipt,
        "receipt_path": receipt_path,
        "arrays_path": arrays_path,
        **value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage0-lock", type=Path, required=True)
    parser.add_argument("--stage1-lock", type=Path, required=True)
    parser.add_argument("--endpoints-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.time()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    stage0 = json.loads(args.stage0_lock.read_text(encoding="utf-8"))
    stage1 = json.loads(args.stage1_lock.read_text(encoding="utf-8"))
    if stage0.get("decision") != "PASS_SRSA_STAGE0_PUBLIC_METADATA":
        raise Stage2ClosureError("Stage 0 lock is not PASS")
    if stage1.get("decision") != "PASS_SRSA_STAGE1_DATA_SEAL":
        raise Stage2ClosureError("Stage 1 lock is not PASS")

    tasks: dict[str, Any] = {}
    failed_gates: list[str] = []
    for task in ("cifar100", "cifar10"):
        endpoints: list[dict[str, Any]] = []
        common_uid: np.ndarray | None = None
        common_partition: np.ndarray | None = None
        for suffix in config["root_suffixes_in_order"]:
            model_name = f"{task}_{suffix}"
            endpoint = load_endpoint(
                args.endpoints_root / model_name,
                task,
                model_name,
            )
            if common_uid is None:
                common_uid = endpoint["uids"]
                common_partition = endpoint["partition_codes"]
            elif not np.array_equal(common_uid, endpoint["uids"]):
                raise Stage2ClosureError(f"UID mismatch within {task}")
            elif not np.array_equal(common_partition, endpoint["partition_codes"]):
                raise Stage2ClosureError(f"partition mismatch within {task}")
            expected_stage1 = stage1["tasks"][task]
            if endpoint["receipt"]["stage1_binding"]["public_manifest_sha256"] != expected_stage1["public_manifest_sha256"]:
                raise Stage2ClosureError(f"Stage-1 manifest binding drift: {model_name}")
            if endpoint["receipt"]["checkpoint"]["sha256"] != stage0["models"][model_name]["sha256"]:
                raise Stage2ClosureError(f"Stage-0 checkpoint binding drift: {model_name}")
            endpoints.append(endpoint)

        assert common_uid is not None and common_partition is not None
        checkpoint_hashes = [
            endpoint["receipt"]["checkpoint"]["sha256"] for endpoint in endpoints
        ]
        prediction_hashes = [
            endpoint["receipt"]["outputs"]["predictions_sha256"]
            for endpoint in endpoints
        ]
        checkpoint_distinct = len(set(checkpoint_hashes)) == len(checkpoint_hashes)
        prediction_distinct = len(set(prediction_hashes)) == len(prediction_hashes)
        if not checkpoint_distinct:
            failed_gates.append(f"{task}:checkpoint_hashes_pairwise_distinct")
        if not prediction_distinct:
            failed_gates.append(f"{task}:prediction_vectors_pairwise_distinct")
        tasks[task] = {
            "root_order": [
                f"{task}_{suffix}" for suffix in config["root_suffixes_in_order"]
            ],
            "uids_sha256": digest_strings(common_uid.astype(str)),
            "partition_codes_sha256": digest_array(common_partition),
            "checkpoint_hashes": checkpoint_hashes,
            "prediction_hashes": prediction_hashes,
            "checkpoint_hashes_pairwise_distinct": checkpoint_distinct,
            "prediction_vectors_pairwise_distinct": prediction_distinct,
            "endpoints": {
                endpoint["receipt"]["model_name"]: {
                    "receipt_sha256": endpoint["receipt"]["receipt_sha256"],
                    "receipt_file_sha256": file_digest(endpoint["receipt_path"]),
                    "arrays_sha256": file_digest(endpoint["arrays_path"]),
                    "checkpoint_sha256": endpoint["receipt"]["checkpoint"]["sha256"],
                    "predictions_sha256": endpoint["receipt"]["outputs"]["predictions_sha256"],
                    "logits_sha256": endpoint["receipt"]["outputs"]["logits_sha256"],
                }
                for endpoint in endpoints
            },
        }

    decision = (
        "PASS_SRSA_STAGE2_CLEAN_ENDPOINTS"
        if not failed_gates
        else "STOP_SRSA_STAGE2_CLEAN_ENDPOINTS"
    )
    ledger: dict[str, Any] = {
        "schema": SCHEMA,
        "decision": decision,
        "config_sha256": file_digest(args.config),
        "stage0_lock_sha256": file_digest(args.stage0_lock),
        "stage1_lock_sha256": file_digest(args.stage1_lock),
        "tasks": tasks,
        "failed_gates": failed_gates,
        "information_boundary": {
            "public_inputs_accessed": True,
            "sealed_labels_accessed": False,
            "labels_accessed": False,
            "clean_endpoint_inference_executed": True,
            "structured_wrappers_constructed": False,
            "selector_executed": False,
            "candidate_quality_computed": False,
            "primary_or_deployment_outcome_opened": False,
        },
        "elapsed_seconds": time.time() - started,
    }
    ledger["ledger_sha256"] = canonical_sha256(ledger)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "SRSA_STAGE2_CLEAN_ENDPOINT_LEDGER.json"
    path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": decision,
                "ledger_sha256": ledger["ledger_sha256"],
                "failed_gates": failed_gates,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
