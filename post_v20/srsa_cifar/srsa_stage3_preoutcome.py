#!/usr/bin/env python3
"""SRSA Stage 3: bind wrappers, pools, and the pre-outcome execution lock.

This stage has no access to the sealed CIFAR labels. It verifies all clean
endpoint predictions, constructs the deterministic structured and exact
refinements, checks semantic/root-aware invariance, freezes all paired pools,
and writes the only lock from which the Stage-4 outcome job may execute.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from srsa_core import (
    canonical_sha256,
    canonicalize_root_aware,
    clean_registry,
    digest_array,
    exact_refinement,
    make_alias_tags,
    sample_pools,
    semantic_alias_equality,
    structured_refinement,
)

SCHEMA = "srsa.cifar.stage3.preoutcome_lock.v1"


class Stage3Error(RuntimeError):
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_public_task(
    root: Path,
    task: str,
    stage1: dict[str, Any],
) -> dict[str, Any]:
    directory = root / task
    manifest_path = directory / "manifest.json"
    arrays_path = directory / "inputs.npz"
    if not manifest_path.is_file() or not arrays_path.is_file():
        raise Stage3Error(f"missing Stage-1 public task artifact: {task}")
    manifest = read_json(manifest_path)
    task_lock = stage1["tasks"][task]
    if manifest.get("manifest_sha256") != task_lock["public_manifest_sha256"]:
        raise Stage3Error(f"Stage-1 public manifest drift: {task}")
    if file_digest(arrays_path) != task_lock["public_npz_sha256"]:
        raise Stage3Error(f"Stage-1 public arrays drift: {task}")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        expected = {
            "images",
            "uids",
            "image_sha256",
            "partition_sha256",
            "archive_positions",
            "partition_codes",
        }
        if set(arrays.files) != expected:
            raise Stage3Error((task, arrays.files))
        value = {name: arrays[name].copy() for name in arrays.files}
    if value["images"].shape != (10000, 32, 32, 3):
        raise Stage3Error((task, value["images"].shape))
    if value["partition_codes"].shape != (10000,):
        raise Stage3Error((task, value["partition_codes"].shape))
    counts = np.bincount(
        value["partition_codes"].astype(np.int64),
        minlength=3,
    ).tolist()
    if counts != [2000, 6000, 2000]:
        raise Stage3Error(f"partition count drift: {task}")
    if len(set(value["uids"].astype(str).tolist())) != 10000:
        raise Stage3Error(f"UID uniqueness drift: {task}")
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "arrays_path": arrays_path,
        **value,
    }


def load_endpoint(
    directory: Path,
    task: str,
    model_name: str,
    stage2: dict[str, Any],
) -> dict[str, Any]:
    receipt_path = directory / "receipt.json"
    arrays_path = directory / "predictions.npz"
    if not receipt_path.is_file() or not arrays_path.is_file():
        raise Stage3Error(f"missing endpoint artifact: {model_name}")
    receipt = read_json(receipt_path)
    if receipt.get("decision") != "PASS_SRSA_STAGE2_CLEAN_ENDPOINT":
        raise Stage3Error(f"endpoint is not PASS: {model_name}")
    if receipt.get("task") != task or receipt.get("model_name") != model_name:
        raise Stage3Error(f"endpoint identity drift: {model_name}")
    binding = stage2["tasks"][task]["endpoints"][model_name]
    if receipt.get("receipt_sha256") != binding["receipt_sha256"]:
        raise Stage3Error(f"endpoint receipt drift: {model_name}")
    if file_digest(receipt_path) != binding["receipt_file_sha256"]:
        raise Stage3Error(f"endpoint receipt file drift: {model_name}")
    if file_digest(arrays_path) != binding["arrays_sha256"]:
        raise Stage3Error(f"endpoint arrays drift: {model_name}")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        expected = {"uids", "partition_codes", "predictions", "logits"}
        if set(arrays.files) != expected:
            raise Stage3Error((model_name, arrays.files))
        value = {name: arrays[name].copy() for name in arrays.files}
    if value["predictions"].shape != (10000,):
        raise Stage3Error((model_name, value["predictions"].shape))
    if value["logits"].shape[0] != 10000:
        raise Stage3Error((model_name, value["logits"].shape))
    if not np.all(np.isfinite(value["logits"])):
        raise Stage3Error(f"non-finite endpoint logits: {model_name}")
    if digest_array(value["predictions"]) != binding["predictions_sha256"]:
        raise Stage3Error(f"prediction digest drift: {model_name}")
    if digest_array(value["logits"]) != binding["logits_sha256"]:
        raise Stage3Error(f"logit digest drift: {model_name}")
    return {
        "receipt": receipt,
        "receipt_path": receipt_path,
        "arrays_path": arrays_path,
        **value,
    }


def registry_binding(registry: Any) -> dict[str, Any]:
    return {
        "labels_sha256": digest_array(registry.labels),
        "tags_sha256": digest_array(registry.tags),
        "roots_sha256": digest_array(registry.roots),
        "entry_ids_sha256": canonical_sha256(list(registry.entry_ids)),
        "rows": int(registry.labels.shape[0]),
        "entries": int(registry.labels.shape[1]),
    }


def build_task(
    task: str,
    config: dict[str, Any],
    stage1: dict[str, Any],
    stage2: dict[str, Any],
    public_root: Path,
    endpoints_root: Path,
    output_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    public = load_public_task(public_root, task, stage1)
    root_ids = [
        f"{task}_{suffix}" for suffix in config["root_suffixes_in_order"]
    ]
    endpoints = [
        load_endpoint(endpoints_root / model_name, task, model_name, stage2)
        for model_name in root_ids
    ]
    for endpoint in endpoints:
        model_name = endpoint["receipt"]["model_name"]
        if not np.array_equal(endpoint["uids"], public["uids"]):
            raise Stage3Error(f"endpoint/public UID mismatch: {model_name}")
        if not np.array_equal(
            endpoint["partition_codes"],
            public["partition_codes"],
        ):
            raise Stage3Error(f"endpoint/public partition mismatch: {model_name}")

    clean_predictions = np.column_stack(
        [endpoint["predictions"] for endpoint in endpoints]
    ).astype(np.int16)
    clean = clean_registry(clean_predictions, root_ids)
    image_keys = public["image_sha256"].astype(str)
    item_uids = public["uids"].astype(str)
    response_config = config["response"]
    alias_tags = make_alias_tags(
        image_keys,
        alias_count=int(response_config["alias_count"]),
        numerator=int(response_config["canary_numerator"]),
        denominator=int(response_config["canary_denominator"]),
        salt=str(response_config["canary_salt"]),
    )
    primary_mask = public["partition_codes"] == 1
    canary_fractions = np.mean(alias_tags[primary_mask] > 0, axis=0)
    lower = float(config["primary_gates"]["realized_canary_fraction_min"])
    upper = float(config["primary_gates"]["realized_canary_fraction_max"])
    canary_gate = bool(
        np.all((canary_fractions >= lower) & (canary_fractions <= upper))
    )

    root_digests = [
        stage2["tasks"][task]["endpoints"][model_name]["receipt_sha256"]
        for model_name in root_ids
    ]
    if len(set(root_digests)) != len(root_digests):
        raise Stage3Error(f"root-manifest digest collision: {task}")

    attacks: dict[str, Any] = {}
    for attack_root, root_id in enumerate(root_ids):
        structured = structured_refinement(
            clean,
            image_keys,
            attack_root,
            root_ids,
            alias_count=int(response_config["alias_count"]),
            numerator=int(response_config["canary_numerator"]),
            denominator=int(response_config["canary_denominator"]),
            salt=str(response_config["canary_salt"]),
        )
        exact = exact_refinement(
            clean,
            attack_root,
            root_ids,
            alias_count=int(response_config["alias_count"]),
        )
        if not semantic_alias_equality(clean, structured):
            raise Stage3Error(f"structured semantic equality failed: {root_id}")
        if not semantic_alias_equality(clean, exact):
            raise Stage3Error(f"exact semantic equality failed: {root_id}")
        if not np.array_equal(
            structured.tags[:, clean.labels.shape[1] :],
            alias_tags,
        ):
            raise Stage3Error(f"structured canary binding failed: {root_id}")
        canonical = canonicalize_root_aware(structured, clean)
        canonical_equal = (
            np.array_equal(canonical.labels, clean.labels)
            and np.array_equal(canonical.tags, clean.tags)
            and np.array_equal(canonical.roots, clean.roots)
            and canonical.entry_ids == clean.entry_ids
        )
        if not canonical_equal:
            raise Stage3Error(f"root-aware canonicalization failed: {root_id}")
        attacks[root_id] = {
            "attack_root_index": attack_root,
            "structured": registry_binding(structured),
            "exact": registry_binding(exact),
            "root_aware_canonical_equals_clean": canonical_equal,
            "semantic_prediction_identical_all_rows": True,
        }

    primary_rows = int(np.sum(primary_mask))
    task_config = config["tasks"][task]
    seeds = np.arange(
        int(task_config["seed_start"]),
        int(task_config["seed_stop_exclusive"]),
        dtype=np.int64,
    )
    expected_runs = int(config["selector"]["paired_runs_per_task"])
    if len(seeds) != expected_runs:
        raise Stage3Error((task, len(seeds), expected_runs))
    pools = sample_pools(
        primary_rows,
        seeds.tolist(),
        int(config["selector"]["pool_size"]),
    )
    primary_indices = np.flatnonzero(primary_mask).astype(np.int32)

    task_output = output_root / task
    task_output.mkdir(parents=True, exist_ok=True)
    arrays_path = task_output / "preoutcome.npz"
    np.savez_compressed(
        arrays_path,
        uids=item_uids.astype("<U64"),
        image_sha256=image_keys.astype("<U64"),
        partition_codes=public["partition_codes"].astype(np.uint8),
        clean_predictions=clean_predictions,
        alias_tags=alias_tags.astype(np.int16),
        primary_indices=primary_indices,
        seeds=seeds,
        pools=pools.astype(np.int32),
        root_digests=np.asarray(root_digests, dtype="<U64"),
    )
    arrays = {
        "uids_sha256": digest_strings(item_uids),
        "image_sha256_array_sha256": digest_strings(image_keys),
        "partition_codes_sha256": digest_array(public["partition_codes"]),
        "clean_predictions_sha256": digest_array(clean_predictions),
        "alias_tags_sha256": digest_array(alias_tags),
        "primary_indices_sha256": digest_array(primary_indices),
        "seeds_sha256": digest_array(seeds),
        "pools_sha256": digest_array(pools),
        "root_digests_sha256": digest_strings(
            np.asarray(root_digests, dtype=str)
        ),
        "file_sha256": file_digest(arrays_path),
        "file_bytes": arrays_path.stat().st_size,
    }
    failed: list[str] = []
    if task == "cifar100" and not canary_gate:
        failed.append("cifar100:primary_canary_fraction_4_to_6_percent")

    task_lock = {
        "role": task_config["role"],
        "root_order": root_ids,
        "root_manifest_digests": root_digests,
        "clean_registry": registry_binding(clean),
        "attacks": attacks,
        "canary": {
            "key": "image_sha256 = SHA256(canonical raw RGB bytes)",
            "fractions_primary": [float(value) for value in canary_fractions],
            "lower": lower,
            "upper": upper,
            "gate_pass": canary_gate,
        },
        "selector_inputs": {
            "population_rows": primary_rows,
            "pool_size": int(config["selector"]["pool_size"]),
            "budget": int(config["selector"]["budget"]),
            "temperature": float(config["selector"]["temperature"]),
            "paired_runs": len(seeds),
            "query_tie": "smallest bound unique uid",
            "root_tie": "smallest root-manifest digest",
        },
        "arrays": arrays,
        "endpoint_arrays": {
            endpoint["receipt"]["model_name"]: {
                "arrays_sha256": file_digest(endpoint["arrays_path"]),
                "predictions_sha256": digest_array(endpoint["predictions"]),
                "logits_sha256": digest_array(endpoint["logits"]),
            }
            for endpoint in endpoints
        },
        "replication_canary_gate_pass": canary_gate,
    }
    return task_lock, failed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage0-lock", type=Path, required=True)
    parser.add_argument("--stage1-lock", type=Path, required=True)
    parser.add_argument("--stage2-lock", type=Path, required=True)
    parser.add_argument("--stage1-public-root", type=Path, required=True)
    parser.add_argument("--endpoints-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = read_json(args.config)
    stage0 = read_json(args.stage0_lock)
    stage1 = read_json(args.stage1_lock)
    stage2 = read_json(args.stage2_lock)
    if stage0.get("decision") != "PASS_SRSA_STAGE0_PUBLIC_METADATA":
        raise Stage3Error("Stage 0 lock is not PASS")
    if stage1.get("decision") != "PASS_SRSA_STAGE1_DATA_SEAL":
        raise Stage3Error("Stage 1 lock is not PASS")
    if stage2.get("decision") != "PASS_SRSA_STAGE2_CLEAN_ENDPOINTS":
        raise Stage3Error("Stage 2 lock is not PASS")
    if int(config["selector"]["budget"]) != 2:
        raise Stage3Error("frozen Stage-4 implementation requires budget two")

    args.output.mkdir(parents=True, exist_ok=True)
    tasks: dict[str, Any] = {}
    failed_gates: list[str] = []
    for task in ("cifar100", "cifar10"):
        task_lock, failed = build_task(
            task,
            config,
            stage1,
            stage2,
            args.stage1_public_root,
            args.endpoints_root,
            args.output,
        )
        tasks[task] = task_lock
        failed_gates.extend(failed)

    decision = (
        "PASS_SRSA_STAGE3_PREOUTCOME"
        if not failed_gates
        else "STOP_SRSA_STAGE3_PREOUTCOME"
    )
    here = Path(__file__).resolve()
    lock: dict[str, Any] = {
        "schema": SCHEMA,
        "decision": decision,
        "config_sha256": file_digest(args.config),
        "stage0_lock_sha256": file_digest(args.stage0_lock),
        "stage1_lock_sha256": file_digest(args.stage1_lock),
        "stage2_lock_sha256": file_digest(args.stage2_lock),
        "tasks": tasks,
        "failed_gates": failed_gates,
        "code": {
            "stage3_sha256": file_digest(here),
            "core_sha256": file_digest(here.with_name("srsa_core.py")),
            "frozen_selector_sha256": file_digest(
                here.with_name("srsa_selector_frozen.py")
            ),
            "stage4_sha256": file_digest(
                here.with_name("srsa_stage4_outcome.py")
            ),
        },
        "information_boundary": {
            "public_inputs_accessed": True,
            "clean_predictions_accessed": True,
            "sealed_label_artifact_downloaded": False,
            "sealed_labels_accessed": False,
            "labels_accessed": False,
            "structured_wrappers_constructed": True,
            "semantic_equality_checked_all_rows": True,
            "selector_pools_frozen": True,
            "selector_executed": False,
            "candidate_quality_computed": False,
            "primary_or_deployment_outcome_opened": False,
        },
    }
    lock["lock_sha256"] = canonical_sha256(lock)
    path = args.output / "SRSA_STAGE3_PREOUTCOME_LOCK.json"
    path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": decision,
                "lock_sha256": lock["lock_sha256"],
                "failed_gates": failed_gates,
                "cifar100_canary_fractions": tasks["cifar100"]["canary"][
                    "fractions_primary"
                ],
                "cifar10_canary_fractions": tasks["cifar10"]["canary"][
                    "fractions_primary"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
