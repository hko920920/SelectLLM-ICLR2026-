#!/usr/bin/env python3
"""Validate PRAA Stage-4 outputs and create the label-free pre-outcome lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping

import jsonschema
import numpy as np

import stage2_clean_safety_endpoint as stage2
from stage3_learned_common import array_sha256, canonical_json, digest_lines, sha256_file

ROOT = Path(__file__).resolve().parent
STAGE0_LOCK = ROOT / "PRAA_STAGE0_METADATA_LOCK_2026-08-14.json"
STAGE1_LOCK = ROOT / "PRAA_STAGE1_ARTIFACT_AND_DATA_SEAL_LOCK_2026-08-14.json"
STAGE2_LOCK = ROOT / "PRAA_STAGE2_FINAL_LOCK_2026-08-14.json"
STAGE4_PROTOCOL = ROOT / "PRAA_STAGE4_INPUT_ONLY_PREOUTCOME_PROTOCOL_2026-08-14.md"
PRIMARY_PROTOCOL = ROOT / "PRAA_PRIMARY_AND_DEPLOYMENT_DECISION_PROTOCOL_2026-08-14.md"
MANIFEST_SCHEMA_PATH = ROOT / "PRAA_ADMISSION_MANIFEST_SCHEMA_V1.json"
ADMISSION_POLICY_PATH = ROOT / "PRAA_ROOT_AWARE_ADMISSION_POLICY_V1.md"
ATTEMPT_REGISTER_PATH = ROOT / "PRAA_PROSPECTIVE_ATTEMPT_REGISTER_2026-08-14.md"
TASKS = ("emotion", "language_identification")
ROOTS = {
    "emotion": [
        "emotion-roberta-dk409",
        "emotion-bert-nateraw",
        "emotion-distilbert-bhadresh",
        "emotion-albert-bhadresh",
    ],
    "language_identification": [
        "lid-xlmroberta-papluca",
        "lid-fasttext-facebook",
        "lid-fasttext-glotlid",
        "lid-langid-package",
    ],
}
TARGET_LABELS = {
    "emotion": stage2.EMOTION_LABELS,
    "language_identification": stage2.LID_LABELS,
}
PARTITIONS = ("primary_outcome", "deployment")


class Stage4Error(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Stage4Error(path)
    return value


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def find_root_dir(root: Path, root_id: str) -> Path:
    candidates = [path for path in root.rglob(root_id) if path.is_dir()]
    if len(candidates) != 1:
        raise Stage4Error(
            f"expected one directory for {root_id}, observed {[str(p) for p in candidates]}"
        )
    return candidates[0]


def find_task_file(root: Path, task: str, name: str) -> Path:
    candidates = [
        path
        for path in root.rglob(name)
        if path.is_file() and path.parent.name == task
    ]
    if len(candidates) != 1:
        raise Stage4Error(
            f"expected one {name} for {task}, observed {[str(p) for p in candidates]}"
        )
    return candidates[0]


def input_uids(public_root: Path, task: str, partition: str, stage1: Mapping[str, Any]) -> list[str]:
    expected = stage1["tasks"][task]["partitions"][partition]
    path = public_root / "inputs" / task / f"{partition}.jsonl"
    if not path.is_file() or sha256_file(path) != expected["input_sha256"]:
        raise Stage4Error(f"input hash drift for {task}/{partition}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    uids = [str(row["uid"]) for row in rows]
    if len(uids) != int(expected["rows"]) or digest_lines(uids) != expected["uid_sha256"]:
        raise Stage4Error(f"input UID drift for {task}/{partition}")
    return uids


def load_clean_endpoint(
    outputs_root: Path,
    task: str,
    root_id: str,
    expected_uids: Mapping[str, list[str]],
    stage1_lock_sha256: str,
) -> dict[str, Any]:
    directory = find_root_dir(outputs_root, root_id)
    receipt_path = directory / "receipt.json"
    arrays_path = directory / "primary_deployment_outputs.npz"
    receipt = load_json(receipt_path)
    if receipt.get("schema") != "praa.stage4.clean_endpoint.v1":
        raise Stage4Error(f"clean receipt schema drift for {root_id}")
    if receipt.get("decision") != "PASS_PRAA_STAGE4_CLEAN_ENDPOINT":
        raise Stage4Error(f"clean endpoint not PASS for {root_id}")
    if receipt.get("task") != task or receipt.get("root_id") != root_id:
        raise Stage4Error(f"clean endpoint identity drift for {root_id}")
    if receipt.get("stage1_lock_sha256") != stage1_lock_sha256:
        raise Stage4Error(f"Stage-1 binding drift for {root_id}")
    if sha256_file(arrays_path) != receipt.get("arrays_sha256"):
        raise Stage4Error(f"clean arrays hash drift for {root_id}")
    boundary = receipt.get("information_boundary") or {}
    forbidden = [
        key
        for key in (
            "labels_received",
            "non_primary_or_deployment_inputs_opened",
            "aliases_executed",
            "selector_executed",
            "primary_labels_opened",
            "deployment_labels_opened",
        )
        if bool(boundary.get(key))
    ]
    if forbidden:
        raise Stage4Error(f"clean endpoint boundary violation for {root_id}: {forbidden}")
    arrays = np.load(arrays_path, allow_pickle=False)
    partitions: dict[str, Any] = {}
    for partition, prefix in (("primary_outcome", "primary"), ("deployment", "deployment")):
        uids = arrays[f"{prefix}_uids"].astype(str).tolist()
        if uids != expected_uids[partition]:
            raise Stage4Error(f"UID order drift for {root_id}/{partition}")
        predictions = arrays[f"{prefix}_predictions"].astype(np.int16)
        scores = arrays[f"{prefix}_scores"].astype(np.float32)
        confidence = arrays[f"{prefix}_confidence"].astype(np.float32)
        recorded = receipt["partitions"][partition]
        if array_sha256(predictions) != recorded["prediction_sha256"]:
            raise Stage4Error(f"prediction digest drift for {root_id}/{partition}")
        if array_sha256(scores) != recorded["score_sha256"]:
            raise Stage4Error(f"score digest drift for {root_id}/{partition}")
        if array_sha256(confidence) != recorded["confidence_sha256"]:
            raise Stage4Error(f"confidence digest drift for {root_id}/{partition}")
        partitions[partition] = {
            "predictions": predictions,
            "scores": scores,
            "confidence": confidence,
            "prediction_sha256": recorded["prediction_sha256"],
            "score_sha256": recorded["score_sha256"],
            "confidence_sha256": recorded["confidence_sha256"],
        }
    return {
        "receipt": receipt,
        "receipt_file_sha256": sha256_file(receipt_path),
        "arrays_file_sha256": sha256_file(arrays_path),
        "partitions": partitions,
    }


def load_derived(
    derived_root: Path,
    task: str,
    expected_uids: Mapping[str, list[str]],
    stage1_hash: str,
    stage2_hash: str,
) -> dict[str, Any]:
    receipt_path = find_task_file(derived_root, task, "PRAA_STAGE4_DERIVED_RECEIPT.json")
    arrays_path = find_task_file(derived_root, task, "PRAA_STAGE4_DERIVED_OUTPUTS.npz")
    receipt = load_json(receipt_path)
    if receipt.get("schema") != "praa.stage4.derived_endpoints.v1":
        raise Stage4Error(f"derived receipt schema drift for {task}")
    if receipt.get("decision") != "PASS_PRAA_STAGE4_DERIVED_ENDPOINTS":
        raise Stage4Error(f"derived endpoints not PASS for {task}")
    if receipt.get("stage1_lock_sha256") != stage1_hash:
        raise Stage4Error(f"derived Stage-1 binding drift for {task}")
    if receipt.get("stage2_lock_sha256") != stage2_hash:
        raise Stage4Error(f"derived Stage-2 binding drift for {task}")
    if receipt.get("failed_gates"):
        raise Stage4Error(f"derived failed gates for {task}: {receipt['failed_gates']}")
    if not all(bool(value) for value in (receipt.get("gates") or {}).values()):
        raise Stage4Error(f"derived gate ledger drift for {task}")
    boundary = receipt.get("information_boundary") or {}
    if any(bool(boundary.get(key)) for key in ("labels_received", "primary_labels_opened", "deployment_labels_opened", "selector_executed")):
        raise Stage4Error(f"derived information-boundary violation for {task}")
    if sha256_file(arrays_path) != receipt.get("arrays_sha256"):
        raise Stage4Error(f"derived arrays hash drift for {task}")
    arrays = np.load(arrays_path, allow_pickle=False)
    partitions: dict[str, Any] = {}
    for partition, prefix in (("primary_outcome", "primary"), ("deployment", "deployment")):
        uids = arrays[f"{prefix}_uids"].astype(str).tolist()
        if uids != expected_uids[partition]:
            raise Stage4Error(f"derived UID order drift for {task}/{partition}")
        aliases = arrays[f"{prefix}_aliases"].astype(np.int16)
        triggers = arrays[f"{prefix}_triggers"].astype(bool)
        parent = arrays[f"{prefix}_parent_predictions"].astype(np.int16)
        scores = arrays[f"{prefix}_head_scores"].astype(np.float64)
        record = receipt["partitions"][partition]
        if array_sha256(parent) != record["parent_prediction_sha256"]:
            raise Stage4Error(f"derived parent digest drift for {task}/{partition}")
        if array_sha256(scores) != record["head_score_sha256"]:
            raise Stage4Error(f"derived score digest drift for {task}/{partition}")
        if array_sha256(triggers) != record["trigger_sha256"]:
            raise Stage4Error(f"derived trigger digest drift for {task}/{partition}")
        if [array_sha256(aliases[:, index]) for index in range(4)] != record["alias_response_sha256"]:
            raise Stage4Error(f"derived alias digest drift for {task}/{partition}")
        partitions[partition] = {
            "parent": parent,
            "aliases": aliases,
            "triggers": triggers,
            "head_scores": scores,
            "record": record,
        }
    return {
        "receipt": receipt,
        "receipt_file_sha256": sha256_file(receipt_path),
        "arrays_file_sha256": sha256_file(arrays_path),
        "partitions": partitions,
    }


def dependency_records(wrapper_path: Path) -> list[dict[str, Any]]:
    python_path = Path(sys.executable).resolve()
    return [
        {
            "name": "python-runtime",
            "version": sys.version.split()[0],
            "artifact_sha256": sha256_file(python_path),
        },
        {
            "name": "endpoint-wrapper",
            "version": sha256_file(wrapper_path)[:16],
            "artifact_sha256": sha256_file(wrapper_path),
        },
    ]


def source_artifact(stage1: Mapping[str, Any], root_id: str) -> dict[str, Any]:
    spec = stage1["runtime_artifacts"][root_id]
    if "source" in spec:
        weight = spec["weight"]
        return {
            "source_type": "huggingface_model",
            "source_id": str(spec["source"]),
            "revision": str(spec["revision"]),
            "files": [
                {
                    "path": str(weight["path"]),
                    "bytes": int(weight["bytes"]),
                    "sha256": str(weight["sha256"]),
                }
            ],
        }
    distribution = spec["distribution"]
    return {
        "source_type": "pypi_package",
        "source_id": str(spec["source"]),
        "revision": str(spec["version"]),
        "files": [
            {
                "path": str(distribution["path"]),
                "bytes": int(distribution["bytes"]),
                "sha256": str(distribution["sha256"]),
            }
        ],
    }


def native_mapping(task: str, root_id: str, clean_receipt: Mapping[str, Any]) -> dict[str, str | int]:
    if task == "emotion":
        mapping = clean_receipt["runtime"]["native_index_to_target_index"]
        return {str(key): int(value) for key, value in mapping.items()}
    if root_id == "lid-xlmroberta-papluca":
        mapping = clean_receipt["runtime"]["native_index_to_target_index"]
        return {str(key): int(value) for key, value in mapping.items()}
    if root_id == "lid-langid-package":
        return {label: label for label in TARGET_LABELS[task]}
    return {native: target for native, target in stage2.LID_THREE_TO_TWO.items() if target in TARGET_LABELS[task]}


def build_manifest(
    *,
    entry_id: str,
    provider: str,
    root_id: str,
    artifact: dict[str, Any],
    family: str,
    root_relation: str,
    upstream: list[dict[str, Any]],
    runtime_entrypoint: str,
    wrapper_path: Path,
    task: str,
    native_to_target: dict[str, str | int],
    private_codes: list[int],
    extra_determinism: dict[str, str | int | float | bool | None],
) -> dict[str, Any]:
    preprocessing = {
        "raw_field": "text",
        "normalization": "none",
        "transformer_max_length": 128,
        "task": task,
    }
    body: dict[str, Any] = {
        "schema": "praa.admission_manifest.v1",
        "entry_id": entry_id,
        "provider_id": provider,
        "declared_root_id": root_id,
        "artifact": artifact,
        "lineage": {
            "implementation_family": family,
            "root_relation": root_relation,
            "upstream_artifacts": upstream,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "container_digest": None,
            "dependencies": dependency_records(wrapper_path),
            "entrypoint": runtime_entrypoint,
            "deterministic_settings": extra_determinism,
        },
        "input_schema": {
            "media_type": "application/json",
            "fields": [{"name": "text", "type": "string"}],
            "preprocessing_sha256": canonical_sha256(preprocessing),
        },
        "output_schema": {
            "hard_label_field": "prediction",
            "confidence_field": "confidence",
            "private_abstention_codes": private_codes,
        },
        "label_mapping": {
            "target_labels": list(TARGET_LABELS[task]),
            "native_to_target": native_to_target,
            "out_of_target_policy": "map_to_private_always_incorrect_code",
        },
    }
    body_hash = canonical_sha256(body)
    body["attestation"] = {
        "manifest_sha256": body_hash,
        "signer_id": "praa-development-registry",
        "signature_algorithm": "development-null-signature",
        "signature": f"development-null:{body_hash}",
    }
    return body


def registry_snapshot(
    task: str,
    clean_manifests: list[dict[str, Any]],
    alias_manifests: list[dict[str, Any]],
    selected_parent: str,
) -> dict[str, Any]:
    clean_entries = [manifest["entry_id"] for manifest in clean_manifests]
    alias_entries = [manifest["entry_id"] for manifest in alias_manifests]
    clean_roots = [manifest["declared_root_id"] for manifest in clean_manifests]
    if len(set(clean_roots)) != 4:
        raise Stage4Error(f"clean root cardinality drift for {task}")
    canonical = {manifest["declared_root_id"]: manifest["entry_id"] for manifest in clean_manifests}
    if len(canonical) != 4:
        raise Stage4Error(f"canonical endpoint ambiguity for {task}")
    root_map = {
        manifest["entry_id"]: manifest["declared_root_id"]
        for manifest in clean_manifests + alias_manifests
    }
    if any(root_map[entry] != selected_parent for entry in alias_entries):
        raise Stage4Error(f"alias root mapping drift for {task}")
    clean_entry_mass = {entry: 1.0 / 4.0 for entry in clean_entries}
    refined_entry_mass = {entry: 1.0 / 8.0 for entry in clean_entries + alias_entries}
    root_mass = {root: 1.0 / 4.0 for root in clean_roots}
    snapshot = {
        "schema": "praa.registry_snapshot.v1",
        "task": task,
        "manifest_file_sha256": {
            manifest["entry_id"]: canonical_sha256(manifest)
            for manifest in clean_manifests + alias_manifests
        },
        "entry_indexed": {
            "clean_entries": clean_entries,
            "refined_entries": clean_entries + alias_entries,
            "clean_entry_mass": clean_entry_mass,
            "refined_entry_mass": refined_entry_mass,
            "entry_to_root": root_map,
        },
        "root_aware_canonical": {
            "clean_roots": clean_roots,
            "refined_roots": clean_roots,
            "root_mass_clean": root_mass,
            "root_mass_refined": root_mass,
            "canonical_entry_clean": canonical,
            "canonical_entry_refined": canonical,
            "derived_entries_admitted_without_independent_mass": alias_entries,
        },
    }
    if snapshot["root_aware_canonical"]["root_mass_clean"] != snapshot["root_aware_canonical"]["root_mass_refined"]:
        raise Stage4Error("root mass invariance failure")
    if snapshot["root_aware_canonical"]["canonical_entry_clean"] != snapshot["root_aware_canonical"]["canonical_entry_refined"]:
        raise Stage4Error("canonical endpoint invariance failure")
    return snapshot


def markdown(closure: Mapping[str, Any]) -> str:
    lines = [
        "# PRAA Stage-4 input-only pre-outcome closure",
        "",
        f"Decision: `{closure['decision']}`",
        "",
        f"Closure SHA-256: `{closure['closure_sha256']}`",
        "",
        "No primary/deployment label, quality, selector path, regret, or deployment outcome was accessed.",
        "",
    ]
    for task, record in closure["tasks"].items():
        lines.extend(
            [
                f"## {task}",
                "",
                f"- selected parent: `{record['selected_parent']}`",
                f"- primary trigger counts: `{record['derived']['primary_outcome']['trigger_counts']}`",
                f"- deployment trigger counts: `{record['derived']['deployment']['trigger_counts']}`",
                f"- endpoint manifests: `{len(record['manifest_sha256'])}`",
                f"- registry snapshot SHA-256: `{record['registry_snapshot_sha256']}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-public-root", type=Path, required=True)
    parser.add_argument("--clean-outputs-root", type=Path, required=True)
    parser.add_argument("--derived-outputs-root", type=Path, required=True)
    parser.add_argument("--stage3-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stage0 = load_json(STAGE0_LOCK)
    stage1 = load_json(STAGE1_LOCK)
    stage2_lock = load_json(STAGE2_LOCK)
    manifest_schema = load_json(MANIFEST_SCHEMA_PATH)
    stage1_hash = sha256_file(STAGE1_LOCK)
    stage2_hash = sha256_file(STAGE2_LOCK)

    task_records: dict[str, Any] = {}
    manifests_root = args.output_dir / "manifests"
    registries_root = args.output_dir / "registries"
    manifests_root.mkdir(parents=True, exist_ok=True)
    registries_root.mkdir(parents=True, exist_ok=True)

    for task in TASKS:
        uids = {
            partition: input_uids(args.stage1_public_root, task, partition, stage1)
            for partition in PARTITIONS
        }
        clean = {
            root_id: load_clean_endpoint(
                args.clean_outputs_root,
                task,
                root_id,
                uids,
                stage1_hash,
            )
            for root_id in ROOTS[task]
        }
        derived = load_derived(
            args.derived_outputs_root, task, uids, stage1_hash, stage2_hash
        )
        selected_parent = str(stage2_lock["tasks"][task]["selected_parent"])
        for partition in PARTITIONS:
            if not np.array_equal(
                clean[selected_parent]["partitions"][partition]["predictions"],
                derived["partitions"][partition]["parent"],
            ):
                raise Stage4Error(f"clean/derived parent mismatch for {task}/{partition}")

        roster = {
            entry["root_id"]: entry
            for entry in stage0["tasks"][task]["retained_roster"]
        }
        clean_manifests: list[dict[str, Any]] = []
        alias_manifests: list[dict[str, Any]] = []
        manifest_hashes: dict[str, str] = {}
        task_manifest_dir = manifests_root / task
        task_manifest_dir.mkdir(parents=True, exist_ok=True)

        for root_id in ROOTS[task]:
            meta = roster[root_id]
            manifest = build_manifest(
                entry_id=f"{task}:{root_id}:canonical",
                provider=str(meta["provider"]),
                root_id=root_id,
                artifact=source_artifact(stage1, root_id),
                family=str(meta["family"]),
                root_relation="canonical_root_endpoint",
                upstream=[],
                runtime_entrypoint="stage4_clean_endpoint.py",
                wrapper_path=ROOT / "stage4_clean_endpoint.py",
                task=task,
                native_to_target=native_mapping(task, root_id, clean[root_id]["receipt"]),
                private_codes=[],
                extra_determinism={
                    "raw_text": True,
                    "max_length": 128 if root_id in stage2.TRANSFORMER_ROOTS else None,
                    "seed": 0,
                    "labels_available": False,
                },
            )
            jsonschema.validate(instance=manifest, schema=manifest_schema)
            path = task_manifest_dir / f"{root_id}.canonical.json"
            path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            manifest_hashes[manifest["entry_id"]] = sha256_file(path)
            clean_manifests.append(manifest)

        stage3_receipt_path = find_task_file(args.stage3_root, task, "PRAA_STAGE3_RECEIPT.json")
        stage3_receipt = load_json(stage3_receipt_path)
        for index, head in enumerate(stage3_receipt["heads"]):
            checkpoint_name = str(head["checkpoint"]["path"])
            checkpoint_candidates = [
                path for path in args.stage3_root.rglob(checkpoint_name) if path.is_file()
            ]
            if len(checkpoint_candidates) != 1:
                raise Stage4Error(f"head checkpoint resolution failure: {checkpoint_name}")
            checkpoint = checkpoint_candidates[0]
            checkpoint_record = {
                "path": checkpoint_name,
                "bytes": int(checkpoint.stat().st_size),
                "sha256": sha256_file(checkpoint),
            }
            if checkpoint_record["sha256"] != head["checkpoint"]["sha256"]:
                raise Stage4Error(f"head checkpoint hash drift: {checkpoint_name}")
            parent_artifact = source_artifact(stage1, selected_parent)
            manifest = build_manifest(
                entry_id=f"{task}:{selected_parent}:derived:{index}",
                provider=str(roster[selected_parent]["provider"]),
                root_id=selected_parent,
                artifact={
                    "source_type": "local_signed_artifact",
                    "source_id": f"praa-stage3-head-{task}-{index}",
                    "revision": checkpoint_record["sha256"],
                    "files": [checkpoint_record],
                },
                family=f"{roster[selected_parent]['family']}-linear-error-wrapper",
                root_relation="derived_endpoint",
                upstream=[
                    {
                        "relation": "wrapper_of",
                        "source_id": str(parent_artifact["source_id"]),
                        "revision_or_sha256": str(parent_artifact["revision"]),
                    }
                ],
                runtime_entrypoint="stage4_alias_endpoint.py",
                wrapper_path=ROOT / "stage4_alias_endpoint.py",
                task=task,
                native_to_target=native_mapping(
                    task, selected_parent, clean[selected_parent]["receipt"]
                ),
                private_codes=[len(TARGET_LABELS[task]) + index],
                extra_determinism={
                    "head_checkpoint_sha256": checkpoint_record["sha256"],
                    "threshold": float(head["threshold"]),
                    "private_code": len(TARGET_LABELS[task]) + index,
                    "labels_available": False,
                },
            )
            jsonschema.validate(instance=manifest, schema=manifest_schema)
            path = task_manifest_dir / f"{selected_parent}.derived.{index}.json"
            path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            manifest_hashes[manifest["entry_id"]] = sha256_file(path)
            alias_manifests.append(manifest)

        snapshot = registry_snapshot(
            task, clean_manifests, alias_manifests, selected_parent
        )
        snapshot_path = registries_root / f"{task}.registry.json"
        snapshot_path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")

        task_records[task] = {
            "selected_parent": selected_parent,
            "clean": {
                root_id: {
                    "receipt_file_sha256": record["receipt_file_sha256"],
                    "arrays_file_sha256": record["arrays_file_sha256"],
                    "partitions": {
                        partition: {
                            "prediction_sha256": record["partitions"][partition]["prediction_sha256"],
                            "score_sha256": record["partitions"][partition]["score_sha256"],
                            "confidence_sha256": record["partitions"][partition]["confidence_sha256"],
                        }
                        for partition in PARTITIONS
                    },
                }
                for root_id, record in clean.items()
            },
            "derived": {
                partition: {
                    "trigger_counts": derived["receipt"]["partitions"][partition]["trigger_counts"],
                    "maximum_trigger_count": derived["receipt"]["partitions"][partition]["maximum_trigger_count"],
                    "alias_response_sha256": derived["receipt"]["partitions"][partition]["alias_response_sha256"],
                    "trigger_sha256": derived["receipt"]["partitions"][partition]["trigger_sha256"],
                }
                for partition in PARTITIONS
            },
            "derived_receipt_file_sha256": derived["receipt_file_sha256"],
            "derived_arrays_file_sha256": derived["arrays_file_sha256"],
            "manifest_sha256": manifest_hashes,
            "registry_snapshot_sha256": sha256_file(snapshot_path),
        }

    closure: dict[str, Any] = {
        "schema": "praa.stage4.input_only_preoutcome_closure.v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "decision": "PASS_PRAA_STAGE4_INPUT_ONLY_PREOUTCOME",
        "stage0_lock_sha256": sha256_file(STAGE0_LOCK),
        "stage1_lock_sha256": stage1_hash,
        "stage2_lock_sha256": stage2_hash,
        "stage4_protocol_sha256": sha256_file(STAGE4_PROTOCOL),
        "primary_protocol_sha256": sha256_file(PRIMARY_PROTOCOL),
        "manifest_schema_sha256": sha256_file(MANIFEST_SCHEMA_PATH),
        "admission_policy_sha256": sha256_file(ADMISSION_POLICY_PATH),
        "attempt_register_sha256": sha256_file(ATTEMPT_REGISTER_PATH),
        "tasks": task_records,
        "information_boundary": {
            "primary_inputs_opened_for_endpoint_execution": True,
            "deployment_inputs_opened_for_endpoint_execution": True,
            "primary_labels_opened": False,
            "deployment_labels_opened": False,
            "candidate_quality_computed": False,
            "selector_executed": False,
            "regret_computed": False,
            "deployment_metric_computed": False,
        },
    }
    closure["closure_sha256"] = canonical_sha256(closure)
    json_path = args.output_dir / "PRAA_STAGE4_INPUT_ONLY_PREOUTCOME_CLOSURE.json"
    md_path = args.output_dir / "PRAA_STAGE4_INPUT_ONLY_PREOUTCOME_CLOSURE.md"
    json_path.write_text(json.dumps(closure, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown(closure), encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
