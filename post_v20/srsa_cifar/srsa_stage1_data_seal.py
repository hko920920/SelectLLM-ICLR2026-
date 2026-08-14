#!/usr/bin/env python3
"""SRSA Stage 1: create public raw-input partitions and a separate label seal.

The executable decodes the official CIFAR test rows exactly once. Labels are
read only to write the physically separate sealed artifact. The public artifact
contains canonical uint8 RGB inputs, digests, unique UIDs, archive positions,
and partition markers, but no label value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from srsa_core import canonical_sha256, digest_array

SCHEMA = "srsa.cifar.stage1.data_seal.v1"
PARTITIONS = (
    ("safety", 0, 2000),
    ("primary_outcome", 2000, 8000),
    ("deployment", 8000, 10000),
)


class Stage1Error(RuntimeError):
    pass


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "SRSA-CIFAR/1.0 stage1-data-seal"},
    )
    with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)


def safe_extract(archive: Path, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    root = directory.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            target = (directory / member.name).resolve()
            if root != target and root not in target.parents:
                raise Stage1Error(f"unsafe tar member: {member.name}")
            if member.issym() or member.islnk():
                raise Stage1Error(f"links are prohibited in dataset archive: {member.name}")
        handle.extractall(directory, members=members)


def load_official_test(task: str, extracted: Path) -> tuple[np.ndarray, np.ndarray]:
    if task == "cifar10":
        path = extracted / "cifar-10-batches-py" / "test_batch"
        label_key = "labels"
    elif task == "cifar100":
        path = extracted / "cifar-100-python" / "test"
        label_key = "fine_labels"
    else:
        raise Stage1Error(task)
    if not path.is_file():
        raise Stage1Error(f"missing official test file: {path}")
    with path.open("rb") as handle:
        value = pickle.load(handle, encoding="latin1")
    data = np.asarray(value["data"], dtype=np.uint8)
    labels = np.asarray(value[label_key], dtype=np.int16)
    if data.shape != (10000, 3072) or labels.shape != (10000,):
        raise Stage1Error((task, data.shape, labels.shape))
    images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    images = np.ascontiguousarray(images, dtype=np.uint8)
    return images, labels


def raw_records(task: str, images: np.ndarray) -> dict[str, np.ndarray]:
    rows = len(images)
    image_sha256 = np.empty(rows, dtype="<U64")
    partition_sha256 = np.empty(rows, dtype="<U64")
    uid = np.empty(rows, dtype="<U64")
    for position, image in enumerate(images):
        raw = np.ascontiguousarray(image, dtype=np.uint8).tobytes(order="C")
        image_hash = hashlib.sha256(raw).hexdigest()
        partition_hash = hashlib.sha256(
            b"srsa-cifar-partition-v1" + raw
        ).hexdigest()
        uid_hash = hashlib.sha256(
            b"srsa-cifar-uid-v1"
            + task.encode("ascii")
            + bytes.fromhex(image_hash)
            + int(position).to_bytes(8, "big", signed=False)
        ).hexdigest()
        image_sha256[position] = image_hash
        partition_sha256[position] = partition_hash
        uid[position] = uid_hash
    archive_position = np.arange(rows, dtype=np.int32)
    order = np.asarray(
        sorted(
            range(rows),
            key=lambda index: (
                str(partition_sha256[index]),
                int(archive_position[index]),
            ),
        ),
        dtype=np.int32,
    )
    return {
        "order": order,
        "image_sha256": image_sha256,
        "partition_sha256": partition_sha256,
        "uid": uid,
        "archive_position": archive_position,
    }


def digest_strings(values: np.ndarray) -> str:
    payload = "\n".join(str(value) for value in values.tolist()) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_task(
    task: str,
    task_config: dict[str, Any],
    stage0_dataset: dict[str, Any],
    public_root: Path,
    sealed_root: Path,
    work_root: Path,
) -> dict[str, Any]:
    archive = work_root / task_config["archive_filename"]
    download(task_config["dataset_url"], archive)
    observed = {
        "bytes": archive.stat().st_size,
        "md5": file_digest(archive, "md5"),
        "sha256": file_digest(archive),
    }
    for key in ("md5", "sha256"):
        if observed[key] != str(stage0_dataset[key]):
            raise Stage1Error(
                {"task": task, "binding": key, "expected": stage0_dataset[key], "observed": observed[key]}
            )
    if observed["md5"] != task_config["archive_md5"]:
        raise Stage1Error(f"official MD5 drift for {task}")

    extracted = work_root / f"extract-{task}"
    safe_extract(archive, extracted)
    images, labels = load_official_test(task, extracted)
    if int(np.min(labels)) < 0 or int(np.max(labels)) >= int(task_config["classes"]):
        raise Stage1Error(f"label domain drift for {task}")

    identity = raw_records(task, images)
    order = identity.pop("order")
    images = images[order]
    labels = labels[order]
    ordered = {key: value[order] for key, value in identity.items()}
    if len(set(ordered["uid"].tolist())) != len(ordered["uid"]):
        raise Stage1Error(f"UID collision for {task}")

    partition_codes = np.empty(10000, dtype=np.uint8)
    partition_manifest: dict[str, Any] = {}
    for code, (name, start, stop) in enumerate(PARTITIONS):
        partition_codes[start:stop] = code
        partition_manifest[name] = {
            "code": code,
            "start": start,
            "stop": stop,
            "rows": stop - start,
            "uid_sha256": digest_strings(ordered["uid"][start:stop]),
            "image_digest_sha256": digest_strings(
                ordered["image_sha256"][start:stop]
            ),
            "image_array_sha256": digest_array(images[start:stop]),
            "sealed_label_array_sha256": digest_array(labels[start:stop]),
        }

    public_dir = public_root / task
    sealed_dir = sealed_root / task
    public_dir.mkdir(parents=True, exist_ok=True)
    sealed_dir.mkdir(parents=True, exist_ok=True)
    public_npz = public_dir / "inputs.npz"
    sealed_npz = sealed_dir / "labels.npz"
    np.savez_compressed(
        public_npz,
        images=images,
        uids=ordered["uid"],
        image_sha256=ordered["image_sha256"],
        partition_sha256=ordered["partition_sha256"],
        archive_positions=ordered["archive_position"],
        partition_codes=partition_codes,
    )
    np.savez_compressed(
        sealed_npz,
        uids=ordered["uid"],
        partition_codes=partition_codes,
        labels=labels,
    )

    with np.load(public_npz, allow_pickle=False) as public_arrays:
        if set(public_arrays.files) != {
            "images",
            "uids",
            "image_sha256",
            "partition_sha256",
            "archive_positions",
            "partition_codes",
        }:
            raise Stage1Error(f"public key drift for {task}: {public_arrays.files}")
    with np.load(sealed_npz, allow_pickle=False) as sealed_arrays:
        if set(sealed_arrays.files) != {"uids", "partition_codes", "labels"}:
            raise Stage1Error(f"sealed key drift for {task}: {sealed_arrays.files}")
        if not np.array_equal(sealed_arrays["uids"], ordered["uid"]):
            raise Stage1Error(f"sealed UID drift for {task}")

    public_manifest: dict[str, Any] = {
        "schema": "srsa.cifar.stage1.public_task.v1",
        "task": task,
        "classes": int(task_config["classes"]),
        "archive": {**observed, "url": task_config["dataset_url"]},
        "canonical_image": {
            "dtype": "uint8",
            "shape": [10000, 32, 32, 3],
            "channel_order": "RGB",
            "serialization": "C-contiguous bytes",
        },
        "partitions": partition_manifest,
        "arrays": {
            "images_sha256": digest_array(images),
            "uids_sha256": digest_strings(ordered["uid"]),
            "image_sha256_array_sha256": digest_strings(ordered["image_sha256"]),
            "partition_sha256_array_sha256": digest_strings(
                ordered["partition_sha256"]
            ),
            "archive_positions_sha256": digest_array(
                ordered["archive_position"]
            ),
            "partition_codes_sha256": digest_array(partition_codes),
        },
        "public_npz": {
            "filename": public_npz.name,
            "bytes": public_npz.stat().st_size,
            "sha256": file_digest(public_npz),
        },
        "contains_labels": False,
    }
    public_manifest["manifest_sha256"] = canonical_sha256(public_manifest)
    public_manifest_path = public_dir / "manifest.json"
    public_manifest_path.write_text(
        json.dumps(public_manifest, indent=2) + "\n", encoding="utf-8"
    )

    sealed_manifest: dict[str, Any] = {
        "schema": "srsa.cifar.stage1.sealed_task.v1",
        "task": task,
        "rows": 10000,
        "classes": int(task_config["classes"]),
        "uids_sha256": digest_strings(ordered["uid"]),
        "partition_codes_sha256": digest_array(partition_codes),
        "labels_sha256": digest_array(labels),
        "partitions": {
            name: {
                "rows": stop - start,
                "uid_sha256": partition_manifest[name]["uid_sha256"],
                "label_array_sha256": partition_manifest[name][
                    "sealed_label_array_sha256"
                ],
            }
            for name, start, stop in PARTITIONS
        },
        "sealed_npz": {
            "filename": sealed_npz.name,
            "bytes": sealed_npz.stat().st_size,
            "sha256": file_digest(sealed_npz),
        },
    }
    sealed_manifest["manifest_sha256"] = canonical_sha256(sealed_manifest)
    sealed_manifest_path = sealed_dir / "manifest.json"
    sealed_manifest_path.write_text(
        json.dumps(sealed_manifest, indent=2) + "\n", encoding="utf-8"
    )

    return {
        "task": task,
        "public_manifest_sha256": public_manifest["manifest_sha256"],
        "public_npz_sha256": public_manifest["public_npz"]["sha256"],
        "sealed_manifest_sha256": sealed_manifest["manifest_sha256"],
        "sealed_npz_sha256": sealed_manifest["sealed_npz"]["sha256"],
        "partitions": partition_manifest,
        "exact_duplicate_image_groups": int(
            len(images) - len(set(ordered["image_sha256"].tolist()))
        ),
    }


def inventory(root: Path) -> dict[str, Any]:
    records = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_digest(path),
            }
        )
    return {
        "files": len(records),
        "bytes": sum(record["bytes"] for record in records),
        "inventory_sha256": canonical_sha256(records),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage0-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.time()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    lock = json.loads(args.stage0_lock.read_text(encoding="utf-8"))
    if config.get("status") != "PREREGISTERED_BEFORE_CIFAR_EXECUTION":
        raise Stage1Error("config is not preregistered")
    if lock.get("decision") != "PASS_SRSA_STAGE0_PUBLIC_METADATA":
        raise Stage1Error("Stage 0 lock is not PASS")

    public_root = args.output / "public"
    sealed_root = args.output / "sealed"
    public_root.mkdir(parents=True, exist_ok=True)
    sealed_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="srsa-stage1-") as temporary:
        work_root = Path(temporary)
        tasks = {
            task: build_task(
                task,
                task_config,
                lock["datasets"][task],
                public_root,
                sealed_root,
                work_root,
            )
            for task, task_config in config["tasks"].items()
        }

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "decision": "PASS_SRSA_STAGE1_DATA_SEAL",
        "config_sha256": file_digest(args.config),
        "stage0_lock_sha256": file_digest(args.stage0_lock),
        "tasks": tasks,
        "public_inventory": inventory(public_root),
        "sealed_inventory": inventory(sealed_root),
        "information_boundary": {
            "dataset_rows_decoded": True,
            "labels_accessed_only_to_create_separate_seal": True,
            "labels_written_to_public_artifact": False,
            "model_source_loaded": False,
            "model_tensor_loaded": False,
            "model_inference_executed": False,
            "candidate_prediction_accessed": False,
            "selector_executed": False,
            "primary_or_deployment_outcome_opened": False,
        },
        "elapsed_seconds": time.time() - started,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    receipt_path = args.output / "SRSA_STAGE1_DATA_SEAL_RECEIPT.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy2(receipt_path, public_root / receipt_path.name)
    print(
        json.dumps(
            {
                "decision": receipt["decision"],
                "receipt_sha256": receipt["receipt_sha256"],
                "public_inventory": receipt["public_inventory"][
                    "inventory_sha256"
                ],
                "sealed_inventory": receipt["sealed_inventory"][
                    "inventory_sha256"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
