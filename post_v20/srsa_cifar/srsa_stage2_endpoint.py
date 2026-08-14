#!/usr/bin/env python3
"""Execute one frozen SRSA clean CIFAR endpoint without labels."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from srsa_core import canonical_sha256, digest_array

SCHEMA = "srsa.cifar.stage2.clean_endpoint.v1"


class Stage2Error(RuntimeError):
    pass


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_strings(values: np.ndarray) -> str:
    payload = "\n".join(str(value) for value in values.tolist()) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "SRSA-CIFAR/1.0 stage2-clean-endpoint"},
    )
    with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)


def import_model_package(repo: Path):
    init_path = repo / "pytorch_cifar_models" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "pytorch_cifar_models",
        init_path,
        submodule_search_locations=[str(init_path.parent)],
    )
    if spec is None or spec.loader is None:
        raise Stage2Error("could not import pinned model package")
    module = importlib.util.module_from_spec(spec)
    sys.modules["pytorch_cifar_models"] = module
    spec.loader.exec_module(module)
    return module


def normalize_state(value: Any) -> dict[str, torch.Tensor]:
    if isinstance(value, dict) and value and all(
        isinstance(key, str) and isinstance(tensor, torch.Tensor)
        for key, tensor in value.items()
    ):
        return value
    if isinstance(value, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            candidate = value.get(key)
            if isinstance(candidate, dict):
                return normalize_state(candidate)
    raise Stage2Error(f"unsupported checkpoint type: {type(value)!r}")


def load_public_inputs(
    public_root: Path,
    task: str,
    stage1_lock: dict[str, Any],
) -> dict[str, Any]:
    task_lock = stage1_lock["tasks"][task]
    directory = public_root / task
    manifest_path = directory / "manifest.json"
    arrays_path = directory / "inputs.npz"
    if not manifest_path.is_file() or not arrays_path.is_file():
        raise Stage2Error(f"missing Stage-1 public input for {task}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed_manifest_hash = manifest.get("manifest_sha256")
    expected_manifest_hash = task_lock["public_manifest_sha256"]
    if observed_manifest_hash != expected_manifest_hash:
        raise Stage2Error(
            {
                "task": task,
                "manifest_expected": expected_manifest_hash,
                "manifest_observed": observed_manifest_hash,
            }
        )
    if file_digest(arrays_path) != task_lock["public_npz_sha256"]:
        raise Stage2Error(f"public NPZ file binding failed for {task}")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        expected_keys = {
            "images",
            "uids",
            "image_sha256",
            "partition_sha256",
            "archive_positions",
            "partition_codes",
        }
        if set(arrays.files) != expected_keys:
            raise Stage2Error((task, arrays.files))
        value = {key: arrays[key].copy() for key in arrays.files}
    if value["images"].shape != (10000, 32, 32, 3):
        raise Stage2Error((task, value["images"].shape))
    if value["images"].dtype != np.uint8:
        raise Stage2Error((task, value["images"].dtype))
    if len(set(value["uids"].astype(str).tolist())) != 10000:
        raise Stage2Error(f"UID uniqueness failed for {task}")
    if digest_array(value["images"]) != manifest["arrays"]["images_sha256"]:
        raise Stage2Error(f"image array binding failed for {task}")
    if digest_strings(value["uids"].astype(str)) != manifest["arrays"]["uids_sha256"]:
        raise Stage2Error(f"UID binding failed for {task}")
    if digest_strings(value["image_sha256"].astype(str)) != manifest["arrays"]["image_sha256_array_sha256"]:
        raise Stage2Error(f"image digest binding failed for {task}")
    if digest_array(value["partition_codes"]) != manifest["arrays"]["partition_codes_sha256"]:
        raise Stage2Error(f"partition binding failed for {task}")
    if np.bincount(value["partition_codes"].astype(np.int64), minlength=3).tolist() != [2000, 6000, 2000]:
        raise Stage2Error(f"partition counts drift for {task}")
    return {"manifest": manifest, "arrays_path": arrays_path, **value}


def load_model(
    package: Any,
    model_name: str,
    weight_path: Path,
    expected: dict[str, Any],
    classes: int,
) -> torch.nn.Module:
    constructor = getattr(package, model_name, None)
    if constructor is None:
        raise Stage2Error(f"missing constructor: {model_name}")
    model = constructor(pretrained=False)
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
    state = normalize_state(checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()
    parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    if parameters != int(expected["parameter_count"]):
        raise Stage2Error(
            {
                "model": model_name,
                "expected_parameters": expected["parameter_count"],
                "observed_parameters": parameters,
            }
        )
    torch.manual_seed(0)
    with torch.inference_mode():
        zero_output = model(torch.zeros((1, 3, 32, 32), dtype=torch.float32))
    if tuple(zero_output.shape) != (1, classes):
        raise Stage2Error((model_name, tuple(zero_output.shape), classes))
    zero_digest = hashlib.sha256(
        zero_output.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()
    if zero_digest != expected["zero_output_sha256"]:
        raise Stage2Error(
            {
                "model": model_name,
                "zero_expected": expected["zero_output_sha256"],
                "zero_observed": zero_digest,
            }
        )
    return model


def infer(
    model: torch.nn.Module,
    images: np.ndarray,
    mean: list[float],
    std: list[float],
    classes: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(0)
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    logits_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            stop = min(len(images), start + batch_size)
            batch = (
                torch.from_numpy(images[start:stop])
                .permute(0, 3, 1, 2)
                .contiguous()
                .float()
                .div_(255.0)
            )
            batch = (batch - mean_tensor) / std_tensor
            output = model(batch)
            if tuple(output.shape) != (stop - start, classes):
                raise Stage2Error(tuple(output.shape))
            logits_parts.append(output.float().cpu().numpy().astype(np.float32))
    logits = np.concatenate(logits_parts, axis=0)
    if logits.shape != (len(images), classes) or not np.all(np.isfinite(logits)):
        raise Stage2Error(logits.shape)
    predictions = np.argmax(logits, axis=1).astype(np.int16)
    return predictions, logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["cifar100", "cifar10"], required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage0-lock", type=Path, required=True)
    parser.add_argument("--stage1-lock", type=Path, required=True)
    parser.add_argument("--stage1-public-root", type=Path, required=True)
    parser.add_argument("--model-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    started = time.time()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    stage0 = json.loads(args.stage0_lock.read_text(encoding="utf-8"))
    stage1 = json.loads(args.stage1_lock.read_text(encoding="utf-8"))
    if stage0.get("decision") != "PASS_SRSA_STAGE0_PUBLIC_METADATA":
        raise Stage2Error("Stage 0 lock is not PASS")
    if stage1.get("decision") != "PASS_SRSA_STAGE1_DATA_SEAL":
        raise Stage2Error("Stage 1 lock is not PASS")
    expected_model_name = args.model_name
    expected_task_prefix = f"{args.task}_"
    if not expected_model_name.startswith(expected_task_prefix):
        raise Stage2Error((args.task, args.model_name))
    if expected_model_name not in config["weight_urls"]:
        raise Stage2Error(f"model not in frozen roster: {args.model_name}")

    expected_commit = config["sources"]["model_repository"]["commit"]
    observed_commit = subprocess.check_output(
        ["git", "-C", str(args.model_repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if observed_commit != expected_commit:
        raise Stage2Error((expected_commit, observed_commit))

    public = load_public_inputs(args.stage1_public_root, args.task, stage1)
    expected_model = stage0["models"][args.model_name]
    args.output.mkdir(parents=True, exist_ok=True)
    weight_path = args.output / f"{args.model_name}.pt"
    download(config["weight_urls"][args.model_name], weight_path)
    if file_digest(weight_path) != expected_model["sha256"]:
        raise Stage2Error(f"checkpoint SHA-256 drift for {args.model_name}")

    package = import_model_package(args.model_repo)
    task_config = config["tasks"][args.task]
    classes = int(task_config["classes"])
    model = load_model(
        package,
        args.model_name,
        weight_path,
        expected_model,
        classes,
    )
    predictions, logits = infer(
        model,
        public["images"],
        task_config["normalization"]["mean"],
        task_config["normalization"]["std"],
        classes,
        args.batch_size,
    )

    arrays_path = args.output / "predictions.npz"
    np.savez_compressed(
        arrays_path,
        uids=public["uids"].astype("<U64"),
        partition_codes=public["partition_codes"].astype(np.uint8),
        predictions=predictions,
        logits=logits,
    )
    partition_results: dict[str, Any] = {}
    for name, code in (("safety", 0), ("primary_outcome", 1), ("deployment", 2)):
        mask = public["partition_codes"] == code
        partition_results[name] = {
            "rows": int(np.sum(mask)),
            "uid_sha256": digest_strings(public["uids"][mask].astype(str)),
            "prediction_sha256": digest_array(predictions[mask]),
            "logits_sha256": digest_array(logits[mask]),
        }

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "decision": "PASS_SRSA_STAGE2_CLEAN_ENDPOINT",
        "task": args.task,
        "model_name": args.model_name,
        "root_suffix": args.model_name[len(expected_task_prefix) :],
        "source_commit": observed_commit,
        "checkpoint": {
            "url": config["weight_urls"][args.model_name],
            "bytes": weight_path.stat().st_size,
            "sha256": file_digest(weight_path),
        },
        "stage0_model_binding": {
            "parameter_count": expected_model["parameter_count"],
            "model_class": expected_model["model_class"],
            "zero_output_sha256": expected_model["zero_output_sha256"],
        },
        "stage1_binding": {
            "public_manifest_sha256": stage1["tasks"][args.task][
                "public_manifest_sha256"
            ],
            "public_npz_sha256": stage1["tasks"][args.task][
                "public_npz_sha256"
            ],
            "uids_sha256": digest_strings(public["uids"].astype(str)),
            "partition_codes_sha256": digest_array(public["partition_codes"]),
        },
        "preprocessing": {
            "input_dtype": "uint8",
            "input_shape": [10000, 32, 32, 3],
            "conversion": "float32 / 255, NHWC to NCHW",
            "mean": task_config["normalization"]["mean"],
            "std": task_config["normalization"]["std"],
            "augmentation": False,
            "batch_size": args.batch_size,
        },
        "outputs": {
            "rows": len(predictions),
            "classes": classes,
            "predictions_sha256": digest_array(predictions),
            "logits_sha256": digest_array(logits),
            "arrays_file": arrays_path.name,
            "arrays_bytes": arrays_path.stat().st_size,
            "arrays_sha256": file_digest(arrays_path),
            "partitions": partition_results,
        },
        "code_sha256": file_digest(Path(__file__).resolve()),
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "packages": package_versions(["torch", "numpy"]),
            "elapsed_seconds": time.time() - started,
        },
        "information_boundary": {
            "stage1_public_inputs_accessed": True,
            "stage1_sealed_labels_accessed": False,
            "labels_accessed": False,
            "clean_endpoint_inference_executed": True,
            "structured_wrappers_constructed": False,
            "selector_executed": False,
            "candidate_quality_computed": False,
            "primary_or_deployment_outcome_opened": False,
        },
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    receipt_path = args.output / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    weight_path.unlink()
    print(
        json.dumps(
            {
                "decision": receipt["decision"],
                "task": args.task,
                "model": args.model_name,
                "receipt_sha256": receipt["receipt_sha256"],
                "prediction_sha256": receipt["outputs"]["predictions_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
