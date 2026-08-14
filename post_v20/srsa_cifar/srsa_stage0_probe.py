#!/usr/bin/env python3
"""SRSA Stage 0: bind public archives, source, and checkpoint runtimes.

This stage does not extract a CIFAR archive or decode any dataset row. It never
reads a task label. It downloads only public immutable inputs, validates model
construction, and writes a metadata receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import torch

from srsa_core import canonical_sha256

SCHEMA = "srsa.cifar.stage0.public_metadata.v1"


class Stage0Error(RuntimeError):
    pass


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "SRSA-CIFAR/1.0 public-metadata-probe"},
    )
    with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as out:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            out.write(block)
    return {
        "url": url,
        "path": target.name,
        "bytes": target.stat().st_size,
        "md5": file_digest(target, "md5"),
        "sha256": file_digest(target, "sha256"),
    }


def source_inventory(repo: Path) -> dict[str, Any]:
    files = sorted(
        path
        for path in repo.rglob("*")
        if path.is_file() and ".git" not in path.parts
    )
    records = []
    total = 0
    for path in files:
        rel = path.relative_to(repo).as_posix()
        size = path.stat().st_size
        total += size
        records.append(
            {
                "path": rel,
                "bytes": size,
                "sha256": file_digest(path),
            }
        )
    return {
        "files": len(records),
        "bytes": total,
        "inventory_sha256": canonical_sha256(records),
        "records": records,
    }


def import_model_package(repo: Path):
    init_path = repo / "pytorch_cifar_models" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "pytorch_cifar_models",
        init_path,
        submodule_search_locations=[str(init_path.parent)],
    )
    if spec is None or spec.loader is None:
        raise Stage0Error("could not import pinned model package")
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
    raise Stage0Error(f"unsupported checkpoint type: {type(value)!r}")


def model_probe(
    package: Any,
    model_name: str,
    weight_url: str,
    target: Path,
    classes: int,
) -> dict[str, Any]:
    weight = download(weight_url, target)
    constructor = getattr(package, model_name, None)
    if constructor is None:
        raise Stage0Error(f"missing model constructor: {model_name}")
    model = constructor(pretrained=False)
    checkpoint = torch.load(target, map_location="cpu", weights_only=True)
    state = normalize_state(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise Stage0Error(
            {
                "model": model_name,
                "missing": list(missing),
                "unexpected": list(unexpected),
            }
        )
    model.eval()
    torch.manual_seed(0)
    with torch.inference_mode():
        output = model(torch.zeros((1, 3, 32, 32), dtype=torch.float32))
    if tuple(output.shape) != (1, classes):
        raise Stage0Error((model_name, tuple(output.shape), classes))
    output_bytes = output.detach().cpu().contiguous().numpy().tobytes()
    weight.update(
        {
            "model_name": model_name,
            "classes": classes,
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
            "trainable_parameter_count": int(
                sum(p.numel() for p in model.parameters() if p.requires_grad)
            ),
            "constructor_module": str(constructor.__module__),
            "model_class": model.__class__.__name__,
            "zero_output_sha256": hashlib.sha256(output_bytes).hexdigest(),
            "state_tensor_count": len(state),
        }
    )
    return weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.time()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("status") != "PREREGISTERED_BEFORE_CIFAR_EXECUTION":
        raise Stage0Error("config is not preregistered")

    observed_commit = subprocess.check_output(
        ["git", "-C", str(args.model_repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    expected_commit = config["sources"]["model_repository"]["commit"]
    if observed_commit != expected_commit:
        raise Stage0Error(
            {"expected_model_commit": expected_commit, "observed": observed_commit}
        )

    args.output.mkdir(parents=True, exist_ok=True)
    downloads = args.output / "downloads"
    dataset_records: dict[str, Any] = {}
    for task, task_config in config["tasks"].items():
        record = download(
            task_config["dataset_url"],
            downloads / task_config["archive_filename"],
        )
        if record["md5"] != task_config["archive_md5"]:
            raise Stage0Error(
                {
                    "task": task,
                    "expected_md5": task_config["archive_md5"],
                    "observed_md5": record["md5"],
                }
            )
        dataset_records[task] = record

    package = import_model_package(args.model_repo)
    model_records: dict[str, Any] = {}
    for task, task_config in config["tasks"].items():
        classes = int(task_config["classes"])
        for suffix in config["root_suffixes_in_order"]:
            model_name = f"{task}_{suffix}"
            model_records[model_name] = model_probe(
                package,
                model_name,
                config["weight_urls"][model_name],
                downloads / f"{model_name}.pt",
                classes,
            )

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "decision": "PASS_SRSA_STAGE0_PUBLIC_METADATA",
        "config_sha256": file_digest(args.config),
        "model_repository": {
            "commit": observed_commit,
            **source_inventory(args.model_repo),
        },
        "datasets": dataset_records,
        "models": model_records,
        "information_boundary": {
            "dataset_archives_downloaded": True,
            "dataset_archives_extracted": False,
            "dataset_rows_decoded": False,
            "labels_accessed": False,
            "model_tensors_loaded": True,
            "zero_input_inference_only": True,
            "cifar_predictions_accessed": False,
            "selector_executed": False,
            "primary_or_deployment_outcome_opened": False,
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "platform": sys.platform,
            "elapsed_seconds": time.time() - started,
        },
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    path = args.output / "SRSA_STAGE0_PUBLIC_METADATA_RECEIPT.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": receipt["decision"],
        "receipt_sha256": receipt["receipt_sha256"],
        "datasets": {
            key: value["sha256"] for key, value in dataset_records.items()
        },
        "models": {
            key: value["sha256"] for key, value in model_records.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
