#!/usr/bin/env python3
"""Execute one frozen clean root on primary/deployment inputs without labels."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from huggingface_hub import hf_hub_download

import stage2_clean_safety_endpoint as stage2
from stage3_learned_common import array_sha256, digest_lines, sha256_file

ROOT = Path(__file__).resolve().parent
STAGE1_LOCK = ROOT / "PRAA_STAGE1_ARTIFACT_AND_DATA_SEAL_LOCK_2026-08-14.json"
STAGE4_PROTOCOL = ROOT / "PRAA_STAGE4_INPUT_ONLY_PREOUTCOME_PROTOCOL_2026-08-14.md"
SCHEMA = "praa.stage4.clean_endpoint.v1"
PARTITIONS = ("primary_outcome", "deployment")
ROOT_TO_TASK = stage2.ROOT_TO_TASK
TRANSFORMER_ROOTS = stage2.TRANSFORMER_ROOTS
FASTTEXT_ROOTS = stage2.FASTTEXT_ROOTS
EMOTION_LABELS = stage2.EMOTION_LABELS
LID_LABELS = stage2.LID_LABELS
LID_OUT_OF_TARGET = stage2.LID_OUT_OF_TARGET


class Stage4Error(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Stage4Error(path)
    return value


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    output: dict[str, str | None] = {}
    for name in names:
        try:
            output[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            output[name] = None
    return output


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
    if any(row.get("partition") != partition for row in rows):
        raise Stage4Error(f"partition marker drift for {task}/{partition}")
    uids = [str(row["uid"]) for row in rows]
    if digest_lines(uids) != expected["uid_sha256"]:
        raise Stage4Error(f"UID digest drift for {task}/{partition}")
    return {
        "path": path,
        "uids": uids,
        "texts": [str(row["input"]["text"]) for row in rows],
    }


def transformer_partitions(
    task: str,
    spec: Mapping[str, Any],
    texts_by_partition: Mapping[str, list[str]],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(0)
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    with tempfile.TemporaryDirectory(prefix="praa-stage4-transformer-") as tmp:
        model_dir, binding = stage2.materialize_transformer(spec, Path(tmp) / "model")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        use_safetensors = str(spec["weight"]["path"]).endswith(".safetensors")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            local_files_only=True,
            use_safetensors=use_safetensors,
        )
        model.eval()
        mapping = stage2.transformer_mapping(task, model.config)
        n_classes = len(EMOTION_LABELS) if task == "emotion" else len(LID_LABELS)
        batch_size = 64 if task == "emotion" else 32
        output: dict[str, dict[str, np.ndarray]] = {}
        with torch.inference_mode():
            for partition, texts in texts_by_partition.items():
                prediction_parts: list[np.ndarray] = []
                probability_parts: list[np.ndarray] = []
                confidence_parts: list[np.ndarray] = []
                for start in range(0, len(texts), batch_size):
                    encoded = tokenizer(
                        texts[start : start + batch_size],
                        padding=True,
                        truncation=True,
                        max_length=128,
                        return_tensors="pt",
                    )
                    native_logits = model(**encoded).logits.float()
                    mapped_logits = torch.empty(
                        (native_logits.shape[0], n_classes), dtype=torch.float32
                    )
                    for native_index, target_index in mapping.items():
                        mapped_logits[:, target_index] = native_logits[:, native_index]
                    probability = torch.softmax(mapped_logits, dim=-1)
                    prediction_parts.append(
                        torch.argmax(probability, dim=-1).cpu().numpy().astype(np.int16)
                    )
                    probability_parts.append(probability.cpu().numpy().astype(np.float32))
                    confidence_parts.append(
                        torch.max(probability, dim=-1).values.cpu().numpy().astype(np.float32)
                    )
                output[partition] = {
                    "predictions": np.concatenate(prediction_parts),
                    "scores": np.concatenate(probability_parts),
                    "confidence": np.concatenate(confidence_parts),
                }
        runtime = {
            **binding,
            "runtime_kind": "transformer_mapped_softmax",
            "model_class": model.__class__.__name__,
            "tokenizer_class": tokenizer.__class__.__name__,
            "batch_size": batch_size,
            "max_length": 128,
            "score_kind": "mapped_softmax_probability",
            "nll_eligible": True,
            "native_index_to_target_index": {
                str(key): int(value) for key, value in mapping.items()
            },
        }
        return output, runtime


def _mapped_top_probabilities(labels: list[str], probabilities: np.ndarray) -> np.ndarray:
    mapped = np.zeros(len(LID_LABELS) + 1, dtype=np.float32)
    for label, probability in zip(labels, probabilities, strict=True):
        mapped[stage2.native_lid_to_target(label)] += float(probability)
    return mapped


def fasttext_partitions(
    root_id: str,
    spec: Mapping[str, Any],
    texts_by_partition: Mapping[str, list[str]],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    import fasttext

    weight = spec["weight"]
    with tempfile.TemporaryDirectory(prefix="praa-stage4-fasttext-") as tmp:
        path = Path(
            hf_hub_download(
                repo_id=str(spec["source"]),
                filename=str(weight["path"]),
                revision=str(spec["revision"]),
                cache_dir=tmp,
            )
        )
        stage2.verify_file(path, weight)
        model = fasttext.load_model(str(path))
        output: dict[str, dict[str, np.ndarray]] = {}
        for partition, texts in texts_by_partition.items():
            predictions: list[int] = []
            scores: list[np.ndarray] = []
            confidence: list[float] = []
            for text in texts:
                if "\n" in text or "\r" in text:
                    raise Stage4Error("fastText input contains an unregistered newline")
                native_labels, native_probabilities = model.predict(text, k=8)
                labels = [str(value) for value in native_labels]
                probabilities = np.asarray(native_probabilities, dtype=np.float64)
                if not labels or len(labels) != len(probabilities):
                    raise Stage4Error(f"malformed fastText output from {root_id}")
                predictions.append(stage2.native_lid_to_target(labels[0]))
                scores.append(_mapped_top_probabilities(labels, probabilities))
                confidence.append(float(probabilities[0]))
            output[partition] = {
                "predictions": np.asarray(predictions, dtype=np.int16),
                "scores": np.stack(scores).astype(np.float32),
                "confidence": np.asarray(confidence, dtype=np.float32),
            }
        runtime = {
            "runtime_kind": "fasttext_top8_mapped_scores",
            "repo_id": str(spec["source"]),
            "revision": str(spec["revision"]),
            "weight_path": str(weight["path"]),
            "weight_bytes": int(weight["bytes"]),
            "weight_sha256": str(weight["sha256"]),
            "score_kind": "mapped_top8_native_probability",
            "nll_eligible": False,
            "out_of_target_code": LID_OUT_OF_TARGET,
        }
        return output, runtime


def langid_partitions(
    spec: Mapping[str, Any], texts_by_partition: Mapping[str, list[str]]
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    import langid

    installed = importlib.metadata.version("langid")
    if installed != str(spec["version"]):
        raise Stage4Error(f"langid version drift: {installed} != {spec['version']}")
    identifier = getattr(langid, "identifier", None)
    if identifier is None:
        langid.classify("")
        identifier = getattr(langid, "identifier", None)
    if identifier is None:
        raise Stage4Error("langid identifier unavailable")

    output: dict[str, dict[str, np.ndarray]] = {}
    for partition, texts in texts_by_partition.items():
        predictions: list[int] = []
        scores: list[np.ndarray] = []
        confidence: list[float] = []
        for text in texts:
            ranking = identifier.rank(text)
            if not ranking:
                raise Stage4Error("empty langid ranking")
            labels = [str(label) for label, _ in ranking]
            native_scores = np.asarray([float(score) for _, score in ranking], dtype=np.float64)
            shifted = native_scores - float(np.max(native_scores))
            probabilities = np.exp(shifted)
            probabilities /= np.sum(probabilities)
            predictions.append(stage2.native_lid_to_target(labels[0]))
            scores.append(_mapped_top_probabilities(labels, probabilities))
            confidence.append(float(np.max(probabilities)))
        output[partition] = {
            "predictions": np.asarray(predictions, dtype=np.int16),
            "scores": np.stack(scores).astype(np.float32),
            "confidence": np.asarray(confidence, dtype=np.float32),
        }
    runtime = {
        "runtime_kind": "langid_full_ranking_mapped_scores",
        "package": "langid",
        "version": installed,
        "distribution_path": str(spec["distribution"]["path"]),
        "distribution_bytes": int(spec["distribution"]["bytes"]),
        "distribution_sha256": str(spec["distribution"]["sha256"]),
        "score_kind": "softmax_normalized_native_ranking",
        "nll_eligible": False,
        "out_of_target_code": LID_OUT_OF_TARGET,
    }
    return output, runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-id", required=True, choices=sorted(ROOT_TO_TASK))
    parser.add_argument("--stage1-public-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    started = time.time()
    lock = load_json(STAGE1_LOCK)
    if lock.get("decision") != "PASS_PRAA_STAGE1_ARTIFACT_AND_DATA_SEAL":
        raise Stage4Error("Stage-1 lock is not PASS")
    task = ROOT_TO_TASK[args.root_id]
    partitions = {
        name: read_partition(args.stage1_public_root, task, name, lock)
        for name in PARTITIONS
    }
    texts = {name: value["texts"] for name, value in partitions.items()}
    spec = lock["runtime_artifacts"][args.root_id]

    if args.root_id in TRANSFORMER_ROOTS:
        inferred, runtime = transformer_partitions(task, spec, texts)
    elif args.root_id in FASTTEXT_ROOTS:
        inferred, runtime = fasttext_partitions(args.root_id, spec, texts)
    elif args.root_id == "lid-langid-package":
        inferred, runtime = langid_partitions(spec, texts)
    else:
        raise Stage4Error(args.root_id)

    for partition in PARTITIONS:
        expected_rows = len(partitions[partition]["uids"])
        for key in ("predictions", "scores", "confidence"):
            if len(inferred[partition][key]) != expected_rows:
                raise Stage4Error(f"output row drift for {args.root_id}/{partition}/{key}")
        if not np.all(np.isfinite(inferred[partition]["scores"])):
            raise Stage4Error(f"non-finite scores for {args.root_id}/{partition}")

    output_dir = args.output_root / args.root_id
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "primary_deployment_outputs.npz"
    np.savez_compressed(
        arrays_path,
        primary_uids=np.asarray(partitions["primary_outcome"]["uids"], dtype="<U64"),
        deployment_uids=np.asarray(partitions["deployment"]["uids"], dtype="<U64"),
        primary_predictions=inferred["primary_outcome"]["predictions"],
        deployment_predictions=inferred["deployment"]["predictions"],
        primary_scores=inferred["primary_outcome"]["scores"],
        deployment_scores=inferred["deployment"]["scores"],
        primary_confidence=inferred["primary_outcome"]["confidence"],
        deployment_confidence=inferred["deployment"]["confidence"],
    )

    receipt = {
        "schema": SCHEMA,
        "task": task,
        "root_id": args.root_id,
        "decision": "PASS_PRAA_STAGE4_CLEAN_ENDPOINT",
        "stage1_lock_sha256": sha256_file(STAGE1_LOCK),
        "stage4_protocol_sha256": sha256_file(STAGE4_PROTOCOL),
        "partitions": {
            name: {
                "rows": len(partitions[name]["uids"]),
                "input_sha256": sha256_file(partitions[name]["path"]),
                "uid_sha256": digest_lines(partitions[name]["uids"]),
                "prediction_sha256": array_sha256(inferred[name]["predictions"]),
                "score_sha256": array_sha256(inferred[name]["scores"]),
                "confidence_sha256": array_sha256(inferred[name]["confidence"]),
                "score_dimension": int(inferred[name]["scores"].shape[1]),
            }
            for name in PARTITIONS
        },
        "runtime": runtime,
        "arrays_file": arrays_path.name,
        "arrays_sha256": sha256_file(arrays_path),
        "wrapper_sha256": sha256_file(Path(__file__).resolve()),
        "dependencies": package_versions(
            [
                "numpy",
                "torch",
                "transformers",
                "tokenizers",
                "safetensors",
                "sentencepiece",
                "fasttext-wheel",
                "langid",
                "huggingface-hub",
            ]
        ),
        "elapsed_seconds": time.time() - started,
        "information_boundary": {
            "labels_received": False,
            "non_primary_or_deployment_inputs_opened": False,
            "aliases_executed": False,
            "selector_executed": False,
            "primary_labels_opened": False,
            "deployment_labels_opened": False,
        },
    }
    receipt_path = output_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
