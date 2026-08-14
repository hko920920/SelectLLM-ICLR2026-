#!/usr/bin/env python3
"""Run one frozen PRAA clean endpoint on safety inputs only.

Labels are deliberately absent from this executable. The output is an ordered
UID/prediction vector and an execution receipt; no candidate quality or selector
quantity is computed here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download

SCRIPT_DIR = Path(__file__).resolve().parent
LOCK_PATH = SCRIPT_DIR / "PRAA_STAGE1_ARTIFACT_AND_DATA_SEAL_LOCK_2026-08-14.json"
SCHEMA = "praa.stage2.clean_safety_endpoint.v1"

EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
LID_LABELS = [
    "ar", "bg", "de", "el", "en", "es", "fr", "hi", "it", "ja",
    "nl", "pl", "pt", "ru", "sw", "th", "tr", "ur", "vi", "zh",
]
LID_OUT_OF_TARGET = len(LID_LABELS)
LID_THREE_TO_TWO = {
    "arb": "ar", "ara": "ar",
    "bul": "bg",
    "deu": "de", "ger": "de",
    "ell": "el", "gre": "el",
    "eng": "en",
    "spa": "es",
    "fra": "fr", "fre": "fr",
    "hin": "hi",
    "ita": "it",
    "jpn": "ja",
    "nld": "nl", "dut": "nl",
    "pol": "pl",
    "por": "pt",
    "rus": "ru",
    "swh": "sw", "swa": "sw",
    "tha": "th",
    "tur": "tr",
    "urd": "ur",
    "vie": "vi",
    "zho": "zh", "cmn": "zh", "chi": "zh",
}

ROOT_TO_TASK = {
    "emotion-roberta-dk409": "emotion",
    "emotion-bert-nateraw": "emotion",
    "emotion-distilbert-bhadresh": "emotion",
    "emotion-albert-bhadresh": "emotion",
    "lid-xlmroberta-papluca": "language_identification",
    "lid-fasttext-facebook": "language_identification",
    "lid-fasttext-glotlid": "language_identification",
    "lid-langid-package": "language_identification",
}
TRANSFORMER_ROOTS = {
    "emotion-roberta-dk409",
    "emotion-bert-nateraw",
    "emotion-distilbert-bhadresh",
    "emotion-albert-bhadresh",
    "lid-xlmroberta-papluca",
}
FASTTEXT_ROOTS = {"lid-fasttext-facebook", "lid-fasttext-glotlid"}


class Stage2Error(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Stage2Error(f"expected JSON object at {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(b"|")
    digest.update(json.dumps(contiguous.shape).encode("utf-8"))
    digest.update(b"|")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def digest_lines(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def normalize_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def read_safety_inputs(
    public_root: Path, task: str, lock: Mapping[str, Any]
) -> tuple[list[str], list[str], Path]:
    path = public_root / "inputs" / task / "safety.jsonl"
    expected = lock["tasks"][task]["partitions"]["safety"]
    if not path.is_file():
        raise Stage2Error(f"missing Stage-1 safety input: {path}")
    if sha256_file(path) != expected["input_sha256"]:
        raise Stage2Error(f"safety input hash drift for {task}")
    uids: list[str] = []
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("partition") != "safety":
                raise Stage2Error("non-safety row encountered")
            if set(row.get("input", {})) != {"text"}:
                raise Stage2Error("unexpected input schema")
            uids.append(str(row["uid"]))
            texts.append(str(row["input"]["text"]))
    if len(uids) != int(expected["rows"]):
        raise Stage2Error(f"safety row count drift for {task}: {len(uids)}")
    if digest_lines(uids) != expected["uid_sha256"]:
        raise Stage2Error(f"safety UID digest drift for {task}")
    if len(set(uids)) != len(uids):
        raise Stage2Error("duplicate safety UID")
    return uids, texts, path


def verify_file(path: Path, expected: Mapping[str, Any]) -> None:
    if not path.is_file():
        raise Stage2Error(f"missing runtime artifact {path}")
    if path.stat().st_size != int(expected["bytes"]):
        raise Stage2Error(
            f"runtime artifact byte drift for {path}: {path.stat().st_size} != {expected['bytes']}"
        )
    observed = sha256_file(path)
    if observed != expected["sha256"]:
        raise Stage2Error(
            f"runtime artifact SHA-256 drift for {path}: {observed} != {expected['sha256']}"
        )


def transformer_mapping(task: str, config: Any) -> dict[int, int]:
    id2label = {int(key): str(value) for key, value in config.id2label.items()}
    if task == "emotion":
        expected = {normalize_label(label): idx for idx, label in enumerate(EMOTION_LABELS)}
    else:
        expected = {normalize_label(label): idx for idx, label in enumerate(LID_LABELS)}
    mapping: dict[int, int] = {}
    for native, label in id2label.items():
        normalized = normalize_label(label)
        if normalized not in expected:
            raise Stage2Error(f"native label {label!r} outside frozen target set")
        mapping[native] = expected[normalized]
    if set(mapping.values()) != set(range(len(expected))):
        raise Stage2Error(f"incomplete transformer label map: {mapping}")
    return mapping


def materialize_transformer(
    spec: Mapping[str, Any], destination: Path
) -> tuple[Path, dict[str, Any]]:
    repo_id = str(spec["source"])
    revision = str(spec["revision"])
    weight = spec["weight"]
    patterns = [
        str(weight["path"]),
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.txt",
        "vocab.json",
        "merges.txt",
        "sentencepiece.bpe.model",
        "spiece.model",
    ]
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=destination,
        allow_patterns=patterns,
    )
    weight_path = destination / str(weight["path"])
    verify_file(weight_path, weight)
    config_path = destination / "config.json"
    if not config_path.is_file():
        raise Stage2Error(f"missing config.json for {repo_id}")
    expected_config = str(spec.get("config_sha256") or "")
    if expected_config and sha256_file(config_path) != expected_config:
        raise Stage2Error(f"config hash drift for {repo_id}")
    if spec.get("tokenizer_json_sha256"):
        token_path = destination / "tokenizer.json"
        if not token_path.is_file() or sha256_file(token_path) != spec["tokenizer_json_sha256"]:
            raise Stage2Error(f"tokenizer.json hash drift for {repo_id}")
    if spec.get("vocab_sha256"):
        vocab_path = destination / "vocab.txt"
        if not vocab_path.is_file() or sha256_file(vocab_path) != spec["vocab_sha256"]:
            raise Stage2Error(f"vocab hash drift for {repo_id}")
    if spec.get("sentencepiece_sha256"):
        sp_path = destination / "sentencepiece.bpe.model"
        if not sp_path.is_file() or sha256_file(sp_path) != spec["sentencepiece_sha256"]:
            raise Stage2Error(f"sentencepiece hash drift for {repo_id}")
    return destination, {
        "repo_id": repo_id,
        "revision": revision,
        "weight_path": str(weight["path"]),
        "weight_bytes": int(weight["bytes"]),
        "weight_sha256": str(weight["sha256"]),
        "config_sha256": sha256_file(config_path),
    }


def run_transformer(
    task: str,
    spec: Mapping[str, Any],
    texts: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(0)
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)

    with tempfile.TemporaryDirectory(prefix="praa-stage2-transformer-") as tmp:
        model_dir, binding = materialize_transformer(spec, Path(tmp) / "model")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        use_safetensors = str(spec["weight"]["path"]).endswith(".safetensors")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            local_files_only=True,
            use_safetensors=use_safetensors,
        )
        model.eval()
        mapping = transformer_mapping(task, model.config)
        batch_size = 64 if task == "emotion" else 32
        predictions: list[int] = []
        confidence: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                logits = model(**encoded).logits
                probabilities = torch.softmax(logits, dim=-1)
                native = torch.argmax(probabilities, dim=-1).cpu().numpy()
                top = torch.max(probabilities, dim=-1).values.cpu().numpy()
                predictions.extend(mapping[int(value)] for value in native)
                confidence.extend(float(value) for value in top)
        return (
            np.asarray(predictions, dtype=np.int16),
            np.asarray(confidence, dtype=np.float32),
            {
                **binding,
                "runtime_kind": "transformer_sequence_classification",
                "batch_size": batch_size,
                "max_length": 128,
                "native_index_to_target_index": {
                    str(key): int(value) for key, value in mapping.items()
                },
            },
        )


def native_lid_to_target(label: str) -> int:
    value = str(label).strip()
    if value.startswith("__label__"):
        value = value[len("__label__") :]
    value = value.strip().lower()
    if value in LID_LABELS:
        return LID_LABELS.index(value)
    first = re.split(r"[_-]", value, maxsplit=1)[0]
    if first in LID_LABELS:
        return LID_LABELS.index(first)
    if first[:3] in LID_THREE_TO_TWO:
        return LID_LABELS.index(LID_THREE_TO_TWO[first[:3]])
    return LID_OUT_OF_TARGET


def run_fasttext(
    root_id: str,
    spec: Mapping[str, Any],
    texts: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import fasttext

    repo_id = str(spec["source"])
    revision = str(spec["revision"])
    weight = spec["weight"]
    with tempfile.TemporaryDirectory(prefix="praa-stage2-fasttext-") as tmp:
        materialized = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=str(weight["path"]),
                revision=revision,
                cache_dir=tmp,
            )
        )
        verify_file(materialized, weight)
        model = fasttext.load_model(str(materialized))
        predictions: list[int] = []
        confidence: list[float] = []
        for text in texts:
            if "\n" in text or "\r" in text:
                raise Stage2Error(
                    f"{root_id} safety input contains a newline; no unregistered normalization is allowed"
                )
            labels, probabilities = model.predict(text, k=1)
            if not labels or not probabilities:
                raise Stage2Error(f"{root_id} returned no prediction")
            predictions.append(native_lid_to_target(labels[0]))
            confidence.append(float(probabilities[0]))
        return (
            np.asarray(predictions, dtype=np.int16),
            np.asarray(confidence, dtype=np.float32),
            {
                "runtime_kind": "fasttext",
                "repo_id": repo_id,
                "revision": revision,
                "weight_path": str(weight["path"]),
                "weight_bytes": int(weight["bytes"]),
                "weight_sha256": str(weight["sha256"]),
                "out_of_target_code": LID_OUT_OF_TARGET,
            },
        )


def run_langid(
    spec: Mapping[str, Any], texts: list[str]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import langid

    distribution = spec["distribution"]
    installed = importlib.metadata.version("langid")
    if installed != str(spec["version"]):
        raise Stage2Error(f"langid version drift: {installed} != {spec['version']}")
    predictions: list[int] = []
    scores: list[float] = []
    for text in texts:
        label, score = langid.classify(text)
        predictions.append(native_lid_to_target(label))
        scores.append(float(score))
    return (
        np.asarray(predictions, dtype=np.int16),
        np.asarray(scores, dtype=np.float32),
        {
            "runtime_kind": "langid",
            "package": "langid",
            "version": installed,
            "distribution_path": str(distribution["path"]),
            "distribution_bytes": int(distribution["bytes"]),
            "distribution_sha256": str(distribution["sha256"]),
            "out_of_target_code": LID_OUT_OF_TARGET,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-id", required=True, choices=sorted(ROOT_TO_TASK))
    parser.add_argument("--stage1-public-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    task = ROOT_TO_TASK[args.root_id]
    lock = load_json(LOCK_PATH)
    if lock.get("decision") != "PASS_PRAA_STAGE1_ARTIFACT_AND_DATA_SEAL":
        raise Stage2Error("Stage-1 lock is not PASS")
    spec = lock["runtime_artifacts"][args.root_id]
    uids, texts, input_path = read_safety_inputs(args.stage1_public_root, task, lock)

    started = time.time()
    if args.root_id in TRANSFORMER_ROOTS:
        predictions, confidence, runtime = run_transformer(task, spec, texts)
    elif args.root_id in FASTTEXT_ROOTS:
        predictions, confidence, runtime = run_fasttext(args.root_id, spec, texts)
    elif args.root_id == "lid-langid-package":
        predictions, confidence, runtime = run_langid(spec, texts)
    else:
        raise Stage2Error(f"unsupported root {args.root_id}")

    if predictions.shape != (len(uids),) or confidence.shape != (len(uids),):
        raise Stage2Error("endpoint output shape mismatch")
    upper = len(EMOTION_LABELS) - 1 if task == "emotion" else LID_OUT_OF_TARGET
    if np.any(predictions < 0) or np.any(predictions > upper):
        raise Stage2Error("mapped prediction outside frozen code range")
    if not np.all(np.isfinite(confidence)):
        raise Stage2Error("non-finite endpoint confidence/score")

    output_dir = args.output_root / args.root_id
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "safety_predictions.npz"
    np.savez_compressed(
        arrays_path,
        uids=np.asarray(uids, dtype="<U64"),
        predictions=predictions,
        confidence=confidence,
    )
    receipt = {
        "schema": SCHEMA,
        "task": task,
        "root_id": args.root_id,
        "stage1_lock_sha256": sha256_file(LOCK_PATH),
        "safety_input_file": str(input_path),
        "safety_input_sha256": sha256_file(input_path),
        "safety_uid_sha256": digest_lines(uids),
        "rows": len(uids),
        "prediction_vector_sha256": sha256_array(predictions),
        "confidence_vector_sha256": sha256_array(confidence),
        "arrays_file": arrays_path.name,
        "arrays_sha256": sha256_file(arrays_path),
        "runtime": runtime,
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
        "wrapper_sha256": sha256_file(Path(__file__).resolve()),
        "elapsed_seconds": time.time() - started,
        "information_boundary": {
            "labels_received": False,
            "non_safety_inputs_opened": False,
            "candidate_quality_computed": False,
            "alias_executed": False,
            "selector_executed": False,
            "primary_outcome_opened": False,
            "deployment_outcome_opened": False,
        },
        "decision": "PASS_PRAA_STAGE2_ENDPOINT",
    }
    receipt_path = output_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
