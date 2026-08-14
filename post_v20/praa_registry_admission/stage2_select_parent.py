#!/usr/bin/env python3
"""Certify PRAA clean safety vectors and select one deterministic parent.

This program reconstructs safety labels from the exact public dataset revision.
It never downloads the Stage-1 sealed-label artifact and never reads any
non-safety label or selector outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from datasets import DatasetDict, load_dataset

SCRIPT_DIR = Path(__file__).resolve().parent
LOCK_PATH = SCRIPT_DIR / "PRAA_STAGE1_ARTIFACT_AND_DATA_SEAL_LOCK_2026-08-14.json"
ENDPOINT_SCHEMA = "praa.stage2.clean_safety_endpoint.v1"
LEDGER_SCHEMA = "praa.stage2.clean_safety_and_parent.v1"
PARENT_TIE_DOMAIN = "PRAA-parent-tie-v1"

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
LID_LABELS = [
    "ar", "bg", "de", "el", "en", "es", "fr", "hi", "it", "ja",
    "nl", "pl", "pt", "ru", "sw", "th", "tr", "ur", "vi", "zh",
]


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


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest_lines(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def parent_tie_key(root_id: str) -> str:
    return hashlib.sha256(f"{PARENT_TIE_DOMAIN}|{root_id}".encode("utf-8")).hexdigest()


def load_exact_dataset(task: str, repo_id: str, revision: str) -> DatasetDict:
    with tempfile.TemporaryDirectory(prefix="praa-stage2-label-cache-") as cache:
        if task == "emotion":
            dataset = load_dataset(
                repo_id,
                "split",
                revision=revision,
                cache_dir=cache,
            )
        else:
            dataset = load_dataset(repo_id, revision=revision, cache_dir=cache)
        if not isinstance(dataset, DatasetDict):
            raise Stage2Error(f"{repo_id} did not return DatasetDict")
        # Dataset objects reference cache files, so materialize only the safety
        # labels needed before the temporary cache is removed.
        labels = DatasetDict()
        labels["safety_source"] = dataset["train"]
        return labels


def read_safety_input(
    public_root: Path, task: str, lock: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], Path]:
    path = public_root / "inputs" / task / "safety.jsonl"
    expected = lock["tasks"][task]["partitions"]["safety"]
    if not path.is_file() or sha256_file(path) != expected["input_sha256"]:
        raise Stage2Error(f"Stage-1 safety input mismatch for {task}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("partition") != "safety" or row.get("source_split") != "train":
                raise Stage2Error(f"unexpected safety input row for {task}")
            rows.append(row)
    if len(rows) != int(expected["rows"]):
        raise Stage2Error(f"safety cardinality drift for {task}")
    uids = [str(row["uid"]) for row in rows]
    if digest_lines(uids) != expected["uid_sha256"]:
        raise Stage2Error(f"safety UID digest drift for {task}")
    return rows, path


def reconstruct_safety_labels(
    task: str,
    rows: list[Mapping[str, Any]],
    lock: Mapping[str, Any],
) -> tuple[np.ndarray, str]:
    task_lock = lock["tasks"][task]
    repo_id = str(task_lock["dataset"])
    revision = str(task_lock["revision"])

    with tempfile.TemporaryDirectory(prefix="praa-stage2-dataset-") as cache:
        if task == "emotion":
            dataset = load_dataset(repo_id, "split", revision=revision, cache_dir=cache)
        else:
            dataset = load_dataset(repo_id, revision=revision, cache_dir=cache)
        train = dataset["train"]
        labels: list[int] = []
        serialized: list[str] = []
        for row in rows:
            index = int(row["source_index"])
            source = train[index]
            if task == "emotion":
                label = int(source["label"])
            else:
                code = str(source["labels"]).strip().lower()
                try:
                    label = LID_LABELS.index(code)
                except ValueError as exc:
                    raise Stage2Error(f"unexpected LID safety label {code!r}") from exc
            uid = str(row["uid"])
            labels.append(label)
            serialized.append(canonical_json({"uid": uid, "label": label}))

    # Stage 1 hashed the exact canonical JSONL bytes with one newline per row.
    payload = "\n".join(serialized) + "\n"
    observed_file_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    expected_file_hash = task_lock["partitions"]["safety"]["sealed_label_sha256"]
    if observed_file_hash != expected_file_hash:
        raise Stage2Error(
            f"reconstructed safety label seal mismatch for {task}: "
            f"{observed_file_hash} != {expected_file_hash}"
        )
    return np.asarray(labels, dtype=np.int16), observed_file_hash


def find_root_dir(predictions_root: Path, root_id: str) -> Path:
    direct = predictions_root / root_id
    if direct.is_dir():
        return direct
    matches = [path for path in predictions_root.rglob(root_id) if path.is_dir()]
    if len(matches) != 1:
        raise Stage2Error(f"cannot resolve unique artifact directory for {root_id}: {matches}")
    return matches[0]


def load_endpoint(
    predictions_root: Path,
    task: str,
    root_id: str,
    expected_uids: list[str],
    stage1_lock_sha256: str,
) -> dict[str, Any]:
    root_dir = find_root_dir(predictions_root, root_id)
    receipt_path = root_dir / "receipt.json"
    arrays_path = root_dir / "safety_predictions.npz"
    receipt = load_json(receipt_path)
    if receipt.get("schema") != ENDPOINT_SCHEMA or receipt.get("decision") != "PASS_PRAA_STAGE2_ENDPOINT":
        raise Stage2Error(f"endpoint receipt not PASS for {root_id}")
    if receipt.get("task") != task or receipt.get("root_id") != root_id:
        raise Stage2Error(f"endpoint identity mismatch for {root_id}")
    if receipt.get("stage1_lock_sha256") != stage1_lock_sha256:
        raise Stage2Error(f"Stage-1 lock binding mismatch for {root_id}")
    if sha256_file(arrays_path) != receipt.get("arrays_sha256"):
        raise Stage2Error(f"arrays file hash mismatch for {root_id}")
    boundary = receipt.get("information_boundary") or {}
    forbidden_true = [
        key
        for key in [
            "labels_received",
            "non_safety_inputs_opened",
            "candidate_quality_computed",
            "alias_executed",
            "selector_executed",
            "primary_outcome_opened",
            "deployment_outcome_opened",
        ]
        if bool(boundary.get(key))
    ]
    if forbidden_true:
        raise Stage2Error(f"endpoint boundary violation for {root_id}: {forbidden_true}")

    arrays = np.load(arrays_path, allow_pickle=False)
    uids = arrays["uids"].astype(str).tolist()
    predictions = arrays["predictions"].astype(np.int16)
    confidence = arrays["confidence"].astype(np.float32)
    if uids != expected_uids:
        raise Stage2Error(f"endpoint UID order mismatch for {root_id}")
    if predictions.shape != (len(expected_uids),):
        raise Stage2Error(f"endpoint prediction shape mismatch for {root_id}")
    if confidence.shape != predictions.shape or not np.all(np.isfinite(confidence)):
        raise Stage2Error(f"endpoint confidence shape/value failure for {root_id}")
    if sha256_array(predictions) != receipt.get("prediction_vector_sha256"):
        raise Stage2Error(f"prediction digest mismatch for {root_id}")
    if sha256_array(confidence) != receipt.get("confidence_vector_sha256"):
        raise Stage2Error(f"confidence digest mismatch for {root_id}")
    return {
        "root_id": root_id,
        "receipt": receipt,
        "receipt_sha256": sha256_file(receipt_path),
        "arrays_sha256": sha256_file(arrays_path),
        "predictions": predictions,
        "confidence": confidence,
    }


def certify_task(
    task: str,
    public_root: Path,
    predictions_root: Path,
    lock: Mapping[str, Any],
    stage1_lock_sha256: str,
) -> dict[str, Any]:
    rows, input_path = read_safety_input(public_root, task, lock)
    uids = [str(row["uid"]) for row in rows]
    labels, label_seal = reconstruct_safety_labels(task, rows, lock)
    endpoints = [
        load_endpoint(predictions_root, task, root_id, uids, stage1_lock_sha256)
        for root_id in ROOTS[task]
    ]

    prediction_hashes = {
        endpoint["root_id"]: sha256_array(endpoint["predictions"])
        for endpoint in endpoints
    }
    if len(set(prediction_hashes.values())) != len(prediction_hashes):
        duplicates: list[list[str]] = []
        for left, right in combinations(ROOTS[task], 2):
            if prediction_hashes[left] == prediction_hashes[right]:
                duplicates.append([left, right])
        raise Stage2Error(f"clean safety response vectors are not distinct for {task}: {duplicates}")

    correct_counts = {
        endpoint["root_id"]: int(np.sum(endpoint["predictions"] == labels))
        for endpoint in endpoints
    }
    rows_count = len(labels)
    accuracies = {
        root_id: count / rows_count for root_id, count in correct_counts.items()
    }
    maximum = max(correct_counts.values())
    best_root_set = sorted(
        root_id for root_id, count in correct_counts.items() if count == maximum
    )
    parent = min(best_root_set, key=parent_tie_key)

    pairwise: dict[str, dict[str, Any]] = {}
    by_root = {endpoint["root_id"]: endpoint for endpoint in endpoints}
    for left, right in combinations(ROOTS[task], 2):
        disagreements = int(
            np.sum(by_root[left]["predictions"] != by_root[right]["predictions"])
        )
        pairwise[f"{left}|{right}"] = {
            "disagreements": disagreements,
            "rate": disagreements / rows_count,
        }

    max_code = 5 if task == "emotion" else 20
    mapped_range_pass = all(
        bool(np.all(endpoint["predictions"] >= 0))
        and bool(np.all(endpoint["predictions"] <= max_code))
        for endpoint in endpoints
    )
    if not mapped_range_pass:
        raise Stage2Error(f"mapped prediction range failure for {task}")

    return {
        "task": task,
        "safety_input_sha256": sha256_file(input_path),
        "safety_uid_sha256": digest_lines(uids),
        "reconstructed_safety_label_sha256": label_seal,
        "rows": rows_count,
        "roots": {
            endpoint["root_id"]: {
                "correct_count": correct_counts[endpoint["root_id"]],
                "accuracy": accuracies[endpoint["root_id"]],
                "prediction_vector_sha256": prediction_hashes[endpoint["root_id"]],
                "endpoint_receipt_sha256": endpoint["receipt_sha256"],
                "endpoint_arrays_sha256": endpoint["arrays_sha256"],
                "runtime": endpoint["receipt"]["runtime"],
            }
            for endpoint in endpoints
        },
        "pairwise_disagreement": pairwise,
        "best_correct_count": maximum,
        "best_root_set": best_root_set,
        "selected_parent": parent,
        "selected_parent_tie_key": parent_tie_key(parent),
        "unique_best_required": False,
        "response_vectors_pairwise_distinct": True,
        "mapped_prediction_range_pass": mapped_range_pass,
        "selector_executed": False,
        "non_safety_input_opened": False,
        "primary_outcome_opened": False,
        "deployment_outcome_opened": False,
        "decision": "PASS",
    }


def markdown_ledger(ledger: Mapping[str, Any]) -> str:
    lines = [
        "# PRAA Stage-2 clean safety and parent-selection ledger",
        "",
        f"Decision: `{ledger['decision']}`",
        "",
        f"Ledger SHA-256: `{ledger['ledger_sha256']}`",
        "",
        "Only frozen safety inputs and reconstructed safety labels were used. No alias, non-safety input, selector, primary outcome, or deployment outcome was opened.",
        "",
    ]
    for task, record in ledger["tasks"].items():
        lines.extend(
            [
                f"## {task}",
                "",
                f"- selected parent: `{record['selected_parent']}`",
                f"- best-root set: {', '.join(f'`{root}`' for root in record['best_root_set'])}",
                f"- unique-best required: `{record['unique_best_required']}`",
                "",
                "| Root | Correct | Accuracy | Prediction digest |",
                "|---|---:|---:|---|",
            ]
        )
        for root, root_record in record["roots"].items():
            lines.append(
                f"| `{root}` | {root_record['correct_count']} / {record['rows']} | "
                f"{100.0 * root_record['accuracy']:.4f}% | `{root_record['prediction_vector_sha256']}` |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-public-root", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lock = load_json(LOCK_PATH)
    if lock.get("decision") != "PASS_PRAA_STAGE1_ARTIFACT_AND_DATA_SEAL":
        raise Stage2Error("Stage-1 lock is not PASS")
    stage1_lock_sha256 = sha256_file(LOCK_PATH)

    tasks = {
        task: certify_task(
            task,
            args.stage1_public_root,
            args.predictions_root,
            lock,
            stage1_lock_sha256,
        )
        for task in ROOTS
    }
    ledger: dict[str, Any] = {
        "schema": LEDGER_SCHEMA,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage1_lock_sha256": stage1_lock_sha256,
        "tasks": tasks,
        "information_boundary": {
            "safety_labels_opened": True,
            "non_safety_labels_opened": False,
            "aliases_executed": False,
            "error_heads_trained": False,
            "selector_executed": False,
            "primary_outcome_opened": False,
            "deployment_outcome_opened": False,
        },
        "decision": "PASS_PRAA_STAGE2_CLEAN_SAFETY_AND_PARENT",
    }
    ledger["ledger_sha256"] = hashlib.sha256(
        canonical_json(ledger).encode("utf-8")
    ).hexdigest()
    json_path = args.output_dir / "PRAA_STAGE2_CLEAN_SAFETY_AND_PARENT_LEDGER.json"
    md_path = args.output_dir / "PRAA_STAGE2_CLEAN_SAFETY_AND_PARENT_LEDGER.md"
    json_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown_ledger(ledger), encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
