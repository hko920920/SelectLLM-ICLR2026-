#!/usr/bin/env python3
"""Independent reconstruction of the PRAA Stage-2 safety ledger.

This validator intentionally does not import the execution or parent-selection
modules. It reconstructs safety labels from the locked public datasets, reads
endpoint arrays, and recomputes every decision-changing Stage-2 quantity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent
LOCK_PATH = ROOT / "PRAA_STAGE1_ARTIFACT_AND_DATA_SEAL_LOCK_2026-08-14.json"
TIE_DOMAIN = "PRAA-parent-tie-v1"
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
LID_CODES = [
    "ar", "bg", "de", "el", "en", "es", "fr", "hi", "it", "ja",
    "nl", "pl", "pt", "ru", "sw", "th", "tr", "ur", "vi", "zh",
]


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(path)
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_lines(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def array_digest(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(b"|")
    digest.update(json.dumps(value.shape).encode("utf-8"))
    digest.update(b"|")
    digest.update(value.tobytes())
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def tie_key(root_id: str) -> str:
    return hashlib.sha256(f"{TIE_DOMAIN}|{root_id}".encode("utf-8")).hexdigest()


def find_dir(base: Path, name: str) -> Path:
    candidates = [path for path in base.rglob(name) if path.is_dir()]
    if len(candidates) != 1:
        raise AssertionError({"directory": name, "matches": [str(p) for p in candidates]})
    return candidates[0]


def read_inputs(public_root: Path, task: str, lock: dict[str, Any]) -> list[dict[str, Any]]:
    path = public_root / "inputs" / task / "safety.jsonl"
    expected = lock["tasks"][task]["partitions"]["safety"]
    assert sha256(path) == expected["input_sha256"]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == expected["rows"]
    assert digest_lines(str(row["uid"]) for row in rows) == expected["uid_sha256"]
    assert all(row["partition"] == "safety" and row["source_split"] == "train" for row in rows)
    return rows


def labels_for(task: str, rows: list[dict[str, Any]], lock: dict[str, Any]) -> np.ndarray:
    task_lock = lock["tasks"][task]
    with tempfile.TemporaryDirectory(prefix="praa-stage2-independent-") as cache:
        if task == "emotion":
            dataset = load_dataset(
                task_lock["dataset"],
                "split",
                revision=task_lock["revision"],
                cache_dir=cache,
            )["train"]
            values = [int(dataset[int(row["source_index"])]["label"]) for row in rows]
        else:
            dataset = load_dataset(
                task_lock["dataset"],
                revision=task_lock["revision"],
                cache_dir=cache,
            )["train"]
            values = [
                LID_CODES.index(str(dataset[int(row["source_index"])]["labels"]).lower())
                for row in rows
            ]
    serialized = "".join(
        canonical({"uid": str(row["uid"]), "label": int(label)}) + "\n"
        for row, label in zip(rows, values, strict=True)
    )
    expected = task_lock["partitions"]["safety"]["sealed_label_sha256"]
    assert hashlib.sha256(serialized.encode("utf-8")).hexdigest() == expected
    return np.asarray(values, dtype=np.int16)


def endpoint_arrays(base: Path, root_id: str, expected_uids: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    directory = find_dir(base, root_id)
    receipt = load_json(directory / "receipt.json")
    arrays_path = directory / "safety_predictions.npz"
    assert receipt["decision"] == "PASS_PRAA_STAGE2_ENDPOINT"
    assert receipt["root_id"] == root_id
    assert sha256(arrays_path) == receipt["arrays_sha256"]
    arrays = np.load(arrays_path, allow_pickle=False)
    uids = arrays["uids"].astype(str).tolist()
    predictions = arrays["predictions"].astype(np.int16)
    assert uids == expected_uids
    assert array_digest(predictions) == receipt["prediction_vector_sha256"]
    return predictions, receipt


def validate_task(
    task: str,
    public_root: Path,
    predictions_root: Path,
    ledger_task: dict[str, Any],
    lock: dict[str, Any],
) -> dict[str, Any]:
    rows = read_inputs(public_root, task, lock)
    uids = [str(row["uid"]) for row in rows]
    labels = labels_for(task, rows, lock)
    predictions: dict[str, np.ndarray] = {}
    receipts: dict[str, dict[str, Any]] = {}
    for root_id in ROOTS[task]:
        predictions[root_id], receipts[root_id] = endpoint_arrays(
            predictions_root, root_id, uids
        )

    digests = {root_id: array_digest(vector) for root_id, vector in predictions.items()}
    assert len(set(digests.values())) == len(digests)
    for left, right in combinations(ROOTS[task], 2):
        observed = int(np.sum(predictions[left] != predictions[right]))
        recorded = ledger_task["pairwise_disagreement"][f"{left}|{right}"]["disagreements"]
        assert observed == recorded

    counts = {
        root_id: int(np.sum(vector == labels))
        for root_id, vector in predictions.items()
    }
    best_count = max(counts.values())
    best_set = sorted(root for root, count in counts.items() if count == best_count)
    parent = min(best_set, key=tie_key)

    assert ledger_task["rows"] == len(labels)
    assert ledger_task["best_correct_count"] == best_count
    assert ledger_task["best_root_set"] == best_set
    assert ledger_task["selected_parent"] == parent
    assert ledger_task["selected_parent_tie_key"] == tie_key(parent)
    assert ledger_task["unique_best_required"] is False
    assert ledger_task["response_vectors_pairwise_distinct"] is True
    for root_id in ROOTS[task]:
        recorded = ledger_task["roots"][root_id]
        assert recorded["correct_count"] == counts[root_id]
        assert abs(float(recorded["accuracy"]) - counts[root_id] / len(labels)) <= 1e-15
        assert recorded["prediction_vector_sha256"] == digests[root_id]

    return {
        "task": task,
        "rows": len(labels),
        "correct_counts": counts,
        "best_root_set": best_set,
        "selected_parent": parent,
        "checks": {
            "input_and_uid_binding": True,
            "safety_label_seal_reconstructed": True,
            "endpoint_arrays_reconstructed": True,
            "pairwise_distinctness_reconstructed": True,
            "accuracy_reconstructed": True,
            "best_set_reconstructed": True,
            "parent_tie_break_reconstructed": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-public-root", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lock = load_json(LOCK_PATH)
    ledger = load_json(args.ledger)
    assert lock["decision"] == "PASS_PRAA_STAGE1_ARTIFACT_AND_DATA_SEAL"
    assert ledger["decision"] == "PASS_PRAA_STAGE2_CLEAN_SAFETY_AND_PARENT"
    assert ledger["stage1_lock_sha256"] == sha256(LOCK_PATH)

    reconstructed = {
        task: validate_task(
            task,
            args.stage1_public_root,
            args.predictions_root,
            ledger["tasks"][task],
            lock,
        )
        for task in ROOTS
    }
    result = {
        "schema": "praa.stage2.independent_validation.v1",
        "decision": "PASS_PRAA_STAGE2_INDEPENDENT_VALIDATION",
        "stage1_lock_sha256": sha256(LOCK_PATH),
        "stage2_ledger_sha256": sha256(args.ledger),
        "tasks": reconstructed,
        "information_boundary": {
            "safety_labels_opened": True,
            "non_safety_labels_opened": False,
            "aliases_executed": False,
            "selector_executed": False,
            "primary_outcome_opened": False,
            "deployment_outcome_opened": False,
        },
    }
    result["result_sha256"] = hashlib.sha256(canonical(result).encode("utf-8")).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
