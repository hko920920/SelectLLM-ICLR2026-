from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset


ROOT = Path(__file__).resolve().parent
DATE = "2026-08-12"
DATASET = "nyu-mll/glue"
REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
SALT = "step90-high-confidence-v1"
TASKS: dict[str, dict[str, Any]] = {
    "mnli": {
        "split": "validation_matched",
        "fields": ("premise", "hypothesis"),
        "examples": 9815,
        "candidates": 82,
        "model": "cross-encoder/nli-distilroberta-base",
        "model_revision": "b14d131f9d32668a5e6a982729b57ff6ed5dfcbd",
    },
    "qqp": {
        "split": "validation",
        "fields": ("question1", "question2"),
        "examples": 40430,
        "candidates": 101,
        "model": "cross-encoder/quora-distilroberta-base",
        "model_revision": "f62e7a4b20b97195c2868e53ec59126df5eac743",
    },
}
PROTOCOL = ROOT / f"STEP90_HIGH_CONFIDENCE_ADAPTER_STAGE0_PROTOCOL_{DATE}.md"
MANIFEST = ROOT / f"STEP90_HIGH_CONFIDENCE_ADAPTER_STAGE0_MANIFEST_{DATE}.json"
SEALED_DIR = ROOT / "external_data" / "step90_sealed"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def assigned(task: str, idx: int) -> str:
    digest = hashlib.sha256(f"{SALT}|{task}|{idx}".encode()).digest()
    return "calibration" if int.from_bytes(digest[:8], "big") % 2 == 0 else "holdout"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    if not PROTOCOL.exists():
        raise FileNotFoundError(PROTOCOL)
    SEALED_DIR.mkdir(parents=True, exist_ok=True)
    task_manifest: dict[str, Any] = {}
    for task, spec in TASKS.items():
        dataset = load_dataset(DATASET, task, revision=REVISION, split=spec["split"])
        official_dir = ROOT / "external" / "model-selector" / "resources" / "datasets" / "glue" / task
        oracle_path = official_dir / "oracle.npy"
        prediction_path = official_dir / "predictions.npy"
        oracle = np.load(oracle_path, allow_pickle=False).astype(np.int64)
        predictions = np.load(prediction_path, allow_pickle=False).astype(np.int8)
        labels = np.asarray(dataset["label"], dtype=np.int64)
        if len(dataset) != spec["examples"] or predictions.shape != (spec["examples"], spec["candidates"]):
            raise AssertionError((task, len(dataset), predictions.shape))
        if not np.array_equal(labels, oracle):
            raise AssertionError(f"{task}: Hugging Face and official MODEL SELECTOR order differ")

        calibration: list[dict[str, Any]] = []
        holdout_inputs: list[dict[str, Any]] = []
        holdout_positions: list[int] = []
        holdout_labels: list[int] = []
        fields = tuple(spec["fields"])
        for position, row in enumerate(dataset):
            item = {
                "position": position,
                "idx": int(row["idx"]),
                "text_a": str(row[fields[0]]),
                "text_b": str(row[fields[1]]),
            }
            if assigned(task, int(row["idx"])) == "calibration":
                item["label"] = int(row["label"])
                item["official_predictions"] = predictions[position].tolist()
                calibration.append(item)
            else:
                holdout_inputs.append(item)
                holdout_positions.append(position)
                holdout_labels.append(int(row["label"]))

        calibration_path = ROOT / f"STEP90_{task.upper()}_CALIBRATION_PACKAGE_{DATE}.json"
        input_path = ROOT / f"STEP90_{task.upper()}_HOLDOUT_INPUTS_{DATE}.json"
        sealed_path = SEALED_DIR / f"STEP90_{task.upper()}_SEALED_OUTCOMES_{DATE}.npz"
        write_json(calibration_path, {"task": task, "revision": REVISION, "rows": calibration})
        write_json(input_path, {"task": task, "revision": REVISION, "rows": holdout_inputs})
        positions_array = np.asarray(holdout_positions, dtype=np.int64)
        np.savez_compressed(
            sealed_path,
            positions=positions_array,
            labels=np.asarray(holdout_labels, dtype=np.int8),
            official_predictions=predictions[positions_array],
        )
        task_manifest[task] = {
            "counts": {
                "total": len(dataset),
                "calibration": len(calibration),
                "holdout": len(holdout_inputs),
            },
            "model": spec["model"],
            "model_revision": spec["model_revision"],
            "source_sha256": {
                "oracle": sha256_path(oracle_path),
                "predictions": sha256_path(prediction_path),
            },
            "output_sha256": {
                calibration_path.name: sha256_path(calibration_path),
                input_path.name: sha256_path(input_path),
                str(sealed_path.relative_to(ROOT)): sha256_path(sealed_path),
            },
        }

    manifest = {
        "manifest_id": "STEP90_HIGH_CONFIDENCE_ADAPTER_STAGE0_V1",
        "date": DATE,
        "dataset_revision": REVISION,
        "split_salt": SALT,
        "protocol_sha256": sha256_path(PROTOCOL),
        "tasks": task_manifest,
        "pre_outcome_statement": (
            "No MNLI/QQP frozen-parent, adapter, selector-path, root, regret, or attack-effect outcome was computed before this lock."
        ),
    }
    write_json(MANIFEST, manifest)
    print(json.dumps({
        "manifest": MANIFEST.name,
        "manifest_sha256": sha256_path(MANIFEST),
        "counts": {task: row["counts"] for task, row in task_manifest.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
