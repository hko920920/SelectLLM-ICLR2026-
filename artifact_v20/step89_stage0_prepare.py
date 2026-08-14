from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset


ROOT = Path(__file__).resolve().parent
DATE = "2026-08-12"
DATASET_NAME = "nyu-mll/glue"
DATASET_CONFIG = "qnli"
DATASET_REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
MODEL_REVISION = "7dd04ee0a6040c06fb381ad7edcb8585f4d937fd"
SPLIT_SALT = "step89-qnli-v1"

OFFICIAL_DIR = ROOT / "external" / "model-selector" / "resources" / "datasets" / "glue" / "qnli"
CALIBRATION_PATH = ROOT / f"STEP89_QNLI_CALIBRATION_PACKAGE_{DATE}.json"
HOLDOUT_INPUT_PATH = ROOT / f"STEP89_QNLI_HOLDOUT_INPUTS_{DATE}.json"
SEALED_DIR = ROOT / "external_data" / "step89_sealed"
SEALED_OUTCOME_PATH = SEALED_DIR / f"STEP89_QNLI_SEALED_HOLDOUT_OUTCOMES_{DATE}.npz"
MANIFEST_PATH = ROOT / f"STEP89_QNLI_STAGE0_SPLIT_MANIFEST_{DATE}.json"
PROTOCOL_PATH = ROOT / f"STEP89_QNLI_EXECUTABLE_ADAPTER_STAGE0_PROTOCOL_{DATE}.md"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def partition(idx: int) -> str:
    digest = hashlib.sha256(f"{SPLIT_SALT}|{idx}".encode("utf-8")).digest()
    return "calibration" if int.from_bytes(digest[:8], "big") % 2 == 0 else "holdout"


def dump_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError(PROTOCOL_PATH)

    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        revision=DATASET_REVISION,
        split="validation",
    )
    official_predictions = np.load(OFFICIAL_DIR / "predictions.npy", allow_pickle=False)
    official_oracle = np.load(OFFICIAL_DIR / "oracle.npy", allow_pickle=False).astype(np.int64)

    if len(dataset) != 5463 or official_predictions.shape != (5463, 90):
        raise AssertionError((len(dataset), official_predictions.shape))
    hf_labels = np.asarray(dataset["label"], dtype=np.int64)
    if not np.array_equal(hf_labels, official_oracle):
        raise AssertionError("Hugging Face QNLI validation order does not match official MODEL SELECTOR oracle")

    calibration_rows: list[dict[str, object]] = []
    holdout_inputs: list[dict[str, object]] = []
    holdout_positions: list[int] = []
    holdout_labels: list[int] = []

    for position, item in enumerate(dataset):
        record = {
            "position": position,
            "idx": int(item["idx"]),
            "question": str(item["question"]),
            "sentence": str(item["sentence"]),
        }
        if partition(int(item["idx"])) == "calibration":
            record["label"] = int(item["label"])
            record["official_predictions"] = official_predictions[position].astype(np.int8).tolist()
            calibration_rows.append(record)
        else:
            holdout_inputs.append(record)
            holdout_positions.append(position)
            holdout_labels.append(int(item["label"]))

    if set(row["position"] for row in calibration_rows) & set(holdout_positions):
        raise AssertionError("split overlap")
    if len(calibration_rows) + len(holdout_inputs) != len(dataset):
        raise AssertionError("split does not cover validation set")

    dump_json(
        CALIBRATION_PATH,
        {
            "dataset": DATASET_NAME,
            "config": DATASET_CONFIG,
            "revision": DATASET_REVISION,
            "split_salt": SPLIT_SALT,
            "rows": calibration_rows,
        },
    )
    dump_json(
        HOLDOUT_INPUT_PATH,
        {
            "dataset": DATASET_NAME,
            "config": DATASET_CONFIG,
            "revision": DATASET_REVISION,
            "split_salt": SPLIT_SALT,
            "rows": holdout_inputs,
        },
    )
    SEALED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        SEALED_OUTCOME_PATH,
        positions=np.asarray(holdout_positions, dtype=np.int64),
        labels=np.asarray(holdout_labels, dtype=np.int8),
        official_predictions=official_predictions[np.asarray(holdout_positions, dtype=np.int64)].astype(np.int8),
    )

    inputs = {
        "protocol": sha256_path(PROTOCOL_PATH),
        "official_oracle": sha256_path(OFFICIAL_DIR / "oracle.npy"),
        "official_predictions": sha256_path(OFFICIAL_DIR / "predictions.npy"),
    }
    outputs = {
        str(CALIBRATION_PATH.relative_to(ROOT)): sha256_path(CALIBRATION_PATH),
        str(HOLDOUT_INPUT_PATH.relative_to(ROOT)): sha256_path(HOLDOUT_INPUT_PATH),
        str(SEALED_OUTCOME_PATH.relative_to(ROOT)): sha256_path(SEALED_OUTCOME_PATH),
    }
    manifest = {
        "manifest_id": "STEP89_QNLI_EXECUTABLE_ADAPTER_STAGE0_V1",
        "date": DATE,
        "dataset_revision": DATASET_REVISION,
        "model_revision": MODEL_REVISION,
        "split_salt": SPLIT_SALT,
        "counts": {
            "total": len(dataset),
            "calibration": len(calibration_rows),
            "holdout": len(holdout_inputs),
        },
        "inputs_sha256": inputs,
        "outputs_sha256": outputs,
        "pre_outcome_statement": (
            "No frozen-parent, adapter, active-path, selected-root, regret, or holdout-effect outcome was computed before this split lock."
        ),
    }
    dump_json(MANIFEST_PATH, manifest)
    print(json.dumps({
        "manifest": str(MANIFEST_PATH.name),
        "counts": manifest["counts"],
        "manifest_sha256": sha256_path(MANIFEST_PATH),
        "holdout_outcomes_sha256": outputs[str(SEALED_OUTCOME_PATH.relative_to(ROOT))],
    }, indent=2))


if __name__ == "__main__":
    main()
