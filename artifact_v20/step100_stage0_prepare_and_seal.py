from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

import step100_common as common


ROOT = Path(__file__).resolve().parent
DEV = ROOT / f"STEP100_IMDB_DEVELOPMENT_ROWS_{common.DATE}.json"
INPUT_DIR = ROOT / "step100_test_inputs"
SEALED_DIR = ROOT / "external_data" / "step100_sealed"
MANIFEST = ROOT / f"STEP100_STAGE0_DATA_AND_SEAL_MANIFEST_{common.DATE}.json"


def uid(index: int, text: str) -> str:
    return hashlib.sha256(
        f"{common.DATASET_REVISION}\x1f{index}\x1f{text}".encode("utf-8")
    ).hexdigest()


def main() -> None:
    if any(path.exists() for path in (DEV, INPUT_DIR, SEALED_DIR, MANIFEST)):
        raise FileExistsError("Step 100 stage-0 output already exists")
    dataset = load_dataset(
        common.DATASET_ID, common.DATASET_CONFIG, split="test",
        revision=common.DATASET_REVISION,
    )
    by_label: dict[int, list[dict[str, object]]] = {0: [], 1: []}
    for index, raw in enumerate(dataset):
        label = int(raw["label"])
        text = str(raw["text"])
        if label not in by_label:
            raise AssertionError({"unexpected_label": label})
        by_label[label].append({
            "source_index": index,
            "text": text,
            "label": label,
            "order_sha256": common.row_digest(index, text, label),
            "uid_sha256": uid(index, text),
        })
    if {key: len(value) for key, value in by_label.items()} != {0: 12500, 1: 12500}:
        raise AssertionError("public class-count metadata drift")
    for rows in by_label.values():
        rows.sort(key=lambda row: (str(row["order_sha256"]), int(row["source_index"])))

    dev_rows: list[dict[str, object]] = []
    heldout_rows: list[dict[str, object]] = []
    for label in (0, 1):
        cursor = 0
        for partition, count in common.DEV_QUOTAS_PER_LABEL.items():
            for row in by_label[label][cursor : cursor + count]:
                dev_rows.append({**row, "partition": partition})
            cursor += count
        if cursor != 6000:
            raise AssertionError(cursor)
        heldout_rows.extend(by_label[label][cursor:])
    dev_rows.sort(key=lambda row: (str(row["partition"]), int(row["label"]), str(row["order_sha256"])))
    heldout_rows.sort(key=lambda row: int(row["source_index"]))
    partition_counts = {
        name: sum(row["partition"] == name for row in dev_rows)
        for name in common.DEV_QUOTAS_PER_LABEL
    }
    expected_counts = {name: 2 * count for name, count in common.DEV_QUOTAS_PER_LABEL.items()}
    if partition_counts != expected_counts or len(dev_rows) != 12000 or len(heldout_rows) != 13000:
        raise AssertionError({"partitions": partition_counts, "heldout": len(heldout_rows)})

    INPUT_DIR.mkdir(parents=True)
    SEALED_DIR.mkdir(parents=True)
    input_path = INPUT_DIR / "imdb_test_heldout_inputs.json"
    sealed_path = SEALED_DIR / "imdb_test_heldout_sealed_outcomes.npz"
    common.json_dump(DEV, {
        "dataset_id": common.DATASET_ID,
        "revision": common.DATASET_REVISION,
        "source_split": "test",
        "rows": dev_rows,
        "partition_counts": partition_counts,
    })
    common.json_dump(input_path, {
        "dataset_id": common.DATASET_ID,
        "revision": common.DATASET_REVISION,
        "source_split": "test",
        "labels_present": False,
        "rows": [
            {
                "source_index": int(row["source_index"]),
                "uid_sha256": str(row["uid_sha256"]),
                "text": str(row["text"]),
            }
            for row in heldout_rows
        ],
    })
    np.savez_compressed(
        sealed_path,
        source_indices=np.asarray([row["source_index"] for row in heldout_rows], dtype=np.int64),
        labels=np.asarray([row["label"] for row in heldout_rows], dtype=np.int16),
        uid_sha256=np.asarray([row["uid_sha256"] for row in heldout_rows], dtype="U64"),
    )
    manifest = {
        "manifest_id": "STEP100_STAGE0_DATA_AND_SEAL_V1",
        "date": common.DATE,
        "protocol_sha256": common.sha256_path(common.PROTOCOL),
        "dataset_id": common.DATASET_ID,
        "dataset_revision": common.DATASET_REVISION,
        "source_split_requested": "test",
        "development_rows": len(dev_rows),
        "development_file": DEV.name,
        "development_sha256": common.sha256_path(DEV),
        "partition_counts": partition_counts,
        "heldout_rows": len(heldout_rows),
        "input_file": str(input_path.relative_to(ROOT)).replace("\\", "/"),
        "input_sha256": common.sha256_path(input_path),
        "input_contains_label_field": False,
        "sealed_file": str(sealed_path.relative_to(ROOT)).replace("\\", "/"),
        "sealed_sha256": common.sha256_path(sealed_path),
        "sealed_outcome_opened_for_analysis": False,
        "console_exposed_labels_or_class_specific_outcomes": False,
        "code_sha256": {
            "step93_common.py": common.sha256_path(ROOT / "step93_common.py"),
            "step100_common.py": common.sha256_path(ROOT / "step100_common.py"),
            "step100_stage0_prepare_and_seal.py": common.sha256_path(Path(__file__)),
        },
    }
    common.json_dump(MANIFEST, manifest)
    print(json.dumps({
        "development_rows": len(dev_rows),
        "heldout_rows": len(heldout_rows),
        "development_sha256": manifest["development_sha256"],
        "input_sha256": manifest["input_sha256"],
        "sealed_sha256": manifest["sealed_sha256"],
        "sealed_outcome_opened_for_analysis": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
