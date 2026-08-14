from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

import step105_common as common


ROOT = Path(__file__).resolve().parent
DEV = ROOT / f"STEP105_YELP_DEVELOPMENT_ROWS_{common.DATE}.json"
INPUT_DIR = ROOT / "step105_test_inputs"
SEALED_DIR = ROOT / "external_data" / "step105_sealed"
MANIFEST = ROOT / f"STEP105_STAGE0_DATA_AND_SEAL_MANIFEST_{common.DATE}.json"


def main() -> None:
    if not common.PREDATA_LOCK.is_file():
        raise FileNotFoundError("pre-data lock must exist before dataset loading")
    if any(path.exists() for path in (DEV, INPUT_DIR, SEALED_DIR, MANIFEST)):
        raise FileExistsError("Step 105 stage-0 output already exists")
    dataset = load_dataset(
        common.DATASET_ID, common.DATASET_CONFIG, revision=common.DATASET_REVISION,
    )
    train = dataset["train"]
    test = dataset["test"]
    if len(train) != 560_000 or len(test) != 38_000:
        raise AssertionError({"train": len(train), "test": len(test)})
    by_label: dict[int, list[dict[str, object]]] = {0: [], 1: []}
    for index, raw in enumerate(train):
        label = int(raw["label"])
        text = str(raw["text"])
        if label not in by_label:
            raise AssertionError({"unexpected_train_label": label})
        by_label[label].append({
            "source_index": index,
            "text": text,
            "label": label,
            "order_sha256": common.row_digest(index, text, label),
            "uid_sha256": common.uid(index, text),
        })
    if {key: len(value) for key, value in by_label.items()} != {0: 280_000, 1: 280_000}:
        raise AssertionError("public train class-count metadata drift")
    for rows in by_label.values():
        rows.sort(key=lambda row: (str(row["order_sha256"]), int(row["source_index"])))
    dev_rows: list[dict[str, object]] = []
    for label in (0, 1):
        cursor = 0
        for partition, count in common.DEV_QUOTAS_PER_LABEL.items():
            for row in by_label[label][cursor : cursor + count]:
                dev_rows.append({**row, "partition": partition})
            cursor += count
        if cursor != 4500:
            raise AssertionError(cursor)
    dev_rows.sort(key=lambda row: (str(row["partition"]), int(row["label"]), str(row["order_sha256"])))
    partition_counts = {
        name: sum(row["partition"] == name for row in dev_rows)
        for name in common.DEV_QUOTAS_PER_LABEL
    }
    expected_counts = {name: 2 * count for name, count in common.DEV_QUOTAS_PER_LABEL.items()}
    if partition_counts != expected_counts or len(dev_rows) != 9000:
        raise AssertionError(partition_counts)

    test_rows: list[dict[str, object]] = []
    test_counts = {0: 0, 1: 0}
    for index, raw in enumerate(test):
        label = int(raw["label"])
        text = str(raw["text"])
        if label not in test_counts:
            raise AssertionError({"unexpected_test_label": label})
        test_counts[label] += 1
        test_rows.append({
            "source_index": index,
            "text": text,
            "label": label,
            "uid_sha256": common.uid(index, text),
        })
    if test_counts != {0: 19_000, 1: 19_000}:
        raise AssertionError(test_counts)

    INPUT_DIR.mkdir(parents=True)
    SEALED_DIR.mkdir(parents=True)
    input_path = INPUT_DIR / "yelp_polarity_test_inputs.json"
    sealed_path = SEALED_DIR / "yelp_polarity_test_sealed_outcomes.npz"
    common.json_dump(DEV, {
        "dataset_id": common.DATASET_ID,
        "revision": common.DATASET_REVISION,
        "source_split": "train",
        "rows": dev_rows,
        "partition_counts": partition_counts,
        "selector_search_rows": 0,
        "selector_verify_rows": 0,
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
            for row in test_rows
        ],
    })
    np.savez_compressed(
        sealed_path,
        source_indices=np.asarray([row["source_index"] for row in test_rows], dtype=np.int64),
        labels=np.asarray([row["label"] for row in test_rows], dtype=np.int16),
        uid_sha256=np.asarray([row["uid_sha256"] for row in test_rows], dtype="U64"),
    )
    manifest = {
        "schema": "step105.stage0_data_and_seal.v1",
        "date": common.DATE,
        "predata_lock_sha256": common.sha256_path(common.PREDATA_LOCK),
        "protocol_sha256": common.sha256_path(common.PROTOCOL),
        "metadata_sha256": common.sha256_path(common.METADATA),
        "dataset_id": common.DATASET_ID,
        "dataset_revision": common.DATASET_REVISION,
        "development_rows": len(dev_rows),
        "development_file": DEV.name,
        "development_sha256": common.sha256_path(DEV),
        "partition_counts": partition_counts,
        "target_selector_search_rows": 0,
        "target_selector_verify_rows": 0,
        "test_rows": len(test_rows),
        "test_input_file": str(input_path.relative_to(ROOT)).replace("\\", "/"),
        "test_input_sha256": common.sha256_path(input_path),
        "test_input_contains_label_field": False,
        "sealed_file": str(sealed_path.relative_to(ROOT)).replace("\\", "/"),
        "sealed_sha256": common.sha256_path(sealed_path),
        "sealed_outcome_opened_for_analysis": False,
        "console_exposed_test_labels_or_class_specific_predictions": False,
        "code_sha256": {
            "step105_common.py": common.sha256_path(ROOT / "step105_common.py"),
            "step105_stage0_prepare_and_seal.py": common.sha256_path(Path(__file__)),
        },
    }
    common.json_dump(MANIFEST, manifest)
    print(json.dumps({
        "development_rows": len(dev_rows),
        "partition_counts": partition_counts,
        "test_rows": len(test_rows),
        "test_input_sha256": manifest["test_input_sha256"],
        "sealed_sha256": manifest["sealed_sha256"],
        "sealed_outcome_opened_for_analysis": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
