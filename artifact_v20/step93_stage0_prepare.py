from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download

import step93_common as common


ROOT = Path(__file__).resolve().parent
DATE = common.DATE
PROTOCOL = ROOT / f"STEP93_SOURCE_FAITHFUL_LEARNED_ABSTENTION_PREREGISTRATION_{DATE}.md"
AMENDMENT = ROOT / f"STEP93_PREREGISTRATION_AMENDMENT_A_{DATE}.md"
AMENDMENT_B = ROOT / f"STEP93_PREREGISTRATION_AMENDMENT_B_{DATE}.md"
MANIFEST = ROOT / f"STEP93_STAGE0_MANIFEST_{DATE}.json"
CALIBRATION = ROOT / f"STEP93_BANKING77_CALIBRATION_PACKAGE_{DATE}.json"
HOLDOUT_INPUTS = ROOT / f"STEP93_BANKING77_HOLDOUT_INPUTS_{DATE}.json"
SEALED_DIR = ROOT / "external_data" / "step93_sealed"
SEALED_OUTCOMES = SEALED_DIR / f"STEP93_BANKING77_SEALED_OUTCOMES_{DATE}.npz"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    for authority in (PROTOCOL, AMENDMENT, AMENDMENT_B):
        if not authority.exists():
            raise FileNotFoundError(authority)
    for output in (MANIFEST, CALIBRATION, HOLDOUT_INPUTS, SEALED_OUTCOMES):
        if output.exists():
            raise RuntimeError(f"refusing to overwrite Step 93 Stage 0 output: {output}")

    raw_paths: dict[str, Path] = {}
    for split in ("train", "test"):
        raw_paths[split] = Path(
            hf_hub_download(
                repo_id=common.DATASET_ID,
                filename=f"data/{split}-00000-of-00001.parquet",
                repo_type="dataset",
                revision=common.DATASET_REVISION,
            )
        )

    payload = load_dataset(common.DATASET_ID, revision=common.DATASET_REVISION)
    train = payload["train"]
    test = payload["test"]
    if len(train) != 9_993 or len(test) != 3_076:
        raise AssertionError((len(train), len(test)))
    expected_columns = {"text", "label", "label_text"}
    if set(train.column_names) != expected_columns or set(test.column_names) != expected_columns:
        raise AssertionError((train.column_names, test.column_names))

    counts = {"root_train": 0, "adapter_train": 0, "stagea_calibration": 0}
    split_indices: dict[str, list[int]] = {key: [] for key in counts}
    calibration_rows: list[dict[str, Any]] = []
    for index, row in enumerate(train):
        partition = common.train_partition(index)
        counts[partition] += 1
        split_indices[partition].append(index)
        if partition == "stagea_calibration":
            calibration_rows.append(
                {"source_index": index, "text": str(row["text"]), "label": int(row["label"])}
            )

    if sum(counts.values()) != len(train) or min(counts.values()) <= 0:
        raise AssertionError(counts)
    train_labels = np.asarray(train["label"], dtype=np.int16)
    test_labels = np.asarray(test["label"], dtype=np.int16)
    expected_classes = set(range(common.N_CLASSES))
    if set(np.unique(train_labels).tolist()) != expected_classes:
        raise AssertionError("train class set mismatch")
    if set(np.unique(test_labels).tolist()) != expected_classes:
        raise AssertionError("test class set mismatch")

    holdout_rows = [
        {"source_index": index, "text": str(row["text"])}
        for index, row in enumerate(test)
    ]
    write_json(
        CALIBRATION,
        {
            "dataset": common.DATASET_ID,
            "revision": common.DATASET_REVISION,
            "partition": "stagea_calibration",
            "rows": calibration_rows,
        },
    )
    write_json(
        HOLDOUT_INPUTS,
        {
            "dataset": common.DATASET_ID,
            "revision": common.DATASET_REVISION,
            "partition": "sealed_test_inputs_only",
            "rows": holdout_rows,
        },
    )
    SEALED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        SEALED_OUTCOMES,
        source_indices=np.arange(len(test), dtype=np.int64),
        labels=test_labels,
    )

    manifest = {
        "manifest_id": "STEP93_BANKING77_STAGE0_V1",
        "date": DATE,
        "protocol_sha256": common.sha256_path(PROTOCOL),
        "amendment_a_sha256": common.sha256_path(AMENDMENT),
        "amendment_b_sha256": common.sha256_path(AMENDMENT_B),
        "dataset": {
            "id": common.DATASET_ID,
            "revision": common.DATASET_REVISION,
            "upstream_id": common.UPSTREAM_DATASET_ID,
            "upstream_revision": common.UPSTREAM_DATASET_REVISION,
            "raw_parquet_sha256": {
                split: common.sha256_path(path) for split, path in raw_paths.items()
            },
            "fingerprints": {"train": train._fingerprint, "test": test._fingerprint},
            "counts": {"train": len(train), "test": len(test), **counts},
            "split_index_sha256": {
                key: common.canonical_json_sha256(value) for key, value in split_indices.items()
            },
            "train_label_array_sha256": common.array_sha256(train_labels),
        },
        "base_model": {
            "id": common.BASE_MODEL,
            "revision": common.BASE_REVISION,
            "model_safetensors_sha256": common.BASE_MODEL_SHA256,
        },
        "single_similarity": {
            "definition": "s(a,b)=1[a=b]",
            "candidate_candidate": "step93_common.exact_match_similarity",
            "reference_candidate": "step93_common.exact_match_similarity",
        },
        "outputs": {
            CALIBRATION.name: common.sha256_path(CALIBRATION),
            HOLDOUT_INPUTS.name: common.sha256_path(HOLDOUT_INPUTS),
            str(SEALED_OUTCOMES.relative_to(ROOT)).replace("\\", "/"): common.sha256_path(
                SEALED_OUTCOMES
            ),
        },
        "pre_outcome_statement": (
            "Stage 0 emitted only counts and hashes; no Banking77 text, label, prediction, "
            "active path, selected root, effect, or regret value was printed."
        ),
    }
    write_json(MANIFEST, manifest)
    print(
        json.dumps(
            {
                "manifest": MANIFEST.name,
                "manifest_sha256": common.sha256_path(MANIFEST),
                "counts": manifest["dataset"]["counts"],
                "output_sha256": manifest["outputs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
