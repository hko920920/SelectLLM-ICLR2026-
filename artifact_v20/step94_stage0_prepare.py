from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download

import step94_common as common


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / f"STEP94_STAGE0_MANIFEST_{common.DATE}.json"
INPUT_DIR = ROOT / "step94_test_inputs"
SEALED_DIR = ROOT / "external_data" / "step94_sealed"


def load_pinned_task(task: common.TaskSpec) -> tuple[dict[str, Any], dict[str, Path]]:
    raw_paths: dict[str, Path] = {}
    for split, filename in task.filenames.items():
        raw_paths[split] = Path(
            hf_hub_download(
                repo_id=task.repo_id,
                filename=filename,
                repo_type="dataset",
                revision=task.revision,
            )
        )
    payload = load_dataset(
        "json",
        data_files={split: str(path) for split, path in raw_paths.items()},
    )
    return payload, raw_paths


def validate_split(task: common.TaskSpec, split: Any) -> None:
    required = {task.text_column, task.label_column}
    if not required.issubset(set(split.column_names)):
        raise AssertionError(
            {"task": task.key, "required": sorted(required), "columns": split.column_names}
        )
    labels = np.asarray(split[task.label_column], dtype=np.int64)
    if len(labels) == 0 or int(np.min(labels)) < 0 or int(np.max(labels)) >= task.n_classes:
        raise AssertionError({"task": task.key, "bad_label_range": True})


def main() -> None:
    for authority in (common.PROTOCOL, common.AMENDMENT_A):
        if not authority.exists():
            raise FileNotFoundError(authority)
    for authority in (
        ROOT / f"STEP94_PREREGISTRATION_AMENDMENT_B_{common.DATE}.md",
        ROOT / f"STEP94_PREREGISTRATION_AMENDMENT_C_{common.DATE}.md",
    ):
        if not authority.exists():
            raise FileNotFoundError(authority)
    if MANIFEST_PATH.exists() or INPUT_DIR.exists() or SEALED_DIR.exists():
        raise RuntimeError("refusing to overwrite Step 94 Stage-0 material")

    INPUT_DIR.mkdir(parents=True, exist_ok=False)
    SEALED_DIR.mkdir(parents=True, exist_ok=False)
    task_rows: list[dict[str, Any]] = []
    for task in common.TASKS:
        payload, raw_paths = load_pinned_task(task)
        for split_name in task.filenames:
            validate_split(task, payload[split_name])

        development_texts: list[str] = []
        development_labels: list[int] = []
        development_origins: list[dict[str, Any]] = []
        for split_name in task.development_splits:
            split = payload[split_name]
            for source_index, (text, label) in enumerate(
                zip(split[task.text_column], split[task.label_column])
            ):
                development_texts.append(str(text))
                development_labels.append(int(label))
                development_origins.append(
                    {"split": split_name, "source_index": int(source_index)}
                )
        labels = np.asarray(development_labels, dtype=np.int16)
        if set(np.unique(labels).tolist()) != set(range(task.n_classes)):
            raise AssertionError({"task": task.key, "development_class_coverage": False})

        partitions = [
            common.development_partition(task.key, index)
            for index in range(len(development_texts))
        ]
        partition_indices = {
            name: [index for index, value in enumerate(partitions) if value == name]
            for name in (
                "root_train",
                "gate_train",
                "stagea_search",
                "stagea_verification",
            )
        }
        if min(len(value) for value in partition_indices.values()) <= 0:
            raise AssertionError({"task": task.key, "empty_partition": True})

        test = payload[task.test_split]
        test_texts = [str(value) for value in test[task.text_column]]
        test_labels = np.asarray(test[task.label_column], dtype=np.int16)
        if (
            len(test_labels) == 0
            or int(np.min(test_labels)) < 0
            or int(np.max(test_labels)) >= task.n_classes
        ):
            raise AssertionError({"task": task.key, "test_label_range": False})

        input_path = INPUT_DIR / f"{task.key}_test_inputs.json"
        common.json_dump(
            input_path,
            {
                "task_key": task.key,
                "dataset_id": task.repo_id,
                "revision": task.revision,
                "split": task.test_split,
                "rows": [
                    {"source_index": index, "text": text}
                    for index, text in enumerate(test_texts)
                ],
            },
        )
        sealed_path = SEALED_DIR / f"{task.key}_sealed_outcomes.npz"
        np.savez_compressed(
            sealed_path,
            source_indices=np.arange(len(test_labels), dtype=np.int64),
            labels=test_labels,
        )

        task_rows.append(
            {
                "task_key": task.key,
                "order": task.order,
                "dataset_id": task.repo_id,
                "revision": task.revision,
                "upstream_id": task.upstream_id,
                "upstream_revision": task.upstream_revision,
                "n_classes": task.n_classes,
                "development_splits": list(task.development_splits),
                "test_split": task.test_split,
                "raw_file_sha256": {
                    split: common.sha256_path(path) for split, path in raw_paths.items()
                },
                "split_counts": {
                    split: len(payload[split]) for split in task.filenames
                },
                "development_count": len(development_texts),
                "development_text_sha256": common.canonical_json_sha256(development_texts),
                "development_label_sha256": common.array_sha256(labels),
                "development_origin_sha256": common.canonical_json_sha256(
                    development_origins
                ),
                "partition_counts": {
                    key: len(value) for key, value in partition_indices.items()
                },
                "partition_index_sha256": {
                    key: common.canonical_json_sha256(value)
                    for key, value in partition_indices.items()
                },
                "test_count": len(test_texts),
                "test_input_file": str(input_path.relative_to(ROOT)).replace("\\", "/"),
                "test_input_sha256": common.sha256_path(input_path),
                "sealed_outcome_file": str(sealed_path.relative_to(ROOT)).replace("\\", "/"),
                "sealed_outcome_sha256": common.sha256_path(sealed_path),
            }
        )

    manifest = {
        "manifest_id": "STEP94_FRESH_TASK_STAGE0_V1",
        "date": common.DATE,
        "authority_sha256": {
            common.PROTOCOL.name: common.sha256_path(common.PROTOCOL),
            common.AMENDMENT_A.name: common.sha256_path(common.AMENDMENT_A),
            f"STEP94_PREREGISTRATION_AMENDMENT_B_{common.DATE}.md": common.sha256_path(
                ROOT / f"STEP94_PREREGISTRATION_AMENDMENT_B_{common.DATE}.md"
            ),
            f"STEP94_PREREGISTRATION_AMENDMENT_C_{common.DATE}.md": common.sha256_path(
                ROOT / f"STEP94_PREREGISTRATION_AMENDMENT_C_{common.DATE}.md"
            ),
        },
        "base_model": {
            "id": common.BASE_MODEL,
            "revision": common.BASE_REVISION,
            "model_safetensors_sha256": common.BASE_MODEL_SHA256,
        },
        "split_salt": common.SPLIT_SALT,
        "tasks": task_rows,
        "pre_outcome_statement": (
            "Stage 0 printed no task row, class frequency, label, prediction, accuracy, "
            "path, root, effect, or regret. Test labels are stored only under the sealed directory."
        ),
    }
    common.json_dump(MANIFEST_PATH, manifest)
    print(
        json.dumps(
            {
                "manifest": MANIFEST_PATH.name,
                "manifest_sha256": common.sha256_path(MANIFEST_PATH),
                "tasks": [
                    {
                        "task_key": row["task_key"],
                        "split_counts": row["split_counts"],
                        "partition_counts": row["partition_counts"],
                        "test_count": row["test_count"],
                        "test_input_sha256": row["test_input_sha256"],
                        "sealed_outcome_sha256": row["sealed_outcome_sha256"],
                    }
                    for row in task_rows
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
