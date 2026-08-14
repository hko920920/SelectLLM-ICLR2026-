from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

import step97_common as common


ROOT = Path(__file__).resolve().parent
DEVELOPMENT = ROOT / f"STEP97_SCITAIL_DEVELOPMENT_ROWS_{common.DATE}.json"
TEST_INPUT = ROOT / "step97_test_inputs" / "scitail_test_inputs.json"
SEALED = ROOT / "external_data" / "step97_sealed" / "scitail_test_sealed_outcomes.npz"
MANIFEST = ROOT / f"STEP97_STAGE0_MANIFEST_{common.DATE}.json"
LABEL_MAP = {"entails": 0, "neutral": 1}


def table_rows(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path, columns=["premise", "hypothesis", "label"])
    values = table.to_pydict()
    rows: list[dict[str, Any]] = []
    for index in range(table.num_rows):
        raw_label = str(values["label"][index]).strip().lower()
        premise = str(values["premise"][index])
        hypothesis = str(values["hypothesis"][index])
        if raw_label not in LABEL_MAP or not premise.strip() or not hypothesis.strip():
            continue
        rows.append(
            {
                "source_index": index,
                "premise": premise,
                "hypothesis": hypothesis,
                "label": LABEL_MAP[raw_label],
            }
        )
    return rows


def build_train(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    strata: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[int(row["label"])].append(row)
    required = sum(common.PARTITION_QUOTAS.values())
    output: list[dict[str, Any]] = []
    audit = {}
    for label in (0, 1):
        ordered = sorted(strata[label], key=common.row_order_key)
        if len(ordered) < required:
            raise AssertionError({"label": label, "available": len(ordered), "required": required})
        cursor = 0
        counts = {}
        for partition, quota in common.PARTITION_QUOTAS.items():
            selected = ordered[cursor : cursor + quota]
            cursor += quota
            counts[partition] = len(selected)
            output.extend({**row, "partition": partition} for row in selected)
        audit[str(label)] = {"available": len(ordered), "selected": required, "partition_counts": counts}
    output.sort(key=lambda row: (row["partition"], row["label"], common.row_order_key(row)))
    return output, audit


def main() -> None:
    for output in (DEVELOPMENT, TEST_INPUT, SEALED, MANIFEST):
        if output.exists():
            raise RuntimeError(f"refusing to overwrite {output}")
    if not common.PROTOCOL.is_file():
        raise FileNotFoundError(common.PROTOCOL)
    paths = {
        split: Path(
            hf_hub_download(
                common.DATASET_ID,
                f"tsv_format/{split}-00000-of-00001.parquet",
                repo_type="dataset",
                revision=common.DATASET_REVISION,
            )
        )
        for split in ("train", "validation", "test")
    }
    train = table_rows(paths["train"])
    validation = table_rows(paths["validation"])
    test = table_rows(paths["test"])
    selected_train, stratum_audit = build_train(train)
    common.json_dump(
        DEVELOPMENT,
        {
            "dataset_id": common.DATASET_ID,
            "config": common.DATASET_CONFIG,
            "revision": common.DATASET_REVISION,
            "label_map": LABEL_MAP,
            "train_partitions": common.PARTITION_QUOTAS,
            "train_rows": selected_train,
            "validation_rows": [{**row, "partition": "selector_verify"} for row in validation],
        },
    )
    common.json_dump(
        TEST_INPUT,
        {
            "dataset_id": common.DATASET_ID,
            "config": common.DATASET_CONFIG,
            "revision": common.DATASET_REVISION,
            "source_split": "test",
            "runtime_fields": ["premise", "hypothesis"],
            "rows": [
                {
                    "source_index": row["source_index"],
                    "premise": row["premise"],
                    "hypothesis": row["hypothesis"],
                }
                for row in test
            ],
        },
    )
    SEALED.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        SEALED,
        source_indices=np.asarray([row["source_index"] for row in test], dtype=np.int64),
        labels=np.asarray([row["label"] for row in test], dtype=np.int16),
    )
    manifest = {
        "manifest_id": "STEP97_SCITAIL_STAGE0_V1",
        "date": common.DATE,
        "authority_sha256": {common.PROTOCOL.name: common.sha256_path(common.PROTOCOL)},
        "pre_outcome_statement": {
            "protocol_existed_before_scitail_shard_download": True,
            "dataset_card_schema_split_sizes_and_task_description_seen": True,
            "no_scitail_row_label_prediction_or_effect_seen_before_lock": True,
        },
        "dataset": {
            "id": common.DATASET_ID,
            "config": common.DATASET_CONFIG,
            "revision": common.DATASET_REVISION,
            "raw_sha256": {split: common.sha256_path(path) for split, path in paths.items()},
            "valid_rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        },
        "development": {
            "file": DEVELOPMENT.name,
            "sha256": common.sha256_path(DEVELOPMENT),
            "selected_train_rows": len(selected_train),
            "validation_rows": len(validation),
            "stratum_audit": stratum_audit,
        },
        "test_input": {
            "file": str(TEST_INPUT.relative_to(ROOT)).replace("\\", "/"),
            "sha256": common.sha256_path(TEST_INPUT),
            "rows": len(test),
            "contains_labels": False,
        },
        "sealed_outcome": {
            "file": str(SEALED.relative_to(ROOT)).replace("\\", "/"),
            "sha256": common.sha256_path(SEALED),
            "rows": len(test),
        },
        "root_bank": [
            {
                "key": spec.key,
                "repo_id": spec.repo_id,
                "revision": spec.revision,
                "raw_to_dataset": list(spec.raw_to_dataset),
                "parent": spec.parent,
            }
            for spec in common.ROOT_SPECS
        ],
        "source_code_sha256": {
            "step97_common.py": common.sha256_path(ROOT / "step97_common.py"),
            "step97_stage0_prepare.py": common.sha256_path(ROOT / "step97_stage0_prepare.py"),
        },
    }
    common.json_dump(MANIFEST, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
