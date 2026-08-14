from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

import step96_common as common


ROOT = Path(__file__).resolve().parent
DEVELOPMENT_PATH = ROOT / f"STEP96_SNLI_DEVELOPMENT_ROWS_{common.DATE}.json"
TEST_INPUT_PATH = ROOT / "step96_test_inputs" / "snli_test_inputs.json"
SEALED_PATH = ROOT / "external_data" / "step96_sealed" / "snli_test_sealed_outcomes.npz"
MANIFEST_PATH = ROOT / f"STEP96_STAGE0_MANIFEST_{common.DATE}.json"


def table_rows(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path, columns=["premise", "hypothesis", "label"])
    values = table.to_pydict()
    rows: list[dict[str, Any]] = []
    for index in range(table.num_rows):
        label = int(values["label"][index])
        premise = str(values["premise"][index])
        hypothesis = str(values["hypothesis"][index])
        if label not in (0, 1, 2) or not premise.strip() or not hypothesis.strip():
            continue
        rows.append(
            {
                "source_index": index,
                "premise": premise,
                "hypothesis": hypothesis,
                "label": label,
            }
        )
    return rows


def build_train_development(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    strata: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[int(row["label"])].append(row)
    required = sum(common.PARTITION_QUOTAS.values())
    if sorted(strata) != [0, 1, 2]:
        raise AssertionError(sorted(strata))
    output: list[dict[str, Any]] = []
    audit: dict[str, Any] = {}
    for label in (0, 1, 2):
        ordered = sorted(strata[label], key=common.row_order_key)
        if len(ordered) < required:
            raise AssertionError({"label": label, "available": len(ordered), "required": required})
        cursor = 0
        counts: dict[str, int] = {}
        for partition, quota in common.PARTITION_QUOTAS.items():
            selected = ordered[cursor : cursor + quota]
            cursor += quota
            counts[partition] = len(selected)
            output.extend({**row, "partition": partition} for row in selected)
        audit[str(label)] = {"available": len(ordered), "selected": required, "partition_counts": counts}
    output.sort(key=lambda row: (row["partition"], row["label"], common.row_order_key(row)))
    return output, audit


def main() -> None:
    outputs = (DEVELOPMENT_PATH, TEST_INPUT_PATH, SEALED_PATH, MANIFEST_PATH)
    for output in outputs:
        if output.exists():
            raise RuntimeError(f"refusing to overwrite Step 96 Stage-0 output: {output}")
    if not common.PROTOCOL.is_file():
        raise FileNotFoundError(common.PROTOCOL)
    protocol_hash = common.sha256_path(common.PROTOCOL)
    paths = {
        split: Path(
            hf_hub_download(
                common.DATASET_ID,
                f"plain_text/{split}-00000-of-00001.parquet",
                repo_type="dataset",
                revision=common.DATASET_REVISION,
            )
        )
        for split in ("train", "validation", "test")
    }
    train_rows = table_rows(paths["train"])
    validation_rows = table_rows(paths["validation"])
    test_rows = table_rows(paths["test"])
    selected_train, stratum_audit = build_train_development(train_rows)
    development = {
        "dataset_id": common.DATASET_ID,
        "config": common.DATASET_CONFIG,
        "revision": common.DATASET_REVISION,
        "train_partitions": common.PARTITION_QUOTAS,
        "train_rows": selected_train,
        "validation_partition": "selector_verify",
        "validation_rows": [
            {**row, "partition": "selector_verify"} for row in validation_rows
        ],
    }
    common.json_dump(DEVELOPMENT_PATH, development)
    test_input = {
        "dataset_id": common.DATASET_ID,
        "config": common.DATASET_CONFIG,
        "revision": common.DATASET_REVISION,
        "source_split": "test",
        "runtime_fields": ["premise", "hypothesis"],
        "rows": [
            {
                "source_index": int(row["source_index"]),
                "premise": row["premise"],
                "hypothesis": row["hypothesis"],
            }
            for row in test_rows
        ],
    }
    common.json_dump(TEST_INPUT_PATH, test_input)
    SEALED_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        SEALED_PATH,
        source_indices=np.asarray([row["source_index"] for row in test_rows], dtype=np.int64),
        labels=np.asarray([row["label"] for row in test_rows], dtype=np.int16),
    )
    partition_counts = {
        partition: sum(row["partition"] == partition for row in selected_train)
        for partition in common.PARTITION_QUOTAS
    }
    manifest = {
        "manifest_id": "STEP96_SNLI_STAGE0_V1",
        "date": common.DATE,
        "authority_sha256": {common.PROTOCOL.name: protocol_hash},
        "pre_outcome_statement": {
            "protocol_existed_before_data_shard_download": True,
            "dataset_card_schema_split_sizes_and_one_card_example_seen": True,
            "no_snli_candidate_prediction_or_effect_seen_before_lock": True,
            "no_validation_or_test_outcome_seen_before_lock": True,
        },
        "dataset": {
            "id": common.DATASET_ID,
            "config": common.DATASET_CONFIG,
            "revision": common.DATASET_REVISION,
            "raw_sha256": {split: common.sha256_path(path) for split, path in paths.items()},
            "valid_rows": {
                "train": len(train_rows),
                "validation": len(validation_rows),
                "test": len(test_rows),
            },
        },
        "development": {
            "file": DEVELOPMENT_PATH.name,
            "sha256": common.sha256_path(DEVELOPMENT_PATH),
            "selected_train_rows": len(selected_train),
            "validation_rows": len(validation_rows),
            "partition_counts": partition_counts,
            "stratum_audit": stratum_audit,
        },
        "test_input": {
            "file": str(TEST_INPUT_PATH.relative_to(ROOT)).replace("\\", "/"),
            "sha256": common.sha256_path(TEST_INPUT_PATH),
            "rows": len(test_rows),
            "contains_labels": False,
        },
        "sealed_outcome": {
            "file": str(SEALED_PATH.relative_to(ROOT)).replace("\\", "/"),
            "sha256": common.sha256_path(SEALED_PATH),
            "rows": len(test_rows),
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
            "step96_common.py": common.sha256_path(ROOT / "step96_common.py"),
            "step96_stage0_prepare.py": common.sha256_path(ROOT / "step96_stage0_prepare.py"),
        },
    }
    common.json_dump(MANIFEST_PATH, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
