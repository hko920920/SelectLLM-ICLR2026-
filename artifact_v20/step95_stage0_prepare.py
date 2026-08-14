from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

import step95_common as common


ROOT = Path(__file__).resolve().parent
DEVELOPMENT_PATH = ROOT / f"STEP95_MNLI_DEVELOPMENT_ROWS_{common.DATE}.json"
TEST_INPUT_PATH = ROOT / "step95_test_inputs" / "mnli_matched_test_inputs.json"
SEALED_PATH = ROOT / "external_data" / "step95_sealed" / "mnli_matched_sealed_outcomes.npz"
MANIFEST_PATH = ROOT / f"STEP95_STAGE0_MANIFEST_{common.DATE}.json"


def table_rows(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path, columns=["pairID", "premise", "hypothesis", "genre", "label"])
    rows: list[dict[str, Any]] = []
    values = table.to_pydict()
    for index in range(table.num_rows):
        label = int(values["label"][index])
        if label not in (0, 1, 2):
            continue
        rows.append(
            {
                "source_index": index,
                "pair_id": str(values["pairID"][index]),
                "premise": str(values["premise"][index]),
                "hypothesis": str(values["hypothesis"][index]),
                "genre": str(values["genre"][index]),
                "label": label,
            }
        )
    return rows


def build_development(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    strata: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(row["genre"], int(row["label"]))].append(row)
    expected_per_stratum = sum(common.PARTITION_QUOTAS.values())
    if len(strata) != 15:
        raise AssertionError({"expected_strata": 15, "observed": sorted(strata)})
    output: list[dict[str, Any]] = []
    audit: dict[str, Any] = {}
    for key in sorted(strata):
        ordered = sorted(strata[key], key=common.row_order_key)
        if len(ordered) < expected_per_stratum:
            raise AssertionError({"stratum": key, "available": len(ordered), "required": expected_per_stratum})
        cursor = 0
        partition_counts: dict[str, int] = {}
        for partition, quota in common.PARTITION_QUOTAS.items():
            selected = ordered[cursor : cursor + quota]
            cursor += quota
            partition_counts[partition] = len(selected)
            for row in selected:
                output.append({**row, "partition": partition})
        audit[f"{key[0]}::{key[1]}"] = {
            "available": len(ordered),
            "selected": expected_per_stratum,
            "partition_counts": partition_counts,
        }
    output.sort(key=lambda row: (row["partition"], row["genre"], row["label"], common.row_order_key(row)))
    return output, audit


def main() -> None:
    for output in (DEVELOPMENT_PATH, TEST_INPUT_PATH, SEALED_PATH, MANIFEST_PATH):
        if output.exists():
            raise RuntimeError(f"refusing to overwrite Step 95 Stage-0 output: {output}")
    if not common.PROTOCOL.is_file():
        raise FileNotFoundError(common.PROTOCOL)
    protocol_sha = common.sha256_path(common.PROTOCOL)
    train_path = Path(
        hf_hub_download(
            common.DATASET_ID,
            "data/train-00000-of-00001.parquet",
            repo_type="dataset",
            revision=common.DATASET_REVISION,
        )
    )
    test_path = Path(
        hf_hub_download(
            common.DATASET_ID,
            "data/validation_matched-00000-of-00001.parquet",
            repo_type="dataset",
            revision=common.DATASET_REVISION,
        )
    )
    train_rows = table_rows(train_path)
    development_rows, stratum_audit = build_development(train_rows)
    common.json_dump(
        DEVELOPMENT_PATH,
        {
            "dataset_id": common.DATASET_ID,
            "revision": common.DATASET_REVISION,
            "source_split": "train",
            "partition_quotas_per_genre_label": common.PARTITION_QUOTAS,
            "rows": development_rows,
        },
    )

    test_rows = table_rows(test_path)
    test_inputs = {
        "dataset_id": common.DATASET_ID,
        "revision": common.DATASET_REVISION,
        "source_split": "validation_matched",
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
    common.json_dump(TEST_INPUT_PATH, test_inputs)
    SEALED_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        SEALED_PATH,
        source_indices=np.asarray([row["source_index"] for row in test_rows], dtype=np.int64),
        labels=np.asarray([row["label"] for row in test_rows], dtype=np.int16),
    )

    partition_counts = {
        partition: sum(row["partition"] == partition for row in development_rows)
        for partition in common.PARTITION_QUOTAS
    }
    manifest = {
        "manifest_id": "STEP95_MNLI_STAGE0_V1",
        "date": common.DATE,
        "authority_sha256": {common.PROTOCOL.name: protocol_sha},
        "pre_outcome_statement": {
            "protocol_existed_before_dataset_file_download": True,
            "no_validation_row_or_outcome_previously_inspected": True,
            "public_cards_configs_and_revision_metadata_only": True,
        },
        "dataset": {
            "id": common.DATASET_ID,
            "revision": common.DATASET_REVISION,
            "development_split": "train",
            "test_split": "validation_matched",
            "unused_split": "validation_mismatched",
            "raw_train_sha256": common.sha256_path(train_path),
            "raw_test_sha256": common.sha256_path(test_path),
            "valid_train_rows": len(train_rows),
            "test_rows": len(test_rows),
        },
        "development": {
            "file": DEVELOPMENT_PATH.name,
            "sha256": common.sha256_path(DEVELOPMENT_PATH),
            "rows": len(development_rows),
            "partition_counts": partition_counts,
            "stratum_audit": stratum_audit,
        },
        "test_input": {
            "file": str(TEST_INPUT_PATH.relative_to(ROOT)).replace("\\", "/"),
            "sha256": common.sha256_path(TEST_INPUT_PATH),
            "rows": len(test_rows),
            "contains_labels": False,
            "contains_genre": False,
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
                "eligible_parent": spec.eligible_parent,
            }
            for spec in common.ROOT_SPECS
        ],
        "source_code_sha256": {
            "step95_common.py": common.sha256_path(ROOT / "step95_common.py"),
            "step95_stage0_prepare.py": common.sha256_path(ROOT / "step95_stage0_prepare.py"),
        },
    }
    common.json_dump(MANIFEST_PATH, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
