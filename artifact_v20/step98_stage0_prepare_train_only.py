from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

import step98_common as common


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / f"STEP98_ANLI_TRAIN_ONLY_ROWS_{common.DATE}.json"
MANIFEST = ROOT / f"STEP98_STAGE0_TRAIN_ONLY_MANIFEST_{common.DATE}.json"


def normalize(row: dict[str, Any], source_index: int, round_key: str) -> dict[str, Any]:
    label = int(row["label"])
    if label not in (0, 1, 2):
        raise AssertionError({"round": round_key, "source_index": source_index, "label": label})
    return {
        "round": round_key,
        "source_index": int(source_index),
        "uid": str(row["uid"]),
        "premise": str(row["premise"]),
        "hypothesis": str(row["hypothesis"]),
        "label": label,
    }


def main() -> None:
    for path in (OUTPUT, MANIFEST):
        if path.exists():
            raise FileExistsError(path)
    if not common.PROTOCOL.is_file():
        raise FileNotFoundError(common.PROTOCOL)
    rounds: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, dict[str, int]] = {}
    ineligible: dict[str, dict[str, Any]] = {}
    total_per_label = sum(common.TRAIN_QUOTAS_PER_LABEL.values())
    for round_key in common.ROUNDS:
        dataset = load_dataset(
            common.DATASET_ID,
            common.DATASET_CONFIG,
            split=f"train_{round_key}",
            revision=common.DATASET_REVISION,
        )
        strata: dict[int, list[dict[str, Any]]] = {0: [], 1: [], 2: []}
        for index, raw in enumerate(dataset):
            row = normalize(dict(raw), index, round_key)
            strata[row["label"]].append(row)
        available_by_label = {str(label): len(rows) for label, rows in strata.items()}
        deficient = {
            label: count for label, count in available_by_label.items()
            if count < total_per_label
        }
        if deficient:
            ineligible[round_key] = {
                "reason": "fewer_than_frozen_rows_in_at_least_one_label",
                "available_by_label": available_by_label,
                "required_per_label": total_per_label,
            }
            print(json.dumps({"round": round_key, "eligible": False, **ineligible[round_key]}), flush=True)
            continue
        selected: list[dict[str, Any]] = []
        round_counts: dict[str, int] = {}
        for label, rows in strata.items():
            rows.sort(key=lambda row: (common.row_digest(round_key, row), row["source_index"]))
            offset = 0
            for partition, quota in common.TRAIN_QUOTAS_PER_LABEL.items():
                block = rows[offset : offset + quota]
                offset += quota
                for row in block:
                    row["partition"] = partition
                    row["order_digest"] = common.row_digest(round_key, row)
                    selected.append(row)
                round_counts[partition] = round_counts.get(partition, 0) + len(block)
        selected.sort(key=lambda row: (
            list(common.TRAIN_QUOTAS_PER_LABEL).index(row["partition"]),
            row["label"], row["order_digest"], row["source_index"],
        ))
        expected = {key: 3 * value for key, value in common.TRAIN_QUOTAS_PER_LABEL.items()}
        if round_counts != expected:
            raise AssertionError({"round": round_key, "counts": round_counts, "expected": expected})
        rounds[round_key] = selected
        counts[round_key] = round_counts
        print(json.dumps({"round": round_key, "train_rows_selected": len(selected)}), flush=True)
    payload = {
        "dataset_id": common.DATASET_ID,
        "config": common.DATASET_CONFIG,
        "revision": common.DATASET_REVISION,
        "dev_accessed": False,
        "salt": common.SALT,
        "quotas_per_label": common.TRAIN_QUOTAS_PER_LABEL,
        "ineligible_rounds": ineligible,
        "rounds": rounds,
    }
    common.json_dump(OUTPUT, payload)
    manifest = {
        "manifest_id": "STEP98_STAGE0_TRAIN_ONLY_V1",
        "date": common.DATE,
        "protocol_sha256": common.sha256_path(common.PROTOCOL),
        "code_sha256": {
            "step98_common.py": common.sha256_path(ROOT / "step98_common.py"),
            "step98_stage0_prepare_train_only.py": common.sha256_path(Path(__file__)),
        },
        "dataset": {
            "id": common.DATASET_ID,
            "config": common.DATASET_CONFIG,
            "revision": common.DATASET_REVISION,
            "train_splits_accessed": [f"train_{key}" for key in common.ROUNDS],
            "dev_splits_accessed": [],
            "test_splits_accessed": [],
            "builder_cache_materialized_all_splits": True,
            "experiment_code_iterated_only_requested_train_objects": True,
            "label_names": ["entailment", "neutral", "contradiction"],
        },
        "train_rows_file": OUTPUT.name,
        "train_rows_sha256": common.sha256_path(OUTPUT),
        "round_partition_counts": counts,
        "ineligible_rounds": ineligible,
        "selected_round": None,
        "sealed_dev_created": False,
    }
    common.json_dump(MANIFEST, manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
